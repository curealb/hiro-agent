"""Interactive CLI; policy decisions and execution remain in services."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table

from hiro_agent.bootstrap import create_harness
from hiro_agent.config import Config, load_config
from hiro_agent.contracts import AgentService, RunResult
from hiro_agent.errors import HiroError, RunCancelled
from hiro_agent.session import list_sessions
from hiro_agent.tasks import TaskSummary, list_tasks, summarize_tasks

app = typer.Typer(help="Hiro：可恢复、支持只读子 Agent 的 Coding Harness", no_args_is_help=False)
console = Console(highlight=False)
HELP = "/help 查看帮助 · /tasks 最近一轮子任务 · /continue 继续未完成任务（新预算） · /exit 退出"


class Terminal:
    def __init__(self) -> None:
        self._prompt: PromptSession[str] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._transport: asyncio.ReadTransport | None = None
        self.agent: AgentService | None = None

    async def read(self, prompt: str, *, approval: bool = False) -> str:
        # A pipe transport closes its input file on EOF, but its StreamReader
        # can still contain complete task/approval lines. Drain those first.
        if self._reader is not None:
            console.print(prompt, end="", markup=False)
            buffered = await self._reader.readline()
            if not buffered:
                raise EOFError
            return buffered.decode("utf-8", errors="replace").rstrip("\r\n")
        if sys.stdin.isatty():
            if self._prompt is None:
                self._prompt = PromptSession()
            with patch_stdout():
                return await self._prompt.prompt_async(prompt, handle_sigint=not approval)
        console.print(prompt, end="", markup=False)
        # Regular redirected files cannot be registered with epoll. Their reads
        # do not wait on a producer; pipes use an asyncio reader for cancellation.
        if stat.S_ISREG(os.fstat(sys.stdin.fileno()).st_mode):
            line = sys.stdin.readline()
        else:
            if self._reader is None:
                self._reader = asyncio.StreamReader(limit=1_048_576)
                protocol = asyncio.StreamReaderProtocol(self._reader)
                self._transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                    lambda: protocol, sys.stdin
                )
            line = (await self._reader.readline()).decode("utf-8", errors="replace")
        if not line:
            raise EOFError
        return line.rstrip("\r\n")

    async def approve(self, tool: str, arguments: dict[str, Any]) -> bool:
        console.print(f"\n审批：{tool}", style="yellow", markup=False)
        console.print(json.dumps(arguments, ensure_ascii=False, indent=2), markup=False)
        try:
            answer = await self.read("允许这一次操作？[y/N] ", approval=True)
        except KeyboardInterrupt:
            if self.agent is not None:
                self.agent.cancel()
            raise RunCancelled("用户取消") from None
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def close(self) -> None:
        if self._transport:
            self._transport.close()


def observe(name: str, data: dict[str, Any]) -> None:
    role = data.get("role", "main")
    child = data.get("child_session_id") or data.get("session_id") or "未创建"
    task_id = data.get("task_id", data.get("delegation_call_id", data.get("call_id", "未知")))
    prefix = f"[任务 {task_id} · {role} · 子会话 {child}] " if role != "main" else ""
    if name == "assistant.message":
        console.print(prefix + str(data.get("text", "")), markup=False)
    elif name == "delegation.queued":
        console.print(
            f"委派 {role} · 任务 {task_id} · 子会话 待创建 · 等待", style="dim", markup=False
        )
    elif name == "delegation.started":
        console.print(
            f"委派 {role} · 任务 {task_id} · 子会话 {child} · 运行", style="dim", markup=False
        )
    elif name == "delegation.finished":
        status = data.get("status", "completed" if data.get("ok") else "failed")
        console.print(
            f"委派 {role} · 任务 {task_id} · 子会话 {child} · {task_status(str(status))}",
            markup=False,
        )
    elif name == "tool.result":
        label = data.get("name", data.get("tool", data.get("call_id", "tool")))
        status = "完成" if data.get("ok") else "未完成"
        console.print(f"  {prefix}{label}: {status}", style="dim", markup=False)


def task_status(status: str) -> str:
    return {
        "queued": "等待",
        "running": "运行",
        "completed": "完成",
        "paused": "暂停",
        "canceled": "取消",
        "failed": "失败",
        "not_started": "未开始",
        "unknown": "结果未知",
    }.get(status, status)


def show_tasks(tasks: list[TaskSummary]) -> None:
    if not tasks:
        console.print("没有子任务。", markup=False)
        return
    for task in tasks:
        console.print(
            f"任务 {task.task_id} · {task.role} · {task_status(task.status)} · "
            f"模型 {task.model_id or '未记录'}",
            markup=False,
        )
        console.print(f"  Run: {task.run_id or '未记录'}", markup=False)
        console.print(
            f"  子会话: {task.child_session_id or '未创建'} · "
            f"局部调用: 模型 {task.model_requests} / 工具 {task.tool_calls}",
            markup=False,
        )
        console.print(f"  任务: {task.task}", markup=False)
        if task.report_summary:
            console.print(f"  报告: {task.report_summary}", markup=False)
        elif task.reason:
            console.print(f"  原因: {task.reason}", markup=False)


def show_result(result: RunResult) -> None:
    if result.output:
        console.print("\n" + result.output, markup=False)
    m = result.metrics
    console.print(
        f"[{result.status.value}] {result.reason} · 模型 {m.model_requests} 次 · "
        f"工具 {m.tool_calls} 次 · {m.elapsed_ms / 1000:.1f}s",
        style="dim",
        markup=False,
    )


async def chat_loop(workspace: Path, config: Config, resume: str | None) -> None:
    terminal = Terminal()
    harness = await create_harness(workspace, config, resume, terminal.approve, observe)
    terminal.agent = harness.agent
    loop = asyncio.get_running_loop()
    console.print(f"Hiro · 工作区 {workspace.resolve()}", markup=False)
    console.print(f"Session: {harness.session.session_id}", markup=False)
    console.print(HELP, style="dim", markup=False)
    if config.model.adapter == "mock":
        extra = "，并委派探索与审查子任务" if config.subagents.enabled else ""
        console.print(
            f"Mock 演示：输入任意任务，将读取、写入并验证 hiro-demo.txt{extra}。", markup=False
        )
    if harness.recovered:
        console.print("已恢复中断调用：", style="yellow", markup=False)
        for tool_result in harness.recovered:
            console.print(
                f"  {tool_result.call_id}: {tool_result.error or '已恢复完成结果'}", markup=False
            )
            if isinstance(tool_result.value, dict) and "delegation" in tool_result.value:
                console.print(
                    f"    子会话：{tool_result.value['delegation']['child_session_id']}",
                    markup=False,
                )
        console.print(
            "未知结果可能包含已修改文件或仍存活的命令。请核实；输入 /continue 才会继续。",
            style="yellow",
            markup=False,
        )
    try:
        while True:
            try:
                text = (await terminal.read("\nhiro> ")).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/help":
                console.print(HELP, markup=False)
                continue
            if text == "/tasks":
                latest_run = next(
                    (e.run_id for e in reversed(harness.session.events) if e.type == "run.started"),
                    None,
                )
                show_tasks(
                    summarize_tasks(harness.session.events, latest_run) if latest_run else []
                )
                continue
            if text.startswith("/") and text != "/continue":
                console.print("未知命令。" + HELP, markup=False)
                continue
            # During a run SIGINT requests cooperative cancellation without
            # killing the REPL. PromptSession handles Ctrl+C while idle.
            previous_sigint = signal.getsignal(signal.SIGINT)
            loop.add_signal_handler(signal.SIGINT, harness.agent.cancel)
            try:
                result = (
                    await harness.agent.continue_run()
                    if text == "/continue"
                    else await harness.agent.run(text)
                )
                show_result(result)
            except HiroError as exc:
                console.print(str(exc), style="red", markup=False)
            finally:
                loop.remove_signal_handler(signal.SIGINT)
                signal.signal(signal.SIGINT, previous_sigint)
    finally:
        terminal.close()
        await harness.close()


def launch(
    workspace: Path,
    config_path: Path | None,
    resume: str | None,
    mock: bool,
    subagents: bool = False,
) -> None:
    if sys.platform != "linux":
        raise typer.BadParameter("此版本仅支持 Linux")
    try:
        config = load_config(config_path)
        # Environment config is deliberately limited to model routing. It never
        # raises tool permissions or injects child-process environment values.
        if "HIRO_BASE_URL" in os.environ:
            config.model.base_url = os.environ["HIRO_BASE_URL"]
        if "HIRO_MODEL" in os.environ:
            config.model.model = os.environ["HIRO_MODEL"]
        if mock:
            config.model.adapter = "mock"
        if subagents:
            config.subagents.enabled = True
        asyncio.run(chat_loop(workspace, config, resume))
    except (HiroError, ValueError, OSError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from None
    except ExceptionGroup:
        console.print("资源清理或会话关闭失败，请重新打开会话检查恢复状态。", style="red")
        raise typer.Exit(1) from None


WorkspaceOption = Annotated[Path, typer.Option("--workspace", "-w", help="工作区目录")]
ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="JSON 配置")]
ResumeOption = Annotated[str | None, typer.Option("--resume", help="打开已有 Session")]
MockOption = Annotated[bool, typer.Option("--mock", help="无需 API Key 的脚本化演示")]
SubagentsOption = Annotated[bool, typer.Option("--subagents", help="启用探索和审查只读子 Agent")]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: WorkspaceOption = Path("."),
    config: ConfigOption = None,
    resume: ResumeOption = None,
    mock: MockOption = False,
    subagents: SubagentsOption = False,
) -> None:
    ctx.obj = {
        "workspace": workspace,
        "config": config,
        "resume": resume,
        "mock": mock,
        "subagents": subagents,
    }
    if ctx.invoked_subcommand is None:
        launch(workspace, config, resume, mock, subagents)


def inherited_option(ctx: typer.Context, name: str, value: Any) -> Any:
    source = ctx.get_parameter_source(name)
    if source is not None and source.name == "DEFAULT" and isinstance(ctx.obj, dict):
        return ctx.obj.get(name, value)
    return value


@app.command()
def chat(
    ctx: typer.Context,
    workspace: WorkspaceOption = Path("."),
    config: ConfigOption = None,
    resume: ResumeOption = None,
    mock: MockOption = False,
    subagents: SubagentsOption = False,
) -> None:
    """交互聊天，或打开已有会话后等待用户输入。"""
    launch(
        inherited_option(ctx, "workspace", workspace),
        inherited_option(ctx, "config", config),
        inherited_option(ctx, "resume", resume),
        inherited_option(ctx, "mock", mock),
        inherited_option(ctx, "subagents", subagents),
    )


@app.command()
def sessions(ctx: typer.Context, workspace: WorkspaceOption = Path(".")) -> None:
    """列出当前工作区的会话，不修改日志。"""
    table = Table("Session", "状态", "更新时间")
    workspace = inherited_option(ctx, "workspace", workspace)
    for item in list_sessions(workspace.resolve()):
        table.add_row(
            str(item.get("session_id", "")),
            str(item.get("status", "")),
            str(item.get("updated_at", "")),
        )
    console.print(table)


@app.command()
def tasks(
    ctx: typer.Context,
    session: Annotated[str, typer.Option("--session", help="主会话 ID")],
    run: Annotated[str | None, typer.Option("--run", help="仅查看指定 Run")] = None,
    workspace: WorkspaceOption = Path("."),
) -> None:
    """只读查询主会话的全部委派，可按 Run 筛选。"""
    try:
        show_tasks(list_tasks(inherited_option(ctx, "workspace", workspace), session, run))
    except (HiroError, ValueError, OSError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
