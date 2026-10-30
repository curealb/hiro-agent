"""Interactive CLI; policy decisions and execution remain in services."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, DummyCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.text import Text

from hiro_agent.bootstrap import Harness, create_harness
from hiro_agent.config import Config, load_config
from hiro_agent.contracts import AgentService, RunResult
from hiro_agent.errors import HiroError, RunCancelled
from hiro_agent.presentation import (
    show_assistant_message,
    show_session_header,
    show_session_history,
)
from hiro_agent.run_display import RunDisplay, usage_summary
from hiro_agent.session import delete_session, list_sessions
from hiro_agent.session_picker import SessionPicker
from hiro_agent.tasks import TaskSummary, list_tasks, summarize_tasks
from hiro_agent.terminal_ui import TERMINAL_STYLE, TerminalInterface

app = typer.Typer(help="Hiro：可恢复、支持只读子 Agent 的 Coding Harness", no_args_is_help=False)
console = Console(highlight=False)
SLASH_COMMANDS = {
    "/help": "查看帮助",
    "/tasks": "最近一轮子任务",
    "/continue": "继续未完成任务（新预算）",
    "/resume": "选择历史会话，或 /resume 会话ID",
    "/exit": "退出",
}
HELP = " · ".join(f"{command} {description}" for command, description in SLASH_COMMANDS.items())
HELP += "\n输入 / 显示命令菜单，继续输入可筛选；↑/↓ 选择，Enter 执行，Tab 补全，Esc 关闭。"


class SlashCommandCompleter(Completer):
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        prefix = document.text.lstrip()
        if (
            document.cursor_position != len(document.text)
            or not prefix.startswith("/")
            or any(character.isspace() for character in prefix)
        ):
            return
        for command, description in SLASH_COMMANDS.items():
            if command.startswith(prefix):
                yield Completion(command, start_position=-len(prefix), display_meta=description)


class Terminal:
    def __init__(self) -> None:
        self._prompt: PromptSession[str] | None = None
        self._interface: TerminalInterface | None = None
        self._reader: asyncio.StreamReader | None = None
        self._transport: asyncio.ReadTransport | None = None
        self.agent: AgentService | None = None
        self.run_display: RunDisplay | None = None

    async def read(self, prompt: str, *, approval: bool = False, selection: bool = False) -> str:
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
            if self._interface is None or self._interface.session is not self._prompt:
                self._interface = TerminalInterface(self._prompt)
            self._interface.approval = approval
            self._interface.selection = selection
            message: StyleAndTextTuples
            if approval:
                message = [("class:composer.approval", " ? " + prompt.strip() + " ")]
            elif selection:
                message = [("class:composer.prefix", " › " + prompt.strip() + " ")]
            else:
                message = [("class:composer.prefix", " › ")]
            with patch_stdout():
                return await self._prompt.prompt_async(
                    message,
                    # None preserves the previous completer on a reused session.
                    completer=(
                        DummyCompleter() if approval or selection else SlashCommandCompleter()
                    ),
                    complete_while_typing=not (approval or selection),
                    reserve_space_for_menu=0,
                    style=TERMINAL_STYLE,
                    placeholder=(
                        ""
                        if approval or selection
                        else [("class:composer.placeholder", "输入任务，或输入 / 选择命令")]
                    ),
                    handle_sigint=not approval,
                )
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
        # Rich and prompt-toolkit must never paint the terminal concurrently.
        with self.run_display.suspend() if self.run_display is not None else nullcontext():
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

    async def pick_session(
        self, items: list[dict[str, Any]], delete: Callable[[str], None]
    ) -> str | None:
        if self._prompt is None:
            self._prompt = PromptSession()
        picker = SessionPicker(
            items, delete, input=self._prompt.app.input, output=self._prompt.app.output
        )
        with patch_stdout():
            return await picker.run()

    def close(self) -> None:
        if self._transport:
            self._transport.close()


def observe(name: str, data: dict[str, Any]) -> None:
    role = data.get("role", "main")
    child = data.get("child_session_id") or data.get("session_id") or "未创建"
    task_id = data.get("task_id", data.get("delegation_call_id", data.get("call_id", "未知")))
    prefix = f"[任务 {task_id} · {role} · 子会话 {child}] " if role != "main" else ""
    if name == "assistant.message":
        text = str(data.get("text", ""))
        if prefix:
            show_assistant_message(console, text, prefix=prefix)
        else:
            show_assistant_message(console, text)
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
        show_assistant_message(console, result.output)
    m = result.metrics
    console.print(
        f"[{result.status.value}] {result.reason} · 模型 {m.model_requests} 次 · "
        f"工具 {m.tool_calls} 次 · {m.elapsed_ms / 1000:.1f}s",
        style="dim",
        markup=False,
    )
    console.print(usage_summary(m), style="dim", markup=False)


def show_sessions(items: list[dict[str, Any]], *, numbered: bool = False) -> None:
    console.print("历史会话 · Session", style="bold", markup=False)
    if not items:
        console.print("暂无会话。", markup=False)
    for index, item in enumerate(items, 1):
        # Keep the ID on its own line so a normal-width terminal can display
        # the complete value for copy/paste into `resume SESSION_ID`.
        prefix = f"{index}. " if numbered else ""
        console.print(Text(prefix + str(item["session_id"]), style="bold", overflow="fold"))
        console.print(
            Text(
                f"  {item.get('status', 'unknown')} · {item.get('updated_at', '时间未知')}",
                style="dim",
                overflow="fold",
            )
        )
        console.print(Text("  " + str(item.get("title") or "（无任务记录）"), overflow="fold"))


async def select_session(
    terminal: Terminal,
    workspace: Path,
    *,
    current_session_id: str | None = None,
    last: bool = False,
) -> str | None:
    items = [
        item
        for item in list_sessions(workspace)
        if item["session_id"] != current_session_id and item.get("task_count") != 0
    ]
    if last:
        items = [item for item in items if item.get("status") not in {"busy", "unreadable"}]
    if not items:
        console.print("当前工作区没有可恢复的历史会话。", markup=False)
        return None
    if last:
        return str(items[0]["session_id"])
    if sys.stdin.isatty():
        return await terminal.pick_session(
            items, lambda session_id: delete_session(workspace, session_id)
        )
    show_sessions(items, numbered=True)
    while True:
        try:
            choice = (
                await terminal.read("选择会话（序号或 ID；d 序号删除；回车取消）：", selection=True)
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice or choice in {"q", "/exit"}:
            return None
        parts = choice.split(maxsplit=1)
        deleting = len(parts) == 2 and parts[0] in {"d", "delete"}
        selected = parts[1] if deleting else choice
        item = next((item for item in items if selected == item["session_id"]), None)
        if (
            item is None
            and selected.isascii()
            and selected.isdecimal()
            and len(selected) <= len(str(len(items)))
        ):
            index = int(selected)
            if 1 <= index <= len(items):
                item = items[index - 1]
        if item is None:
            console.print("请输入列表中的序号或完整 Session ID，回车取消。", markup=False)
            continue
        session_id = str(item["session_id"])
        if not deleting:
            return session_id
        console.print(
            f"删除会话 {session_id}：{item.get('title') or '（无任务记录）'}\n"
            "将删除此会话及关联子会话的历史记录，无法撤销。",
            markup=False,
        )
        try:
            answer = await terminal.read("确认删除？[y/N] ", selection=True)
        except (EOFError, KeyboardInterrupt):
            return None
        if answer.strip().lower() not in {"y", "yes"}:
            continue
        try:
            delete_session(workspace, session_id)
        except (HiroError, OSError) as exc:
            console.print(str(exc), style="red", markup=False)
            continue
        console.print(f"已删除会话 {session_id}。", markup=False)
        items.remove(item)
        if not items:
            console.print("没有其他历史会话。", markup=False)
            return None
        show_sessions(items, numbered=True)


def show_opened_session(
    harness: Harness, workspace: Path, config: Config, *, resumed: bool
) -> None:
    show_session_header(
        console, workspace=workspace, session_id=harness.session.session_id, config=config
    )
    console.print(
        f"会话自动保存：{workspace.resolve() / '.hiro-agent' / 'sessions'}\n"
        "输入 /resume 选择历史会话；下次启动可运行 hiro-agent resume。",
        style="dim",
        markup=False,
    )
    if resumed:
        show_session_history(console, harness.session.events)
        console.print("已恢复会话；输入新任务，或用 /continue 继续未完成任务。", markup=False)
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


async def chat_loop(
    workspace: Path,
    config: Config,
    resume: str | None,
    *,
    select_resume: bool = False,
    last: bool = False,
) -> None:
    terminal = Terminal()
    display = RunDisplay(console)
    terminal.run_display = display

    def on_event(name: str, data: dict[str, Any]) -> None:
        display.observe(name, data)
        observe(name, data)

    harness: Harness | None = None
    try:
        if select_resume or last:
            resume = await select_session(terminal, workspace, last=last)
            if resume is None:
                return
        harness = await create_harness(workspace, config, resume, terminal.approve, on_event)
        terminal.agent = harness.agent
        loop = asyncio.get_running_loop()
        show_opened_session(harness, workspace, config, resumed=resume is not None)
        if not console.is_terminal:
            console.print(HELP, style="dim", markup=False)
        if config.model.adapter == "mock":
            extra = "，并委派探索与审查子任务" if config.subagents.enabled else ""
            console.print(
                f"Mock 演示：输入任意任务，将读取、写入并验证 hiro-demo.txt{extra}。", markup=False
            )
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
            if text.split(maxsplit=1)[0] == "/resume":
                arguments = text.split(maxsplit=1)
                target = arguments[1] if len(arguments) == 2 else None
                try:
                    if target is None or target == "--last":
                        target = await select_session(
                            terminal,
                            workspace,
                            current_session_id=harness.session.session_id,
                            last=target == "--last",
                        )
                    if target is None:
                        continue
                    if target == harness.session.session_id:
                        console.print("当前已在此会话中。", markup=False)
                        continue
                    restored = await create_harness(
                        workspace, config, target, terminal.approve, on_event
                    )
                except (HiroError, ValueError, OSError) as exc:
                    console.print(str(exc), style="red", markup=False)
                    continue
                previous = harness
                harness = restored
                terminal.agent = harness.agent
                await previous.close()
                show_opened_session(harness, workspace, config, resumed=True)
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
                try:
                    display.start()
                    result = (
                        await harness.agent.continue_run()
                        if text == "/continue"
                        else await harness.agent.run(text)
                    )
                finally:
                    display.stop()
                if display.interrupted_output:
                    show_assistant_message(console, display.interrupted_output)
                    console.print("[输出未完成，未写入模型对话历史]", style="dim", markup=False)
                show_result(result)
            except HiroError as exc:
                console.print(str(exc), style="red", markup=False)
            finally:
                loop.remove_signal_handler(signal.SIGINT)
                signal.signal(signal.SIGINT, previous_sigint)
    finally:
        display.stop()
        terminal.close()
        if harness is not None:
            await harness.close()


def launch(
    workspace: Path,
    config_path: Path | None,
    resume: str | None,
    mock: bool,
    subagents: bool = False,
    *,
    select_resume: bool = False,
    last: bool = False,
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
        asyncio.run(chat_loop(workspace, config, resume, select_resume=select_resume, last=last))
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
def resume(
    ctx: typer.Context,
    session_id: Annotated[
        str | None, typer.Argument(help="要恢复的 Session ID，省略时选择")
    ] = None,
    last: Annotated[bool, typer.Option("--last", help="恢复最近有任务记录的可读取会话")] = False,
    workspace: WorkspaceOption = Path("."),
    config: ConfigOption = None,
    mock: MockOption = False,
    subagents: SubagentsOption = False,
) -> None:
    """选择历史会话或按 ID 恢复，显示历史后等待输入。"""
    session_id = session_id or (ctx.obj.get("resume") if isinstance(ctx.obj, dict) else None)
    if last and session_id is not None:
        raise typer.BadParameter("Session ID 和 --last 不能同时使用")
    launch(
        inherited_option(ctx, "workspace", workspace),
        inherited_option(ctx, "config", config),
        session_id,
        inherited_option(ctx, "mock", mock),
        inherited_option(ctx, "subagents", subagents),
        select_resume=session_id is None,
        last=last,
    )


@app.command()
def sessions(ctx: typer.Context, workspace: WorkspaceOption = Path(".")) -> None:
    """列出当前工作区的会话，不修改日志。"""
    workspace = inherited_option(ctx, "workspace", workspace)
    show_sessions(list_sessions(workspace.resolve()))


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
