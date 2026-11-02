"""Process-independent commands for persistent Bots, Tasks, and interactions."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import TypeAdapter, ValidationError

from hiro_agent.agent_communication import enable_communication
from hiro_agent.agent_management import agent_summary, manager_config
from hiro_agent.agent_presets import initialize_agents
from hiro_agent.bot_lifecycle import clear_bot_history
from hiro_agent.config import BudgetConfig, load_config, load_environment
from hiro_agent.errors import ConfigurationError, HiroError
from hiro_agent.kernel_contracts import InteractionReply, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.routines import RoutineCreate

app = typer.Typer(help="持久 Bot/Task、记忆、Skills、定时任务和独立 worker；输出为 JSON。")
Workspace = Annotated[Path, typer.Option("--workspace", "-w", file_okay=False, exists=True)]


def _command(action: Callable[[], Any]) -> None:
    try:
        value = action()
        value = TypeAdapter(Any).dump_python(value, mode="json")
        typer.echo(json.dumps(value, ensure_ascii=False, indent=2))
    except (HiroError, ValidationError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


@app.command("bot-save")
def bot_save(config: Path, workspace: Workspace = Path(".")) -> None:
    """创建/更新完整 Bot 配置；工具权限变化立即递增权限版本。"""
    _command(lambda: KernelRuntime(workspace).bots.save(load_config(config)))


@app.command("bots")
def bots(workspace: Workspace = Path(".")) -> None:
    _command(lambda: KernelRuntime(workspace).bots.list())


@app.command("agent-manager")
def agent_manager(
    workspace: Workspace = Path("."),
    source: str = "default",
    bot_id: str = "agent-manager",
    name: str = "Agent 管理员",
) -> None:
    """从已有 Bot 继承模型和权限，新增能创建永久 Agent 的管理角色。"""

    def create() -> dict[str, object]:
        runtime = KernelRuntime(workspace)
        config = manager_config(runtime.bots.get(source).config, bot_id=bot_id, name=name)
        return agent_summary(runtime.bots.save(config, expected_version=0))

    _command(create)


@app.command("bot")
def bot(bot_id: str, workspace: Workspace = Path(".")) -> None:
    _command(lambda: KernelRuntime(workspace).bots.get(bot_id))


@app.command("enable-communication")
def communication(bot_id: str, workspace: Workspace = Path(".")) -> None:
    """为已有 Agent 开启 list_agents/send_message/read_message，保留模型和其他配置。"""

    def apply() -> dict[str, object]:
        runtime = KernelRuntime(workspace)
        enable_communication(runtime.bots, bot_id)
        return agent_summary(runtime.bots.get(bot_id))

    _command(apply)


@app.command("clear-history")
def clear_history(
    bot_id: str,
    workspace: Workspace = Path("."),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="确认清空历史和记忆")] = False,
) -> None:
    """清空指定 Agent 的任务、会话和记忆，保留身份与配置。"""

    def apply() -> dict:
        if not yes:
            typer.confirm(
                f"清空 Agent {bot_id} 的全部任务、会话和记忆？无法撤销",
                default=False,
                abort=True,
            )
        return clear_bot_history(workspace, bot_id)

    _command(apply)


@app.command("bot-delete")
def bot_delete(
    bot_id: str,
    workspace: Workspace = Path("."),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="确认删除 Agent 及其历史")] = False,
) -> None:
    """删除指定 Agent、全部历史、记忆及其专属设置。"""

    def apply() -> dict:
        if not yes:
            typer.confirm(
                f"删除 Agent {bot_id} 及其全部历史和记忆？无法撤销",
                default=False,
                abort=True,
            )
        return clear_bot_history(workspace, bot_id, delete_bot=True)

    _command(apply)


@app.command("init-agents")
def init_agents(
    workspace: Workspace = Path("."),
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="新 Agent 的模型配置")
    ] = None,
    mock: Annotated[bool, typer.Option("--mock", help="以离线模型初始化")] = False,
) -> None:
    """补齐 Hiro、Agent 管理员、代码编写员；保留已有 Agent 的配置与历史。"""

    def apply() -> dict[str, list[str]]:
        from hiro_agent.cli import configured

        store = KernelRuntime(workspace).bots
        source = (
            configured(config, workspace=workspace, mock=mock)
            if config or mock or not store.list()
            else None
        )
        return initialize_agents(store, source)

    _command(apply)


@app.command("submit")
def submit(
    bot_id: str,
    text: str,
    workspace: Workspace = Path("."),
    idempotency_key: Annotated[str | None, typer.Option("--idempotency-key")] = None,
    chain_max_hops: int | None = None,
    chain_model_requests: int | None = None,
    chain_tool_calls: int | None = None,
    chain_timeout_seconds: float | None = None,
) -> None:
    """只提交持久任务并立即退出；由独立 worker 执行。"""

    def apply() -> Any:
        kernel = KernelRuntime(workspace)
        limits = kernel.bots.get(bot_id).config.budget.model_dump()
        overrides = {
            "max_model_requests": chain_model_requests,
            "max_tool_calls": chain_tool_calls,
            "timeout_seconds": chain_timeout_seconds,
        }
        limits.update({key: value for key, value in overrides.items() if value is not None})
        return kernel.tasks.submit(
            TaskSubmission(
                bot_id=bot_id,
                text=text,
                idempotency_key=idempotency_key,
                chain_max_hops=chain_max_hops,
                chain_budget=(
                    BudgetConfig.model_validate(limits)
                    if any(value is not None for value in overrides.values())
                    else None
                ),
            )
        )

    _command(apply)


@app.command("chain")
def chain(task_id: str, workspace: Workspace = Path(".")) -> None:
    """查询任务所在链的不可重置上限、累计用量和已接受交接次数。"""
    _command(lambda: KernelRuntime(workspace).tasks.chain(task_id))


@app.command("handoff-send")
def handoff_send(
    bot_id: str,
    source_task_id: str,
    target_bot_id: str,
    text: str,
    idempotency_key: str,
    workspace: Workspace = Path("."),
) -> None:
    """持久发送；worker 对账后为目标 Bot 提交任务。源 Bot 须允许 send_handoff。"""
    _command(
        lambda: KernelRuntime(workspace).handoffs.send(
            bot_id,
            source_task_id,
            target_bot_id,
            text,
            idempotency_key=idempotency_key,
        )
    )


@app.command("handoffs")
def handoffs(bot_id: str, workspace: Workspace = Path("."), outgoing: bool = False) -> None:
    """查询 Bot 的收件箱；--outgoing 查询此 Bot 发出的消息。"""
    _command(lambda: KernelRuntime(workspace).handoffs.list(bot_id, outgoing=outgoing))


@app.command("handoff-result")
def handoff_result(bot_id: str, message_id: str, workspace: Workspace = Path(".")) -> None:
    """查询回传结果及接收回传的 Task 状态；仅原发送 Bot 可读取。"""

    def inspect() -> Any:
        kernel = KernelRuntime(workspace)
        result = kernel.handoffs.result(bot_id, message_id)
        return {
            "result": result,
            "delivery_task": (
                kernel.tasks.get(result.delivery_task_id, bot_id=bot_id)
                if result is not None and result.delivery_task_id is not None
                else None
            ),
        }

    _command(inspect)


@app.command("handoff-sync")
def handoff_sync(workspace: Workspace = Path(".")) -> None:
    """手工对账已保存的发送和终态回传；不执行模型或工具。"""

    def apply() -> Any:
        KernelRuntime(workspace).handoffs.sync()
        return {"reconciled": True}

    _command(apply)


@app.command("tasks")
def tasks(workspace: Workspace = Path("."), bot_id: str | None = None) -> None:
    """查询持久任务；旧顶层 tasks 仍查询委派工具日志。"""
    _command(lambda: KernelRuntime(workspace).tasks.list(bot_id=bot_id))


@app.command("task")
def task(task_id: str, workspace: Workspace = Path(".")) -> None:
    def inspect() -> dict[str, Any]:
        kernel = KernelRuntime(workspace)
        return {
            "task": kernel.tasks.get(task_id).model_dump(mode="json"),
            "runs": [run.model_dump(mode="json") for run in kernel.tasks.runs(task_id)],
        }

    _command(inspect)


@app.command("cancel")
def cancel(task_id: str, workspace: Workspace = Path(".")) -> None:
    def apply() -> Any:
        kernel = KernelRuntime(workspace)
        result = kernel.tasks.cancel(task_id)
        if result.state == "cancelled":
            kernel.interactions.cancel_for_task(task_id)
        return result

    _command(apply)


@app.command("resolve")
def resolve(task_id: str, action: str, note: str, workspace: Workspace = Path(".")) -> None:
    """核实未知副作用后显式 retry/fail/cancel；必须填写审计说明。"""

    def apply() -> Any:
        if action not in {"retry", "fail", "cancel"}:
            raise ConfigurationError("action must be retry, fail, or cancel")
        choice: Literal["retry", "fail", "cancel"]
        choice = "retry" if action == "retry" else "fail" if action == "fail" else "cancel"
        return KernelRuntime(workspace).tasks.resolve(task_id, action=choice, note=note)

    _command(apply)


@app.command("interactions")
def interactions(
    workspace: Workspace = Path("."), bot_id: str | None = None, task_id: str | None = None
) -> None:
    _command(lambda: KernelRuntime(workspace).interactions.list(bot_id=bot_id, task_id=task_id))


@app.command("reply")
def reply(
    interaction_id: str,
    workspace: Workspace = Path("."),
    approve: bool = False,
    reject: bool = False,
    text: str | None = None,
) -> None:
    """保存答复；worker 重启后仍会读取，重复相同答复不会重复执行。"""

    def apply() -> Any:
        if int(approve) + int(reject) + int(text is not None) != 1:
            raise ConfigurationError("Choose exactly one of --approve, --reject, or --text")
        response = InteractionReply(
            approved=True if approve else False if reject else None, text=text
        )
        return KernelRuntime(workspace).interactions.reply(interaction_id, response)

    _command(apply)


@app.command("memory-add")
def memory_add(bot_id: str, content: str, workspace: Workspace = Path(".")) -> None:
    """保存此 Bot 的参考记忆；下一次模型请求读取最新内容。"""
    _command(lambda: KernelRuntime(workspace).memory.create(bot_id, content))


@app.command("memories")
def memories(
    bot_id: str, workspace: Workspace = Path("."), query: str = "", limit: int = 100
) -> None:
    _command(lambda: KernelRuntime(workspace).memory.list(bot_id, query=query, limit=limit))


@app.command("memory")
def memory(bot_id: str, memory_id: str, workspace: Workspace = Path(".")) -> None:
    _command(lambda: KernelRuntime(workspace).memory.get(bot_id, memory_id))


@app.command("memory-update")
def memory_update(
    bot_id: str,
    memory_id: str,
    content: str,
    workspace: Workspace = Path("."),
    expected_revision: int | None = None,
) -> None:
    _command(
        lambda: KernelRuntime(workspace).memory.update(
            bot_id, memory_id, content, expected_revision=expected_revision
        )
    )


@app.command("memory-delete")
def memory_delete(bot_id: str, memory_id: str, workspace: Workspace = Path(".")) -> None:
    """删除正文；摘要消费凭证保留，后续轮询不会重新生成已删除摘要。"""
    _command(lambda: KernelRuntime(workspace).memory.delete(bot_id, memory_id))


@app.command("skills")
def skills(bot_id: str, workspace: Workspace = Path("."), enabled_only: bool = False) -> None:
    _command(lambda: KernelRuntime(workspace).skills.list(bot_id, enabled_only=enabled_only))


@app.command("skill")
def skill(
    bot_id: str,
    skill_id: str,
    workspace: Workspace = Path("."),
    include_content: bool = False,
) -> None:
    """查询技能；--include-content 只读取已启用技能的完整正文。"""

    def inspect() -> Any:
        store = KernelRuntime(workspace).skills
        if include_content:
            return {"id": skill_id, "content": store.content(bot_id, skill_id)}
        record = TypeAdapter(Any).dump_python(store.get(bot_id, skill_id), mode="json")
        record["document"].pop("content")
        return record

    _command(inspect)


@app.command("skill-enable")
def skill_enable(bot_id: str, skill_id: str, workspace: Workspace = Path(".")) -> None:
    _command(lambda: KernelRuntime(workspace).skills.enable(bot_id, skill_id))


@app.command("skill-disable")
def skill_disable(bot_id: str, skill_id: str, workspace: Workspace = Path(".")) -> None:
    _command(lambda: KernelRuntime(workspace).skills.disable(bot_id, skill_id))


@app.command("skill-draft")
def skill_draft(bot_id: str, task_id: str, workspace: Workspace = Path(".")) -> None:
    """从真实成功任务生成供人工编辑的持久草稿；生成不会启用技能。"""
    _command(lambda: KernelRuntime(workspace).skills.draft_from_task(bot_id, task_id))


@app.command("routine-add")
def routine_add(
    bot_id: str,
    text: str,
    at: Annotated[str, typer.Option("--at", help="含 Z 或明确偏移量的 ISO 8601 时间")],
    workspace: Workspace = Path("."),
    interval_seconds: int | None = None,
    disabled: bool = False,
) -> None:
    """保存一次性或固定间隔计划；独立 worker 负责持续触发。"""

    def create() -> Any:
        try:
            scheduled_at = datetime.fromisoformat(at)
        except ValueError:
            raise ConfigurationError("--at must be an ISO 8601 timestamp with timezone") from None
        return KernelRuntime(workspace).routines.create(
            RoutineCreate(
                bot_id=bot_id,
                text=text,
                scheduled_at=scheduled_at,
                interval_seconds=interval_seconds,
                enabled=not disabled,
            )
        )

    _command(create)


@app.command("routines")
def routines(workspace: Workspace = Path("."), bot_id: str | None = None) -> None:
    _command(lambda: KernelRuntime(workspace).routines.list(bot_id=bot_id))


@app.command("routine")
def routine(routine_id: str, workspace: Workspace = Path("."), bot_id: str | None = None) -> None:
    def inspect() -> Any:
        store = KernelRuntime(workspace).routines
        return {
            "routine": store.get(routine_id, bot_id=bot_id),
            "triggers": store.triggers(routine_id, bot_id=bot_id),
        }

    _command(inspect)


@app.command("routine-enable")
def routine_enable(
    routine_id: str, workspace: Workspace = Path("."), bot_id: str | None = None
) -> None:
    _command(lambda: KernelRuntime(workspace).routines.enable(routine_id, bot_id=bot_id))


@app.command("routine-disable")
def routine_disable(
    routine_id: str, workspace: Workspace = Path("."), bot_id: str | None = None
) -> None:
    """阻止新触发；已持久接受的 pending 触发仍会提交，已有 Task 须另行 cancel。"""
    _command(lambda: KernelRuntime(workspace).routines.disable(routine_id, bot_id=bot_id))


@app.command("routine-tick")
def routine_tick(workspace: Workspace = Path("."), max_triggers: int = 100) -> None:
    """手动执行一轮有界触发；只提交 Task，不运行模型或工具。"""
    _command(lambda: KernelRuntime(workspace).routines.tick(max_triggers=max_triggers))


async def serve_worker(
    workspace: Path,
    *,
    once: bool,
    poll_interval: float,
    context_max_chars: int = 64_000,
    context_max_messages: int = 64,
) -> None:
    load_environment(workspace)
    kernel = (
        KernelRuntime(workspace)
        if (context_max_chars, context_max_messages) == (64_000, 64)
        else KernelRuntime(
            workspace,
            context_max_chars=context_max_chars,
            context_max_messages=context_max_messages,
        )
    )
    worker = kernel.worker(poll_interval=poll_interval)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        if once:
            await worker.recover()
            await worker.tick()
        else:
            await worker.serve(stop)
    finally:
        await worker.close()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


@app.command("worker")
def worker(
    workspace: Workspace = Path("."),
    once: bool = False,
    poll_interval: float = 1.0,
    context_max_chars: int = 64_000,
    context_max_messages: int = 64,
) -> None:
    """前台独立单 worker；SIGINT/SIGTERM 协作停止，--once 至多处理一个任务。"""
    _command(
        lambda: asyncio.run(
            serve_worker(
                workspace,
                once=once,
                poll_interval=poll_interval,
                context_max_chars=context_max_chars,
                context_max_messages=context_max_messages,
            )
        )
    )
