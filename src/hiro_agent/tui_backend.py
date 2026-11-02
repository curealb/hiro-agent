"""Durable, Bot-scoped display queries and commands for the terminal interface."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from hiro_agent.agent_presets import initialize_agents
from hiro_agent.bot_lifecycle import clear_bot_history
from hiro_agent.config import Config
from hiro_agent.contracts import AssistantMessage, SessionEvent, ToolCall, ToolResult
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    InteractionRecord,
    InteractionReply,
    InteractionState,
    TaskRecord,
    TaskState,
    TaskSubmission,
)
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.session import read_session_snapshot
from hiro_agent.ui_commands import addressed_agent


@dataclass(frozen=True)
class BotSummary:
    bot_id: str
    name: str
    model: str
    status: str
    preview: str
    pending_count: int
    task_count: int


@dataclass(frozen=True)
class ChatEntry:
    role: str
    text: str
    task_id: str
    entry_id: str = ""
    preview: Literal["streaming", "interrupted"] | None = None
    sender_bot_id: str | None = None
    sender_name: str | None = None


@dataclass(frozen=True)
class Conversation:
    bot_id: str
    entries: tuple[ChatEntry, ...]
    tasks: tuple[TaskRecord, ...]
    interactions: tuple[InteractionRecord, ...]
    error: str | None = None


@dataclass(frozen=True)
class _CachedSnapshot:
    signature: tuple[int, ...]
    events: tuple[SessionEvent, ...]
    error: str | None = None


def _live_interaction(record: InteractionRecord, tasks: dict[str, TaskRecord]) -> bool:
    task = tasks.get(record.request.task_id)
    expected = (
        TaskState.WAITING_APPROVAL if record.request.kind == "approval" else TaskState.WAITING_INPUT
    )
    return (
        record.state == InteractionState.PENDING
        and task is not None
        and task.bot_id == record.request.bot_id
        and task.state == expected
        and not task.cancel_requested
        and task.interaction_id == record.id
    )


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TuiBackend:
    """Use service APIs for authority; cache only immutable display snapshots."""

    def __init__(self, workspace: Path, *, kernel: KernelRuntime | None = None) -> None:
        self.workspace = workspace.resolve(strict=True)
        self.kernel = kernel if kernel is not None else KernelRuntime(self.workspace)
        if self.kernel.workspace != self.workspace:
            raise ConfigurationError("TUI 与内核必须使用相同工作区")
        self._snapshots: OrderedDict[str, _CachedSnapshot] = OrderedDict()
        self._worker: subprocess.Popen[bytes] | None = None

    def list_bots(self) -> list[BotSummary]:
        tasks = self.kernel.tasks.list()
        interactions = self.kernel.interactions.list(state=InteractionState.PENDING)
        by_task = {task.id: task for task in tasks}
        summaries: list[BotSummary] = []
        for bot in self.kernel.bots.list():
            owned = [task for task in tasks if task.bot_id == bot.bot_id]
            latest = max(owned, key=lambda task: (task.updated_at, task.id), default=None)
            status = latest.state.value if latest is not None else "idle"
            for state in (
                TaskState.WAITING_APPROVAL,
                TaskState.WAITING_INPUT,
                TaskState.RUNNING,
                TaskState.QUEUED,
            ):
                if any(task.state == state for task in owned):
                    status = state.value
                    break
            assert bot.config.bot is not None
            summaries.append(
                BotSummary(
                    bot_id=bot.bot_id,
                    name=bot.config.bot.name,
                    model=bot.config.model.model or bot.config.model.adapter,
                    status=status,
                    preview=_short(latest.output or latest.text) if latest is not None else "",
                    pending_count=sum(
                        record.request.bot_id == bot.bot_id and _live_interaction(record, by_task)
                        for record in interactions
                    ),
                    task_count=len(owned),
                )
            )
        return summaries

    def _snapshot(self, session_id: str) -> tuple[SessionEvent, ...]:
        directory = self.workspace / ".hiro-agent" / "sessions"
        try:
            if directory.parent.is_symlink() or directory.is_symlink():
                raise PersistenceError("会话日志路径不安全")
            metadata = (directory / f"{session_id}.jsonl").lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise PersistenceError("会话日志必须为普通文件")
        except OSError as exc:
            raise PersistenceError("会话日志不存在或无法访问") from exc
        signature = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        cached = self._snapshots.get(session_id)
        if cached is None or cached.signature != signature:
            try:
                cached = _CachedSnapshot(
                    signature, tuple(read_session_snapshot(self.workspace, session_id))
                )
            except PersistenceError as exc:
                cached = _CachedSnapshot(signature, (), str(exc))
            self._snapshots[session_id] = cached
        self._snapshots.move_to_end(session_id)
        while len(self._snapshots) > 64:
            self._snapshots.popitem(last=False)
        if cached.error is not None:
            raise PersistenceError(cached.error)
        return cached.events

    def conversation(self, bot_id: str) -> Conversation:
        self.kernel.bots.get(bot_id)
        all_tasks = self.kernel.tasks.list(bot_id=bot_id)
        tasks = tuple(all_tasks[-30:])
        interactions = self.kernel.interactions.list(bot_id=bot_id)
        by_task = {task.id: task for task in all_tasks}
        pending = tuple(record for record in interactions if _live_interaction(record, by_task))
        entries: list[ChatEntry] = []
        errors: list[str] = []
        names = {
            record.bot_id: record.config.bot.name
            for record in self.kernel.bots.list()
            if record.config.bot is not None
        }
        for task in tasks:
            entries.append(self._task_input(task, names))
            events: tuple[SessionEvent, ...] = ()
            # A task without any Run has no journal yet, including cancelled queues.
            if self.kernel.tasks.runs(task.id):
                try:
                    events = self._snapshot(task.session_id)
                except PersistenceError as exc:
                    errors.append(f"任务 {task.id[:8]}：{exc}")
            related = [record for record in interactions if record.request.task_id == task.id]
            projected = _task_entries(task, events, related)
            entries.extend(projected)
            if task.output and not any(
                entry.role == "assistant" and entry.text.strip() == task.output.strip()
                for entry in projected
            ):
                entries.append(ChatEntry("assistant", task.output, task.id))
            status = {
                TaskState.FAILED: "任务失败",
                TaskState.CANCELLED: "任务已取消",
                TaskState.NEEDS_RECONCILIATION: (
                    "任务待核实（需要人工明确核实后处理，不会自动重新执行）"
                ),
            }.get(task.state)
            if status is not None:
                reason = task.reason.strip()
                if reason and reason not in {status, task.output.strip()}:
                    status += "：" + reason
                entries.append(ChatEntry("system", status, task.id))
        return Conversation(bot_id, tuple(entries), tasks, pending, "\n".join(errors) or None)

    def _task_input(self, task: TaskRecord, names: dict[str, str]) -> ChatEntry:
        if task.source not in {"handoff", "handoff_result"}:
            original = task.metadata.get("addressed_input")
            text = original if isinstance(original, str) else task.text
            return ChatEntry("user", text, task.id)
        sender_id = None
        # Message tasks point to their sender's Task; result tasks point to the
        # responding Bot's Task. Display identity comes from this durable link,
        # never a name or "from=" field in model-controlled message text.
        if task.parent_task_id is not None:
            try:
                sender_id = self.kernel.tasks.get(task.parent_task_id).bot_id
            except ConfigurationError:
                pass  # The originating Bot's history may have been cleared.
        if sender_id is None and task.source == "handoff":
            saved_id = task.metadata.get("source_bot_id")
            if isinstance(saved_id, str) and saved_id:
                sender_id = saved_id
        return ChatEntry(
            "agent",
            task.text,
            task.id,
            sender_bot_id=sender_id,
            sender_name=names.get(sender_id) if sender_id is not None else None,
        )

    def submit(self, bot_id: str, text: str, *, idempotency_key: str | None = None) -> TaskRecord:
        if not text.strip():
            raise ConfigurationError("任务内容不能为空")
        metadata: dict[str, object] = {}
        if text.lstrip().startswith("/"):
            address = addressed_agent(
                text,
                (
                    (record.bot_id, record.config.bot.name)
                    for record in self.kernel.bots.list()
                    if record.config.bot is not None
                ),
                (),
            )
            assert address is not None
            if not self.kernel.bots.authority(bot_id).allows("send_message"):
                raise ConfigurationError("当前 Agent 没有 send_message 权限，无法发起通信")
            # Stable across a repeated submit from the same client request.
            correlation_id = "comm-" + uuid4().hex
            if idempotency_key is not None:
                import hashlib

                correlation_id = (
                    "comm-"
                    + hashlib.sha256((bot_id + ":" + idempotency_key).encode()).hexdigest()[:24]
                )
            metadata = {
                "addressed_input": text,
                "target_agent_ids": [target.bot_id for target in address.targets],
                "correlation_id": correlation_id,
            }
            if len(address.targets) == 1:
                metadata["target_agent_id"] = address.targets[0].bot_id
            requests = [
                {
                    "target_agent_id": target.bot_id,
                    "name": target.name,
                    "correlation_id": (
                        correlation_id if len(address.targets) == 1 else f"{correlation_id}-{i + 1}"
                    ),
                }
                for i, target in enumerate(address.targets)
            ]
            metadata["target_correlations"] = {
                str(request["target_agent_id"]): str(request["correlation_id"])
                for request in requests
            }
            text = (
                "用户指定本次通信对象（必须逐一联系，不能只选择其中一个）：\n"
                + json.dumps(requests, ensure_ascii=False)
                + "\n"
                f"本次通信/测试标识已自动生成：{correlation_id}。"
                "用户已经确认目标及发起通信，不需要再要求用户提供唯一标识或重复确认。\n"
                "分别使用真实 send_message 接口联系每个目标，"
                "每条消息的 correlation_id 使用该目标对应的上述标识，"
                "idempotency_key 同样使用该目标标识；根据用户要求分工，"
                "根据用户内容和双方协议组织 payload。若是 ping/pong 测试，"
                "协议中的 id/test_id 也使用该目标标识。全部发送后结束本阶段，"
                "由系统回传后使用 read_message 分别核验真实回复并汇总。"
                "部分回复时明确标注其他目标尚未完成；"
                "只有所有目标均有匹配回复才能报告全部完成，失败与未发送的目标也须列出。\n"
                f"用户的消息内容：\n{address.text}"
            )
        return self.kernel.tasks.submit(
            TaskSubmission(
                bot_id=bot_id, text=text, idempotency_key=idempotency_key, metadata=metadata
            )
        )

    def clear_history(self, bot_id: str) -> dict:
        result = clear_bot_history(self.workspace, bot_id)
        self._snapshots.clear()
        return result

    def delete_bot(self, bot_id: str) -> dict:
        result = clear_bot_history(self.workspace, bot_id, delete_bot=True)
        self._snapshots.clear()
        return result

    def initialize_agents(self) -> dict[str, list[str]]:
        source = None
        if not self.kernel.bots.list():
            source = Config()
            if "HIRO_MODEL" in os.environ:
                source.model.model = os.environ["HIRO_MODEL"]
            if "HIRO_BASE_URL" in os.environ:
                source.model.base_url = os.environ["HIRO_BASE_URL"]
        return initialize_agents(self.kernel.bots, source)

    def reply(self, bot_id: str, interaction_id: str, reply: InteractionReply) -> InteractionRecord:
        record = self.kernel.interactions.get(interaction_id, bot_id=bot_id)
        task = self.kernel.tasks.get(record.request.task_id, bot_id=bot_id)
        expected = (
            TaskState.WAITING_APPROVAL
            if record.request.kind == "approval"
            else TaskState.WAITING_INPUT
        )
        if (
            task.interaction_id != record.id
            or task.state != expected
            or task.cancel_requested
            or record.state not in (InteractionState.PENDING, InteractionState.ANSWERED)
        ):
            raise ConfigurationError("交互已过期或任务不再等待此答复")
        if record.request.kind == "approval" and record.request.operation != task.pending_operation:
            raise ConfigurationError("审批操作与任务当前保存的操作不一致")
        return self.kernel.interactions.reply(interaction_id, reply)

    def cancel(self, bot_id: str, task_id: str) -> TaskRecord:
        task = self.kernel.tasks.cancel(task_id, bot_id=bot_id)
        if task.state == TaskState.CANCELLED:
            self.kernel.interactions.cancel_for_task(task.id)
        return task

    def resolve_cancel(self, bot_id: str, task_id: str, note: str) -> TaskRecord:
        self.kernel.tasks.get(task_id, bot_id=bot_id)
        task = self.kernel.tasks.resolve(task_id, action="cancel", note=note)
        self.kernel.interactions.cancel_for_task(task.id)
        return task

    def worker_running(self) -> bool:
        """Probe the existing lock inode; never instantiate or recover a worker."""
        descriptor: int | None = None
        try:
            if (self.workspace / ".hiro-agent").is_symlink():
                raise PersistenceError("Worker 锁目录不安全")
            descriptor = os.open(
                self.workspace / ".hiro-agent" / "worker.lock",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PersistenceError("Worker 锁必须为普通文件")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise PersistenceError("无法检查 Worker 锁") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def start_worker(self) -> subprocess.Popen[bytes] | None:
        if self.worker_running():
            return None
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        log_path = self.workspace / ".hiro-agent" / "worker.log"
        descriptor = os.open(
            log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PersistenceError("Worker 日志必须为普通文件")
            self._worker = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "hiro_agent.cli",
                    "kernel",
                    "worker",
                    "--workspace",
                    str(self.workspace),
                ],
                stdin=subprocess.DEVNULL,
                stdout=descriptor,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(descriptor)
        return self._worker


def _task_entries(
    task: TaskRecord,
    events: tuple[SessionEvent, ...],
    interactions: list[InteractionRecord],
) -> list[ChatEntry]:
    timeline: list[tuple[datetime, int, ChatEntry]] = []
    calls: dict[str, ToolCall] = {}
    shown_calls: set[str] = set()
    interaction_ids = {record.id for record in interactions}
    previews: dict[str, tuple[SessionEvent, list[str]]] = {}
    ended_attempts: set[str] = set()
    ended_runs: set[str | None] = set()

    def append(timestamp: datetime, role: str, text: str, *, entry_id: str = "") -> None:
        if text.strip():
            timeline.append((timestamp, len(timeline), ChatEntry(role, text, task.id, entry_id)))

    for event in events:
        if event.type == "assistant.message":
            previews.pop(event.data.get("attempt_id", ""), None)
            message = AssistantMessage.model_validate(event.data["message"])
            calls.update((call.id, call) for call in message.tool_calls)
            if not event.data.get("prepared_operation"):
                append(event.timestamp, "assistant", message.text or "")
        elif event.type == "assistant.delta":
            attempt_id = event.data["attempt_id"]
            if attempt_id not in previews:
                previews[attempt_id] = (event, [])
            previews[attempt_id][1].append(event.data["text"])
        elif event.type in {"model.interrupted", "model.failed"}:
            ended_attempts.add(event.data["attempt_id"])
        elif event.type == "run.ended":
            ended_runs.add(event.run_id)
        elif event.type in {"tool.called", "tool.started"}:
            call = ToolCall.model_validate(event.data["call"])
            if call.id not in shown_calls and call.name != "ask_user":
                append(event.timestamp, "tool", f"{call.name} · 执行请求")
                shown_calls.add(call.id)
        elif event.type == "tool.result":
            result = ToolResult.model_validate(event.data)
            name = calls[result.call_id].name
            if name == "ask_user":
                continue
            status = "完成" if result.ok else result.error or result.outcome
            value = result.message or result.value
            detail = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
                if value is not None
                else ""
            )
            append(
                event.timestamp,
                "tool",
                f"{name} · {status}\n{detail}".rstrip(),
                entry_id=f"{task.id}:{event.session_id}:{event.seq}",
            )
        elif event.type == "interaction.answer":
            # DB answers appear as soon as saved, before a worker resumes. A
            # journal answer is the fallback if its interaction row is missing.
            if event.data.get("interaction_id") not in interaction_ids:
                text = event.data.get("text")
                if isinstance(text, str):
                    append(event.timestamp, "user", text)
    for attempt_id, (first, chunks) in previews.items():
        text = "".join(chunks)
        if not text.strip():
            continue
        live = (
            task.state == TaskState.RUNNING
            and first.run_id == task.active_run_id
            and first.run_id not in ended_runs
            and attempt_id not in ended_attempts
        )
        timeline.append(
            (
                first.timestamp,
                len(timeline),
                ChatEntry(
                    "assistant", text, task.id, preview="streaming" if live else "interrupted"
                ),
            )
        )
    for record in interactions:
        if record.request.kind == "question":
            append(record.created_at, "assistant", record.request.prompt)
            if record.reply is not None and record.reply.text is not None:
                append(record.answered_at or record.created_at, "user", record.reply.text)
        else:
            append(record.created_at, "system", f"等待审批：{record.request.prompt}")
            if record.reply is not None:
                append(
                    record.answered_at or record.created_at,
                    "user",
                    "已批准操作" if record.reply.approved else "已拒绝操作",
                )
    timeline.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in timeline]
