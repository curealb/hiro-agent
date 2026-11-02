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
        for task in tasks:
            entries.append(ChatEntry("user", task.text, task.id))
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

    def submit(self, bot_id: str, text: str, *, idempotency_key: str | None = None) -> TaskRecord:
        if not text.strip():
            raise ConfigurationError("任务内容不能为空")
        return self.kernel.tasks.submit(
            TaskSubmission(bot_id=bot_id, text=text, idempotency_key=idempotency_key)
        )

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

    def append(timestamp: datetime, role: str, text: str) -> None:
        if text.strip():
            timeline.append((timestamp, len(timeline), ChatEntry(role, text, task.id)))

    for event in events:
        if event.type == "assistant.message":
            message = AssistantMessage.model_validate(event.data["message"])
            calls.update((call.id, call) for call in message.tool_calls)
            if not event.data.get("prepared_operation"):
                append(event.timestamp, "assistant", message.text or "")
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
            append(event.timestamp, "tool", f"{name} · {status}\n{detail[:2000]}".rstrip())
        elif event.type == "interaction.answer":
            # DB answers appear as soon as saved, before a worker resumes. A
            # journal answer is the fallback if its interaction row is missing.
            if event.data.get("interaction_id") not in interaction_ids:
                text = event.data.get("text")
                if isinstance(text, str):
                    append(event.timestamp, "user", text)
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
