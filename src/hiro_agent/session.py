"""Append-only conversation journals and conservative crash recovery."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import os
import re
import stat
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hiro_agent.contracts import (
    AssistantMessage,
    DelegationLink,
    Message,
    RunMetrics,
    RunStatus,
    SessionEvent,
    SessionService,
    ToolCall,
    ToolResult,
    Usage,
    usage_counts,
)
from hiro_agent.errors import PersistenceError

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _session_id(value: str | None) -> str:
    value = uuid.uuid4().hex if value is None else value
    if not isinstance(value, str) or not _SESSION_ID.fullmatch(value):
        raise PersistenceError("Invalid session ID.")
    return value


def _snapshot(event: SessionEvent) -> SessionEvent:
    return event.model_copy(deep=True)


def _error(message: str) -> PersistenceError:
    return PersistenceError(f"Invalid session journal: {message}.")


@dataclass
class _Call:
    call: ToolCall
    run_id: str | None
    called: bool = False
    started: bool = False
    result: ToolResult | None = None
    delegation: DelegationLink | None = None
    delegation_result: ToolResult | None = None


@dataclass
class _State:
    calls: dict[str, _Call] = field(default_factory=dict)
    attempts: dict[str, str | None] = field(default_factory=dict)
    ended_attempts: set[str] = field(default_factory=set)
    runs: dict[str, bool] = field(default_factory=dict)
    child_link: DelegationLink | None = None
    child_finished: bool = False

    def pending(self) -> list[_Call]:
        return [call for call in self.calls.values() if call.result is None]


def _validate_events(events: list[SessionEvent], session_id: str) -> _State:
    """Validate references as well as the event envelope, without changing history."""
    state = _State()
    created = False
    for expected_seq, event in enumerate(events, 1):
        if event.session_id != session_id or event.seq != expected_seq:
            raise _error("session ID or event sequence mismatch")
        if not event.type or event.timestamp.utcoffset() is None:
            raise _error("missing event type or timezone")
        data = event.data
        if event.run_id is not None and event.type != "run.started":
            if event.run_id not in state.runs:
                raise _error("event references an unknown run")
        try:
            match event.type:
                case "session.created":
                    if created or expected_seq != 1 or not isinstance(data.get("workspace"), str):
                        raise _error("invalid session.created event")
                    created = True
                case "prompt.system" | "task.received" | "task.continued":
                    if not isinstance(data.get("text"), str):
                        raise _error("message text must be a string")
                    if event.type in {"task.received", "task.continued"} and state.pending():
                        raise _error("user message before outstanding tool results")
                case "run.started":
                    if not event.run_id or event.run_id in state.runs:
                        raise _error("duplicate or missing run ID")
                    if (
                        state.child_link is not None
                        and event.run_id != state.child_link.child_run_id
                    ):
                        raise _error("child run does not match its delegation linkage")
                    state.runs[event.run_id] = False
                case "run.ended":
                    if event.run_id not in state.runs or state.runs[event.run_id]:
                        raise _error("run.ended does not reference an open run")
                    RunStatus(data["status"])
                    if not isinstance(data["reason"], str):
                        raise _error("run reason must be a string")
                    RunMetrics.model_validate(data["metrics"])
                    state.runs[event.run_id] = True
                case "model.requested":
                    attempt_id = data.get("attempt_id")
                    if not isinstance(attempt_id, str) or not attempt_id:
                        raise _error("missing model attempt ID")
                    if attempt_id in state.attempts:
                        raise _error("duplicate model attempt ID")
                    state.attempts[attempt_id] = event.run_id
                case "model.failed" | "model.interrupted":
                    _end_attempt(state, data.get("attempt_id"), event.run_id)
                case "assistant.message":
                    if state.pending():
                        raise _error("assistant message before outstanding tool results")
                    message = AssistantMessage.model_validate(data["message"])
                    if "attempt_id" in data:
                        _end_attempt(state, data["attempt_id"], event.run_id)
                    if "finish_reason" in data and not isinstance(data["finish_reason"], str):
                        raise _error("invalid model finish reason")
                    if data.get("usage") is not None:
                        Usage.model_validate(data["usage"])
                    for call in message.tool_calls:
                        if call.id in state.calls:
                            raise _error("duplicate tool call ID")
                        state.calls[call.id] = _Call(call, event.run_id)
                case "tool.called" | "tool.started":
                    if "accepted" in data and type(data["accepted"]) is not bool:
                        raise _error("tool admission flag must be a boolean")
                    call = ToolCall.model_validate(data["call"])
                    recorded = state.calls.get(call.id)
                    if recorded is None or recorded.call != call or recorded.run_id != event.run_id:
                        raise _error("tool event does not reference its assistant call")
                    if recorded.result is not None:
                        raise _error("tool event after terminal result")
                    if event.type == "tool.called":
                        if recorded.called or recorded.started:
                            raise _error("duplicate or late tool.called")
                        recorded.called = True
                    else:
                        if recorded.started:
                            raise _error("duplicate tool.started")
                        recorded.started = True
                case "delegation.started" | "subagent.created":
                    link = DelegationLink.model_validate(data)
                    _session_id(link.parent_session_id)
                    _session_id(link.child_session_id)
                    if not link.call_id or not link.parent_run_id or not link.child_run_id:
                        raise _error("missing delegation reference")
                    if link.parent_session_id == link.child_session_id:
                        raise _error("delegation cannot reference its own session")
                    if event.type == "subagent.created":
                        if (
                            link.child_session_id != session_id
                            or event.run_id is not None
                            or state.child_link is not None
                            or state.runs
                        ):
                            raise _error("invalid child session linkage")
                        state.child_link = link
                    else:
                        recorded = state.calls.get(link.call_id)
                        if (
                            link.parent_session_id != session_id
                            or link.parent_run_id != event.run_id
                            or recorded is None
                            or recorded.run_id != event.run_id
                            or recorded.call.name != "delegate_task"
                            or not recorded.started
                            or recorded.result is not None
                            or recorded.delegation is not None
                            or state.child_link is not None
                        ):
                            raise _error("invalid delegation linkage")
                        recorded.delegation = link
                case "delegation.finished" | "subagent.finished":
                    result = ToolResult.model_validate(data["result"])
                    completion_metrics = RunMetrics.model_validate(data["metrics"])
                    if "recovered" in data and type(data["recovered"]) is not bool:
                        raise _error("completion recovery flag must be a boolean")
                    if (
                        not isinstance(data.get("call_id"), str)
                        or data["call_id"] != result.call_id
                    ):
                        raise _error("completion call reference mismatch")
                    if event.type == "subagent.finished":
                        if (
                            state.child_link is None
                            or state.child_link.call_id != result.call_id
                            or state.child_link.child_run_id != event.run_id
                            or state.child_finished
                            or not state.runs.get(event.run_id, False)
                        ):
                            raise _error("invalid child completion reference")
                        ended = next(
                            previous
                            for previous in reversed(events[: expected_seq - 1])
                            if previous.type == "run.ended" and previous.run_id == event.run_id
                        )
                        if completion_metrics != RunMetrics.model_validate(ended.data["metrics"]):
                            raise _error("child completion metrics differ from terminal run")
                        state.child_finished = True
                    else:
                        recorded = state.calls.get(result.call_id)
                        if (
                            recorded is None
                            or recorded.run_id != event.run_id
                            or recorded.delegation is None
                            or recorded.delegation_result is not None
                        ):
                            raise _error("invalid delegation completion reference")
                        recorded.delegation_result = result
                case "tool.result":
                    result = ToolResult.model_validate(data)
                    recorded = state.calls.get(result.call_id)
                    if recorded is None or recorded.run_id != event.run_id:
                        raise _error("tool result does not reference its assistant call")
                    if recorded.result is not None:
                        raise _error("duplicate tool result")
                    if (
                        recorded.delegation_result is not None
                        and result != recorded.delegation_result
                        and not (
                            not result.ok
                            and result.outcome == "unknown"
                            and result.error
                            in {"cancelled", "budget_exceeded", "TOOL_OUTCOME_UNKNOWN"}
                        )
                    ):
                        raise _error("tool result differs from durable delegation completion")
                    recorded.result = result
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise _error("invalid event payload") from exc
    return state


def _end_attempt(state: _State, attempt_id: Any, run_id: str | None) -> None:
    if not isinstance(attempt_id, str) or attempt_id not in state.attempts:
        raise _error("model result references an unknown attempt")
    if attempt_id in state.ended_attempts or state.attempts[attempt_id] != run_id:
        raise _error("model attempt already ended or belongs to a different run")
    state.ended_attempts.add(attempt_id)


def _messages(events: list[SessionEvent], session_id: str) -> list[Message]:
    if _validate_events(events, session_id).pending():
        raise _error("outstanding tool calls require recovery before model projection")
    messages: list[Message] = []
    for event in events:
        match event.type:
            case "prompt.system":
                messages.append(Message(role="system", content=event.data["text"]))
            case "task.received" | "task.continued":
                messages.append(
                    Message(
                        role="user",
                        content=event.data["text"],
                        continuation=event.type == "task.continued",
                    )
                )
            case "assistant.message":
                assistant = AssistantMessage.model_validate(event.data["message"])
                messages.append(
                    Message(
                        role="assistant", content=assistant.text, tool_calls=assistant.tool_calls
                    )
                )
            case "tool.result":
                result = ToolResult.model_validate(event.data)
                messages.append(
                    Message(
                        role="tool", content=result.model_dump_json(), tool_call_id=result.call_id
                    )
                )
    return messages


def _new_event(
    session_id: str,
    events: list[SessionEvent],
    event_type: str,
    data: dict[str, Any],
    run_id: str | None,
) -> SessionEvent:
    try:
        event = SessionEvent(
            session_id=session_id, run_id=run_id, seq=len(events) + 1, type=event_type, data=data
        )
        # JSON round-trip both rejects unserializable values and detaches caller-owned objects.
        event = SessionEvent.model_validate_json(event.model_dump_json())
    except (ValueError, TypeError) as exc:
        raise _error("event is not JSON serializable") from exc
    _validate_events([*events, event], session_id)
    return event


class MemorySession:
    """An isolated in-memory journal with the same snapshot and pairing guarantees."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = _session_id(session_id)
        self._events: list[SessionEvent] = []
        self._closed = False
        self._recovery_lock = asyncio.Lock()

    @property
    def events(self) -> list[SessionEvent]:
        return [_snapshot(event) for event in self._events]

    def _check_open(self) -> None:
        if self._closed:
            raise PersistenceError("Session is closed.")

    async def append(
        self, event_type: str, data: dict[str, Any], run_id: str | None = None
    ) -> SessionEvent:
        self._check_open()
        event = _new_event(self.session_id, self._events, event_type, data, run_id)
        self._events.append(event)
        return _snapshot(event)

    def derive_messages(self) -> list[Message]:
        return _messages(self._events, self.session_id)

    async def flush(self) -> None:
        self._check_open()

    async def close(self) -> None:
        self._closed = True


@dataclass
class _Write:
    kind: str
    future: asyncio.Future[SessionEvent | None]
    event_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


def _consume_future(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise PersistenceError("Session directory must not be a symbolic link.")
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise PersistenceError("Session directory is not a directory.") from None
    else:
        _fsync_directory(path.parent)


def _partial_json(raw: bytes) -> bool:
    """Recognize unfinished JSON prefixes, never a complete malformed final record."""
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data":
            return False
        # A process can stop while writing the last code point of a JSON string.
        value = raw[: exc.start].decode("utf-8")
    if not value.strip():
        return True
    if not value.lstrip().startswith("{"):
        return False
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        remaining = value[exc.pos :].strip()
        if exc.msg.startswith("Unterminated string"):
            return True
        if not remaining:
            return True
        if exc.msg == "Expecting value" and any(
            literal.startswith(remaining) for literal in ("true", "false", "null", "-")
        ):
            return True
        if exc.msg == "Expecting ',' delimiter" and re.fullmatch(r"(?:[eE][+-]?|\.)", remaining):
            return True
    return False


def _read_events(fd: int, session_id: str, *, repair: bool) -> list[SessionEvent]:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    raw = b"".join(chunks)
    lines = raw.split(b"\n")
    final = lines.pop()
    repair_at: int | None = None
    add_newline = False
    if final:
        try:
            json.loads(final)
        except (ValueError, UnicodeDecodeError) as exc:
            if not repair or not _partial_json(final):
                raise _error("invalid final JSON record") from exc
            repair_at = len(raw) - len(final)
        else:
            lines.append(final)
            add_newline = True
    events: list[SessionEvent] = []
    try:
        for line in lines:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise _error("event is not an object")
            if type(data.get("seq")) is not int or type(data.get("version")) is not int:
                raise _error("invalid event sequence or version")
            events.append(SessionEvent.model_validate(data))
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise _error("invalid JSON or event schema") from exc
    _validate_events(events, session_id)
    if events and events[0].type != "session.created":
        raise _error("first persisted event must be session.created")
    # Do not modify even the tail until every complete record and reference is valid.
    if repair:
        if repair_at is not None:
            os.ftruncate(fd, repair_at)
            os.fsync(fd)
        elif add_newline:
            if os.write(fd, b"\n") != 1:
                raise PersistenceError("Failed to repair final journal newline.")
            os.fsync(fd)
    return events


def _open_journal(
    workspace: Path, session_id: str, *, create: bool = True
) -> tuple[Path, int, int, list[SessionEvent]]:
    parent = workspace / ".hiro-agent"
    directory = parent / "sessions"
    _ensure_directory(parent)
    _ensure_directory(directory)
    lock_path = directory / f"{session_id}.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    journal_fd: int | None = None
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PersistenceError("Session is already open in another process.") from None
        os.fsync(lock_fd)
        _fsync_directory(directory)
        path = directory / f"{session_id}.jsonl"
        flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW
        journal_fd = os.open(path, flags | (os.O_CREAT if create else 0), 0o600)
        os.fsync(journal_fd)
        _fsync_directory(directory)
        events = _read_events(journal_fd, session_id, repair=True)
        return path, lock_fd, journal_fd, events
    except BaseException:
        if journal_fd is not None:
            os.close(journal_fd)
        os.close(lock_fd)
        raise


class JsonlSession(MemorySession):
    """An exclusively locked journal; append completes only after its fsync succeeds."""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.workspace: Path
        self.path: Path
        self._lock_fd: int
        self._journal_fd: int
        self._writer: asyncio.Task[None]
        self._queue: asyncio.Queue[_Write] = asyncio.Queue()
        self._failure: PersistenceError | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(
        cls, workspace: Path, session_id: str | None = None, *, create: bool = True
    ) -> JsonlSession:
        instance = cls(_session_id(session_id))
        instance.workspace = Path(workspace).resolve()
        if not instance.workspace.is_dir():
            raise PersistenceError("Workspace must be an existing directory.")
        # Synchronous opening prevents cancellation from orphaning a thread-acquired file lock.
        try:
            path, lock_fd, journal_fd, events = _open_journal(
                instance.workspace, instance.session_id, create=create
            )
        except PersistenceError:
            raise
        except FileNotFoundError:
            if not create:
                raise
            raise PersistenceError("Unable to open session journal.") from None
        except OSError as exc:
            raise PersistenceError("Unable to open session journal.") from exc
        instance.path = path
        instance._lock_fd = lock_fd
        instance._journal_fd = journal_fd
        instance._events = events
        instance._writer = asyncio.create_task(instance._write_loop())
        if not events and create:
            try:
                await instance.append("session.created", {"workspace": str(instance.workspace)})
            except BaseException:
                try:
                    await instance.close()
                except PersistenceError:
                    pass
                raise
        return instance

    def _check_open(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed or self._closing:
            raise PersistenceError("Session is closed.")

    async def _submit(self, item: _Write) -> SessionEvent | None:
        self._check_open()
        item.future.add_done_callback(_consume_future)
        self._queue.put_nowait(item)
        return await asyncio.shield(item.future)

    async def append(
        self, event_type: str, data: dict[str, Any], run_id: str | None = None
    ) -> SessionEvent:
        future: asyncio.Future[SessionEvent | None] = asyncio.get_running_loop().create_future()
        result = await self._submit(
            _Write("append", future, event_type, copy.deepcopy(data), run_id)
        )
        assert result is not None
        return result

    async def flush(self) -> None:
        future: asyncio.Future[SessionEvent | None] = asyncio.get_running_loop().create_future()
        await self._submit(_Write("flush", future))

    def _write_bytes(self, raw: bytes) -> None:
        remaining = memoryview(raw)
        while remaining:
            count = os.write(self._journal_fd, remaining)
            if count <= 0:
                raise OSError("Journal write did not make progress")
            remaining = remaining[count:]
        os.fsync(self._journal_fd)

    async def _write_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item.kind == "stop":
                item.future.set_result(None)
                return
            if self._failure is not None:
                item.future.set_exception(self._failure)
                continue
            event: SessionEvent | None = None
            if item.kind == "append":
                try:
                    event = _new_event(
                        self.session_id, self._events, item.event_type, item.data, item.run_id
                    )
                    raw = (event.model_dump_json() + "\n").encode("utf-8")
                except Exception as exc:
                    item.future.set_exception(exc)
                    continue
            try:
                if event is not None:
                    await asyncio.to_thread(self._write_bytes, raw)
                else:
                    await asyncio.to_thread(os.fsync, self._journal_fd)
            except Exception:
                self._failure = PersistenceError(
                    "Session persistence failed; reopen the session before continuing."
                )
                item.future.set_exception(self._failure)
            else:
                if event is not None:
                    self._events.append(event)
                item.future.set_result(_snapshot(event) if event is not None else None)

    async def _finish_close(self) -> None:
        future: asyncio.Future[SessionEvent | None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Write("stop", future))
        try:
            await self._writer
        finally:
            try:
                os.close(self._journal_fd)
            finally:
                os.close(self._lock_fd)
                self._closed = True
        if self._failure is not None:
            raise self._failure

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._finish_close())
            self._close_task.add_done_callback(_consume_future)
        await asyncio.shield(self._close_task)


async def _settle_tool_calls(session: SessionService) -> list[ToolResult]:
    state = _validate_events(session.events, session.session_id)
    results: list[ToolResult] = []
    for recorded in state.pending():
        result = recorded.delegation_result or ToolResult(
            call_id=recorded.call.id,
            ok=False,
            error="TOOL_OUTCOME_UNKNOWN" if recorded.started else "TOOL_NOT_STARTED",
            message=(
                "The process stopped after this tool started; its outcome is unknown. "
                "Inspect the workspace before deciding whether to retry."
                if recorded.started
                else "This tool did not start before the run stopped."
            ),
            outcome="unknown" if recorded.started else "not_started",
            value=(
                {"delegation": recorded.delegation.model_dump(mode="json")}
                if recorded.delegation is not None
                else None
            ),
        )
        await session.append("tool.result", result.model_dump(mode="json"), recorded.run_id)
        results.append(result)
    return results


async def settle_tool_calls(session: SessionService) -> list[ToolResult]:
    """Terminate unfinished calls without executing tools or ending the current run."""
    lock = getattr(session, "_recovery_lock", None)
    if lock is None:
        return await _settle_tool_calls(session)
    async with lock:
        return await _settle_tool_calls(session)


def delegation_terminal_event(
    session: SessionService, result: ToolResult, *, recovered: bool = False
) -> dict[str, Any] | None:
    """Describe a terminal declaration without creating a synthetic child/link."""
    events = session.events
    if not recovered and any(
        e.type == "delegation.finished" and e.data["call_id"] == result.call_id for e in events
    ):
        return None  # The manager already emitted the linked completion.
    call = next(
        (
            call
            for event in reversed(events)
            if event.type == "assistant.message"
            for call in AssistantMessage.model_validate(event.data["message"]).tool_calls
            if call.id == result.call_id and call.name == "delegate_task"
        ),
        None,
    )
    if call is None:
        return None
    try:
        args = json.loads(call.arguments)
    except ValueError:
        args = {}
    value = result.value if isinstance(result.value, dict) else {}
    link = value.get("delegation", {})
    status = (
        "not_started"
        if result.outcome == "not_started"
        else "paused"
        if result.error == "budget_exceeded"
        else "canceled"
        if result.error == "cancelled"
        else "unknown"
        if result.outcome == "unknown"
        else value.get("status", "completed" if result.ok else "failed")
    )
    return {
        "parent_session_id": session.session_id,
        "child_session_id": None,
        **(link if isinstance(link, dict) else {}),
        "call_id": call.id,
        "role": args.get("role", "unknown") if isinstance(args, dict) else "unknown",
        "status": status,
        "ok": result.ok,
        "error": result.error,
    }


def _recovered_metrics(events: list[SessionEvent], run_id: str) -> RunMetrics:
    """Reconstruct durable counts; elapsed excludes offline and recovery-only time.

    Tool counts use accepted tool.called markers. Legacy markers without accepted
    are counted because they cannot distinguish rejection before budget admission.
    Synthetic recovery results are not new tool executions or runtime errors.
    """
    metrics = RunMetrics()
    recorded = []
    recovered_delegations = {
        event.data["call_id"]
        for event in events
        if event.run_id == run_id
        and event.type == "delegation.finished"
        and event.data.get("recovered") is True
    }
    for event in events:
        if event.run_id != run_id:
            continue
        if event.type == "model.interrupted" and event.data.get("reason") == "interrupted":
            continue
        if event.type == "tool.result" and event.data.get("error") in {
            "TOOL_OUTCOME_UNKNOWN",
            "TOOL_NOT_STARTED",
        }:
            continue
        if not (
            (event.type == "delegation.finished" and event.data.get("recovered") is True)
            or (event.type == "tool.result" and event.data.get("call_id") in recovered_delegations)
        ):
            recorded.append(event)
        match event.type:
            case "delegation.finished":
                child_metrics = RunMetrics.model_validate(event.data["metrics"])
                for name in RunMetrics.model_fields:
                    if name != "elapsed_ms":
                        setattr(
                            metrics, name, getattr(metrics, name) + getattr(child_metrics, name)
                        )
            case "model.requested":
                metrics.model_requests += 1
            case "assistant.message":
                usage = (
                    Usage.model_validate(event.data["usage"])
                    if event.data.get("usage") is not None
                    else None
                )
                for name, count in usage_counts(usage).items():
                    setattr(metrics, name, getattr(metrics, name) + count)
            case "tool.called":
                metrics.tool_calls += int(event.data.get("accepted", True))
            case "tool.result":
                result = ToolResult.model_validate(event.data)
                metrics.tool_errors += int(not result.ok)
                metrics.permission_denials += int(result.error == "permission_denied")
    started = next((event for event in recorded if event.type == "run.started"), None)
    if started is not None and recorded:
        elapsed = (recorded[-1].timestamp - started.timestamp).total_seconds()
        metrics.elapsed_ms = max(0, int(elapsed * 1000))
    return metrics


async def _recover_session(session: SessionService) -> list[ToolResult]:
    before_recovery = session.events
    results = await _settle_tool_calls(session)
    state = _validate_events(session.events, session.session_id)
    for attempt_id, run_id in state.attempts.items():
        if attempt_id not in state.ended_attempts:
            await session.append(
                "model.interrupted", {"attempt_id": attempt_id, "reason": "interrupted"}, run_id
            )
    for run_id, ended in state.runs.items():
        if not ended:
            await session.append(
                "run.ended",
                {
                    "status": "paused",
                    "reason": "interrupted",
                    "metrics": _recovered_metrics(before_recovery, run_id).model_dump(mode="json"),
                },
                run_id,
            )
    return results


async def recover_session(session: SessionService) -> list[ToolResult]:
    """Record uncertain outcomes and interrupted work once, without re-execution."""
    lock = getattr(session, "_recovery_lock", None)
    if lock is None:
        return await _recover_session(session)
    async with lock:
        return await _recover_session(session)


def read_session_events(workspace: Path, session_id: str) -> list[SessionEvent]:
    """Read an existing journal without repair, recovery, creation or a writer."""
    session_id = _session_id(session_id)
    directory = Path(workspace).resolve() / ".hiro-agent" / "sessions"
    if directory.is_symlink() or directory.parent.is_symlink() or not directory.is_dir():
        raise PersistenceError("会话不可读取：日志目录不存在或不安全。")
    lock_fd: int | None = None
    journal_fd: int | None = None
    try:
        lock_fd = os.open(directory / f"{session_id}.lock", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PersistenceError("会话不可读取：文件被占用。") from None
        journal_fd = os.open(directory / f"{session_id}.jsonl", os.O_RDONLY | os.O_NOFOLLOW)
        return _read_events(journal_fd, session_id, repair=False)
    except (OSError, PersistenceError) as exc:
        if isinstance(exc, PersistenceError) and str(exc).startswith("会话不可读取"):
            raise
        raise PersistenceError("会话不可读取：日志不存在、损坏或无法访问。") from exc
    finally:
        if journal_fd is not None:
            os.close(journal_fd)
        if lock_fd is not None:
            os.close(lock_fd)


def delete_session(workspace: Path, session_id: str) -> None:
    """Remove a closed parent and its children after locking and validating all logs.

    Stable lock files are deliberately retained: unlinking a lock file could let
    another process acquire a new inode while an existing opener holds the old one.
    """
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise PersistenceError("会话删除失败：Session ID 不合法。")
    try:
        with ExitStack() as stack:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            workspace_fd = os.open(Path(workspace).resolve(strict=True), flags)
            stack.callback(os.close, workspace_fd)
            parent_fd = os.open(".hiro-agent", flags, dir_fd=workspace_fd)
            stack.callback(os.close, parent_fd)
            directory_fd = os.open("sessions", flags, dir_fd=parent_fd)
            stack.callback(os.close, directory_fd)
            opened: dict[str, int] = {}
            journals: list[str] = []

            def open_file(name: str) -> int:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                stack.callback(os.close, fd)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise PersistenceError("会话删除失败：日志或锁文件不是普通文件。")
                opened[name] = fd
                return fd

            def read_locked(
                target_id: str, *, missing_ok: bool = False
            ) -> list[SessionEvent] | None:
                journal_name = f"{target_id}.jsonl"
                try:
                    lock_fd = open_file(f"{target_id}.lock")
                except FileNotFoundError:
                    if missing_ok:
                        try:
                            os.stat(journal_name, dir_fd=directory_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            return None
                    raise
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise PersistenceError(
                        "会话删除失败：会话或关联子会话正在使用，请关闭后重试。"
                    ) from None
                try:
                    journal_fd = open_file(journal_name)
                except FileNotFoundError:
                    if missing_ok:
                        return None
                    raise
                events = _read_events(journal_fd, target_id, repair=False)
                journals.append(journal_name)
                return events

            events = read_locked(session_id)
            assert events is not None
            if any(event.type == "subagent.created" for event in events):
                raise PersistenceError("会话删除失败：子会话不能单独删除，请选择对应的主会话。")
            children: dict[str, DelegationLink] = {}
            for event in events:
                if event.type != "delegation.started":
                    continue
                link = DelegationLink.model_validate(event.data)
                previous = children.setdefault(link.child_session_id, link)
                if previous != link:
                    raise PersistenceError("会话删除失败：同一子会话存在冲突的委派关联。")
            for child_id, link in sorted(children.items()):
                child_events = read_locked(child_id, missing_ok=True)
                if child_events is None:
                    continue
                reciprocal = [event for event in child_events if event.type == "subagent.created"]
                if reciprocal:
                    if DelegationLink.model_validate(reciprocal[0].data) != link:
                        raise PersistenceError("会话删除失败：父子会话委派关联不一致。")
                elif any(event.type != "session.created" for event in child_events):
                    raise PersistenceError("会话删除失败：子会话缺少父任务关联记录。")

            # All targets are locked and validated before removing any journal.
            # Directory-relative operations also avoid following replaced parent paths.
            for name, fd in opened.items():
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not os.path.samestat(os.fstat(fd), current):
                    raise PersistenceError("会话删除失败：日志或锁文件已变化，请重试。")
            for name in reversed(journals):
                os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except PersistenceError as exc:
        if str(exc).startswith("会话删除失败"):
            raise
        raise PersistenceError("会话删除失败：日志损坏，无法确认父子关联。") from exc
    except OSError as exc:
        raise PersistenceError("会话删除失败：日志不存在、路径不安全或无法访问。") from exc


def list_sessions(workspace: Path) -> list[dict[str, Any]]:
    """Return read-only metadata; busy or corrupt journals are never repaired here."""
    directory = Path(workspace).resolve() / ".hiro-agent" / "sessions"
    if directory.is_symlink() or directory.parent.is_symlink() or not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    updated_at: dict[str, datetime] = {}
    for path in sorted(directory.glob("*.jsonl")):
        session_id = path.stem
        if not _SESSION_ID.fullmatch(session_id):
            continue
        summary: dict[str, Any] = {"session_id": session_id, "status": "unknown"}
        lock_fd: int | None = None
        journal_fd: int | None = None
        is_child = False
        try:
            lock_fd = os.open(path.with_suffix(".lock"), os.O_RDONLY | os.O_NOFOLLOW)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                summary["status"] = "busy"
                continue
            journal_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            events = _read_events(journal_fd, session_id, repair=False)
            is_child = any(event.type == "subagent.created" for event in events)
            if is_child:
                continue
            tasks = [event for event in events if event.type == "task.received"]
            title = " ".join(tasks[0].data["text"].split()) if tasks else ""
            summary["title"] = title if len(title) <= 80 else title[:79] + "…"
            summary["task_count"] = len(tasks)
            if events:
                updated_at[session_id] = events[-1].timestamp
                summary["updated_at"] = events[-1].timestamp.isoformat()
                summary["workspace"] = events[0].data.get("workspace")
                summary["status"] = "idle"
                for event in events:
                    if event.type == "run.started":
                        summary["status"] = "running"
                    elif event.type == "run.ended":
                        summary["status"] = event.data.get("status", "unknown")
        except (OSError, PersistenceError):
            summary["status"] = "unreadable"
        finally:
            if journal_fd is not None:
                os.close(journal_fd)
            if lock_fd is not None:
                os.close(lock_fd)
            if not is_child:
                summaries.append(summary)
    # The initial path order breaks ties by ID. Compare aware datetimes rather
    # than ISO strings, which can order different timezone offsets incorrectly.
    return sorted(
        summaries,
        key=lambda summary: updated_at.get(summary["session_id"], datetime.min.replace(tzinfo=UTC)),
        reverse=True,
    )
