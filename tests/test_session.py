from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from hiro_agent.contracts import AssistantMessage, ToolCall, ToolResult
from hiro_agent.errors import PersistenceError
from hiro_agent.session import JsonlSession, MemorySession, list_sessions, recover_session


async def tool_batch(session: MemorySession, *, two: bool = False) -> list[ToolCall]:
    await session.append("run.started", {}, "run-1")
    await session.append("task.received", {"text": "check files"}, "run-1")
    await session.append("model.requested", {"attempt_id": "attempt-1"}, "run-1")
    calls = [ToolCall(id="call-1", name="write_file", arguments='{"path":"x"}')]
    if two:
        calls.append(ToolCall(id="call-2", name="read_file", arguments='{"path":"x"}'))
    await session.append(
        "assistant.message",
        {
            "message": AssistantMessage(tool_calls=calls).model_dump(mode="json"),
            "attempt_id": "attempt-1",
            "finish_reason": "tool_calls",
            "usage": None,
        },
        "run-1",
    )
    return calls


async def test_memory_snapshots_are_detached() -> None:
    session = MemorySession("snapshot")
    payload = {"nested": {"values": [1]}}
    event = await session.append("custom.event", payload)
    payload["nested"]["values"].append(2)
    event.data["nested"]["values"].append(3)
    events = session.events
    events[0].data["nested"]["values"].append(4)
    events.clear()
    assert session.events[0].data == {"nested": {"values": [1]}}
    await session.close()
    with pytest.raises(PersistenceError, match="closed"):
        await session.append("custom.event", {})


async def test_projection_pairs_tool_results_and_continuation() -> None:
    session = MemorySession()
    await session.append("prompt.system", {"text": "system"})
    calls = await tool_batch(session)
    with pytest.raises(PersistenceError, match="outstanding"):
        session.derive_messages()
    result = ToolResult(call_id=calls[0].id, ok=True, value={"written": True})
    await session.append("tool.result", result.model_dump(mode="json"), "run-1")
    await session.append("task.continued", {"text": "continue"}, "run-1")
    messages = session.derive_messages()
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2].tool_calls == calls
    assert messages[3].tool_call_id == calls[0].id
    assert json.loads(messages[3].content or "") == result.model_dump(mode="json")
    messages[2].tool_calls.clear()
    assert session.derive_messages()[2].tool_calls == calls


@pytest.mark.parametrize("bad_id", ["../escape", "", "/tmp/journal", "a.b", "a/b", "x" * 129])
async def test_invalid_session_ids(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(PersistenceError, match="Invalid session ID"):
        await JsonlSession.open(tmp_path, bad_id)


async def test_durable_ack_lock_and_reopen(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "durable")
    event = await session.append("task.received", {"text": "hello"})
    assert json.loads(session.path.read_bytes().splitlines()[-1])["seq"] == event.seq
    with pytest.raises(PersistenceError, match="already open"):
        await JsonlSession.open(tmp_path, "durable")
    assert list_sessions(tmp_path)[0]["status"] == "busy"
    await session.close()
    reopened = await JsonlSession.open(tmp_path, "durable")
    assert reopened.events[-1] == event
    await reopened.close()
    assert list_sessions(tmp_path)[0]["status"] == "idle"
    assert session.path.with_suffix(".lock").exists()


@pytest.mark.parametrize("tail", [b'{"version":', b'{"data":{"text":"cut', b'{"a":tru'])
async def test_repair_only_incomplete_final_record(tmp_path: Path, tail: bytes) -> None:
    session = await JsonlSession.open(tmp_path, "tail")
    await session.close()
    original = session.path.read_bytes()
    session.path.write_bytes(original + tail)
    assert list_sessions(tmp_path)[0]["status"] == "unreadable"
    assert session.path.read_bytes() == original + tail
    repaired = await JsonlSession.open(tmp_path, "tail")
    assert repaired.path.read_bytes() == original
    await repaired.close()


async def test_valid_final_record_without_newline(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "newline")
    await session.close()
    original = session.path.read_bytes()
    session.path.write_bytes(original.rstrip(b"\n"))
    reopened = await JsonlSession.open(tmp_path, "newline")
    assert reopened.path.read_bytes() == original
    assert len(reopened.events) == 1
    await reopened.close()


@pytest.mark.parametrize("tail", [b"broken\n", b'{"a":broken}', b"{}", b"[]", b'{"bad":true}\n'])
async def test_corrupt_complete_records_are_never_repaired(tmp_path: Path, tail: bytes) -> None:
    session = await JsonlSession.open(tmp_path, "corrupt")
    await session.close()
    damaged = session.path.read_bytes() + tail
    session.path.write_bytes(damaged)
    with pytest.raises(PersistenceError):
        await JsonlSession.open(tmp_path, "corrupt")
    assert session.path.read_bytes() == damaged


async def test_interior_corruption_blocks_tail_repair(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "interior")
    await session.close()
    damaged = session.path.read_bytes() + b"broken\n" + b'{"version":'
    session.path.write_bytes(damaged)
    with pytest.raises(PersistenceError):
        await JsonlSession.open(tmp_path, "interior")
    assert session.path.read_bytes() == damaged


async def test_sequence_corruption_is_fatal(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "seq")
    await session.close()
    data = json.loads(session.path.read_text())
    data["seq"] = 2
    damaged = json.dumps(data).encode()
    session.path.write_bytes(damaged)
    with pytest.raises(PersistenceError, match="sequence"):
        await JsonlSession.open(tmp_path, "seq")
    assert session.path.read_bytes() == damaged


async def test_fsync_failure_poison_and_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await JsonlSession.open(tmp_path, "fault")
    original_events = session.events
    original_fsync = os.fsync

    def fail_journal(fd: int) -> None:
        if fd == session._journal_fd:
            raise OSError("simulated disk failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_journal)
    with pytest.raises(PersistenceError, match="persistence failed"):
        await session.append("task.received", {"text": "not acknowledged"})
    assert session.events == original_events
    with pytest.raises(PersistenceError, match="persistence failed"):
        await session.append("task.received", {"text": "must not write"})
    with pytest.raises(PersistenceError):
        await session.flush()
    with pytest.raises(PersistenceError):
        await session.close()
    monkeypatch.setattr(os, "fsync", original_fsync)
    reopened = await JsonlSession.open(tmp_path, "fault")
    assert all(e.data.get("text") != "must not write" for e in reopened.events)
    await reopened.close()


@pytest.mark.parametrize("cut", ["assistant", "called", "started", "result"])
async def test_recovery_at_tool_crash_cuts(tmp_path: Path, cut: str) -> None:
    session = await JsonlSession.open(tmp_path, "crash")
    calls = await tool_batch(session, two=True)
    if cut in {"called", "started", "result"}:
        await session.append("tool.called", {"call": calls[0].model_dump()}, "run-1")
    if cut in {"started", "result"}:
        await session.append("tool.started", {"call": calls[0].model_dump()}, "run-1")
    if cut == "result":
        await session.append(
            "tool.result", ToolResult(call_id="call-1", ok=True).model_dump(), "run-1"
        )
    await session.close()
    reopened = await JsonlSession.open(tmp_path, "crash")
    results = await recover_session(reopened)
    expected_ids = ["call-2"] if cut == "result" else ["call-1", "call-2"]
    assert [r.call_id for r in results] == expected_ids
    if cut == "started":
        assert results[0].error == "TOOL_OUTCOME_UNKNOWN"
        assert results[0].outcome == "unknown"
    else:
        assert all(r.error == "TOOL_NOT_STARTED" for r in results)
    assert reopened.events[-1].type == "run.ended"
    assert reopened.events[-1].data["status"] == "paused"
    assert not any(e.type == "model.interrupted" for e in reopened.events)
    recovered = reopened.events
    assert await recover_session(reopened) == []
    assert reopened.events == recovered
    assert len([m for m in reopened.derive_messages() if m.role == "tool"]) == 2
    await reopened.close()


async def test_interrupted_attempt_and_concurrent_recovery_are_idempotent() -> None:
    session = MemorySession()
    await session.append("run.started", {}, "run")
    await session.append("model.requested", {"attempt_id": "attempt"}, "run")
    await asyncio.gather(recover_session(session), recover_session(session))
    assert [e.type for e in session.events] == [
        "run.started",
        "model.requested",
        "model.interrupted",
        "run.ended",
    ]


async def test_reject_malformed_references_without_poisoning() -> None:
    session = MemorySession()
    calls = await tool_batch(session)
    with pytest.raises(PersistenceError, match="outstanding"):
        await session.append("task.received", {"text": "bad"})
    with pytest.raises(PersistenceError, match="reference"):
        await session.append("tool.result", ToolResult(call_id="unknown", ok=True).model_dump())
    await session.append(
        "tool.result", ToolResult(call_id=calls[0].id, ok=True).model_dump(), "run-1"
    )
    with pytest.raises(PersistenceError, match="duplicate tool result"):
        await session.append(
            "tool.result", ToolResult(call_id=calls[0].id, ok=True).model_dump(), "run-1"
        )
    await session.append("task.continued", {"text": "okay"})


@pytest.mark.parametrize("target", ["directory", "journal", "lock"])
async def test_symlinks_are_rejected(tmp_path: Path, target: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "directory":
        (tmp_path / ".hiro-agent").symlink_to(outside, target_is_directory=True)
    else:
        directory = tmp_path / ".hiro-agent" / "sessions"
        directory.mkdir(parents=True)
        (outside / "target").write_text("untouched")
        suffix = "jsonl" if target == "journal" else "lock"
        (directory / f"symlink.{suffix}").symlink_to(outside / "target")
    with pytest.raises(PersistenceError):
        await JsonlSession.open(tmp_path, "symlink")
    if target != "directory":
        assert (outside / "target").read_text() == "untouched"


async def test_concurrent_appends_acknowledge_contiguous_durable_events(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "concurrent")
    events = await asyncio.gather(*(session.append("custom", {"index": i}) for i in range(20)))
    assert [event.seq for event in events] == list(range(2, 22))
    assert [json.loads(line)["seq"] for line in session.path.read_bytes().splitlines()] == list(
        range(1, 22)
    )
    await session.close()


async def test_cancelled_append_finishes_durable_write_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    session = await JsonlSession.open(tmp_path, "cancelled")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    write = session._write_bytes

    def slow_write(raw: bytes) -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        write(raw)

    monkeypatch.setattr(session, "_write_bytes", slow_write)
    task = asyncio.create_task(session.append("task.received", {"text": "persist anyway"}))
    await asyncio.wait_for(entered.wait(), 2)
    assert len(session.events) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    closing = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    reopened = await JsonlSession.open(tmp_path, "cancelled")
    assert reopened.events[-1].data["text"] == "persist anyway"
    await reopened.close()


async def test_queued_writes_all_fail_after_first_disk_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await JsonlSession.open(tmp_path, "queued-failure")
    writes = 0

    def fail_write(raw: bytes) -> None:
        nonlocal writes
        writes += 1
        raise OSError("simulated full device")

    monkeypatch.setattr(session, "_write_bytes", fail_write)
    results = await asyncio.gather(
        *(session.append("custom", {"index": i}) for i in range(5)), return_exceptions=True
    )
    assert all(isinstance(result, PersistenceError) for result in results)
    assert writes == 1
    assert len(session.events) == 1
    with pytest.raises(PersistenceError):
        await session.close()


async def test_valid_json_prefixes_are_repairable() -> None:
    from hiro_agent.contracts import SessionEvent
    from hiro_agent.session import _partial_json

    raw = (
        SessionEvent(
            session_id="prefix",
            seq=1,
            type="custom",
            data={"text": "你好\\line\ncut", "bool": True, "null": None, "float": 1.1e20},
        )
        .model_dump_json()
        .encode()
    )
    for index in range(1, len(raw)):
        assert _partial_json(raw[:index]), raw[:index]


async def test_recovery_preserves_durable_usage_and_tool_metrics() -> None:
    from datetime import UTC, datetime, timedelta

    from hiro_agent.contracts import RunMetrics

    session = MemorySession()
    await session.append("run.started", {}, "metrics-run")
    await session.append("task.received", {"text": "task"}, "metrics-run")
    await session.append("model.requested", {"attempt_id": "used"}, "metrics-run")
    calls = [ToolCall(id="denied", name="write_file", arguments="{}")]
    await session.append(
        "assistant.message",
        {
            "attempt_id": "used",
            "message": AssistantMessage(tool_calls=calls).model_dump(),
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 100, "output_tokens": 7},
        },
        "metrics-run",
    )
    await session.append("tool.called", {"call": calls[0].model_dump()}, "metrics-run")
    await session.append(
        "tool.result",
        ToolResult(
            call_id="denied", ok=False, error="permission_denied", outcome="not_started"
        ).model_dump(),
        "metrics-run",
    )
    await session.append("model.requested", {"attempt_id": "missing"}, "metrics-run")
    await session.append(
        "assistant.message",
        {
            "attempt_id": "missing",
            "message": AssistantMessage(text="unfinished").model_dump(),
            "finish_reason": "length",
            "usage": None,
        },
        "metrics-run",
    )
    await session.append("model.requested", {"attempt_id": "interrupted"}, "metrics-run")
    base = datetime(2000, 1, 1, tzinfo=UTC)
    for index, event in enumerate(session._events):
        event.timestamp = base + timedelta(seconds=index)
    await recover_session(session)
    assert (
        session.events[-1].data["metrics"]
        == RunMetrics(
            model_requests=3,
            worker_steps=2,
            tool_calls=1,
            tool_errors=1,
            permission_denials=1,
            input_tokens=100,
            output_tokens=7,
            usage_missing=1,
            elapsed_ms=8000,
        ).model_dump()
    )


async def test_repeated_recovery_excludes_offline_gap_from_elapsed_metrics() -> None:
    from datetime import UTC, datetime, timedelta

    from hiro_agent.session import settle_tool_calls

    session = MemorySession()
    calls = await tool_batch(session)
    await session.append("tool.called", {"call": calls[0].model_dump()}, "run-1")
    await session.append("tool.started", {"call": calls[0].model_dump()}, "run-1")
    base = datetime(2000, 1, 1, tzinfo=UTC)
    for index, event in enumerate(session._events):
        event.timestamp = base + timedelta(seconds=index)
    # Simulate a crash in recovery after it persisted a synthetic tool result.
    await settle_tool_calls(session)
    assert session.events[-1].timestamp.year > 2000
    await recover_session(session)
    metrics = session.events[-1].data["metrics"]
    assert metrics["elapsed_ms"] == 5000
    assert metrics["tool_calls"] == 1
    assert metrics["tool_errors"] == 0
    assert metrics["model_requests"] == 1
    assert metrics["worker_steps"] == 1
    assert metrics["usage_missing"] == 1


async def test_recovered_tool_budget_counts_accepted_calls_only() -> None:
    session = MemorySession()
    calls = await tool_batch(session, two=True)
    for index, call in enumerate(calls):
        await session.append(
            "tool.called", {"call": call.model_dump(), "accepted": index == 0}, "run-1"
        )
        result = ToolResult(
            call_id=call.id,
            ok=index == 0,
            error="budget_exceeded" if index else None,
            outcome="not_started" if index else "completed",
        )
        await session.append("tool.result", result.model_dump(), "run-1")
    await recover_session(session)
    metrics = session.events[-1].data["metrics"]
    assert metrics["tool_calls"] == 1
    assert metrics["tool_errors"] == 1
