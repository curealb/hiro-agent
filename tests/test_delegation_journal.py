"""Delegation journals preserve outcomes without executing children on recovery."""

from pathlib import Path

import pytest

from hiro_agent.contracts import AssistantMessage, DelegationLink, RunMetrics, ToolCall, ToolResult
from hiro_agent.errors import PersistenceError
from hiro_agent.session import JsonlSession, MemorySession, list_sessions, recover_session


async def pending_delegate(session: MemorySession) -> DelegationLink:
    await session.append("run.started", {}, "parent-run")
    call = ToolCall(id="delegate", name="delegate_task", arguments="{}")
    await session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=[call]).model_dump()},
        "parent-run",
    )
    await session.append("tool.called", {"call": call.model_dump()}, "parent-run")
    await session.append("tool.started", {"call": call.model_dump()}, "parent-run")
    return DelegationLink(
        call_id=call.id,
        parent_session_id=session.session_id,
        parent_run_id="parent-run",
        child_session_id="child",
        child_run_id="child-run",
        role="explorer",
    )


async def test_cached_completion_is_settled_once_and_metrics_are_aggregated() -> None:
    session = MemorySession("parent")
    link = await pending_delegate(session)
    await session.append("delegation.started", link.model_dump(), "parent-run")
    result = ToolResult(call_id=link.call_id, ok=True, value={"summary": "found it"})
    await session.append(
        "delegation.finished",
        {
            "call_id": link.call_id,
            "result": result.model_dump(),
            "metrics": RunMetrics(
                model_requests=2, tool_calls=3, input_tokens=7, elapsed_ms=999999
            ).model_dump(),
        },
        "parent-run",
    )
    assert await recover_session(session) == [result]
    metrics = session.events[-1].data["metrics"]
    assert metrics["model_requests"] == 2
    assert metrics["tool_calls"] == 4
    assert metrics["input_tokens"] == 7
    assert metrics["elapsed_ms"] < 999999
    before = session.events
    assert await recover_session(session) == []
    assert session.events == before
    assert len([m for m in session.derive_messages() if m.role == "tool"]) == 1


async def test_unfinished_delegation_preserves_linkage() -> None:
    session = MemorySession("parent")
    link = await pending_delegate(session)
    await session.append("delegation.started", link.model_dump(), "parent-run")
    results = await recover_session(session)
    assert results[0].error == "TOOL_OUTCOME_UNKNOWN"
    assert results[0].value == {"delegation": link.model_dump()}


async def test_linkage_and_completion_references_are_validated() -> None:
    session = MemorySession("parent")
    link = await pending_delegate(session)
    for change in (
        {"call_id": "missing"},
        {"parent_session_id": "other"},
        {"parent_run_id": "other"},
        {"child_session_id": "parent"},
    ):
        with pytest.raises(PersistenceError):
            await session.append(
                "delegation.started", link.model_copy(update=change).model_dump(), "parent-run"
            )
    await session.append("delegation.started", link.model_dump(), "parent-run")
    with pytest.raises(PersistenceError):
        await session.append("delegation.started", link.model_dump(), "parent-run")
    result = ToolResult(call_id=link.call_id, ok=True)
    payload = {
        "call_id": link.call_id,
        "result": result.model_dump(),
        "metrics": RunMetrics().model_dump(),
    }
    with pytest.raises(PersistenceError):
        await session.append("delegation.finished", {**payload, "call_id": "wrong"}, "parent-run")
    await session.append("delegation.finished", payload, "parent-run")
    with pytest.raises(PersistenceError):
        await session.append("delegation.finished", payload, "parent-run")
    with pytest.raises(PersistenceError):
        await session.append(
            "tool.result", ToolResult(call_id=link.call_id, ok=False).model_dump(), "parent-run"
        )


async def test_child_journal_linkage_and_listing(tmp_path: Path) -> None:
    parent = MemorySession("parent")
    link = await pending_delegate(parent)
    child = await JsonlSession.open(tmp_path, "child")
    try:
        with pytest.raises(PersistenceError):
            await child.append(
                "subagent.created",
                link.model_copy(update={"child_session_id": "other"}).model_dump(),
            )
        await child.append("subagent.created", link.model_dump())
        with pytest.raises(PersistenceError):
            await child.append("subagent.created", link.model_dump())
        with pytest.raises(PersistenceError):
            await child.append("run.started", {}, "wrong-run")
        await child.append("run.started", {}, "child-run")
        payload = {
            "call_id": link.call_id,
            "result": ToolResult(call_id=link.call_id, ok=True).model_dump(),
            "metrics": RunMetrics().model_dump(),
        }
        with pytest.raises(PersistenceError):
            await child.append("subagent.finished", payload, "child-run")
        await child.append(
            "run.ended",
            {
                "status": "completed",
                "reason": "done",
                "metrics": RunMetrics().model_dump(),
            },
            "child-run",
        )
        with pytest.raises(PersistenceError):
            await child.append(
                "subagent.finished",
                {
                    **payload,
                    "metrics": RunMetrics(input_tokens=99).model_dump(),
                },
                "child-run",
            )
        await child.append("subagent.finished", payload, "child-run")
        with pytest.raises(PersistenceError):
            await child.append("subagent.finished", payload, "child-run")
    finally:
        await child.close()
    assert list_sessions(tmp_path) == []


async def test_existing_only_open_never_creates_journal(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await JsonlSession.open(tmp_path, "missing", create=False)
    assert not (tmp_path / ".hiro-agent/sessions/missing.jsonl").exists()
    path = tmp_path / ".hiro-agent/sessions/empty.jsonl"
    path.touch()
    session = await JsonlSession.open(tmp_path, "empty", create=False)
    try:
        assert session.events == []
    finally:
        await session.close()
    assert path.read_bytes() == b""


async def test_completion_metadata_can_follow_terminal_interruption() -> None:
    session = MemorySession("parent")
    link = await pending_delegate(session)
    await session.append("delegation.started", link.model_dump(), "parent-run")
    interrupted = ToolResult(call_id=link.call_id, ok=False, error="cancelled", outcome="unknown")
    await session.append("tool.result", interrupted.model_dump(), "parent-run")
    await session.append(
        "delegation.finished",
        {
            "call_id": link.call_id,
            "result": ToolResult(call_id=link.call_id, ok=True).model_dump(),
            "metrics": RunMetrics(input_tokens=10).model_dump(),
        },
        "parent-run",
    )
    assert await recover_session(session) == []
    assert session.events[-1].data["metrics"]["input_tokens"] == 10
    assert [event.data for event in session.events if event.type == "tool.result"] == [
        interrupted.model_dump()
    ]


@pytest.mark.parametrize("completed", [False, True])
async def test_recovered_child_metrics_exclude_offline_parent_elapsed(completed: bool) -> None:
    from datetime import UTC, datetime, timedelta

    from hiro_agent.session import settle_tool_calls

    session = MemorySession("parent")
    link = await pending_delegate(session)
    await session.append("delegation.started", link.model_dump(), "parent-run")
    base = datetime(2000, 1, 1, tzinfo=UTC)
    for index, event in enumerate(session._events):
        event.timestamp = base + timedelta(seconds=index)
    await session.append(
        "delegation.finished",
        {
            "call_id": link.call_id,
            "result": ToolResult(
                call_id=link.call_id,
                ok=completed,
                error=None if completed else "TOOL_OUTCOME_UNKNOWN",
                outcome="completed" if completed else "unknown",
            ).model_dump(),
            "metrics": RunMetrics(input_tokens=17, elapsed_ms=500000).model_dump(),
            "recovered": True,
        },
        "parent-run",
    )
    # A second recovery after the cached result was settled must also ignore offline time.
    await settle_tool_calls(session)
    await recover_session(session)
    metrics = session.events[-1].data["metrics"]
    assert metrics["input_tokens"] == 17
    assert metrics["elapsed_ms"] == 4000
