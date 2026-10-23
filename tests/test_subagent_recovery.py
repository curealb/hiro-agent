"""Simulate process loss at durable delegation boundaries, without replaying work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, SubagentsConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import AssistantMessage, ModelResponse, RunStatus, ToolCall
from hiro_agent.errors import PersistenceError
from hiro_agent.models import ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import JsonlSession, recover_session
from hiro_agent.subagents import SubagentManager, recover_delegations
from hiro_agent.tools import ToolExecutor, build_file_tools


def response(name: str | None = None, arguments: dict | None = None, text: str = ""):
    return ModelResponse(
        assistant=AssistantMessage(
            text=text,
            tool_calls=[ToolCall(id=name, name=name, arguments=json.dumps(arguments or {}))]
            if name
            else [],
        ),
        finish_reason="tool_calls" if name else "stop",
    )


async def completed_fixture(workspace: Path):
    (workspace / "source.py").write_text("answer = 42\n")
    report = json.dumps({"summary": "Found answer", "findings": [], "suggestions": []})
    model = ScriptedModel(
        [
            response("delegate_task", {"role": "explorer", "task": "inspect source.py"}),
            response("read_file", {"path": "source.py"}),
            response(text=report),
            response(text="Done"),
        ]
    )
    parent = await JsonlSession.open(workspace, "parent")
    children = []

    async def sessions(session_id):
        child = await JsonlSession.open(workspace, session_id)
        children.append(child)
        return child

    registry = build_file_tools(workspace, ToolsConfig())
    manager = SubagentManager(model, registry, sessions, SubagentsConfig())
    manager.register(registry)
    agent = Agent(
        parent,
        model,
        ToolExecutor(registry, parent, PermissionPolicy("read-only")),
        PassthroughContext(),
        IdentityPreprocessor(),
        BudgetConfig(),
    )
    result = await agent.run("Inspect")
    await agent.close()
    await manager.close()
    await parent.close()
    assert result.status == RunStatus.COMPLETED, result.model_dump()
    return parent, children[0], model


def truncate_after(session: JsonlSession, kind: str) -> None:
    events = session.events
    index = next(index for index, event in enumerate(events) if event.type == kind)
    session.path.write_text(
        "".join(event.model_dump_json() + "\n" for event in events[: index + 1])
    )


@pytest.mark.parametrize(
    ("parent_cut", "child_cut", "expected_error"),
    [
        ("tool.called", None, "TOOL_NOT_STARTED"),
        ("tool.started", None, "TOOL_OUTCOME_UNKNOWN"),
        ("delegation.started", None, "TOOL_OUTCOME_UNKNOWN"),
        ("delegation.started", "session.created", "TOOL_OUTCOME_UNKNOWN"),
        ("delegation.started", "subagent.created", "TOOL_OUTCOME_UNKNOWN"),
        ("delegation.started", "tool.started", "SUBAGENT_INCOMPLETE"),
        ("delegation.started", "run.ended", None),
        ("delegation.started", "subagent.finished", None),
        ("delegation.finished", "subagent.finished", None),
        ("tool.result", "subagent.finished", None),
        ("run.ended", "subagent.finished", None),
    ],
)
async def test_crash_cut_recovery_pairs_once_without_replaying_models_or_tools(
    tmp_path: Path, parent_cut: str, child_cut: str | None, expected_error: str | None
) -> None:
    old_parent, old_child, model = await completed_fixture(tmp_path)
    truncate_after(old_parent, parent_cut)
    if child_cut is None:
        old_child.path.unlink()
    else:
        truncate_after(old_child, child_cut)
    original_requests = len(model.requests)
    parent = await JsonlSession.open(tmp_path, "parent")
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    results = [event for event in parent.events if event.type == "tool.result"]
    assert len(results) == 1
    assert results[0].data["error"] == expected_error
    assert results[0].data["ok"] == (expected_error is None)
    if expected_error is None:
        assert results[0].data["value"]["report"]["summary"] == "Found answer"
    parent.derive_messages()
    stable = parent.path.read_bytes()
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    assert parent.path.read_bytes() == stable
    assert len(model.requests) == original_requests
    assert (tmp_path / "source.py").read_text() == "answer = 42\n"
    if child_cut is None:
        assert not old_child.path.exists(), "Recovery must not create an unstarted child journal"
    else:
        child = await JsonlSession.open(tmp_path, old_child.session_id)
        child.derive_messages()
        if child_cut == "tool.started":
            child_results = [event for event in child.events if event.type == "tool.result"]
            assert len(child_results) == 1
            assert child_results[0].data["error"] == "TOOL_OUTCOME_UNKNOWN"
        await child.close()
    await parent.close()


async def test_recovery_refuses_a_busy_child_until_its_owner_releases_it(tmp_path: Path) -> None:
    old_parent, old_child, _ = await completed_fixture(tmp_path)
    truncate_after(old_parent, "delegation.started")
    parent = await JsonlSession.open(tmp_path, "parent")
    child = await JsonlSession.open(tmp_path, old_child.session_id)
    try:
        with pytest.raises(PersistenceError, match="already open"):
            await recover_delegations(tmp_path, parent, 16000)
        assert not any(event.type == "delegation.finished" for event in parent.events)
        assert not any(event.type == "tool.result" for event in parent.events)
    finally:
        await child.close()
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    assert len([event for event in parent.events if event.type == "tool.result"]) == 1
    parent.derive_messages()
    await parent.close()


async def test_corrupt_child_stops_recovery_without_settling_parent(tmp_path: Path) -> None:
    old_parent, old_child, _ = await completed_fixture(tmp_path)
    truncate_after(old_parent, "delegation.started")
    lines = old_child.path.read_text().splitlines(keepends=True)
    lines.insert(2, "not-json\n")
    old_child.path.write_text("".join(lines))
    parent = await JsonlSession.open(tmp_path, "parent")
    try:
        with pytest.raises(PersistenceError, match="Invalid session journal"):
            await recover_delegations(tmp_path, parent, 16000)
        assert not any(event.type == "delegation.finished" for event in parent.events)
        assert not any(event.type == "tool.result" for event in parent.events)
    finally:
        await parent.close()


async def test_mismatched_child_parent_link_is_not_accepted(tmp_path: Path) -> None:
    old_parent, old_child, _ = await completed_fixture(tmp_path)
    truncate_after(old_parent, "delegation.started")
    # A syntactically valid child with a different parent is not this delegation.
    events = old_child.events
    for event in events:
        if event.type == "subagent.created":
            event.data["parent_session_id"] = "another-parent"
    old_child.path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    parent = await JsonlSession.open(tmp_path, "parent")
    try:
        with pytest.raises(PersistenceError):
            await recover_delegations(tmp_path, parent, 16000)
        assert not any(event.type == "tool.result" for event in parent.events)
    finally:
        await parent.close()


async def test_recovery_aggregates_child_metrics_without_double_counting(tmp_path: Path) -> None:
    old_parent, old_child, _ = await completed_fixture(tmp_path)
    truncate_after(old_parent, "delegation.started")
    truncate_after(old_child, "run.ended")
    parent = await JsonlSession.open(tmp_path, "parent")
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    ended = next(event for event in parent.events if event.type == "run.ended")
    assert ended.data["metrics"]["model_requests"] == 3
    assert ended.data["metrics"]["tool_calls"] == 2
    assert ended.data["metrics"]["worker_steps"] == 3
    assert ended.data["metrics"]["usage_missing"] == 3
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    assert [event for event in parent.events if event.type == "run.ended"] == [ended]
    await parent.close()


async def test_next_model_request_after_recovery_has_complete_pairing(tmp_path: Path) -> None:
    old_parent, old_child, _ = await completed_fixture(tmp_path)
    truncate_after(old_parent, "delegation.started")
    truncate_after(old_child, "run.ended")
    parent = await JsonlSession.open(tmp_path, "parent")
    await recover_delegations(tmp_path, parent, 16000)
    await recover_session(parent)
    model = ScriptedModel([response(text="Resumed using child result")])
    agent = Agent(
        parent,
        model,
        ToolExecutor(build_file_tools(tmp_path, ToolsConfig()), parent, PermissionPolicy()),
        PassthroughContext(),
        IdentityPreprocessor(),
        BudgetConfig(),
    )
    result = await agent.continue_run()
    assert result.status == RunStatus.COMPLETED
    messages = model.requests[0].messages
    declared = [call.id for message in messages for call in message.tool_calls]
    paired = [message.tool_call_id for message in messages if message.role == "tool"]
    assert declared == paired == ["delegate_task"]
    assert sum(event.type == "delegation.started" for event in parent.events) == 1
    assert sum(event.type == "task.received" for event in parent.events) == 1
    await agent.close()
    await parent.close()
