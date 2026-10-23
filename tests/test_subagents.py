"""Exercise delegation through the real Agent/Executor/Session boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, SubagentsConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import (
    AssistantMessage,
    ModelResponse,
    RunStatus,
    ToolCall,
    ToolResult,
    Usage,
)
from hiro_agent.core import EventBus
from hiro_agent.errors import HiroError, PersistenceError
from hiro_agent.models import ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession
from hiro_agent.subagents import READ_TOOLS, SubagentManager
from hiro_agent.tools import ToolExecutor, build_file_tools


def call(name: str, arguments: dict[str, Any], call_id: str = "delegate") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def delegate(role: str = "explorer", call_id: str = "delegate") -> ToolCall:
    return call("delegate_task", {"role": role, "task": "Inspect source.py"}, call_id)


def response(
    *calls: ToolCall, text: str | None = None, usage: Usage | None = None
) -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason="tool_calls" if calls else "stop",
        usage=usage,
    )


def report() -> str:
    return json.dumps(
        {
            "summary": "Located the answer",
            "findings": [{"path": "source.py", "line": 1, "description": "Defines answer"}],
            "suggestions": ["Check callers"],
        }
    )


def tool_results(session: MemorySession) -> list[ToolResult]:
    return [
        ToolResult.model_validate(event.data)
        for event in session.events
        if event.type == "tool.result"
    ]


def setup(
    workspace: Path,
    responses: list[ModelResponse] | None = None,
    *,
    model: Any = None,
    budget: BudgetConfig | None = None,
    config: SubagentsConfig | None = None,
    parent: MemorySession | None = None,
    child_type: type[MemorySession] = MemorySession,
) -> tuple[Agent, SubagentManager, MemorySession, list[MemorySession], Any, list[Any]]:
    parent = parent or MemorySession("parent")
    children: list[MemorySession] = []
    model = model or ScriptedModel(responses or [])
    registry = build_file_tools(workspace, ToolsConfig())
    observed: list[Any] = []
    events = EventBus()
    events.subscribe(lambda name, data: observed.append((name, data)))

    async def sessions(session_id: str) -> MemorySession:
        # The factory cannot start child work before the parent linkage commits.
        latest = parent.events[-1]
        assert latest.type == "delegation.started"
        assert latest.data["child_session_id"] == session_id
        session = child_type(session_id)
        children.append(session)
        return session

    async def approve(*_args: Any) -> bool:
        return True

    manager = SubagentManager(model, registry, sessions, config or SubagentsConfig(), events)
    manager.register(registry)
    agent = Agent(
        parent,
        model,
        ToolExecutor(registry, parent, PermissionPolicy(callback=approve), events.emit),
        PassthroughContext(),
        IdentityPreprocessor(),
        budget or BudgetConfig(),
        events,
    )
    return agent, manager, parent, children, model, observed


async def test_delegation_isolates_history_and_accounts_usage(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("answer = 42\n")
    agent, manager, parent, children, model, observed = setup(
        tmp_path,
        [
            response(delegate(), usage=Usage(input_tokens=10, output_tokens=2)),
            response(call("read_file", {"path": "source.py"}, "read")),
            response(text=report(), usage=Usage(input_tokens=20, output_tokens=5)),
            response(text="Parent answer", usage=Usage(input_tokens=30, output_tokens=7)),
        ],
    )
    result = await agent.run("parent-private task and context")
    assert result.status == RunStatus.COMPLETED
    assert result.output == "Parent answer"
    assert len(children) == 1
    child = children[0]
    assert child.session_id != parent.session_id
    for request in model.requests[1:3]:
        assert {tool["function"]["name"] for tool in request.tools} == set(READ_TOOLS)
        assert "parent-private" not in json.dumps(request.model_dump())
        assert [message.content for message in request.messages if message.role == "user"] == [
            "Inspect source.py"
        ]
    parent_final_input = model.requests[-1]
    assert "answer = 42" not in json.dumps(parent_final_input.model_dump())
    assert "Located the answer" in json.dumps(parent_final_input.model_dump())
    assert [item.call_id for item in tool_results(parent)] == ["delegate"]
    assert [item.call_id for item in tool_results(child)] == ["read"]
    delegated = tool_results(parent)[0]
    assert delegated.ok and delegated.value["report"]["findings"][0]["line"] == 1
    assert result.metrics.model_requests == 4
    assert result.metrics.tool_calls == 2
    assert result.metrics.input_tokens == 60
    assert result.metrics.output_tokens == 14
    assert result.metrics.usage_missing == 1
    child_end = next(event for event in child.events if event.type == "run.ended")
    assert child_end.data["metrics"]["model_requests"] == 2
    assert child_end.data["metrics"]["input_tokens"] == 20
    assert any(name == "run.ended" and data.get("role") == "explorer" for name, data in observed)
    child.derive_messages()
    parent.derive_messages()
    with pytest.raises(PersistenceError, match="closed"):
        await child.append("probe", {})
    await agent.close()
    await manager.close()


async def test_child_cannot_mutate_execute_or_delegate_even_if_model_requests_it(
    tmp_path: Path,
) -> None:
    agent, manager, parent, children, _, _ = setup(
        tmp_path,
        [
            response(delegate("reviewer")),
            response(
                call("write_file", {"path": "written", "content": "bad"}, "write"),
                call("run_command", {"executable": "touch", "args": ["executed"]}, "command"),
                delegate("explorer", "nested"),
            ),
            response(text=report()),
            response(text="Review complete"),
        ],
    )
    result = await agent.run("Review")
    assert result.status == RunStatus.COMPLETED
    assert len(children) == 1
    assert not (tmp_path / "written").exists()
    assert not (tmp_path / "executed").exists()
    denied = tool_results(children[0])
    assert len(denied) == 3
    assert all(item.error == "unknown_tool" and item.outcome == "not_started" for item in denied)
    assert not any(event.type == "tool.started" for event in children[0].events)
    assert result.metrics.tool_calls == 4 and result.metrics.tool_errors == 3
    assert tool_results(parent)[0].ok
    await agent.close()
    await manager.close()


@pytest.mark.parametrize(
    "child_output",
    [
        "plain text is not a report",
        json.dumps({"summary": "found", "unexpected": True}),
        json.dumps(
            {"summary": "found", "findings": [{"path": "a", "line": 0, "description": "x"}]}
        ),
        json.dumps({"summary": "x" * 300}),
    ],
)
async def test_invalid_or_oversized_reports_are_not_returned_as_success(
    tmp_path: Path, child_output: str
) -> None:
    agent, manager, parent, _, _, _ = setup(
        tmp_path,
        [response(delegate()), response(text=child_output), response(text="Child report rejected")],
        config=SubagentsConfig(max_output_chars=256),
    )
    await agent.run("Inspect")
    result = tool_results(parent)[0]
    assert result.error == "SUBAGENT_INVALID_REPORT"
    assert "report" not in result.value
    assert child_output not in result.model_dump_json()
    await agent.close()
    await manager.close()


async def test_root_model_allowance_is_shared_across_successive_delegations(tmp_path: Path) -> None:
    agent, manager, parent, children, model, _ = setup(
        tmp_path,
        [
            response(delegate(call_id="first"), delegate("reviewer", "second")),
            response(text=report()),
            response(text=report()),
            response(text="Must not be requested"),
        ],
        budget=BudgetConfig(max_model_requests=3),
    )
    result = await agent.run("Inspect and review")
    assert result.status == RunStatus.PAUSED
    assert len(model.requests) == result.metrics.model_requests == 3
    assert len(children) == 2
    assert len({child.session_id for child in children}) == 2
    assert len(tool_results(parent)) == 2
    assert all(item.ok for item in tool_results(parent))
    parent.derive_messages()
    await agent.close()
    await manager.close()


async def test_child_local_cap_returns_incomplete_and_parent_can_answer(tmp_path: Path) -> None:
    agent, manager, parent, children, _, _ = setup(
        tmp_path,
        [response(delegate()), response(call("list_files", {}, "list")), response(text="Partial")],
        config=SubagentsConfig(max_model_requests=1),
    )
    result = await agent.run("Inspect")
    assert result.status == RunStatus.COMPLETED
    assert result.metrics.model_requests == 3
    delegated = tool_results(parent)[0]
    assert delegated.error == "SUBAGENT_INCOMPLETE"
    assert delegated.value["status"] == "paused"
    assert delegated.value["metrics"]["model_requests"] == 1
    assert len(tool_results(children[0])) == 1
    children[0].derive_messages()
    await agent.close()
    await manager.close()


async def test_root_tool_limit_exhausted_inside_child_pauses_parent(tmp_path: Path) -> None:
    agent, manager, parent, children, model, _ = setup(
        tmp_path,
        [
            response(delegate()),
            response(call("list_files", {}, "first"), call("list_files", {}, "over-budget")),
            response(text="Must not answer after global tool budget exhaustion"),
        ],
        budget=BudgetConfig(max_tool_calls=2),
    )
    result = await agent.run("Inspect")
    assert result.status == RunStatus.PAUSED
    assert result.metrics.tool_calls == 2
    assert len(model.requests) == 2
    assert tool_results(parent)[0].error == "budget_exceeded"
    child_results = tool_results(children[0])
    assert child_results[0].ok
    assert child_results[1].error == "budget_exceeded"
    parent.derive_messages()
    children[0].derive_messages()
    await agent.close()
    await manager.close()


@pytest.mark.parametrize("stop", ["cancel", "close_manager", "close_agent", "cancel_task"])
async def test_cancellation_and_disposal_wait_for_child_cleanup(tmp_path: Path, stop: str) -> None:
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class WaitingModel(ScriptedModel):
        async def complete(self, request):
            if self.requests:
                self.requests.append(request.model_copy(deep=True))
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await release_cleanup.wait()
            return await super().complete(request)

    model = WaitingModel([response(delegate())])
    agent, manager, parent, children, _, _ = setup(tmp_path, model=model)
    running = asyncio.create_task(agent.run("Inspect"))
    await asyncio.wait_for(entered.wait(), 2)
    closing = None
    if stop == "cancel":
        agent.cancel()
    elif stop == "cancel_task":
        running.cancel()
    elif stop == "close_manager":
        closing = asyncio.create_task(manager.close())
    else:
        closing = asyncio.create_task(agent.close())
    await asyncio.wait_for(cleanup_started.wait(), 2)
    assert not running.done()
    if closing:
        assert not closing.done()
    release_cleanup.set()
    result = await asyncio.wait_for(running, 2)
    if closing:
        await asyncio.wait_for(closing, 2)
    assert result.status == RunStatus.CANCELED
    assert len(model.requests) == 2
    child = children[0]
    assert any(event.type == "model.interrupted" for event in child.events)
    assert (
        next(event for event in child.events if event.type == "run.ended").data["status"]
        == "canceled"
    )
    assert len(tool_results(parent)) == 1
    parent.derive_messages()
    child.derive_messages()
    with pytest.raises(PersistenceError, match="closed"):
        await child.append("probe", {})
    await agent.close()
    await manager.close()


@pytest.mark.parametrize("boundary", ["assistant.message", "run.ended", "subagent.finished"])
async def test_child_persistence_failure_stops_parent_before_next_tool(
    tmp_path: Path, boundary: str
) -> None:
    class BrokenChild(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if event_type == boundary:
                raise PersistenceError("injected child write failure")
            return await super().append(event_type, data, run_id)

    agent, manager, parent, children, model, _ = setup(
        tmp_path,
        [
            response(delegate(), call("write_file", {"path": "never", "content": "x"}, "next")),
            response(text=report()),
        ],
        child_type=BrokenChild,
    )
    result = await agent.run("Inspect then modify")
    assert result.status == RunStatus.FAILED and result.persistence_failed
    assert len(model.requests) == 2
    assert not (tmp_path / "never").exists()
    assert not any(event.type == "run.ended" for event in parent.events)
    assert not any(event.type == "tool.result" for event in parent.events)
    with pytest.raises(HiroError, match="持久化失败"):
        await agent.run("next task")
    with pytest.raises(PersistenceError, match="closed"):
        await children[0].append("probe", {})
    await agent.close()
    await manager.close()


async def test_failed_parent_linkage_never_creates_a_child(tmp_path: Path) -> None:
    class BrokenParent(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if event_type == "delegation.started":
                raise PersistenceError("injected linkage failure")
            return await super().append(event_type, data, run_id)

    agent, manager, _, children, model, _ = setup(
        tmp_path, [response(delegate())], parent=BrokenParent("parent")
    )
    result = await agent.run("Inspect")
    assert result.persistence_failed
    assert not children
    assert len(model.requests) == 1
    await agent.close()
    await manager.close()


async def test_invalid_delegate_arguments_consume_budget_without_creating_children(
    tmp_path: Path,
) -> None:
    agent, manager, parent, children, model, _ = setup(
        tmp_path,
        [
            response(
                call("delegate_task", {"role": "writer", "task": "modify"}, "role"),
                call("delegate_task", {"role": "explorer", "task": "x", "budget": 10000}, "forge"),
                delegate(call_id="would-spawn"),
            )
        ],
        budget=BudgetConfig(max_tool_calls=2),
    )
    result = await agent.run("Inspect")
    assert result.status == RunStatus.PAUSED
    assert result.metrics.tool_calls == 2
    assert len(model.requests) == 1
    assert not children
    results = tool_results(parent)
    assert [item.error for item in results[:2]] == ["invalid_arguments", "invalid_arguments"]
    assert results[2].error == "budget_exceeded"
    parent.derive_messages()
    await agent.close()
    await manager.close()
