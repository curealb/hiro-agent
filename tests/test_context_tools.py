from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.budget import Budget
from hiro_agent.config import BudgetConfig, ToolsConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    DataModel,
    ToolCall,
    ToolExecutionContext,
)
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession
from hiro_agent.tools import ToolExecutor, ToolOutput, ToolRegistry, build_file_tools


class TaskArgs(DataModel):
    task: str


async def declare(session: MemorySession, *calls: ToolCall) -> None:
    if not session.events:
        await session.append("run.started", {"continued": False}, "run")
    await session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=list(calls)).model_dump(mode="json")},
        "run",
    )


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


async def test_context_is_internal_and_durable_start_precedes_handler() -> None:
    session = MemorySession("parent")
    budget = Budget(BudgetConfig())
    cancellation = Cancellation()
    contexts = []
    observed = []

    async def handler(args: TaskArgs, context: ToolExecutionContext) -> dict[str, str]:
        contexts.append(context)
        assert session.events[-1].type == "tool.started"
        assert args.task == "inspect"
        return {"summary": "found"}

    registry = ToolRegistry()
    registry.register_contextual("delegate", "delegate", TaskArgs, handler)
    executor = ToolExecutor(
        registry,
        session,
        PermissionPolicy(),
        lambda name, data: observed.append((name, data)),
    )
    schema = executor.schemas()[0]["function"]["parameters"]
    assert set(schema["properties"]) == {"task"}
    call = tool_call("delegate", {"task": "inspect"})
    await declare(session, call)
    result = await executor.execute(call, "run", budget, cancellation)
    assert result.ok and result.value == {"summary": "found"}
    assert len(contexts) == 1
    context = contexts[0]
    assert context.session is session
    assert context.budget is budget
    assert context.cancellation is cancellation
    assert context.run_id == "run" and context.call_id == "call"
    assert session.events[-1].data == result.model_dump(mode="json")
    assert observed[-1][1] == {
        **result.model_dump(mode="json"),
        "session_id": "parent",
        "run_id": "run",
    }

    forged = tool_call("delegate", {"task": "inspect", "budget": {}, "run_id": "fake"}, "bad")
    await declare(session, forged)
    rejected = await executor.execute(forged, "run", budget, cancellation)
    assert rejected.error == "invalid_arguments"
    assert len(contexts) == 1
    assert budget.metrics.tool_calls == 2


async def test_contextual_mutation_still_requires_permission() -> None:
    called = False

    async def handler(_args: TaskArgs, _context: ToolExecutionContext) -> None:
        nonlocal called
        called = True

    registry = ToolRegistry()
    registry.register_contextual("delegate", "delegate", TaskArgs, handler, mutating=True)
    session = MemorySession()
    budget = Budget(BudgetConfig())
    call = tool_call("delegate", {"task": "inspect"})
    await declare(session, call)
    result = await ToolExecutor(registry, session, PermissionPolicy("read-only")).execute(
        call, "run", budget, Cancellation()
    )
    assert result.error == "permission_denied"
    assert not called
    assert not any(event.type == "tool.started" for event in session.events)


async def test_contextual_unknown_outcome_is_preserved_in_journal() -> None:
    async def handler(_args: TaskArgs, _context: ToolExecutionContext) -> ToolOutput:
        return ToolOutput(
            value={"child_session_id": "child"},
            ok=False,
            error="SUBAGENT_INTERRUPTED",
            message="Child outcome requires inspection",
            outcome="unknown",
        )

    session = MemorySession()
    registry = ToolRegistry()
    registry.register_contextual("delegate", "delegate", TaskArgs, handler)
    call = tool_call("delegate", {"task": "inspect"})
    await declare(session, call)
    result = await ToolExecutor(registry, session, PermissionPolicy()).execute(
        call, "run", Budget(BudgetConfig()), Cancellation()
    )
    assert result.outcome == "unknown"
    assert result.error == "SUBAGENT_INTERRUPTED"
    assert session.events[-1].data == result.model_dump(mode="json")


async def test_restricted_registry_prevents_execution_even_with_approval(tmp_path: Path) -> None:
    registry = build_file_tools(tmp_path, ToolsConfig())
    restricted = registry.restricted(["list_files", "read_file", "search_text"])
    assert {item["function"]["name"] for item in restricted.schemas()} == {
        "list_files",
        "read_file",
        "search_text",
    }
    assert len(registry.schemas()) == 6
    (tmp_path / "source.py").write_text("answer = 42\n")

    async def allow(*_args: Any) -> bool:
        pytest.fail("Forbidden tools must not reach approval")

    session = MemorySession()
    executor = ToolExecutor(restricted, session, PermissionPolicy(callback=allow))
    budget = Budget(BudgetConfig())
    calls = [
        tool_call("write_file", {"path": "changed", "content": "bad"}, "write"),
        tool_call("run_command", {"executable": "touch", "args": ["command-ran"]}, "command"),
        tool_call("delegate_task", {"role": "explorer", "task": "recurse"}, "recursive"),
        tool_call("read_file", {"path": "source.py"}, "read"),
    ]
    await declare(session, *calls)
    results = [await executor.execute(call, "run", budget, Cancellation()) for call in calls]
    assert all(result.error == "unknown_tool" for result in results[:3])
    assert results[-1].ok
    assert not (tmp_path / "changed").exists()
    assert not (tmp_path / "command-ran").exists()
    assert budget.metrics.tool_calls == 4
    assert len([event for event in session.events if event.type == "tool.started"]) == 1
    session.derive_messages()
    with pytest.raises(ValueError, match="Unknown tools"):
        registry.restricted(["missing"])


async def test_child_tool_failures_are_counted_once_in_root_budget(tmp_path: Path) -> None:
    root = Budget(BudgetConfig())
    child = Budget(BudgetConfig(), parent=root)
    session = MemorySession()
    registry = build_file_tools(tmp_path, ToolsConfig())
    executor = ToolExecutor(registry, session, PermissionPolicy("read-only"))
    calls = [
        tool_call("write_file", {"path": "changed", "content": "bad"}, "write"),
        tool_call("unknown", {}, "unknown"),
    ]
    await declare(session, *calls)
    for call in calls:
        await executor.execute(call, "run", child, Cancellation())
    for metrics in [root.metrics, child.metrics]:
        assert metrics.tool_calls == 2
        assert metrics.tool_errors == 2
        assert metrics.permission_denials == 1


def test_contextual_duplicate_registration_is_rejected() -> None:
    async def handler(_args: TaskArgs, _context: ToolExecutionContext) -> None:
        pass

    registry = ToolRegistry()
    registry.register_contextual("delegate", "delegate", TaskArgs, handler)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_contextual("delegate", "delegate", TaskArgs, handler)
