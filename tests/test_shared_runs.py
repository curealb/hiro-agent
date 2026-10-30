"""Child runs keep their own history while accounting to one root allowance."""

import asyncio
import json

import pytest

from hiro_agent.agent import Agent
from hiro_agent.budget import Budget
from hiro_agent.config import BudgetConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    ModelResponse,
    RunScope,
    RunStatus,
    ToolCall,
    Usage,
    usage_counts,
)
from hiro_agent.core import EventBus
from hiro_agent.errors import BudgetExceeded, PersistenceError
from hiro_agent.models import ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession
from hiro_agent.tools import ToolExecutor, build_file_tools


def child_scope(budget, cancellation=None):
    return RunScope(
        run_id="child-run",
        budget=budget,
        cancellation=cancellation or Cancellation(),
        root_session_id="root-session",
        root_run_id="root-run",
        parent_session_id="root-session",
        parent_run_id="root-run",
        role="explorer",
    )


def child_agent(tmp_path, model, session=None, events=None):
    session = session or MemorySession("child-session")
    executor = ToolExecutor(
        build_file_tools(tmp_path, ToolsConfig()), session, PermissionPolicy("read-only")
    )
    return Agent(
        session,
        model,
        executor,
        PassthroughContext(),
        IdentityPreprocessor(),
        BudgetConfig(),
        events,
    )


def test_root_allowance_applies_across_siblings_and_claims_are_atomic():
    root = Budget(BudgetConfig(max_model_requests=2, max_tool_calls=2))
    first = Budget(BudgetConfig(max_model_requests=1, max_tool_calls=1), parent=root)
    second = Budget(BudgetConfig(), parent=root)
    first.claim_model()
    first.claim_tool()
    for claim in (first.claim_model, first.claim_tool):
        with pytest.raises(BudgetExceeded):
            claim()
    assert root.metrics.model_requests == root.metrics.tool_calls == 1
    second.claim_model()
    second.claim_tool()
    for claim in (second.claim_model, second.claim_tool):
        with pytest.raises(BudgetExceeded):
            claim()
    assert second.metrics.model_requests == second.metrics.tool_calls == 1
    assert root.metrics.model_requests == root.metrics.tool_calls == 2


def test_child_deadline_cannot_extend_root_and_elapsed_is_not_summed(monkeypatch):
    now = 100.0
    monkeypatch.setattr("hiro_agent.budget.time.monotonic", lambda: now)
    root = Budget(BudgetConfig(timeout_seconds=10))
    now = 108.0
    child = Budget(BudgetConfig(timeout_seconds=600), parent=root)
    assert child.remaining == root.remaining == 2
    now = 110.0
    with pytest.raises(BudgetExceeded):
        child.claim_model()
    assert child.finish().elapsed_ms == 2000
    assert root.finish().elapsed_ms == 10000


async def test_scoped_run_tracks_local_and_total_usage_with_lineage(tmp_path):
    root = Budget(BudgetConfig())
    root.claim_model()
    root.record(**usage_counts(Usage(input_tokens=7, output_tokens=3, cached_input_tokens=3)))
    child = Budget(BudgetConfig(), parent=root)
    events = EventBus()
    observed = []
    events.subscribe(lambda name, data: observed.append((name, data)))
    model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(text="child answer"),
                finish_reason="stop",
                usage=Usage(input_tokens=11, output_tokens=5, cached_input_tokens=6),
            )
        ]
    )
    agent = child_agent(tmp_path, model, events=events)
    scope = child_scope(child)
    result = await agent.run("investigate", scope=scope)
    assert result.status == RunStatus.COMPLETED
    assert result.run_id == "child-run"
    assert result.metrics.model_requests == 1
    assert result.metrics.input_tokens == 11
    assert root.metrics.model_requests == 2
    assert root.metrics.input_tokens == 18
    assert root.metrics.output_tokens == 8
    assert root.metrics.worker_steps == 2
    assert result.metrics.cached_input_tokens == 6
    assert result.metrics.cache_reported_input_tokens == 11
    assert root.metrics.cached_input_tokens == 9
    assert root.metrics.cache_reported_input_tokens == 18
    assert root.metrics.cache_usage_missing == 0
    for _, data in observed:
        assert data["session_id"] == "child-session"
        assert data["run_id"] == "child-run"
        assert data["root_run_id"] == "root-run"
        assert data["role"] == "explorer"
    started = next(event for event in agent.session.events if event.type == "run.started")
    assert started.data["parent_session_id"] == "root-session"
    assert agent.session.events[-1].data["output"] == "child answer"
    finished = [data for name, data in observed if name == "model.finished"]
    assert len(finished) == 1
    assert finished[0]["usage"]["cached_input_tokens"] == 6
    assert finished[0]["attempt_id"]
    await agent.close()
    assert not scope.cancellation.cancelled, "Closing a completed child must not cancel the root"


async def test_child_cap_pauses_locally_without_spending_extra_root_request(tmp_path):
    root = Budget(BudgetConfig(max_model_requests=10))
    child = Budget(BudgetConfig(max_model_requests=1), parent=root)
    model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(
                    tool_calls=[ToolCall(id="list", name="list_files", arguments="{}")]
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    agent = child_agent(tmp_path, model)
    result = await agent.run("inspect", scope=child_scope(child))
    assert result.status == RunStatus.PAUSED
    assert root.metrics.model_requests == child.metrics.model_requests == 1
    assert root.metrics.tool_calls == child.metrics.tool_calls == 1
    root.claim_model()
    assert root.metrics.model_requests == 2
    assert [message.role for message in agent.session.derive_messages()] == [
        "user",
        "assistant",
        "tool",
    ]


async def test_root_cancellation_waits_for_child_cleanup(tmp_path):
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class WaitingModel:
        async def complete(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await finish_cleanup.wait()

        async def close(self):
            pass

    cancellation = Cancellation()
    root = Budget(BudgetConfig())
    agent = child_agent(tmp_path, WaitingModel())
    running = asyncio.create_task(
        agent.run("inspect", scope=child_scope(Budget(BudgetConfig(), root), cancellation))
    )
    await entered.wait()
    cancellation.cancel()
    await cleanup_started.wait()
    assert not running.done()
    finish_cleanup.set()
    result = await asyncio.wait_for(running, 2)
    assert result.status == RunStatus.CANCELED
    assert agent.session.events[-1].type == "run.ended"
    assert any(event.type == "model.interrupted" for event in agent.session.events)


@pytest.mark.parametrize("boundary", ["assistant.message", "run.ended"])
async def test_child_persistence_failure_is_explicit_in_result(tmp_path, boundary):
    class BrokenSession(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if event_type == boundary:
                raise PersistenceError("injected persistence failure")
            return await super().append(event_type, data, run_id)

    agent = child_agent(
        tmp_path,
        ScriptedModel(
            [ModelResponse(assistant=AssistantMessage(text="done"), finish_reason="stop")]
        ),
        session=BrokenSession("child-session"),
    )
    result = await agent.run("inspect", scope=child_scope(Budget(BudgetConfig())))
    assert result.status == RunStatus.FAILED
    assert result.persistence_failed


async def test_child_denials_and_errors_are_counted_in_root(tmp_path):
    root = Budget(BudgetConfig())
    child = Budget(BudgetConfig(), root)
    model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="write",
                            name="write_file",
                            arguments=json.dumps({"path": "blocked", "content": "never"}),
                        ),
                        ToolCall(id="invalid", name="read_file", arguments="{"),
                    ]
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(assistant=AssistantMessage(text="reported"), finish_reason="stop"),
        ]
    )
    result = await child_agent(tmp_path, model).run("inspect", scope=child_scope(child))
    assert result.status == RunStatus.COMPLETED
    assert root.metrics.permission_denials == child.metrics.permission_denials == 1
    assert root.metrics.tool_errors == child.metrics.tool_errors == 2
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize("interruption", ["deadline", "token", "caller"])
async def test_cleanup_persistence_failure_outranks_interruption(interruption):
    from hiro_agent.execution import cancellable

    cancellation = Cancellation()
    entered = asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            raise PersistenceError("cleanup fsync failed") from None

    running = asyncio.create_task(
        cancellable(operation(), cancellation, 0.01 if interruption == "deadline" else 10)
    )
    await entered.wait()
    if interruption == "token":
        cancellation.cancel()
    elif interruption == "caller":
        running.cancel()
    with pytest.raises(PersistenceError, match="cleanup fsync failed"):
        await asyncio.wait_for(running, 1)


async def test_deadline_stays_budget_pause_when_cleanup_signals_shared_cancellation():
    from hiro_agent.execution import cancellable

    cancellation = Cancellation()

    async def operation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation.cancel()
            return "settled child"

    with pytest.raises(BudgetExceeded, match="时间预算"):
        await cancellable(operation(), cancellation, 0.01)
    assert cancellation.cancelled


async def test_root_exhaustion_in_child_stops_parent_already_declared_write(tmp_path):
    from hiro_agent.contracts import DataModel

    parent_session = MemorySession("root-session")
    child_model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(
                    tool_calls=[ToolCall(id="child-list", name="list_files", arguments="{}")]
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    child = child_agent(tmp_path, child_model)
    registry = build_file_tools(tmp_path, ToolsConfig())
    child_results = []

    async def delegate(arguments, context):
        result = await child.run(
            "inspect",
            scope=child_scope(Budget(BudgetConfig(), context.budget), context.cancellation),
        )
        child_results.append(result)
        return result.model_dump(mode="json")

    registry.register_contextual("inspect_task", "Inspect", DataModel, delegate)

    async def allow(*args):
        return True

    parent_model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(
                    tool_calls=[
                        ToolCall(id="delegate", name="inspect_task", arguments="{}"),
                        ToolCall(
                            id="write",
                            name="write_file",
                            arguments=json.dumps({"path": "blocked", "content": "never"}),
                        ),
                    ]
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    parent = Agent(
        parent_session,
        parent_model,
        ToolExecutor(registry, parent_session, PermissionPolicy("ask-write", allow)),
        PassthroughContext(),
        IdentityPreprocessor(),
        BudgetConfig(max_model_requests=2),
    )
    result = await parent.run("inspect before changing")
    assert result.status == RunStatus.PAUSED
    assert child_results[0].status == RunStatus.PAUSED
    assert result.metrics.model_requests == 2
    assert not (tmp_path / "blocked").exists()
    tools = [json.loads(m.content) for m in parent_session.derive_messages() if m.role == "tool"]
    assert tools[-1]["call_id"] == "write"
    assert tools[-1]["outcome"] == "not_started"


async def test_result_metrics_match_durable_end_even_when_append_takes_time(tmp_path, monkeypatch):
    now = 100.0
    monkeypatch.setattr("hiro_agent.budget.time.monotonic", lambda: now)

    class SlowFinalWrite(MemorySession):
        async def append(self, event_type, data, run_id=None):
            nonlocal now
            event = await super().append(event_type, data, run_id)
            if event_type == "run.ended":
                now += 0.25
            return event

    agent = child_agent(
        tmp_path,
        ScriptedModel(
            [ModelResponse(assistant=AssistantMessage(text="done"), finish_reason="stop")]
        ),
        session=SlowFinalWrite("child-session"),
    )
    result = await agent.run("inspect", scope=child_scope(Budget(BudgetConfig())))
    assert result.metrics.model_dump() == agent.session.events[-1].data["metrics"]
