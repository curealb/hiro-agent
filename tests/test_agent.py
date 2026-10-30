import asyncio
import json

import pytest

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import AssistantMessage, ModelResponse, RunStatus, ToolCall, Usage
from hiro_agent.errors import ContextLimitError, HiroError, ModelError, PersistenceError
from hiro_agent.models import MockModel, ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession, recover_session
from hiro_agent.tools import ToolExecutor, build_file_tools


def call(name, arguments, id="call-1"):
    return ToolCall(id=id, name=name, arguments=json.dumps(arguments))


def response(*calls, text=None, finish=None, usage=None):
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason=finish or ("tool_calls" if calls else "stop"),
        usage=usage,
    )


async def allow(*args):
    return True


def harness(tmp_path, responses=None, model=None, config=None, approval=allow, session=None):
    session = session or MemorySession()
    model = model or ScriptedModel(responses or [])
    tools = ToolExecutor(
        build_file_tools(tmp_path, ToolsConfig()), session, PermissionPolicy("ask-write", approval)
    )
    agent = Agent(
        session,
        model,
        tools,
        PassthroughContext(),
        IdentityPreprocessor(),
        config or BudgetConfig(),
    )
    return agent, session, model


async def test_mock_read_write_verify_answer_and_new_turn(tmp_path):
    agent, session, _ = harness(tmp_path, model=MockModel())
    result = await agent.run("演示")
    assert result.status == RunStatus.COMPLETED
    assert "演示完成" in result.output
    assert (tmp_path / "hiro-demo.txt").read_text() == "Hello from Hiro!\n"
    assert result.metrics.model_requests == 4
    assert result.metrics.tool_calls == 3
    assert [m.role for m in session.derive_messages()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    with pytest.raises(HiroError, match="没有可继续"):
        await agent.continue_run()
    second = await agent.run("再演示")
    assert second.status == RunStatus.COMPLETED
    assert second.run_id != result.run_id


async def test_mock_continues_after_write_without_repeating_side_effect(tmp_path):
    agent, session, _ = harness(
        tmp_path, model=MockModel(), config=BudgetConfig(max_model_requests=2)
    )
    first = await agent.run("演示")
    assert first.status == RunStatus.PAUSED
    second = await agent.continue_run()
    assert second.status == RunStatus.COMPLETED
    calls = [e.data["call"]["name"] for e in session.events if e.type == "tool.started"]
    assert calls == ["read_file", "write_file", "run_command"]


async def test_invalid_args_denial_count_and_complete_history(tmp_path):
    broken = ToolCall(id="invalid", name="read_file", arguments="{broken")
    denied = call("write_file", {"path": "blocked.txt", "content": "blocked"}, "denied")
    agent, session, _ = harness(
        tmp_path, [response(broken, denied), response(text="权限不足")], approval=None
    )
    result = await agent.run("test")
    assert result.status == RunStatus.COMPLETED
    assert result.metrics.tool_calls == 2
    assert result.metrics.tool_errors == 2
    assert result.metrics.permission_denials == 1
    assert not (tmp_path / "blocked.txt").exists()
    assert len([m for m in session.derive_messages() if m.role == "tool"]) == 2


async def test_budget_pause_then_explicit_continue_fresh_budget_no_duplicate_task(tmp_path):
    agent, session, model = harness(
        tmp_path,
        [response(call("list_files", {})), response(text="done")],
        config=BudgetConfig(max_model_requests=1),
    )
    first = await agent.run("original task")
    assert first.status == RunStatus.PAUSED
    assert first.metrics.model_requests == 1
    second = await agent.continue_run()
    assert second.status == RunStatus.COMPLETED
    assert second.metrics.model_requests == 1
    assert len([e for e in session.events if e.type == "task.received"]) == 1
    assert len([e for e in session.events if e.type == "task.continued"]) == 1
    assert model.requests[-1].messages[-1].role == "user"


async def test_tool_budget_settles_unstarted_calls(tmp_path):
    calls = [call("list_files", {}, str(n)) for n in range(3)]
    agent, session, _ = harness(tmp_path, [response(*calls)], config=BudgetConfig(max_tool_calls=1))
    result = await agent.run("test")
    assert result.status == RunStatus.PAUSED
    assert result.metrics.tool_calls == 1
    tools = [json.loads(m.content) for m in session.derive_messages() if m.role == "tool"]
    assert [t["call_id"] for t in tools] == ["0", "1", "2"]
    assert [t["outcome"] for t in tools] == ["completed", "not_started", "not_started"]


async def test_length_response_never_executes_its_calls(tmp_path):
    agent, session, _ = harness(
        tmp_path,
        [
            response(
                call("write_file", {"path": "bad.txt", "content": "bad"}),
                text="partial",
                finish="length",
            )
        ],
    )
    result = await agent.run("test")
    assert result.status == RunStatus.PAUSED
    assert not (tmp_path / "bad.txt").exists()
    assert result.metrics.tool_calls == 0
    assert json.loads(session.derive_messages()[-1].content)["error"] == "TOOL_NOT_STARTED"


@pytest.mark.parametrize(
    "error,status",
    [
        (ModelError("network failed"), RunStatus.FAILED),
        (ContextLimitError("too long"), RunStatus.PAUSED),
    ],
)
async def test_model_failure_journal_and_no_auto_retry(tmp_path, error, status):
    agent, session, model = harness(tmp_path, [error])
    result = await agent.run("test")
    assert result.status == status
    assert len(model.requests) == 1
    assert any(e.type == "model.failed" for e in session.events)
    assert [m.role for m in session.derive_messages()] == ["user"]


async def test_cancel_during_model_and_concurrent_run_rejected(tmp_path):
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class WaitingModel:
        async def complete(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def close(self):
            pass

    agent, session, _ = harness(tmp_path, model=WaitingModel())
    running = asyncio.create_task(agent.run("test"))
    await entered.wait()
    with pytest.raises(HiroError, match="正在执行"):
        await agent.run("other")
    agent.cancel()
    result = await asyncio.wait_for(running, 2)
    assert result.status == RunStatus.CANCELED
    assert cleaned.is_set()
    assert any(e.type == "model.interrupted" for e in session.events)


async def test_cancel_approval_balances_all_calls_and_no_writes(tmp_path):
    entered = asyncio.Event()

    async def approve(*args):
        entered.set()
        await asyncio.Event().wait()

    agent, session, _ = harness(
        tmp_path,
        [
            response(
                call("write_file", {"path": "a", "content": "a"}, "a"),
                call("write_file", {"path": "b", "content": "b"}, "b"),
            )
        ],
        approval=approve,
    )
    running = asyncio.create_task(agent.run("test"))
    await entered.wait()
    agent.cancel()
    result = await asyncio.wait_for(running, 2)
    assert result.status == RunStatus.CANCELED
    assert not list(tmp_path.iterdir())
    assert len([m for m in session.derive_messages() if m.role == "tool"]) == 2


async def test_total_timeout_includes_approval(tmp_path):
    async def approve(*args):
        await asyncio.Event().wait()

    agent, session, _ = harness(
        tmp_path,
        [response(call("write_file", {"path": "a", "content": "a"}))],
        approval=approve,
        config=BudgetConfig(timeout_seconds=0.05),
    )
    result = await asyncio.wait_for(agent.run("test"), 2)
    assert result.status == RunStatus.PAUSED
    assert not list(tmp_path.iterdir())
    session.derive_messages()


async def test_persistence_failure_stops_next_tool_and_poison_agent(tmp_path):
    class FailingSession(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if event_type == "tool.result":
                raise PersistenceError("injected fsync failure")
            return await super().append(event_type, data, run_id)

    agent, session, _ = harness(
        tmp_path,
        [
            response(
                call("write_file", {"path": "a", "content": "a"}, "a"),
                call("write_file", {"path": "b", "content": "b"}, "b"),
            )
        ],
        session=FailingSession(),
    )
    result = await agent.run("test")
    assert result.status == RunStatus.FAILED
    assert (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()
    assert not any(e.type == "run.ended" for e in session.events)
    with pytest.raises(HiroError, match="持久化失败"):
        await agent.run("next")


async def test_usage_and_missing_usage_are_separate(tmp_path):
    agent, _, _ = harness(
        tmp_path,
        [
            response(call("list_files", {}), usage=Usage(input_tokens=10, output_tokens=4)),
            response(text="done"),
        ],
    )
    result = await agent.run("test")
    assert result.metrics.input_tokens == 10
    assert result.metrics.output_tokens == 4
    assert result.metrics.usage_missing == 1
    assert result.metrics.cache_usage_missing == 2
    assert result.metrics.cache_reported_input_tokens == 0


async def test_model_finished_reports_each_durable_response_usage_once(tmp_path):
    agent, session, _ = harness(
        tmp_path,
        [
            response(
                call("list_files", {}),
                usage=Usage(input_tokens=10, output_tokens=4, cached_input_tokens=0),
            ),
            response(
                text="done", usage=Usage(input_tokens=20, output_tokens=5, cached_input_tokens=8)
            ),
        ],
    )
    observed = []
    agent.events.subscribe(lambda name, data: observed.append((name, data)))
    result = await agent.run("test")
    assert result.metrics.cached_input_tokens == 8
    assert result.metrics.cache_reported_input_tokens == 30
    assert result.metrics.cache_usage_missing == 0
    assert result.metrics.input_tokens + result.metrics.output_tokens == 39
    finished = [data for name, data in observed if name == "model.finished"]
    durable = [event for event in session.events if event.type == "assistant.message"]
    assert len(finished) == len(durable) == 2
    assert len({data["attempt_id"] for data in finished}) == 2
    for data, event in zip(finished, durable, strict=True):
        assert data["attempt_id"] == event.data["attempt_id"]
        assert data["usage"] == event.data["usage"]
        assert data["finish_reason"] == event.data["finish_reason"]
        assert data["session_id"] == session.session_id
        assert data["run_id"] == result.run_id


async def test_empty_crashed_new_run_does_not_continue_prior_completed_task(tmp_path):
    agent, session, _ = harness(tmp_path, [response(text="done")])
    assert (await agent.run("already finished")).status == RunStatus.COMPLETED
    await session.append("run.started", {"continued": False}, "empty-run")
    await recover_session(session)
    with pytest.raises(HiroError, match="尚未记录任务"):
        await agent.continue_run()
