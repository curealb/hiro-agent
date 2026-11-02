"""Offline cross-Bot acceptance through real stores, Runs, and restart boundaries."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig
from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall
from hiro_agent.kernel_contracts import InteractionReply, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.routines import RoutineCreate
from hiro_agent.session import read_session_events


def response(name=None, arguments=None, *, text=None):
    calls = (
        []
        if name is None
        else [ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments))]
    )
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop",
    )


def save_bot(runtime, bot_id, tools):
    runtime.bots.save(
        Config(
            bot=BotConfig(id=bot_id, allowed_tools=tuple(tools)),
            model=ModelConfig(adapter="mock"),
            budget=BudgetConfig(max_model_requests=100, max_tool_calls=100),
        )
    )


async def tick(runtime):
    worker = runtime.worker(poll_interval=0.01)
    try:
        await worker.recover()
        return await worker.tick()
    finally:
        await worker.close()


async def test_routine_write_review_return_survives_restart(tmp_path: Path):
    seed_model = ScriptedModel([response(text="Report method: cite the source and review.")])
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: seed_model)
    save_bot(
        runtime,
        "writer",
        [
            "read_file",
            "read_skill",
            "write_file",
            "send_handoff",
            "read_handoff",
        ],
    )
    (tmp_path / "evidence.txt").write_text("Source evidence: VERIFIED\n")
    save_bot(runtime, "reviewer", ["read_file"])
    runtime.memory.create("writer", "Reports must contain the exact marker VERIFIED.")
    seed = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="Save report method"))
    assert await tick(runtime)
    draft = runtime.skills.draft_from_task("writer", seed.id)
    runtime.skills.enable("writer", draft.document.id)

    first = ScriptedModel(
        [
            response("read_skill", {"skill_id": draft.document.id}),
            response("read_file", {"path": "evidence.txt"}),
            response(
                "write_file",
                {"path": "report.txt", "content": "VERIFIED report\n"},
                text="Report: source evidence confirms VERIFIED.",
            ),
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: first)
    routine = runtime.routines.create(
        RoutineCreate(
            bot_id="writer",
            text="Produce the report using the saved method",
            scheduled_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    assert await tick(runtime)
    source_id = runtime.routines.get(routine.id).last_task_id
    source = runtime.tasks.get(source_id)
    assert source.source == "routine" and source.state == "waiting_approval"
    assert not (tmp_path / "report.txt").exists()
    assert "VERIFIED" in str(first.requests[0])
    assert "Report method" in str(first.requests[1])
    assert "Source evidence: VERIFIED" in str(first.requests[2])
    before_approval = read_session_events(tmp_path, source.session_id)
    report_event = next(
        e
        for e in before_approval
        if "Report: source evidence confirms VERIFIED." in json.dumps(e.data)
    )
    waiting_event = next(e for e in before_approval if e.type == "interaction.waiting")
    assert report_event.seq < waiting_event.seq

    writer = ScriptedModel(
        [
            response(
                "send_handoff",
                {
                    "target_bot_id": "reviewer",
                    "text": "Review report.txt",
                    "idempotency_key": "review",
                },
            ),
            response(text="Report written; review requested."),
        ]
    )
    reviewer = ScriptedModel(
        [
            response("read_file", {"path": "report.txt"}),
            response(text="Approved: VERIFIED report"),
        ]
    )
    restarted = KernelRuntime(
        tmp_path,
        model_factory=lambda config: writer if config.bot.id == "writer" else reviewer,
    )
    reply = InteractionReply(approved=True)
    approved = restarted.interactions.reply(source.interaction_id, reply)
    assert restarted.interactions.reply(source.interaction_id, reply) == approved
    assert await tick(restarted)
    assert (tmp_path / "report.txt").read_text() == "VERIFIED report\n"
    [message] = restarted.handoffs.list("writer", outgoing=True)
    assert message.source_task_id == source.id
    assert await tick(restarted)
    result = restarted.handoffs.result("writer", message.id)
    assert result.output == "Approved: VERIFIED report"
    returned = restarted.tasks.get(result.delivery_task_id)
    assert returned.source == "handoff_result" and returned.chain_id == source.chain_id
    assert returned.metadata == {
        "message_id": message.id,
        "source_task_id": source.id,
        "target_task_id": result.target_task_id,
    }
    final_model = ScriptedModel(
        [
            response("read_handoff", {"message_id": message.id}),
            response(text="Review received."),
        ]
    )
    final = KernelRuntime(tmp_path, model_factory=lambda config: final_model)
    assert await tick(final)
    assert source.id in str(final_model.requests[-1])
    assert result.target_task_id in str(final_model.requests[-1])
    assert "Approved: VERIFIED report" in str(final_model.requests[-1])
    assert final.tasks.get(returned.id).state == "succeeded"
    assert not await tick(final)
    assert len(final.handoffs.results("writer")) == 1
    starts = [
        e
        for e in read_session_events(tmp_path, source.session_id)
        if e.type == "tool.started" and e.data["call"]["name"] == "write_file"
    ]
    assert len(starts) == 1
    assert len(final.interactions.list(task_id=source.id)) == 1
    final.interactions.reply(source.interaction_id, reply)
    final.handoffs.sync()
    final.handoffs.sync()
    assert not await tick(final)
    assert len(final.tasks.list()) == 4  # seed, original, review, returned result
    assert len(final.handoffs.results("writer")) == 1
    members = [task for task in final.tasks.list() if task.chain_id == source.chain_id]
    assert len(members) == 3
    original_runs = final.tasks.runs(source.id)
    assert len(original_runs) == 2
    original = final.tasks.get(source.id)
    chain = final.tasks.chain(source.id)
    for name, total in chain.cumulative_metrics.model_dump().items():
        assert total == sum(getattr(task.cumulative_metrics, name) for task in members)
        assert getattr(original.cumulative_metrics, name) == sum(
            getattr(run.metrics, name) for run in original_runs
        )
    assert final.tasks.get(seed.id).chain_id != source.chain_id
    final_starts = [
        e
        for e in read_session_events(tmp_path, source.session_id)
        if e.type == "tool.started" and e.data["call"]["name"] == "write_file"
    ]
    assert len(final_starts) == 1


class ChatModel:
    """Every fresh Task tries to continue the conversation once, then ends."""

    def __init__(self, bot_id):
        self.bot_id = bot_id

    async def complete(self, request):
        if any(m.role == "tool" for m in request.messages):
            return response(text="sent or stopped")
        return response(
            "send_handoff",
            {
                "target_bot_id": "b" if self.bot_id == "a" else "a",
                "text": "Continue the conversation",
                "idempotency_key": "next",
            },
        )

    async def close(self):
        pass


@pytest.mark.parametrize("limit", ["hops", "models", "tools"])
async def test_automatic_chat_limits_shared_across_bots_and_restarts(tmp_path: Path, limit):
    def reopen():
        return KernelRuntime(tmp_path, model_factory=lambda config: ChatModel(config.bot.id))

    runtime = reopen()
    for bot_id in ("a", "b"):
        save_bot(runtime, bot_id, ["send_handoff"])
    budget = BudgetConfig(
        max_model_requests=5 if limit == "models" else 100,
        max_tool_calls=3 if limit == "tools" else 100,
    )
    root = runtime.tasks.submit(
        TaskSubmission(
            bot_id="a",
            text="Start chatting",
            chain_budget=budget,
            chain_max_hops=3 if limit == "hops" else 50,
        )
    )
    for _ in range(30):
        runtime = reopen()
        if not await tick(runtime):
            break
    else:
        pytest.fail("Automatic cross-Bot conversation did not stop")
    chain = runtime.tasks.chain(root.id)
    tasks = runtime.tasks.list()
    assert {t.bot_id for t in tasks} == {"a", "b"}
    assert all(t.chain_id == root.chain_id for t in tasks)
    assert chain.cumulative_metrics.model_requests == sum(
        t.cumulative_metrics.model_requests for t in tasks
    )
    assert chain.cumulative_metrics.tool_calls == sum(
        t.cumulative_metrics.tool_calls for t in tasks
    )
    assert chain.cumulative_metrics.model_requests <= budget.max_model_requests
    assert chain.cumulative_metrics.tool_calls <= budget.max_tool_calls
    assert not runtime.interactions.list()
    if limit == "hops":
        assert chain.hops_used == 3
    elif limit == "models":
        assert chain.cumulative_metrics.model_requests == 5
    else:
        assert chain.cumulative_metrics.tool_calls == 3


@pytest.mark.parametrize("operation", ["send", "read"])
async def test_handoff_tools_enforce_whitelist_and_result_ownership(tmp_path: Path, operation):
    runtime = KernelRuntime(tmp_path)
    save_bot(runtime, "owner", ["send_handoff", "read_handoff"])
    save_bot(runtime, "reviewer", [])
    save_bot(runtime, "outsider", ["read_handoff"])
    source = runtime.tasks.submit(TaskSubmission(bot_id="owner", text="Original"))
    message = runtime.handoffs.send(
        "owner",
        source.id,
        "reviewer",
        "Review",
        idempotency_key="review",
    )
    name = "send_handoff" if operation == "send" else "read_handoff"
    arguments = (
        {"target_bot_id": "reviewer", "text": "Unauthorized", "idempotency_key": "bad"}
        if operation == "send"
        else {"message_id": message.id}
    )
    model = ScriptedModel([response(name, arguments), response(text="Rejected safely")])
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: model)
    # Remove the unrelated queued source so this tick executes the outsider.
    runtime.tasks.cancel(source.id)
    outsider = runtime.tasks.submit(TaskSubmission(bot_id="outsider", text="Try access"))
    assert await tick(runtime)
    assert runtime.tasks.get(outsider.id).state == "failed"
    results = [
        e.data
        for e in read_session_events(tmp_path, outsider.session_id)
        if e.type == "tool.result"
    ]
    assert len(results) == 1 and results[0]["ok"] is False
    assert not runtime.handoffs.list("outsider", outgoing=True)
    assert not runtime.interactions.list(task_id=outsider.id)
