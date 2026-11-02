"""Persistent-task integration against actual SQLite state and JSONL journals."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig
from hiro_agent.contracts import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    RunStatus,
    SessionEvent,
    ToolCall,
    Usage,
)
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    DurableRunState,
    InteractionReply,
    InteractionState,
    RunClaim,
    TaskState,
    TaskSubmission,
    canonical_arguments,
)
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.session import JsonlSession
from hiro_agent.tools import ToolRegistry, WriteFileArgs


def tool(name: str, arguments: dict, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=canonical_arguments(arguments))


def response(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(input_tokens=7, output_tokens=3, cached_input_tokens=0),
    )


class InspectingModel(ScriptedModel):
    def __init__(
        self, responses: list[ModelResponse], inspect: Callable[[ModelRequest], None]
    ) -> None:
        super().__init__(responses)
        self.inspect = inspect

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.inspect(request)
        return await super().complete(request)


def kernel(
    workspace: Path, model: ScriptedModel, *, budget: BudgetConfig | None = None
) -> KernelRuntime:
    runtime = KernelRuntime(workspace, model_factory=lambda config: model)
    runtime.bots.save(
        Config(
            model=ModelConfig(adapter="mock"),
            bot=BotConfig(id="writer", allowed_tools=("write_file", "read_file", "ask_user")),
            budget=budget or BudgetConfig(),
        )
    )
    return runtime


async def tick(runtime: KernelRuntime) -> bool:
    worker = runtime.worker(poll_interval=0.01)
    try:
        return await worker.tick()
    finally:
        await worker.close()


async def events(runtime: KernelRuntime, session_id: str) -> list[SessionEvent]:
    session = await JsonlSession.open(runtime.workspace, session_id, create=False)
    try:
        return session.events
    finally:
        await session.close()


async def test_approval_restart_executes_exact_saved_call_before_model(tmp_path: Path) -> None:
    write = tool("write_file", {"path": "approved.txt", "content": "saved\n"}, "old-write")
    later = tool("read_file", {"path": "approved.txt"}, "old-later")
    first_model = ScriptedModel([response(write, later), response(text="must not continue")])
    runtime = kernel(tmp_path, first_model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="write the saved text"))

    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.state == TaskState.WAITING_APPROVAL
    assert waiting.active_run_id is None
    assert waiting.interaction_id is not None and waiting.pending_operation is not None
    saved_arguments = canonical_arguments(
        {"path": "approved.txt", "content": "saved\n", "create_parents": False}
    )
    assert waiting.pending_operation.arguments_json == saved_arguments
    assert len(first_model.requests) == 1
    assert not (tmp_path / "approved.txt").exists()
    old_events = await events(runtime, task.session_id)
    old_results = [event.data for event in old_events if event.type == "tool.result"]
    assert {result["call_id"] for result in old_results} == {write.id, later.id}
    assert all(result["outcome"] == "not_started" for result in old_results)
    assert not any(event.type == "tool.started" for event in old_events)
    first_run = runtime.tasks.runs(task.id)[0]
    assert first_run.result is not None and first_run.result.status == RunStatus.PAUSED

    reply = InteractionReply(approved=True)
    runtime.interactions.reply(waiting.interaction_id, reply)
    # The reply transaction has committed, but no in-memory wake was delivered.
    assert runtime.tasks.get(task.id).state == TaskState.WAITING_APPROVAL

    def inspect(request: ModelRequest) -> None:
        assert (tmp_path / "approved.txt").read_text() == "saved\n"
        saved_calls = [
            call
            for message in request.messages
            for call in message.tool_calls
            if call.id != write.id and call.name == "write_file"
        ]
        assert len(saved_calls) == 1 and saved_calls[0].arguments == saved_arguments
        assert any(message.tool_call_id == saved_calls[0].id for message in request.messages)

    resumed_model = InspectingModel([response(text="saved successfully")], inspect)
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: resumed_model)
    assert await tick(restarted)
    done = restarted.tasks.get(task.id)
    assert done.state == TaskState.SUCCEEDED and done.output == "saved successfully"
    assert done.session_id == task.session_id
    runs = restarted.tasks.runs(task.id)
    assert len(runs) == 2 and runs[0].id != runs[1].id
    assert done.cumulative_metrics.model_requests == 2
    assert done.cumulative_metrics.tool_calls == 2
    assert done.cumulative_metrics.input_tokens == 14
    assert done.cumulative_metrics.elapsed_ms == sum(run.metrics.elapsed_ms for run in runs)
    journal = await events(restarted, task.session_id)
    starts = [event for event in journal if event.type == "tool.started"]
    assert len(starts) == 1 and starts[0].run_id == runs[1].id
    assert starts[0].data["call"]["id"] not in {write.id, later.id}
    assert starts[0].data["call"]["arguments"] == saved_arguments
    next_request = next(
        event for event in journal if event.run_id == runs[1].id and event.type == "model.requested"
    )
    new_result = next(
        event for event in journal if event.run_id == runs[1].id and event.type == "tool.result"
    )
    assert starts[0].seq < new_result.seq < next_request.seq
    consumed = restarted.interactions.get(waiting.interaction_id)
    assert consumed.state == InteractionState.CONSUMED
    assert consumed.consumed_by_run_id == runs[1].id
    assert restarted.interactions.reply(waiting.interaction_id, reply) == consumed
    assert not await tick(restarted)
    assert len(resumed_model.requests) == 1
    assert restarted.interactions.ready() == []


async def test_question_reply_becomes_one_durable_input_after_restart(tmp_path: Path) -> None:
    question = "Which city should appear in the report?"
    first = ScriptedModel([response(tool("ask_user", {"question": question}, "question"))])
    runtime = kernel(tmp_path, first)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="draft the report"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.state == TaskState.WAITING_INPUT and waiting.pending_operation is None
    assert waiting.interaction_id is not None
    reply = InteractionReply(text="Use Shanghai, with the exact wording 上海。")
    runtime.interactions.reply(waiting.interaction_id, reply)
    assert runtime.tasks.get(task.id).state == TaskState.WAITING_INPUT

    def inspect(request: ModelRequest) -> None:
        replies = [message for message in request.messages if reply.text in (message.content or "")]
        assert len(replies) == 1 and replies[0].role == "user"
        assert question in replies[0].content

    resumed = InspectingModel([response(text="Report for Shanghai")], inspect)
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: resumed)
    assert await tick(restarted)
    done = restarted.tasks.get(task.id)
    assert done.state == TaskState.SUCCEEDED
    record = restarted.interactions.get(waiting.interaction_id)
    assert record.state == InteractionState.CONSUMED
    assert record.consumed_by_run_id == restarted.tasks.runs(task.id)[1].id
    assert restarted.interactions.consume_answer(record.id, task.id, "another-run") is None
    assert restarted.interactions.reply(record.id, reply) == record
    assert not await tick(restarted)
    journal = await events(restarted, task.session_id)
    answers = [
        event
        for event in journal
        if event.type == "task.continued" and reply.text in event.data.get("text", "")
    ]
    assert len(answers) == 1 and answers[0].run_id == record.consumed_by_run_id
    assert len(resumed.requests) == 1


@pytest.mark.parametrize("change", ["reject", "remove_tool", "read_only", "revoke_restore"])
async def test_saved_approval_cannot_bypass_rejection_or_current_authority(
    tmp_path: Path, change: str
) -> None:
    write = tool("write_file", {"path": "blocked.txt", "content": "blocked"}, "write")
    runtime = kernel(tmp_path, ScriptedModel([response(write)]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="write"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    runtime.interactions.reply(
        waiting.interaction_id, InteractionReply(approved=change != "reject")
    )
    original = runtime.bots.get("writer").config
    changed = original.model_copy(deep=True)
    if change in {"remove_tool", "revoke_restore"}:
        assert changed.bot is not None
        changed.bot.allowed_tools = ("read_file", "ask_user")
        runtime.bots.save(changed)
        if change == "revoke_restore":
            runtime.bots.save(original)
    elif change == "read_only":
        changed.permissions = "read-only"
        runtime.bots.save(changed)
    model = ScriptedModel([response(text="must not execute")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    assert await tick(restarted)
    assert restarted.tasks.get(task.id).state == TaskState.FAILED
    assert not (tmp_path / "blocked.txt").exists()
    assert model.requests == []
    assert not any(
        event.type == "tool.started" for event in await events(restarted, task.session_id)
    )
    assert restarted.interactions.get(waiting.interaction_id).state == InteractionState.CANCELLED
    assert not await tick(restarted)


@pytest.mark.parametrize(
    "budget", [BudgetConfig(max_model_requests=1), BudgetConfig(max_tool_calls=1)]
)
async def test_reply_cannot_reset_an_exhausted_task_budget(
    tmp_path: Path, budget: BudgetConfig
) -> None:
    write = tool("write_file", {"path": "budget.txt", "content": "blocked"}, "write")
    runtime = kernel(tmp_path, ScriptedModel([response(write)]), budget=budget)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="write"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.state == TaskState.WAITING_APPROVAL and waiting.interaction_id is not None
    runtime.interactions.reply(waiting.interaction_id, InteractionReply(approved=True))
    model = ScriptedModel([response(text="must not obtain a fresh budget")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    assert not await tick(restarted)
    failed = restarted.tasks.get(task.id)
    assert failed.state == TaskState.FAILED and "预算" in failed.reason
    assert failed.cumulative_metrics == waiting.cumulative_metrics
    assert failed.cumulative_metrics.model_requests == 1
    assert failed.cumulative_metrics.tool_calls == 1
    assert len(restarted.tasks.runs(task.id)) == 1
    assert not (tmp_path / "budget.txt").exists()
    assert model.requests == []


async def test_repeated_questions_share_model_budget_across_runs(tmp_path: Path) -> None:
    first = response(tool("ask_user", {"question": "First?"}, "first"))
    second = response(tool("ask_user", {"question": "Second?"}, "second"))
    model = ScriptedModel([first, second, response(text="third request must be blocked")])
    runtime = kernel(tmp_path, model, budget=BudgetConfig(max_model_requests=2))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="ask twice"))
    for answer in ("first answer", "second answer"):
        assert await tick(runtime)
        waiting = runtime.tasks.get(task.id)
        assert waiting.state == TaskState.WAITING_INPUT and waiting.interaction_id is not None
        runtime.interactions.reply(waiting.interaction_id, InteractionReply(text=answer))
    assert not await tick(runtime)
    failed = runtime.tasks.get(task.id)
    assert failed.state == TaskState.FAILED
    assert failed.cumulative_metrics.model_requests == 2
    assert failed.cumulative_metrics.tool_calls == 2
    assert failed.cumulative_metrics.input_tokens == 14
    assert len(runtime.tasks.runs(task.id)) == 2
    assert len(model.requests) == 2


async def test_crash_after_sql_claim_requeues_without_reusing_the_run(tmp_path: Path) -> None:
    runtime = kernel(tmp_path, ScriptedModel([]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="answer"))
    claim = runtime.tasks.claim("dead-worker")
    assert claim is not None and claim.run.state == DurableRunState.CLAIMED
    model = ScriptedModel([response(text="recovered")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        recovered = await worker.recover()
        assert len(recovered) == 1 and recovered[0].state == TaskState.QUEUED
        assert await worker.recover() == []
        assert restarted.tasks.get(task.id).cumulative_metrics == task.cumulative_metrics
        assert await worker.tick()
        assert not await worker.tick()
    finally:
        await worker.close()
    runs = restarted.tasks.runs(task.id)
    assert len(runs) == 2 and runs[0].id == claim.run.id and runs[1].id != claim.run.id
    assert restarted.tasks.get(task.id).state == TaskState.SUCCEEDED
    journal = await events(restarted, task.session_id)
    assert [event.run_id for event in journal if event.type == "run.started"] == [runs[1].id]
    assert len(model.requests) == 1


async def test_crash_after_jsonl_end_reconciles_without_model_or_double_charge(
    tmp_path: Path,
) -> None:
    runtime = kernel(tmp_path, ScriptedModel([response(text="journal is authoritative")]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="answer"))
    claim = runtime.tasks.claim("dead-worker")
    assert claim is not None
    claim = RunClaim(task=claim.task, run=runtime.tasks.mark_started(claim.run.id))
    outcome = await runtime.runner.run(claim, lambda metrics: None)
    assert outcome.result.status == RunStatus.COMPLETED
    assert runtime.tasks.get(task.id).state == TaskState.RUNNING
    assert runtime.tasks.get(task.id).cumulative_metrics.model_requests == 0
    journal = await events(runtime, task.session_id)
    assert journal[-1].type == "run.ended"
    model = ScriptedModel([response(text="must not rerun")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        recovered = await worker.recover()
        assert len(recovered) == 1 and recovered[0].state == TaskState.SUCCEEDED
        assert recovered[0].output == "journal is authoritative"
        assert recovered[0].cumulative_metrics == outcome.result.metrics
        assert await worker.recover() == []
        assert not await worker.tick()
    finally:
        await worker.close()
    assert model.requests == []
    assert len(restarted.tasks.runs(task.id)) == 1
    assert restarted.tasks.get(task.id).cumulative_metrics == outcome.result.metrics
    assert await events(restarted, task.session_id) == journal


async def test_crash_after_tool_started_never_replays_uncertain_effect(tmp_path: Path) -> None:
    runtime = kernel(tmp_path, ScriptedModel([]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="write once"))
    claim = runtime.tasks.claim("dead-worker")
    assert claim is not None
    runtime.tasks.mark_started(claim.run.id)
    write = tool("write_file", {"path": "uncertain.txt", "content": "original effect"}, "write")
    session = await JsonlSession.open(tmp_path, task.session_id)
    try:
        authority = runtime.bots.authority(task.bot_id).model_dump(mode="json")
        await session.append("session.bot", authority)
        await session.append("run.started", {"persistent_task_id": task.id}, claim.run.id)
        await session.append("task.received", {"text": task.text}, claim.run.id)
        await session.append("model.requested", {"attempt_id": "model-attempt"}, claim.run.id)
        await session.append(
            "assistant.message",
            {
                "message": AssistantMessage(tool_calls=[write]).model_dump(mode="json"),
                "attempt_id": "model-attempt",
                "usage": Usage(input_tokens=7, output_tokens=3).model_dump(mode="json"),
            },
            claim.run.id,
        )
        call_data = {"call": write.model_dump(mode="json")}
        await session.append("tool.called", {**call_data, "accepted": True}, claim.run.id)
        await session.append("tool.started", {**call_data, "authority": authority}, claim.run.id)
        # Simulate the external effect before its terminal tool record was committed.
        (tmp_path / "uncertain.txt").write_text("original effect")
    finally:
        await session.close()
    model = ScriptedModel([response(text="must not automatically replay")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        recovered = await worker.recover()
        assert len(recovered) == 1 and recovered[0].state == TaskState.NEEDS_RECONCILIATION
        assert recovered[0].cumulative_metrics.model_requests == 1
        assert recovered[0].cumulative_metrics.tool_calls == 1
        assert recovered[0].cumulative_metrics.input_tokens == 7
        assert not await worker.tick()
        assert await worker.recover() == []
    finally:
        await worker.close()
    journal = await events(restarted, task.session_id)
    results = [event for event in journal if event.type == "tool.result"]
    assert len(results) == 1 and results[0].data["outcome"] == "unknown"
    assert results[0].data["call_id"] == write.id
    assert len([event for event in journal if event.type == "run.ended"]) == 1
    assert restarted.tasks.runs(task.id)[0].state == DurableRunState.UNCERTAIN
    assert (tmp_path / "uncertain.txt").read_text() == "original effect"
    assert model.requests == []


async def test_busy_journal_blocks_worker_recovery_without_requeue(tmp_path: Path) -> None:
    runtime = kernel(tmp_path, ScriptedModel([]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="answer"))
    claim = runtime.tasks.claim("dead-worker")
    assert claim is not None
    session = await JsonlSession.open(tmp_path, task.session_id)
    model = ScriptedModel([response(text="must not rerun")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        with pytest.raises(PersistenceError, match="already open"):
            await worker.recover()
        assert restarted.tasks.get(task.id).state == TaskState.RUNNING
        assert len(restarted.tasks.runs(task.id)) == 1
        assert model.requests == []
    finally:
        await worker.close()
        await session.close()


async def test_live_unknown_tool_outcome_requires_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = tool("write_file", {"path": "partial.txt", "content": "approved content"}, "write")
    runtime = kernel(tmp_path, ScriptedModel([response(write)]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="write once"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    runtime.interactions.reply(waiting.interaction_id, InteractionReply(approved=True))
    attempts = []

    async def uncertain_write(arguments: WriteFileArgs) -> None:
        attempts.append(arguments)
        (tmp_path / arguments.path).write_text("partially committed")
        raise RuntimeError("Lost the side-effect acknowledgement")

    configure = runtime.runner._configure_tools

    def configure_uncertain_tool(registry: ToolRegistry) -> None:
        configure(registry)
        registry.definitions["write_file"] = replace(
            registry.definitions["write_file"], handler=uncertain_write
        )

    model = ScriptedModel([response(text="must not continue after an uncertain side effect")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    monkeypatch.setattr(restarted.runner, "_configure_tools", configure_uncertain_tool)
    assert await tick(restarted)
    uncertain = restarted.tasks.get(task.id)
    assert uncertain.state == TaskState.NEEDS_RECONCILIATION
    assert uncertain.active_run_id is None
    assert len(attempts) == 1
    assert (tmp_path / "partial.txt").read_text() == "partially committed"
    assert model.requests == []
    runs = restarted.tasks.runs(task.id)
    assert len(runs) == 2 and runs[1].state == DurableRunState.UNCERTAIN
    assert runs[1].result is not None and runs[1].result.status == RunStatus.FAILED
    journal = await events(restarted, task.session_id)
    results = [
        event for event in journal if event.run_id == runs[1].id and event.type == "tool.result"
    ]
    assert len(results) == 1 and results[0].data["outcome"] == "unknown"
    assert restarted.interactions.get(waiting.interaction_id).state == InteractionState.CONSUMED
    assert not await tick(restarted)
    assert restarted.tasks.get(task.id) == uncertain
    assert len(attempts) == 1


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("invalid_json", "invalid_arguments"),
        ("invalid_schema", "invalid_arguments"),
        ("disabled", "permission_denied"),
        ("unknown", "unknown_tool"),
    ],
)
async def test_invalid_initial_tool_fails_task_before_remaining_calls_or_model(
    tmp_path: Path, failure: str, expected_error: str
) -> None:
    if failure == "invalid_json":
        broken = ToolCall(id="broken", name="write_file", arguments="{not json")
    elif failure == "invalid_schema":
        broken = tool("write_file", {"path": "broken.txt"}, "broken")
    elif failure == "disabled":
        broken = tool("write_file", {"path": "broken.txt", "content": "denied"}, "broken")
    else:
        broken = tool("unregistered_tool", {}, "broken")
    later = tool("write_file", {"path": "later.txt", "content": "must not run"}, "later")
    model = ScriptedModel([response(broken, later), response(text="must not mask tool failure")])
    runtime = kernel(tmp_path, model)
    if failure == "disabled":
        config = runtime.bots.get("writer").config
        assert config.bot is not None
        config.bot.allowed_tools = ("read_file", "ask_user")
        runtime.bots.save(config)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="execute tools"))
    assert await tick(runtime)
    failed = runtime.tasks.get(task.id)
    assert failed.state == TaskState.FAILED
    assert failed.interaction_id is None and failed.pending_operation is None
    assert expected_error in failed.reason
    assert len(model.requests) == 1
    assert not (tmp_path / "broken.txt").exists() and not (tmp_path / "later.txt").exists()
    journal = await events(runtime, task.session_id)
    results = {
        event.data["call_id"]: event.data for event in journal if event.type == "tool.result"
    }
    assert set(results) == {broken.id, later.id}
    assert results[broken.id]["error"] == expected_error
    assert all(result["outcome"] == "not_started" for result in results.values())
    assert not any(event.type == "tool.started" for event in journal)
    assert not await tick(runtime)


async def test_corrupt_journal_halts_recovery_without_requeue_or_rewrite(tmp_path: Path) -> None:
    runtime = kernel(tmp_path, ScriptedModel([]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="answer"))
    claim = runtime.tasks.claim("dead-worker")
    assert claim is not None
    runtime.tasks.mark_started(claim.run.id)
    session = await JsonlSession.open(tmp_path, task.session_id)
    path = session.path
    try:
        await session.append(
            "session.bot", runtime.bots.authority(task.bot_id).model_dump(mode="json")
        )
        await session.append("run.started", {"persistent_task_id": task.id}, claim.run.id)
    finally:
        await session.close()
    with path.open("ab") as journal:
        journal.write(b'{"not-a-valid":"session event"}\n')
    corrupted_bytes = path.read_bytes()
    model = ScriptedModel([response(text="must not reexecute")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        with pytest.raises(PersistenceError):
            await worker.recover()
        assert restarted.tasks.get(task.id).state == TaskState.RUNNING
        assert restarted.tasks.get(task.id).active_run_id == claim.run.id
        assert len(restarted.tasks.runs(task.id)) == 1
        assert model.requests == []
        assert path.read_bytes() == corrupted_bytes
    finally:
        await worker.close()


@pytest.mark.parametrize("failed_stage", ["model", "tool"])
async def test_checkpoint_failure_stops_execution_and_leaves_recoverable_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_stage: str
) -> None:
    read = tool("read_file", {"path": "input.txt"}, "read")
    write = tool("write_file", {"path": "blocked.txt", "content": "must not run"}, "write")
    (tmp_path / "input.txt").write_text("input")
    model = ScriptedModel([response(read, write), response(text="must not continue")])
    runtime = kernel(tmp_path, model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="read and write"))
    untouched = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="run later"))
    checkpoint = runtime.tasks.checkpoint
    failures = []

    def fail_checkpoint(run_id, metrics):
        if (failed_stage == "model" and metrics.model_requests) or (
            failed_stage == "tool" and metrics.tool_calls
        ):
            failures.append(metrics.model_copy(deep=True))
            raise PersistenceError("Injected durable budget write failure")
        return checkpoint(run_id, metrics)

    monkeypatch.setattr(runtime.tasks, "checkpoint", fail_checkpoint)
    with pytest.raises(PersistenceError):
        await tick(runtime)
    assert failures
    claimed = runtime.tasks.get(task.id)
    assert claimed.state == TaskState.RUNNING and claimed.active_run_id is not None
    assert runtime.tasks.get(untouched.id).state == TaskState.QUEUED
    assert len(model.requests) == (0 if failed_stage == "model" else 1)
    journal = await events(runtime, task.session_id)
    assert not any(event.type in {"tool.started", "run.ended"} for event in journal)
    assert not (tmp_path / "blocked.txt").exists()
    assert runtime.interactions.ready() == []


async def test_cancelling_running_task_stops_slow_model_and_releases_worker(tmp_path: Path) -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    closed = asyncio.Event()

    class SlowModel(ScriptedModel):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request.model_copy(deep=True))
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return response(tool("write_file", {"path": "late.txt", "content": "late"}, "late"))

        async def close(self) -> None:
            closed.set()

    model = SlowModel([])
    runtime = kernel(tmp_path, model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="wait for cancellation"))
    worker = runtime.worker(poll_interval=0.01)
    execution = asyncio.create_task(worker.tick())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        requested = runtime.tasks.cancel(task.id)
        assert requested.state == TaskState.RUNNING and requested.cancel_requested
        assert await asyncio.wait_for(execution, timeout=5)
        cancelled = runtime.tasks.get(task.id)
        assert cancelled.state == TaskState.CANCELLED and cancelled.active_run_id is None
        assert cancelled.cumulative_metrics.model_requests == 1
        assert cleaned.is_set() and closed.is_set()
        assert len(model.requests) == 1
        assert not await worker.tick()
    finally:
        await worker.close()
        await asyncio.gather(execution, return_exceptions=True)
    assert not (tmp_path / "late.txt").exists()
    journal = await events(runtime, task.session_id)
    assert not any(event.type == "tool.started" for event in journal)
    ended = [event for event in journal if event.type == "run.ended"]
    assert len(ended) == 1 and ended[0].data["status"] == "canceled"
    # Closing released the workspace's process lock for another worker.
    restarted = runtime.worker(poll_interval=0.01)
    await restarted.close()


@pytest.mark.parametrize("kind", ["approval", "question"])
@pytest.mark.parametrize("answered", [False, True])
async def test_cancelled_wait_cleans_pending_reply_and_never_resumes(
    tmp_path: Path, kind: str, answered: bool
) -> None:
    call = (
        tool("write_file", {"path": "cancelled.txt", "content": "blocked"}, "write")
        if kind == "approval"
        else tool("ask_user", {"question": "Continue?"}, "question")
    )
    reply = InteractionReply(approved=True) if kind == "approval" else InteractionReply(text="yes")
    model = ScriptedModel([response(call), response(text="must not continue")])
    runtime = kernel(tmp_path, model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="wait, then cancel"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    if answered:
        runtime.interactions.reply(waiting.interaction_id, reply)
    cancelled = runtime.tasks.cancel(task.id)
    assert cancelled.state == TaskState.CANCELLED
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    assert not await tick(restarted)
    assert restarted.tasks.get(task.id) == cancelled
    assert restarted.interactions.get(waiting.interaction_id).state == InteractionState.CANCELLED
    assert restarted.interactions.ready() == []
    with pytest.raises(ConfigurationError):
        restarted.interactions.reply(waiting.interaction_id, reply)
    assert len(restarted.tasks.runs(task.id)) == 1
    assert len(model.requests) == 1
    assert not (tmp_path / "cancelled.txt").exists()


async def test_crash_after_answer_consumption_preserves_input_without_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel([response(tool("ask_user", {"question": "Exact title?"}, "question"))])
    runtime = kernel(tmp_path, model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="writer", text="prepare a titled report"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    reply = InteractionReply(text="Saved title with exact wording 上海 2026")
    runtime.interactions.reply(waiting.interaction_id, reply)
    fresh_model = ScriptedModel([response(text="must not run after the journal write fails")])
    resumed = KernelRuntime(tmp_path, model_factory=lambda config: fresh_model)
    append = JsonlSession.append

    async def fail_after_consumption(session, event_type, data, run_id=None):
        if event_type == "interaction.resumed":
            consumed = resumed.interactions.get(waiting.interaction_id)
            assert consumed.state == InteractionState.CONSUMED
            assert consumed.consumed_by_run_id == run_id
            raise PersistenceError("Injected failure after consuming the answer")
        return await append(session, event_type, data, run_id)

    with monkeypatch.context() as injection:
        injection.setattr(JsonlSession, "append", fail_after_consumption)
        with pytest.raises(PersistenceError):
            await tick(resumed)
    interrupted = resumed.tasks.get(task.id)
    assert interrupted.state == TaskState.RUNNING and interrupted.active_run_id is not None
    assert fresh_model.requests == []
    journal = await events(resumed, task.session_id)
    answers = [event for event in journal if event.type == "interaction.answer"]
    assert len(answers) == 1
    assert answers[0].data == {"interaction_id": waiting.interaction_id, "text": reply.text}
    assert answers[0].run_id == interrupted.active_run_id
    assert not any(
        event.run_id == interrupted.active_run_id
        and event.type in {"interaction.resumed", "task.continued", "model.requested", "run.ended"}
        for event in journal
    )

    restarted = KernelRuntime(tmp_path, model_factory=lambda config: fresh_model)
    worker = restarted.worker(poll_interval=0.01)
    try:
        recovered = await worker.recover()
        assert len(recovered) == 1 and recovered[0].state == TaskState.FAILED
        assert recovered[0].active_run_id is None
        assert not await worker.tick()
    finally:
        await worker.close()
    record = restarted.interactions.get(waiting.interaction_id)
    assert record.state == InteractionState.CONSUMED and record.reply == reply
    assert record.consumed_by_run_id == interrupted.active_run_id
    assert restarted.interactions.reply(waiting.interaction_id, reply) == record
    assert restarted.interactions.consume_answer(record.id, task.id, "another-run") is None
    assert restarted.interactions.ready() == []
    assert not await tick(restarted)
    assert len(restarted.tasks.runs(task.id)) == 2
    assert fresh_model.requests == []
    recovered_journal = await events(restarted, task.session_id)
    assert [event for event in recovered_journal if event.type == "interaction.answer"] == answers
