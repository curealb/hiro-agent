"""Persistent task invariants, budget accounting and crash-window reconciliation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hiro_agent.bots import BotStore
from hiro_agent.config import BotConfig, BudgetConfig, Config
from hiro_agent.contracts import RunMetrics, RunResult, RunStatus
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    DurableRunState,
    OperationIntent,
    RunClaim,
    RunEvidence,
    TaskState,
    TaskSubmission,
)
from hiro_agent.kernel_db import KernelDatabase


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    bots = BotStore(database)
    bots.save(Config(bot=BotConfig(id="writer", allowed_tools=("write_file",))))
    bots.save(Config(bot=BotConfig(id="reader")))
    return TaskStore(database, bots)


def claim_task(store: TaskStore, *, budget: BudgetConfig | None = None) -> RunClaim:
    store.submit(TaskSubmission(bot_id="writer", text="Do the work", budget=budget))
    claim = store.claim("worker")
    assert claim is not None
    return claim


def result(
    claim: RunClaim,
    status: RunStatus = RunStatus.COMPLETED,
    metrics: RunMetrics | None = None,
) -> RunResult:
    return RunResult(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        status=status,
        reason="finished" if status == RunStatus.COMPLETED else "stopped",
        output="Saved output",
        metrics=metrics or RunMetrics(),
    )


def operation(store: TaskStore, claim: RunClaim) -> OperationIntent:
    authority = store.bots.authority(claim.task.bot_id)
    return OperationIntent(
        task_id=claim.task.id,
        bot_id=claim.task.bot_id,
        session_id=claim.task.session_id,
        run_id=claim.run.id,
        call_id="original-call",
        tool_name="write_file",
        arguments_json='{"path":"report.txt","content":"exact text"}',
        permission_version=authority.permission_version,
        permission_digest=authority.permission_digest,
    )


def test_submit_persists_full_request_and_bot_budget(store: TaskStore) -> None:
    config = store.bots.get("writer").config
    config.budget = BudgetConfig(max_model_requests=3, max_tool_calls=4, timeout_seconds=7)
    store.bots.save(config)
    request = TaskSubmission(
        bot_id="writer",
        text="report",
        source="routine",
        chain_id="chain",
        parent_task_id="parent",
        metadata={"nested": {"a": [1, 2]}},
    )
    task = store.submit(request)
    request.metadata["nested"]["a"].append(3)
    reopened = TaskStore(KernelDatabase(store.database.path), store.bots)
    saved = reopened.get(task.id, bot_id="writer")
    assert saved.budget == config.budget
    assert saved.source == "routine"
    assert saved.chain_id == "chain" and saved.parent_task_id == "parent"
    assert saved.metadata == {"nested": {"a": [1, 2]}}
    assert saved.id != saved.session_id
    assert saved.state == TaskState.QUEUED
    assert saved.revision == 1


def test_idempotency_is_per_bot_canonical_and_survives_config_change(store: TaskStore) -> None:
    request = TaskSubmission(
        bot_id="writer", text="report", idempotency_key="scheduled", metadata={"b": 2, "a": 1}
    )
    task = store.submit(request)
    config = store.bots.get("writer").config
    config.budget.max_model_requests = 2
    store.bots.save(config)
    assert store.submit(request.model_copy(update={"metadata": {"a": 1, "b": 2}})) == task
    other = store.submit(request.model_copy(update={"bot_id": "reader"}))
    assert other.id != task.id
    assert store.submit(TaskSubmission(bot_id="writer", text="report")).id != task.id
    with pytest.raises(ConfigurationError, match="幂等"):
        store.submit(request.model_copy(update={"text": "different"}))
    with pytest.raises(ConfigurationError, match="幂等"):
        store.submit(request.model_copy(update={"budget": config.budget}))


def test_concurrent_submit_and_claim_are_atomic(store: TaskStore) -> None:
    request = TaskSubmission(bot_id="writer", text="report", idempotency_key="once")
    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = list(executor.map(lambda _: store.submit(request), range(10)))
    assert len({task.id for task in tasks}) == 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        claims = list(executor.map(lambda index: store.claim(f"worker-{index}"), range(10)))
    actual = [claim for claim in claims if claim is not None]
    assert len(actual) == 1
    claim = actual[0]
    assert claim.task.active_run_id == claim.run.id
    assert claim.run.task_id == tasks[0].id
    assert claim.run.session_id == claim.task.session_id
    assert len({claim.run.id, claim.task.id, claim.task.session_id}) == 3
    assert store.runs(claim.task.id) == [claim.run]


def test_bot_ownership_and_query_filters(store: TaskStore) -> None:
    writer = store.submit(TaskSubmission(bot_id="writer", text="one"))
    reader = store.submit(TaskSubmission(bot_id="reader", text="two"))
    with pytest.raises(ConfigurationError, match="Bot"):
        store.get(writer.id, bot_id="reader")
    with pytest.raises(ConfigurationError, match="Bot"):
        store.cancel(writer.id, bot_id="reader")
    with pytest.raises(ConfigurationError):
        store.submit(TaskSubmission(bot_id="missing", text="x"))
    store.cancel(writer.id)
    assert [task.id for task in store.list(bot_id="reader")] == [reader.id]
    assert [task.id for task in store.list(states=(TaskState.CANCELLED,))] == [writer.id]
    assert store.list(bot_id="reader", states=(TaskState.CANCELLED,)) == []


def test_checkpoint_charges_snapshots_once_and_finish_only_delta(store: TaskStore) -> None:
    claim = claim_task(store)
    store.mark_started(claim.run.id)
    first = RunMetrics(model_requests=1, tool_calls=2, input_tokens=20, elapsed_ms=100)
    initial = store.checkpoint(claim.run.id, first)
    assert store.checkpoint(claim.run.id, first) == initial
    final = RunMetrics(model_requests=2, tool_calls=3, input_tokens=30, elapsed_ms=250)
    done = store.finish(claim.run.id, result(claim, metrics=final))
    assert done.cumulative_metrics == final
    assert store.runs(done.id)[0].metrics == final
    assert store.finish(claim.run.id, result(claim, metrics=final)) == done
    assert store.get(done.id) == done
    assert done.output == "Saved output"
    assert done.active_run_id is None


@pytest.mark.parametrize("metrics", [RunMetrics(model_requests=-1), RunMetrics(elapsed_ms=1)])
def test_regressing_checkpoints_roll_back_entire_transaction(
    store: TaskStore, metrics: RunMetrics
) -> None:
    claim = claim_task(store)
    current = store.checkpoint(claim.run.id, RunMetrics(model_requests=2, elapsed_ms=10))
    with pytest.raises(ConfigurationError, match="不能倒退"):
        store.checkpoint(claim.run.id, metrics)
    with pytest.raises(ConfigurationError, match="不能倒退"):
        store.finish(claim.run.id, result(claim, metrics=metrics))
    assert store.get(claim.task.id) == current
    assert store.runs(claim.task.id)[0].result is None


def test_counter_type_is_validated_even_after_model_mutation(store: TaskStore) -> None:
    claim = claim_task(store)
    metrics = RunMetrics()
    metrics.model_requests = True
    with pytest.raises(ConfigurationError):
        store.checkpoint(claim.run.id, metrics)


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (RunStatus.COMPLETED, TaskState.SUCCEEDED),
        (RunStatus.FAILED, TaskState.FAILED),
        (RunStatus.CANCELED, TaskState.CANCELLED),
        (RunStatus.PAUSED, TaskState.FAILED),
    ],
)
def test_finish_mapping_and_terminal_runs_cannot_be_claimed(
    store: TaskStore, status: RunStatus, state: TaskState
) -> None:
    claim = claim_task(store)
    done = store.finish(claim.run.id, result(claim, status))
    assert done.state == state
    assert done.reason == result(claim, status).reason
    assert store.claim("worker") is None
    with pytest.raises(ConfigurationError):
        store.mark_started(claim.run.id)


def test_finish_rejects_conflicting_result_and_wrong_association(store: TaskStore) -> None:
    claim = claim_task(store)
    for updates in ({"run_id": "another"}, {"session_id": "another"}):
        with pytest.raises(ConfigurationError):
            store.finish(claim.run.id, result(claim).model_copy(update=updates))
    done = store.finish(claim.run.id, result(claim))
    with pytest.raises(ConfigurationError):
        store.finish(claim.run.id, result(claim).model_copy(update={"output": "different"}))
    assert store.get(done.id) == done


def test_persistence_failed_result_leaves_task_recoverable(store: TaskStore) -> None:
    claim = claim_task(store)
    with pytest.raises(PersistenceError):
        store.finish(claim.run.id, result(claim).model_copy(update={"persistence_failed": True}))
    assert store.get(claim.task.id).state == TaskState.RUNNING


def test_wait_resume_preserves_session_intent_and_cumulative_budget(store: TaskStore) -> None:
    claim = claim_task(store, budget=BudgetConfig(max_model_requests=3))
    intent = operation(store, claim)
    first_metrics = RunMetrics(model_requests=1, tool_calls=1, elapsed_ms=50)
    waiting = store.finish(
        claim.run.id,
        result(claim, RunStatus.PAUSED, first_metrics),
        pending_operation=intent,
        interaction_id="approval-1",
        waiting_for="approval",
    )
    assert waiting.state == TaskState.WAITING_APPROVAL
    assert store.claim("worker") is None
    assert store.wake(waiting.id, "older") == waiting
    queued = store.wake(waiting.id, "approval-1")
    assert queued.pending_operation == intent
    assert store.wake(waiting.id, "approval-1") == queued
    second = store.claim("worker")
    assert second is not None
    assert second.task.session_id == claim.task.session_id
    assert second.run.id != claim.run.id
    assert second.task.cumulative_metrics == first_metrics
    assert (
        second.task.budget.max_model_requests - second.task.cumulative_metrics.model_requests == 2
    )
    assert store.wake(waiting.id, "approval-1") == second.task
    done = store.finish(
        second.run.id,
        result(second, metrics=RunMetrics(model_requests=1, tool_calls=1, elapsed_ms=25)),
    )
    assert done.cumulative_metrics.model_requests == 2
    assert done.cumulative_metrics.tool_calls == 2
    assert done.cumulative_metrics.elapsed_ms == 75
    assert done.pending_operation is None
    assert done.interaction_id is None
    assert store.wake(done.id, "approval-1") == done


def test_old_interaction_cannot_wake_a_new_wait(store: TaskStore) -> None:
    first = claim_task(store)
    store.finish(
        first.run.id, result(first, RunStatus.PAUSED), interaction_id="q1", waiting_for="question"
    )
    store.wake(first.task.id, "q1")
    second = store.claim("worker")
    assert second is not None
    waiting = store.finish(
        second.run.id, result(second, RunStatus.PAUSED), interaction_id="q2", waiting_for="question"
    )
    assert store.wake(waiting.id, "q1") == waiting
    assert store.claim("worker") is None
    assert store.wake(waiting.id, "q2").state == TaskState.QUEUED


@pytest.mark.parametrize("invalid", ["no_intent", "no_interaction", "wrong_owner", "not_paused"])
def test_invalid_wait_is_rejected_without_spending_budget(store: TaskStore, invalid: str) -> None:
    claim = claim_task(store)
    intent = operation(store, claim)
    if invalid == "wrong_owner":
        intent = intent.model_copy(update={"bot_id": "reader"})
    with pytest.raises(ConfigurationError):
        store.finish(
            claim.run.id,
            result(
                claim,
                RunStatus.COMPLETED if invalid == "not_paused" else RunStatus.PAUSED,
                RunMetrics(model_requests=1),
            ),
            pending_operation=None if invalid == "no_intent" else intent,
            interaction_id=None if invalid == "no_interaction" else "approval",
            waiting_for="approval",
        )
    assert store.get(claim.task.id) == claim.task


@pytest.mark.parametrize(
    "metrics",
    [RunMetrics(model_requests=2), RunMetrics(tool_calls=3), RunMetrics(elapsed_ms=1000)],
)
def test_woken_exhausted_task_is_failed_without_another_claim(
    store: TaskStore, metrics: RunMetrics
) -> None:
    claim = claim_task(
        store, budget=BudgetConfig(max_model_requests=2, max_tool_calls=3, timeout_seconds=1)
    )
    store.finish(
        claim.run.id,
        result(claim, RunStatus.PAUSED, metrics),
        interaction_id="question",
        waiting_for="question",
    )
    store.wake(claim.task.id, "question")
    assert store.claim("worker") is None
    task = store.get(claim.task.id)
    assert task.state == TaskState.FAILED
    assert "预算耗尽" in task.reason
    assert len(store.runs(task.id)) == 1
    assert store.claim("worker") is None


def test_running_cancel_only_requests_and_settles_after_runner(store: TaskStore) -> None:
    claim = claim_task(store)
    requested = store.cancel(claim.task.id)
    assert requested.state == TaskState.RUNNING and requested.cancel_requested
    assert requested.active_run_id == claim.run.id
    assert store.cancel(claim.task.id) == requested
    done = store.finish(
        claim.run.id,
        result(claim, RunStatus.PAUSED, RunMetrics(model_requests=1)),
        interaction_id="late-answer",
        waiting_for="question",
    )
    assert done.state == TaskState.CANCELLED
    assert store.wake(done.id, "late-answer") == done
    assert store.claim("worker") is None


@pytest.mark.parametrize("waiting", [False, True])
def test_queued_and_waiting_cancel_immediately(store: TaskStore, waiting: bool) -> None:
    if waiting:
        claim = claim_task(store)
        task = store.finish(
            claim.run.id,
            result(claim, RunStatus.PAUSED),
            interaction_id="question",
            waiting_for="question",
        )
    else:
        task = store.submit(TaskSubmission(bot_id="writer", text="work"))
    cancelled = store.cancel(task.id)
    assert cancelled.state == TaskState.CANCELLED
    assert cancelled.cancel_requested
    assert store.wake(task.id, "question") == cancelled
    assert store.claim("worker") is None


def test_crash_claim_without_run_evidence_requeues_once(store: TaskStore) -> None:
    first = claim_task(store)
    evidence = RunEvidence(run_id=first.run.id, session_id=first.task.session_id, started=False)
    queued = store.reconcile(evidence)
    assert queued.state == TaskState.QUEUED
    assert queued.active_run_id is None
    assert store.runs(queued.id)[0].state == DurableRunState.FINISHED
    second = store.claim("restarted-worker")
    assert second is not None and second.run.id != first.run.id
    assert store.reconcile(evidence) == second.task
    assert store.get(first.task.id).active_run_id == second.run.id


def test_crash_ended_run_is_settled_once_from_journal(store: TaskStore) -> None:
    claim = claim_task(store)
    store.mark_started(claim.run.id)
    store.checkpoint(claim.run.id, RunMetrics(model_requests=1, elapsed_ms=10))
    final_metrics = RunMetrics(model_requests=2, elapsed_ms=30)
    evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        started=True,
        result=result(claim, metrics=final_metrics),
        metrics=final_metrics,
    )
    done = store.reconcile(evidence)
    assert done.state == TaskState.SUCCEEDED
    assert done.cumulative_metrics == final_metrics
    assert store.reconcile(evidence) == done
    assert store.finish(claim.run.id, evidence.result) == done  # type: ignore[arg-type]
    assert store.claim("worker") is None


def test_recovery_merges_attempt_evidence_missing_from_result(store: TaskStore) -> None:
    claim = claim_task(store)
    evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        started=True,
        result=result(claim, RunStatus.PAUSED),
        metrics=RunMetrics(model_requests=2, tool_calls=1, elapsed_ms=10),
    )
    done = store.reconcile(evidence)
    assert done.state == TaskState.FAILED
    assert done.cumulative_metrics == evidence.metrics
    assert store.reconcile(evidence) == done


@pytest.mark.parametrize("status", [RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.CANCELED])
def test_crash_unknown_side_effect_overrides_recovered_status(
    store: TaskStore, status: RunStatus
) -> None:
    claim = claim_task(store)
    evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        started=True,
        result=result(claim, status, RunMetrics(model_requests=1, tool_calls=1)),
        metrics=RunMetrics(model_requests=1, tool_calls=1),
        unknown_side_effect=True,
    )
    task = store.reconcile(evidence)
    assert task.state == TaskState.NEEDS_RECONCILIATION
    assert store.runs(task.id)[0].state == DurableRunState.UNCERTAIN
    assert task.active_run_id is None
    assert store.reconcile(evidence) == task
    assert store.claim("worker") is None
    for note in ("", "  "):
        with pytest.raises(ConfigurationError):
            store.resolve(task.id, action="retry", note=note)
    resolved = store.resolve(
        task.id, action="retry", note="Operator verified the file was not written"
    )
    assert resolved.cumulative_metrics == task.cumulative_metrics
    assert store.reconcile(evidence) == resolved
    second = store.claim("worker")
    assert second is not None and second.task.id == task.id
    with store.database.connect() as connection:
        row = connection.execute("SELECT * FROM durable_task_resolutions").fetchone()
    assert row["run_id"] == claim.run.id
    assert row["action"] == "retry" and "verified" in row["note"]


@pytest.mark.parametrize("sql_started", [False, True])
def test_incomplete_started_evidence_is_never_replayed(store: TaskStore, sql_started: bool) -> None:
    claim = claim_task(store)
    if sql_started:
        store.mark_started(claim.run.id)
    task = store.reconcile(
        RunEvidence(
            run_id=claim.run.id,
            session_id=claim.task.session_id,
            started=not sql_started,
        )
    )
    assert task.state == TaskState.NEEDS_RECONCILIATION
    assert store.claim("worker") is None


def test_reconciliation_rejects_wrong_session_and_conflicting_evidence(store: TaskStore) -> None:
    claim = claim_task(store)
    with pytest.raises(ConfigurationError):
        store.reconcile(RunEvidence(run_id=claim.run.id, session_id="wrong", started=False))
    evidence = RunEvidence(run_id=claim.run.id, session_id=claim.task.session_id, started=False)
    task = store.reconcile(evidence)
    with pytest.raises(ConfigurationError):
        store.reconcile(evidence.model_copy(update={"started": True}))
    assert store.get(task.id) == task


def test_missing_journal_preserves_already_checkpointed_budget(store: TaskStore) -> None:
    claim = claim_task(store)
    store.mark_started(claim.run.id)
    spent = RunMetrics(model_requests=2, tool_calls=1, elapsed_ms=100)
    store.checkpoint(claim.run.id, spent)
    evidence = RunEvidence(run_id=claim.run.id, session_id=claim.task.session_id, started=False)
    task = store.reconcile(evidence)
    assert task.state == TaskState.NEEDS_RECONCILIATION
    assert task.cumulative_metrics == spent
    assert store.reconcile(evidence) == task


def test_negative_recovery_metrics_do_not_hide_behind_final_metrics(store: TaskStore) -> None:
    claim = claim_task(store)
    with pytest.raises(ConfigurationError):
        store.reconcile(
            RunEvidence(
                run_id=claim.run.id,
                session_id=claim.task.session_id,
                started=True,
                result=result(claim, metrics=RunMetrics(model_requests=2)),
                metrics=RunMetrics(model_requests=-1),
            )
        )
    assert store.get(claim.task.id) == claim.task


@pytest.mark.parametrize("action", ["fail", "cancel"])
def test_manual_resolution_terminal_and_cancel_does_not_erase_uncertainty(
    store: TaskStore, action: str
) -> None:
    claim = claim_task(store)
    task = store.reconcile(
        RunEvidence(
            run_id=claim.run.id,
            session_id=claim.task.session_id,
            started=True,
            unknown_side_effect=True,
        )
    )
    requested = store.cancel(task.id)
    assert requested.state == TaskState.NEEDS_RECONCILIATION
    with pytest.raises(ConfigurationError):
        store.resolve(task.id, action="retry", note="Must not disregard cancellation")
    resolved = store.resolve(task.id, action=action, note="Checked external state")  # type: ignore[arg-type]
    assert resolved.state == (TaskState.FAILED if action == "fail" else TaskState.CANCELLED)
    assert store.claim("worker") is None
    with pytest.raises(ConfigurationError):
        store.resolve(task.id, action="retry", note="Already resolved")


def test_sql_failure_is_propagated_instead_of_success(store: TaskStore) -> None:
    claim = claim_task(store)
    with store.database.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER reject_task_update BEFORE UPDATE ON durable_tasks "
            "BEGIN SELECT RAISE(ABORT, 'injected disk error'); END"
        )
    with pytest.raises(PersistenceError):
        store.finish(claim.run.id, result(claim, metrics=RunMetrics(model_requests=1)))
    assert store.get(claim.task.id) == claim.task
    assert store.runs(claim.task.id)[0] == claim.run
