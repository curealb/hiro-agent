"""Handoff transactions, migration, restart cutpoints, and ownership isolation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config
from hiro_agent.contracts import RunMetrics, RunResult, RunStatus
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import RunEvidence, TaskState, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime


@pytest.fixture
def kernel(tmp_path: Path) -> KernelRuntime:
    runtime = KernelRuntime(tmp_path)
    for bot_id in ("a", "b", "stranger"):
        runtime.bots.save(Config(bot=BotConfig(id=bot_id, allowed_tools=("send_handoff",))))
    return runtime


def source(kernel: KernelRuntime, **kwargs: object):
    return kernel.tasks.submit(TaskSubmission(bot_id="a", text="inspect", **kwargs))


def settle(kernel: KernelRuntime, *, status: RunStatus = RunStatus.COMPLETED):
    claim = kernel.tasks.claim("worker")
    assert claim is not None
    metrics = RunMetrics(model_requests=1, tool_calls=1, elapsed_ms=7)
    kernel.tasks.checkpoint(claim.run.id, metrics)
    result = RunResult(
        session_id=claim.task.session_id,
        run_id=claim.run.id,
        status=status,
        output="reviewed",
        reason="done",
        metrics=metrics,
    )
    kernel.tasks.finish(claim.run.id, result)
    return claim, result


def test_send_is_atomic_idempotent_and_isolated(kernel: KernelRuntime) -> None:
    task = source(kernel, chain_max_hops=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        messages = list(
            pool.map(
                lambda _: kernel.handoffs.send("a", task.id, "b", "review", idempotency_key="same"),
                range(8),
            )
        )
    assert len({message.id for message in messages}) == 1
    assert kernel.tasks.chain(task.id).hops_used == 1
    message = messages[0]
    assert kernel.handoffs.list("b") == [message]
    assert kernel.handoffs.list("a", outgoing=True) == [message]
    with pytest.raises(ConfigurationError, match="payload"):
        kernel.handoffs.send("a", task.id, "b", "different", idempotency_key="same")
    with pytest.raises(ConfigurationError, match="Bot"):
        kernel.handoffs.send("b", task.id, "a", "review", idempotency_key="wrong-owner")
    with pytest.raises(ConfigurationError, match="Bot"):
        kernel.handoffs.get("stranger", message.id)
    with pytest.raises(ConfigurationError, match="sender"):
        kernel.handoffs.result("b", message.id)
    kernel.handoffs.send("a", task.id, "b", "second", idempotency_key="second")
    with pytest.raises(ConfigurationError, match="hop limit"):
        kernel.handoffs.send("a", task.id, "b", "third", idempotency_key="third")
    assert len(kernel.handoffs.list("b")) == 2


@pytest.mark.parametrize("cut", ["saved", "submitted", "terminal", "result_saved", "returned"])
def test_every_delivery_cut_recovers_once(
    kernel: KernelRuntime, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    task = source(kernel)
    settle(kernel)
    message = kernel.handoffs.send("a", task.id, "b", "review", idempotency_key="once")
    original_submit = kernel.tasks.submit

    def crash_after_submit(request: TaskSubmission):
        result = original_submit(request)
        if request.source == ("handoff_result" if cut == "returned" else "handoff"):
            raise RuntimeError("crash after durable submit")
        return result

    if cut == "submitted":
        monkeypatch.setattr(kernel.tasks, "submit", crash_after_submit)
        with pytest.raises(RuntimeError):
            kernel.handoffs.sync()
    elif cut != "saved":
        kernel.handoffs.sync()
        settle(kernel, status=RunStatus.FAILED)
        if cut == "returned":
            monkeypatch.setattr(kernel.tasks, "submit", crash_after_submit)
            with pytest.raises(RuntimeError):
                kernel.handoffs.sync()
        elif cut == "result_saved":

            def crash_before_result_delivery(result):
                raise RuntimeError("crash after durable result")

            monkeypatch.setattr(kernel.handoffs, "_deliver_result", crash_before_result_delivery)
            with pytest.raises(RuntimeError):
                kernel.handoffs.sync()

    restarted = KernelRuntime(kernel.workspace)
    restarted.handoffs.sync()
    target = restarted.handoffs.get("a", message.id).target_task_id
    assert target is not None
    if restarted.tasks.get(target).state == TaskState.QUEUED:
        settle(restarted, status=RunStatus.FAILED)
    restarted.handoffs.sync()
    returned = restarted.handoffs.result("a", message.id)
    assert returned is not None and returned.state == TaskState.FAILED
    assert returned.source_task_id == task.id and returned.target_task_id == target
    assert returned.delivery_task_id is not None
    delivery = restarted.tasks.get(returned.delivery_task_id, bot_id="a")
    assert delivery.source == "handoff_result"
    assert delivery.chain_id == task.chain_id
    for _ in range(3):
        KernelRuntime(kernel.workspace).handoffs.sync()
    assert len(restarted.tasks.list()) == 3
    assert len(restarted.handoffs.results("a")) == 1
    assert restarted.tasks.chain(task.id).hops_used == 1
    assert restarted.tasks.chain(task.id).cumulative_metrics.model_requests == 2


def test_chain_charge_reconcile_restart_and_root_limits(kernel: KernelRuntime) -> None:
    task = source(kernel, chain_max_hops=3, chain_budget=BudgetConfig(max_model_requests=3))
    claim = kernel.tasks.claim("worker")
    assert claim is not None
    metrics = RunMetrics(model_requests=1, tool_calls=2, input_tokens=8, elapsed_ms=20)
    kernel.tasks.checkpoint(claim.run.id, metrics)
    kernel.tasks.checkpoint(claim.run.id, metrics)
    restarted = KernelRuntime(kernel.workspace)
    final_metrics = metrics.model_copy(update={"model_requests": 2, "elapsed_ms": 25})
    result = RunResult(
        run_id=claim.run.id,
        session_id=task.session_id,
        status=RunStatus.COMPLETED,
        reason="done",
        output="done",
        metrics=final_metrics,
    )
    evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=task.session_id,
        started=True,
        metrics=final_metrics,
        result=result,
    )
    restarted.tasks.reconcile(evidence)
    restarted.tasks.reconcile(evidence)
    child = restarted.tasks.submit(
        TaskSubmission(
            bot_id="b",
            text="child",
            parent_task_id=task.id,
            budget=BudgetConfig(max_model_requests=100),
        )
    )
    assert child.chain_id == task.chain_id
    for overrides in (
        {"chain_max_hops": 100},
        {"chain_budget": BudgetConfig(max_model_requests=100)},
        {"chain_id": "fresh"},
    ):
        with pytest.raises(ConfigurationError):
            restarted.tasks.submit(
                TaskSubmission(bot_id="b", text="reset", parent_task_id=child.id, **overrides)
            )
    settle(restarted)
    chain = restarted.tasks.chain(child.id)
    assert chain.cumulative_metrics.model_requests == 3
    assert chain.cumulative_metrics.tool_calls == 3
    assert chain.cumulative_metrics.elapsed_ms == 32
    blocked = restarted.tasks.submit(
        TaskSubmission(bot_id="a", text="blocked", parent_task_id=child.id)
    )
    assert restarted.tasks.claim("worker") is None
    assert restarted.tasks.get(blocked.id).state == TaskState.FAILED
    assert "Chain" in restarted.tasks.get(blocked.id).reason


def test_legacy_submission_and_metrics_backfill_once(kernel: KernelRuntime) -> None:
    request = TaskSubmission(bot_id="a", text="legacy", idempotency_key="legacy")
    task = kernel.tasks.submit(request)
    settle(kernel)
    with kernel.database.transaction() as connection:
        record = kernel.tasks._task(connection, task.id)
        record.chain_id = None
        old_submission = request.model_dump(mode="json")
        del old_submission["chain_budget"]
        del old_submission["chain_max_hops"]
        connection.execute(
            "UPDATE durable_tasks SET record_json=?, submission_json=? WHERE id=?",
            (record.model_dump_json(), json.dumps(old_submission), task.id),
        )
        connection.execute("DELETE FROM chain_members")
        connection.execute("DELETE FROM task_chains")
    for _ in range(3):
        restarted = KernelRuntime(kernel.workspace)
        retried = restarted.tasks.submit(request)
        assert retried.id == task.id
        assert restarted.tasks.chain(task.id).cumulative_metrics == retried.cumulative_metrics
        assert retried.cumulative_metrics.model_requests == 1


def test_exhausted_chain_still_returns_durable_failure(kernel: KernelRuntime) -> None:
    task = source(kernel, chain_budget=BudgetConfig(max_model_requests=1))
    message = kernel.handoffs.send("a", task.id, "b", "review", idempotency_key="once")
    settle(kernel)
    kernel.handoffs.sync()
    assert kernel.tasks.claim("worker") is None
    kernel.handoffs.sync()
    returned = kernel.handoffs.result("a", message.id)
    assert returned is not None and returned.state == TaskState.FAILED
    assert returned.delivery_task_id is not None
    assert kernel.tasks.claim("worker") is None
    assert kernel.tasks.get(returned.delivery_task_id).state == TaskState.FAILED
    assert kernel.tasks.chain(task.id).cumulative_metrics.model_requests == 1
    assert len(kernel.tasks.list()) == 3
