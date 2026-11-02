"""Independent transactional migration/limit probes complementary to process tests."""

import json

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config
from hiro_agent.contracts import RunMetrics, RunResult, RunStatus
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import RunEvidence, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime


def finish(kernel, claim, metrics):
    result = RunResult(
        session_id=claim.task.session_id,
        run_id=claim.run.id,
        status=RunStatus.COMPLETED,
        output="legacy done",
        reason="done",
        metrics=metrics,
    )
    evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        started=True,
        metrics=metrics,
        result=result,
    )
    kernel.tasks.reconcile(evidence)
    return evidence


def test_legacy_shared_chain_backfills_once_then_only_charges_new_run_delta(tmp_path):
    kernel = KernelRuntime(tmp_path)
    for bot_id in ("a", "b"):
        kernel.bots.save(Config(bot=BotConfig(id=bot_id)))
    request = TaskSubmission(
        bot_id="a",
        text="legacy root",
        idempotency_key="old",
        chain_id="legacy-shared",
        budget=BudgetConfig(max_model_requests=5),
    )
    root = kernel.tasks.submit(request)
    first = kernel.tasks.claim("old-worker")
    assert first and first.task.id == root.id
    evidence = finish(kernel, first, RunMetrics(model_requests=2, tool_calls=1, elapsed_ms=17))
    child = kernel.tasks.submit(
        TaskSubmission(
            bot_id="b",
            text="legacy child",
            parent_task_id=root.id,
            budget=BudgetConfig(max_model_requests=100),
        )
    )
    second = kernel.tasks.claim("old-worker")
    assert second and second.task.id == child.id
    kernel.tasks.checkpoint(
        second.run.id, RunMetrics(model_requests=1, tool_calls=2, elapsed_ms=11)
    )
    with kernel.database.transaction() as connection:
        connection.execute("DELETE FROM chain_members")
        connection.execute("DELETE FROM task_chains")
        old = request.model_dump(mode="json")
        del old["chain_budget"]
        del old["chain_max_hops"]
        connection.execute(
            "UPDATE durable_tasks SET submission_json=? WHERE id=?", (json.dumps(old), root.id)
        )
    for _ in range(3):
        kernel = KernelRuntime(tmp_path)
        assert kernel.tasks.submit(request).id == root.id
        chain = kernel.tasks.chain(root.id)
        assert chain.cumulative_metrics == RunMetrics(model_requests=3, tool_calls=3, elapsed_ms=28)
        assert chain.budget.max_model_requests == 5
        assert kernel.tasks.chain(child.id).id == chain.id
        kernel.tasks.reconcile(evidence)
    final = RunMetrics(model_requests=3, tool_calls=4, elapsed_ms=19)
    kernel.tasks.checkpoint(second.run.id, final)
    kernel.tasks.checkpoint(second.run.id, final)
    final_evidence = finish(kernel, second, final)
    for _ in range(3):
        kernel = KernelRuntime(tmp_path)
        kernel.tasks.reconcile(final_evidence)
        chain = kernel.tasks.chain(root.id)
        assert chain.cumulative_metrics == RunMetrics(model_requests=5, tool_calls=5, elapsed_ms=36)
        assert chain.budget.max_model_requests == 5
    grandchild = kernel.tasks.submit(
        TaskSubmission(
            bot_id="a",
            text="cannot refund",
            parent_task_id=child.id,
            budget=BudgetConfig(max_model_requests=1000),
        )
    )
    assert kernel.tasks.claim("new-worker") is None
    assert kernel.tasks.get(grandchild.id).state == "failed"
    assert kernel.tasks.chain(grandchild.id).cumulative_metrics == chain.cumulative_metrics


@pytest.mark.parametrize(
    "override",
    [
        {"chain_id": "fresh"},
        {"chain_max_hops": 100},
        {"chain_budget": BudgetConfig(max_model_requests=1000, max_tool_calls=1000)},
    ],
)
def test_child_and_result_tasks_cannot_raise_or_replace_chain(tmp_path, override):
    kernel = KernelRuntime(tmp_path)
    for bot_id in ("a", "b"):
        kernel.bots.save(Config(bot=BotConfig(id=bot_id, allowed_tools=("send_handoff",))))
    root = kernel.tasks.submit(TaskSubmission(bot_id="a", text="root", chain_max_hops=1))
    for source in ("handoff", "handoff_result"):
        with pytest.raises(ConfigurationError):
            kernel.tasks.submit(
                TaskSubmission(
                    bot_id="b",
                    text="raise limits",
                    source=source,
                    parent_task_id=root.id,
                    **override,
                )
            )
    kernel.handoffs.send("a", root.id, "b", "review", idempotency_key="once")
    original = kernel.tasks.chain(root.id)
    for _ in range(3):
        kernel = KernelRuntime(tmp_path)
        assert kernel.tasks.chain(root.id) == original
        with pytest.raises(ConfigurationError, match="hop limit"):
            kernel.handoffs.send("a", root.id, "b", "again", idempotency_key="twice")
    assert len(kernel.tasks.list()) == 1
