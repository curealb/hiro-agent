"""Nested Agent replies preserve ownership and wait for actual verification."""

import json
import re
from pathlib import Path

import pytest
from test_agent_communication import (
    RESPONDER,
    SENDER,
    ProbeResponder,
    ProbeSender,
    ProbeVerifier,
    response,
    save_bot,
    tick,
)

from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import ModelRequest, ModelResponse, RunMetrics, RunResult, RunStatus
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import RunEvidence, TaskState, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.session import read_session_events

MANAGER = "manager"


class ManagerVerifier(ScriptedModel):
    """Exercise a mistaken downstream read, then recover using forwarded evidence."""

    def __init__(self):
        super().__init__([])
        self.own_id = ""

    def read(self, message_id):
        result = response("read_message", {"message_id": message_id})
        result.assistant.tool_calls[0].id = f"read-{len(self.requests)}"
        return result

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            text = next(m.content for m in request.messages if m.role == "user")
            self.own_id = re.search(r"message_id=([a-f0-9]+)", text)[1]
            return self.read(self.own_id)
        result = json.loads(next(m.content for m in reversed(request.messages) if m.role == "tool"))
        assert result["ok"]
        value = result["value"]
        if len(self.requests) == 2:
            child_id = value["result"]["delegated_results"][0]["message_id"]
            return self.read(child_id)
        if len(self.requests) == 3:
            assert value["status"] == "unavailable"
            assert value["error"] == "message_not_accessible"
            assert "result" not in value
            return self.read(self.own_id)
        assert value["result"]["output"] == "comm-001 通过"
        evidence = next(
            e
            for e in value["result"]["delegated_results"]
            if e["source_bot_id"] == SENDER and e["target_bot_id"] == RESPONDER
        )
        assert evidence["source_bot_id"] == SENDER
        assert evidence["target_bot_id"] == RESPONDER
        assert evidence["state"] == "succeeded"
        assert json.loads(evidence["output"]) == {"correlation_id": "comm-001", "payload": "pong"}
        return response(text="管理员核验 comm-001 通过")


class AckResponder(ProbeResponder):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.requests:
            self.requests.append(request)
            return response(
                "send_message",
                {
                    "target_agent_id": SENDER,
                    "correlation_id": "comm-001",
                    "payload": "ACK comm-001 pong",
                    "idempotency_key": "ack",
                },
            )
        return await super().complete(request)


class AckFinalizer(ScriptedModel):
    def __init__(self):
        super().__init__([])

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            text = next(m.content for m in request.messages if m.role == "user")
            message_id = re.search(r"message_id=([a-f0-9]+)", text)[1]
            return response("read_message", {"message_id": message_id})
        result = json.loads(next(m.content for m in reversed(request.messages) if m.role == "tool"))
        assert result["ok"] and result["value"]["result"]["output"] == "ACK received"
        return response(text=json.dumps({"correlation_id": "comm-001", "payload": "pong"}))


@pytest.mark.parametrize("explicit_ack", [False, True])
async def test_manager_gets_final_verified_report_across_restarts(tmp_path: Path, explicit_ack):
    manager = ScriptedModel(
        [
            response(
                "send_message",
                {
                    "target_agent_id": SENDER,
                    "correlation_id": "comm-001",
                    "payload": "Run ping/pong with responder",
                    "idempotency_key": "delegate",
                },
            ),
            response(text="测试已委派，等待结果。"),
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: manager)
    for bot_id in (MANAGER, SENDER, RESPONDER):
        save_bot(runtime, bot_id)
    root = runtime.tasks.submit(TaskSubmission(bot_id=MANAGER, text="retry communication"))
    assert await tick(runtime)
    [parent] = runtime.handoffs.list(MANAGER, outgoing=True)

    runtime = KernelRuntime(tmp_path, model_factory=lambda config: ProbeSender())
    assert await tick(runtime)
    assert runtime.handoffs.result(MANAGER, parent.id) is None
    [child] = runtime.handoffs.list(SENDER, outgoing=True)

    runtime = KernelRuntime(
        tmp_path,
        model_factory=lambda config: AckResponder() if explicit_ack else ProbeResponder(),
    )
    assert await tick(runtime)
    if explicit_ack:
        assert runtime.handoffs.result(SENDER, child.id) is None
        ack_receiver = ScriptedModel([response(text="ACK received")])
        runtime = KernelRuntime(tmp_path, model_factory=lambda config: ack_receiver)
        assert await tick(runtime)
        text = next(m.content for m in ack_receiver.requests[0].messages if m.role == "user")
        assert json.loads(text.splitlines()[-1])["payload"] == "ACK comm-001 pong"
        assert runtime.handoffs.result(SENDER, child.id) is None
        runtime = KernelRuntime(tmp_path, model_factory=lambda config: AckFinalizer())
        assert await tick(runtime)
    child_result = runtime.handoffs.result(SENDER, child.id)
    assert child_result is not None
    assert runtime.handoffs.result(MANAGER, parent.id) is None

    runtime = KernelRuntime(tmp_path, model_factory=lambda config: ProbeVerifier())
    assert await tick(runtime)
    parent_result = runtime.handoffs.result(MANAGER, parent.id)
    assert parent_result is not None
    assert parent_result.output == "comm-001 通过"
    assert parent_result.response_task_ids == [child_result.delivery_task_id]
    assert parent_result.delegated_results[0].message_id == child.id
    with pytest.raises(ConfigurationError, match="belong"):
        runtime.handoffs.result(MANAGER, child.id)

    verifier = ManagerVerifier()
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: verifier)
    assert await tick(runtime)
    for _ in range(3):
        runtime.handoffs.sync()
    assert not await tick(runtime)
    tasks = [t for t in runtime.tasks.list() if t.chain_id == root.chain_id]
    assert len(tasks) == (7 if explicit_ack else 5)
    assert all(t.state == TaskState.SUCCEEDED for t in tasks)
    final = runtime.tasks.get(parent_result.delivery_task_id)
    assert final.output == "管理员核验 comm-001 通过"
    results = [
        e.data for e in read_session_events(tmp_path, final.session_id) if e.type == "tool.result"
    ]
    assert len(results) == 3 and all(r["ok"] for r in results)


def finish(
    runtime,
    expected_id,
    output,
    status=RunStatus.COMPLETED,
    *,
    metrics=None,
    unknown=False,
):
    claim = runtime.tasks.claim("test-worker")
    assert claim is not None and claim.task.id == expected_id
    result = RunResult(
        run_id=claim.run.id,
        session_id=claim.task.session_id,
        status=status,
        output=output,
        reason="stage finished",
        metrics=metrics or RunMetrics(),
    )
    if unknown:
        runtime.tasks.reconcile(
            RunEvidence(
                run_id=claim.run.id,
                session_id=claim.task.session_id,
                started=True,
                result=result,
                unknown_side_effect=True,
            )
        )
    else:
        runtime.tasks.finish(claim.run.id, result)


def nested(runtime, *, two_children=False, budget=None):
    for bot_id in (MANAGER, SENDER, RESPONDER):
        save_bot(runtime, bot_id)
    root = runtime.tasks.submit(
        TaskSubmission(bot_id=MANAGER, text="delegate", chain_budget=budget)
    )
    parent = runtime.handoffs.send_message(
        MANAGER,
        root.id,
        SENDER,
        "probe",
        correlation_id="comm-001",
        idempotency_key="parent",
    )
    finish(runtime, root.id, "delegated")
    runtime.handoffs.sync()
    sender_id = runtime.handoffs.get(MANAGER, parent.id).target_task_id
    children = [
        runtime.handoffs.send_message(
            SENDER,
            sender_id,
            RESPONDER,
            "ping",
            correlation_id=f"probe-{i}",
            idempotency_key=f"c-{i}",
        )
        for i in range(2 if two_children else 1)
    ]
    finish(runtime, sender_id, "waiting reply, not passed")
    runtime.handoffs.sync()
    assert runtime.handoffs.result(MANAGER, parent.id) is None
    return root, parent, [runtime.handoffs.get(SENDER, c.id) for c in children]


@pytest.mark.parametrize(
    "stage,status",
    [
        ("recipient", RunStatus.FAILED),
        ("recipient", RunStatus.CANCELED),
        ("verification", RunStatus.FAILED),
        ("verification", RunStatus.CANCELED),
        ("unknown", RunStatus.FAILED),
        ("budget", RunStatus.COMPLETED),
    ],
)
def test_nested_failures_are_preserved_and_uncertain_results_stay_pending(tmp_path, stage, status):
    runtime = KernelRuntime(tmp_path)
    budget = BudgetConfig(max_model_requests=2) if stage == "budget" else None
    _, parent, [child] = nested(runtime, budget=budget)
    finish(
        runtime,
        child.target_task_id,
        "pong",
        status if stage == "recipient" else RunStatus.COMPLETED,
        metrics=RunMetrics(model_requests=2) if stage == "budget" else None,
    )
    runtime.handoffs.sync()
    delivery = runtime.handoffs.result(SENDER, child.id).delivery_task_id
    if stage == "budget":
        assert runtime.tasks.claim("test-worker") is None
        assert runtime.tasks.get(delivery).state == TaskState.FAILED
    else:
        finish(
            runtime,
            delivery,
            "verification report",
            status if stage == "verification" else RunStatus.COMPLETED,
            unknown=stage == "unknown",
        )
    runtime = KernelRuntime(tmp_path)
    runtime.handoffs.sync()
    result = runtime.handoffs.result(MANAGER, parent.id)
    if stage == "unknown":
        assert runtime.tasks.get(delivery).state == TaskState.NEEDS_RECONCILIATION
        assert result is None
    else:
        assert result is not None and result.state == TaskState.FAILED
        assert result.response_task_ids == [delivery]


def test_all_branches_must_verify_and_terminal_sources_cannot_add_dependencies(tmp_path):
    runtime = KernelRuntime(tmp_path)
    _, parent, children = nested(runtime, two_children=True)
    for i, child in enumerate(children):
        finish(runtime, child.target_task_id, f"pong-{i}")
        runtime.handoffs.sync()
    deliveries = [runtime.handoffs.result(SENDER, c.id).delivery_task_id for c in children]
    finish(runtime, deliveries[0], "first verified")
    runtime.handoffs.sync()
    assert runtime.handoffs.result(MANAGER, parent.id) is None
    finish(runtime, deliveries[1], "second verified")
    runtime.handoffs.sync()
    result = runtime.handoffs.result(MANAGER, parent.id)
    assert result.output == "first verified\n\nsecond verified"
    assert result.response_task_ids == deliveries
    assert {e.message_id for e in result.delegated_results} == {c.id for c in children}
    child = children[0]
    # Retry remains idempotent after completion; only new dependencies are disallowed.
    assert (
        runtime.handoffs.send_message(
            SENDER,
            child.source_task_id,
            RESPONDER,
            "ping",
            correlation_id="probe-0",
            idempotency_key="c-0",
        ).id
        == child.id
    )
    with pytest.raises(ConfigurationError, match="Terminal"):
        runtime.handoffs.send_message(
            SENDER,
            child.source_task_id,
            RESPONDER,
            "late",
            correlation_id="late",
            idempotency_key="late",
        )
