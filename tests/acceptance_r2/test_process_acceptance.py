"""Independent R2 acceptance using real subprocess failure and restart boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import InteractionReply, OperationIntent, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.session import read_session_events

DRIVER = Path(__file__).with_name("process_driver.py")


def cli(workspace: Path, *args: str, code: int = 0) -> Any:
    result = subprocess.run(
        [sys.executable, "-m", "hiro_agent.cli", "kernel", *args, "-w", str(workspace)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == code, result.stdout + result.stderr
    return json.loads(result.stdout) if code == 0 else result.stderr


def worker_args(workspace: Path, *, once: bool = True) -> list[str]:
    return [
        sys.executable,
        str(DRIVER),
        "kernel",
        "worker",
        "-w",
        str(workspace),
        "--poll-interval",
        "0.02",
        *(["--once"] if once else []),
    ]


def process_env(workspace: Path, cut: str = "") -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "R2_WORKSPACE": str(workspace),
        "R2_CRASH_CUT": cut,
    }


def run_worker(workspace: Path, *, cut: str = "", code: int = 0) -> None:
    result = subprocess.run(
        worker_args(workspace),
        env=process_env(workspace, cut),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == code, result.stdout + result.stderr


def wait_until(predicate: Callable[[], bool], process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 12
    while not predicate():
        assert process.poll() is None, f"worker exited with {process.returncode}"
        assert time.monotonic() < deadline, "worker did not reach the expected durable state"
        time.sleep(0.025)


@pytest.fixture
def kernel(tmp_path: Path) -> KernelRuntime:
    runtime = KernelRuntime(tmp_path)
    runtime.bots.save(
        Config(
            bot=BotConfig(id="r2", allowed_tools=("write_file", "read_file", "ask_user")),
            model=ModelConfig(adapter="mock"),
        )
    )
    return runtime


@pytest.fixture
def live_worker(kernel: KernelRuntime) -> Iterator[subprocess.Popen]:
    with (kernel.workspace / "worker.log").open("w") as log:
        process = subprocess.Popen(
            worker_args(kernel.workspace, once=False),
            env=process_env(kernel.workspace),
            stdout=log,
            stderr=log,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    pytest.fail("worker failed cooperative shutdown")


def submit(kernel: KernelRuntime, text: str) -> str:
    return cli(kernel.workspace, "submit", "r2", text)["id"]


def journal(kernel: KernelRuntime, task_id: str):
    return read_session_events(kernel.workspace, kernel.tasks.get(task_id).session_id)


def assert_paired(events) -> None:
    declarations = [
        call["id"]
        for event in events
        if event.type == "assistant.message"
        for call in event.data["message"].get("tool_calls", [])
    ]
    results = [event.data["call_id"] for event in events if event.type == "tool.result"]
    assert sorted(declarations) == sorted(results)
    assert len(results) == len(set(results))


@pytest.mark.parametrize("kind", ["approval", "question"])
def test_r2_cross_process_reply_exact_intent_and_new_run(kernel: KernelRuntime, kind: str) -> None:
    task_id = submit(kernel, kind)
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    assert waiting.state == ("waiting_approval" if kind == "approval" else "waiting_input")
    assert waiting.interaction_id is not None
    first_run = kernel.tasks.runs(task_id)[0]
    before = journal(kernel, task_id)
    assert_paired(before)
    assert all(e.data["outcome"] == "not_started" for e in before if e.type == "tool.result")
    assert not any(e.type == "tool.started" for e in before)
    assert not (kernel.workspace / "approved.txt").exists()
    if kind == "approval":
        assert waiting.pending_operation is not None
        assert json.loads(waiting.pending_operation.arguments_json) == {
            "path": "approved.txt",
            "content": "精确内容\n",
            "create_parents": False,
        }
    args = ("--approve",) if kind == "approval" else ("--text", "上海，原样保存")
    first_reply = cli(kernel.workspace, "reply", waiting.interaction_id, *args)
    assert cli(kernel.workspace, "reply", waiting.interaction_id, *args) == first_reply
    assert KernelRuntime(kernel.workspace).tasks.get(task_id) == waiting
    conflict = ("--reject",) if kind == "approval" else ("--text", "conflicting answer")
    assert cli(kernel.workspace, "reply", waiting.interaction_id, *conflict, code=1)
    run_worker(kernel.workspace)
    reopened = KernelRuntime(kernel.workspace)
    done = reopened.tasks.get(task_id)
    assert done.state == "succeeded"
    runs = reopened.tasks.runs(task_id)
    assert len(runs) == 2 and runs[1].id != first_run.id
    consumed = reopened.interactions.get(waiting.interaction_id)
    assert consumed.state == "consumed" and consumed.consumed_by_run_id == runs[1].id
    cli(kernel.workspace, "reply", waiting.interaction_id, *args)
    run_worker(kernel.workspace)
    assert reopened.tasks.get(task_id) == done
    after = journal(reopened, task_id)
    assert_paired(after)
    assert done.cumulative_metrics.model_requests == 2
    assert done.cumulative_metrics.input_tokens == 26
    assert done.cumulative_metrics.elapsed_ms == sum(run.metrics.elapsed_ms for run in runs)
    if kind == "approval":
        starts = [event for event in after if event.type == "tool.started"]
        assert len(starts) == 1 and starts[0].run_id == runs[1].id
        assert starts[0].data["call"]["id"] != waiting.pending_operation.call_id
        assert starts[0].data["call"]["arguments"] == waiting.pending_operation.arguments_json
        assert (kernel.workspace / "approved.txt").read_text() == "精确内容\n"
        prepared_result = next(
            e for e in after if e.type == "tool.result" and e.run_id == runs[1].id
        )
        next_model = next(
            e for e in after if e.type == "model.requested" and e.run_id == runs[1].id
        )
        assert prepared_result.seq < next_model.seq
    else:
        assert "上海，原样保存" in done.output
        assert len([e for e in after if e.type == "interaction.answer"]) == 1


def test_r2_cli_exit_independent_worker_singleton_query_cancel(
    kernel: KernelRuntime, live_worker: subprocess.Popen
) -> None:
    task_id = submit(kernel, "slow")
    wait_until(lambda: (kernel.workspace / "model-entered").exists(), live_worker)
    assert live_worker.poll() is None
    status = cli(kernel.workspace, "task", task_id)
    assert status["task"]["state"] == "running"
    second = subprocess.run(
        worker_args(kernel.workspace),
        env=process_env(kernel.workspace),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert second.returncode == 1 and "Another worker" in second.stderr
    cancelled = cli(kernel.workspace, "cancel", task_id)
    assert cancelled["cancel_requested"] is True
    wait_until(lambda: kernel.tasks.get(task_id).state == "cancelled", live_worker)
    done = kernel.tasks.get(task_id)
    assert done.cumulative_metrics.model_requests == 1
    assert done.active_run_id is None and len(kernel.tasks.runs(task_id)) == 1
    next_id = submit(kernel, "finish")
    wait_until(lambda: kernel.tasks.get(next_id).state == "succeeded", live_worker)


@pytest.mark.parametrize(("cut", "exit_code"), [("claim", 71), ("ended", 72)])
def test_r2_process_death_claim_and_jsonl_end_windows(
    kernel: KernelRuntime, cut: str, exit_code: int
) -> None:
    task_id = submit(kernel, "finish")
    run_worker(kernel.workspace, cut=cut, code=exit_code)
    crashed = KernelRuntime(kernel.workspace)
    before = crashed.tasks.get(task_id)
    assert before.state == "running"
    old_run = crashed.tasks.runs(task_id)[0]
    if cut == "claim":
        assert old_run.state == "claimed" and old_run.started_at is None
        assert not (kernel.workspace / "model-attempts").exists()
    else:
        evidence = journal(crashed, task_id)
        assert evidence[-1].type == "run.ended"
        assert old_run.result is None
        assert before.cumulative_metrics.model_requests == 1
    run_worker(kernel.workspace)
    reopened = KernelRuntime(kernel.workspace)
    done = reopened.tasks.get(task_id)
    assert done.state == "succeeded" and done.output == "accepted result"
    assert done.cumulative_metrics.model_requests == 1
    assert (kernel.workspace / "model-attempts").read_text() == "request\n"
    runs = reopened.tasks.runs(task_id)
    assert len(runs) == (2 if cut == "claim" else 1)
    if cut == "claim":
        assert runs[1].id != old_run.id
    else:
        assert journal(reopened, task_id) == evidence
    run_worker(kernel.workspace)
    assert reopened.tasks.get(task_id) == done


def test_r2_process_death_after_real_effect_never_replays_unknown(kernel: KernelRuntime) -> None:
    task_id = submit(kernel, "approval")
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    cli(kernel.workspace, "reply", waiting.interaction_id, "--approve")
    run_worker(kernel.workspace, cut="effect", code=73)
    crashed = KernelRuntime(kernel.workspace)
    active = crashed.tasks.get(task_id)
    assert active.state == "running"
    before = journal(crashed, task_id)
    assert before[-1].type == "tool.started"
    assert (kernel.workspace / "effect-count").read_text() == "effect\n"
    assert (kernel.workspace / "approved.txt").read_text() == "精确内容\n"
    run_worker(kernel.workspace)
    reopened = KernelRuntime(kernel.workspace)
    uncertain = reopened.tasks.get(task_id)
    assert uncertain.state == "needs_reconciliation"
    assert uncertain.cumulative_metrics.model_requests == 1
    assert uncertain.cumulative_metrics.tool_calls == 2
    after = journal(reopened, task_id)
    assert_paired(after)
    unknown = [e for e in after if e.type == "tool.result" and e.data["outcome"] == "unknown"]
    assert len(unknown) == 1 and unknown[0].run_id == active.active_run_id
    cli(kernel.workspace, "reply", waiting.interaction_id, "--approve")
    for _ in range(2):
        run_worker(kernel.workspace)
        assert KernelRuntime(kernel.workspace).tasks.get(task_id) == uncertain
    assert (kernel.workspace / "effect-count").read_text() == "effect\n"
    assert (kernel.workspace / "model-attempts").read_text() == "request\n"
    assert journal(reopened, task_id) == after


def test_r2_wait_time_excluded_and_budget_cumulative(kernel: KernelRuntime) -> None:
    config = kernel.bots.get("r2").config
    config.budget = BudgetConfig(timeout_seconds=0.8, max_model_requests=2)
    kernel.bots.save(config)
    task_id = submit(kernel, "question")
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    assert waiting.state == "waiting_input"
    time.sleep(1.0)
    assert kernel.tasks.get(task_id).cumulative_metrics == waiting.cumulative_metrics
    cli(kernel.workspace, "reply", waiting.interaction_id, "--text", "after long wait")
    run_worker(kernel.workspace)
    done = KernelRuntime(kernel.workspace).tasks.get(task_id)
    assert done.state == "succeeded" and done.cumulative_metrics.model_requests == 2
    assert done.cumulative_metrics.elapsed_ms < 800
    assert "after long wait" in done.output


def test_r2_killed_model_keeps_attempt_and_elapsed_without_free_retry(
    kernel: KernelRuntime, live_worker: subprocess.Popen
) -> None:
    task_id = submit(kernel, "slow")
    wait_until(lambda: (kernel.workspace / "model-entered").exists(), live_worker)
    wait_until(lambda: kernel.tasks.get(task_id).cumulative_metrics.elapsed_ms >= 200, live_worker)
    checkpoint = kernel.tasks.get(task_id).cumulative_metrics
    live_worker.kill()
    live_worker.wait(timeout=5)
    assert not any(event.type == "run.ended" for event in journal(kernel, task_id))
    run_worker(kernel.workspace)
    reopened = KernelRuntime(kernel.workspace)
    done = reopened.tasks.get(task_id)
    assert done.state == "failed"
    assert done.cumulative_metrics.model_requests == checkpoint.model_requests == 1
    assert done.cumulative_metrics.elapsed_ms >= checkpoint.elapsed_ms
    run_worker(kernel.workspace)
    assert reopened.tasks.get(task_id) == done
    assert len(reopened.tasks.runs(task_id)) == 1
    assert (kernel.workspace / "model-attempts").read_text() == "request\n"


@pytest.mark.parametrize("field", ["task_id", "tool_name", "arguments_json", "permission_version"])
def test_r2_saved_authorization_cannot_cross_binding(kernel: KernelRuntime, field: str) -> None:
    task_id = submit(kernel, "approval")
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    operation = waiting.pending_operation
    assert operation is not None and waiting.interaction_id is not None
    cli(kernel.workspace, "reply", waiting.interaction_id, "--approve")
    changed = operation.model_dump()
    changed[field] = {
        "task_id": kernel.tasks.submit(TaskSubmission(bot_id="r2", text="other")).id,
        "tool_name": "run_command",
        "arguments_json": '{"path":"different","content":"changed","create_parents":false}',
        "permission_version": operation.permission_version + 1,
    }[field]
    reopened = KernelRuntime(kernel.workspace)
    assert not reopened.interactions.consume_authorization(
        waiting.interaction_id, OperationIntent.model_validate(changed), "unauthorized-run"
    )
    assert reopened.interactions.get(waiting.interaction_id).state == "answered"
    assert reopened.interactions.consume_authorization(waiting.interaction_id, operation, "new-run")
    assert not reopened.interactions.consume_authorization(
        waiting.interaction_id, operation, "again"
    )
    consumed = reopened.interactions.get(waiting.interaction_id)
    assert (
        reopened.interactions.reply(waiting.interaction_id, InteractionReply(approved=True))
        == consumed
    )
    with pytest.raises(ConfigurationError):
        reopened.interactions.reply(waiting.interaction_id, InteractionReply(approved=False))


@pytest.mark.parametrize("policy", ["reject", "revoke_restore"])
def test_r2_cross_process_rejection_or_expired_permission_blocks_effect(
    kernel: KernelRuntime, policy: str
) -> None:
    task_id = submit(kernel, "approval")
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    cli(
        kernel.workspace,
        "reply",
        waiting.interaction_id,
        "--reject" if policy == "reject" else "--approve",
    )
    if policy == "revoke_restore":
        original = kernel.bots.get("r2").config
        changed = original.model_copy(deep=True)
        changed.bot.allowed_tools = ("read_file", "ask_user")
        kernel.bots.save(changed)
        restored = kernel.bots.save(original)
        assert restored.authority.permission_version > waiting.pending_operation.permission_version
    run_worker(kernel.workspace)
    failed = KernelRuntime(kernel.workspace).tasks.get(task_id)
    assert failed.state == "failed" and failed.reason
    assert failed.cumulative_metrics.model_requests == 1
    assert not (kernel.workspace / "approved.txt").exists()
    assert not any(event.type == "tool.started" for event in journal(kernel, task_id))
    assert_paired(journal(kernel, task_id))


@pytest.mark.parametrize("counter", ["max_model_requests", "max_tool_calls"])
def test_r2_new_process_cannot_reset_exhausted_task_budget(
    kernel: KernelRuntime, counter: str
) -> None:
    task = kernel.tasks.submit(
        TaskSubmission(bot_id="r2", text="approval", budget=BudgetConfig(**{counter: 1}))
    )
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task.id)
    assert waiting.state == "waiting_approval"
    cli(kernel.workspace, "reply", waiting.interaction_id, "--approve")
    run_worker(kernel.workspace)
    done = KernelRuntime(kernel.workspace).tasks.get(task.id)
    assert done.state == "failed" and "预算" in done.reason
    assert done.cumulative_metrics == waiting.cumulative_metrics
    assert len(kernel.tasks.runs(task.id)) == 1
    assert not (kernel.workspace / "approved.txt").exists()


def test_r2_racing_processes_consume_authorization_only_once(kernel: KernelRuntime) -> None:
    task_id = submit(kernel, "approval")
    run_worker(kernel.workspace)
    waiting = kernel.tasks.get(task_id)
    code = """
import json, sys
from pathlib import Path
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.kernel_contracts import InteractionReply
k = KernelRuntime(Path(sys.argv[1]))
record = k.interactions.get(sys.argv[2])
sys.stdin.readline()
k.interactions.reply(record.id, InteractionReply(approved=True))
won = k.interactions.consume_authorization(record.id, record.request.operation, sys.argv[3])
print(json.dumps({'won': won, 'run': sys.argv[3]}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(kernel.workspace), waiting.interaction_id, f"r-{i}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(2)
    ]
    try:
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            outcomes.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert sorted(outcome["won"] for outcome in outcomes) == [False, True]
    winner = next(outcome["run"] for outcome in outcomes if outcome["won"])
    consumed = KernelRuntime(kernel.workspace).interactions.get(waiting.interaction_id)
    assert consumed.state == "consumed" and consumed.consumed_by_run_id == winner
    assert cli(kernel.workspace, "reply", consumed.id, "--approve")["state"] == "consumed"
