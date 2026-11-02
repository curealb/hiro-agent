"""Scheduling uses real SQLite and controlled execution, without Agent integration."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from hiro_agent.bots import BotStore
from hiro_agent.config import BotConfig, Config
from hiro_agent.contracts import RunMetrics, RunResult, RunStatus
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    DurableRunState,
    RunClaim,
    RunEvidence,
    TaskRunOutcome,
    TaskState,
    TaskSubmission,
)
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.worker import TaskWorker


class ControlledRunner:
    def __init__(self, *, gated: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        if not gated:
            self.release.set()
        self.claims: list[RunClaim] = []
        self.recovered: list[RunClaim] = []
        self.cancel_calls: list[str] = []
        self.close_calls = 0
        self.running = 0
        self.max_running = 0
        self.fail: Exception | None = None
        self.close_error: Exception | None = None
        self.evidence: RunEvidence | None = None
        self.persistence_failed = False
        self.unknown_side_effect = False
        self.status: RunStatus | None = None

    async def run(
        self, claim: RunClaim, checkpoint: Callable[[RunMetrics], None]
    ) -> TaskRunOutcome:
        self.claims.append(claim)
        self.running += 1
        self.max_running = max(self.running, self.max_running)
        try:
            self.started.set()
            checkpoint(RunMetrics(model_requests=1, elapsed_ms=10))
            checkpoint(RunMetrics(model_requests=1, elapsed_ms=10))
            await self.release.wait()
            if self.fail is not None:
                raise self.fail
            return TaskRunOutcome(
                result=RunResult(
                    session_id=claim.run.session_id,
                    run_id=claim.run.id,
                    status=self.status
                    or (RunStatus.CANCELED if self.cancelled.is_set() else RunStatus.COMPLETED),
                    reason="cancelled" if self.cancelled.is_set() else "complete",
                    output="done",
                    metrics=RunMetrics(model_requests=1, elapsed_ms=20),
                    persistence_failed=self.persistence_failed,
                ),
                unknown_side_effect=self.unknown_side_effect,
            )
        finally:
            self.running -= 1

    async def recover(self, claim: RunClaim) -> RunEvidence:
        self.recovered.append(claim)
        return self.evidence or RunEvidence(
            run_id=claim.run.id,
            session_id=claim.run.session_id,
            started=False,
        )

    def cancel(self, task_id: str) -> None:
        self.cancel_calls.append(task_id)
        self.cancelled.set()

    async def close(self) -> None:
        assert self.running == 0
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def storage(tmp_path: Path) -> tuple[KernelDatabase, TaskStore]:
    database = KernelDatabase.for_workspace(tmp_path)
    bots = BotStore(database)
    bots.save(Config(bot=BotConfig(id="worker-bot")))
    return database, TaskStore(database, bots)


def submit(tasks: TaskStore, text: str = "work") -> str:
    return tasks.submit(TaskSubmission(bot_id="worker-bot", text=text)).id


async def test_tick_persists_preallocated_ids_checkpoints_and_result(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner, worker_id="test", poll_interval=0.01)
    try:
        assert await worker.tick()
        assert not await worker.tick()
        task = tasks.get(task_id)
        run = tasks.runs(task_id)[0]
        assert runner.claims[0].run.id == run.id
        assert runner.claims[0].run.state == DurableRunState.STARTED
        assert runner.claims[0].task.active_run_id == run.id
        assert run.worker_id == "test"
        assert task.session_id == run.session_id
        assert task.state == TaskState.SUCCEEDED
        assert task.cumulative_metrics.model_requests == 1
        assert task.cumulative_metrics.elapsed_ms == 20
    finally:
        await worker.close()
    assert runner.close_calls == 1
    await worker.close()
    assert runner.close_calls == 1


async def test_running_cancel_is_cooperative_until_runner_settles(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    tick = asyncio.create_task(worker.tick())
    try:
        await asyncio.wait_for(runner.started.wait(), 1)
        tasks.cancel(task_id)
        await asyncio.wait_for(runner.cancelled.wait(), 1)
        assert tasks.get(task_id).state == TaskState.RUNNING
        assert not tick.done()
        assert runner.cancel_calls == [task_id]
        runner.release.set()
        assert await asyncio.wait_for(tick, 1)
        assert tasks.get(task_id).state == TaskState.CANCELLED
    finally:
        runner.release.set()
        await tick
        await worker.close()


async def test_close_keeps_process_lock_until_execution_has_stopped(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    tick = asyncio.create_task(worker.tick())
    await asyncio.wait_for(runner.started.wait(), 1)
    closer = asyncio.create_task(worker.close())
    await asyncio.wait_for(runner.cancelled.wait(), 1)
    assert not closer.done()
    with pytest.raises(ConfigurationError, match="Another worker"):
        TaskWorker(database, tasks, ControlledRunner())
    runner.release.set()
    await asyncio.wait_for(asyncio.gather(tick, closer), 1)
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()
    with pytest.raises(ConfigurationError, match="closed"):
        await worker.tick()


async def test_concurrent_ticks_cannot_execute_two_runs_at_once(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    submit(tasks, "first")
    submit(tasks, "second")
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    try:
        assert await asyncio.gather(worker.tick(), worker.tick()) == [True, True]
        assert runner.max_running == 1
        assert len(runner.claims) == 2
    finally:
        await worker.close()


async def test_first_tick_recovers_placeholder_before_claiming_again(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    abandoned = tasks.claim("dead-worker")
    assert abandoned is not None
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    try:
        assert await worker.tick()
        assert runner.recovered[0].run.id == abandoned.run.id
        assert runner.claims[0].run.id != abandoned.run.id
        assert tasks.get(task_id).state == TaskState.SUCCEEDED
        assert len(tasks.runs(task_id)) == 2
    finally:
        await worker.close()


async def test_recover_checks_the_runner_evidence_identity(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    claim = tasks.claim("dead-worker")
    assert claim is not None
    runner = ControlledRunner()
    runner.evidence = RunEvidence(run_id="wrong", session_id=claim.run.session_id, started=False)
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(PersistenceError, match="different Run"):
        await worker.recover()
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert runner.close_calls == 1
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()


async def test_recover_unknown_side_effect_is_not_dispatched(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    claim = tasks.claim("dead-worker")
    assert claim is not None
    tasks.mark_started(claim.run.id)
    runner = ControlledRunner()
    runner.evidence = RunEvidence(
        run_id=claim.run.id,
        session_id=claim.run.session_id,
        started=True,
        unknown_side_effect=True,
        metrics=RunMetrics(tool_calls=1),
    )
    worker = TaskWorker(database, tasks, runner)
    try:
        assert not await worker.tick()
        assert tasks.get(task_id).state == TaskState.NEEDS_RECONCILIATION
        assert not runner.claims
    finally:
        await worker.close()


@pytest.mark.parametrize("status", [RunStatus.CANCELED, RunStatus.FAILED, RunStatus.PAUSED])
async def test_normal_run_unknown_side_effect_requires_reconciliation(
    tmp_path: Path, status: RunStatus
) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner()
    runner.status = status
    runner.unknown_side_effect = True
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    try:
        assert await worker.tick()
        task = tasks.get(task_id)
        assert task.state == TaskState.NEEDS_RECONCILIATION
        assert task.active_run_id is None
        assert task.cumulative_metrics.model_requests == 1
        assert task.cumulative_metrics.elapsed_ms == 20
        assert tasks.runs(task_id)[0].state == DurableRunState.UNCERTAIN
        assert not await worker.tick()
        assert len(runner.claims) == 1
    finally:
        await worker.close()


async def test_runner_failure_is_reported_and_releases_lock(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner()
    runner.fail = RuntimeError("runner failed")
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(RuntimeError, match="runner failed"):
        await worker.tick()
    assert runner.close_calls == 1
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert tasks.runs(task_id)[0].result is None
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()


async def test_checkpoint_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)

    def broken_checkpoint(run_id: str, metrics: RunMetrics) -> None:
        raise PersistenceError("checkpoint failed")

    monkeypatch.setattr(tasks, "checkpoint", broken_checkpoint)
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(PersistenceError, match="checkpoint failed"):
        await worker.tick()
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert runner.close_calls == 1


async def test_polling_failure_stops_runner_before_releasing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    ticking = asyncio.create_task(worker.tick())
    await asyncio.wait_for(runner.started.wait(), 1)
    original_get = tasks.get

    def broken_get(task_id: str, *, bot_id: str | None = None) -> None:
        raise PersistenceError("polling failed")

    monkeypatch.setattr(tasks, "get", broken_get)
    await asyncio.wait_for(runner.cancelled.wait(), 1)
    assert not ticking.done()
    with pytest.raises(ConfigurationError, match="Another worker"):
        TaskWorker(database, tasks, ControlledRunner())
    runner.release.set()
    with pytest.raises(PersistenceError, match="polling failed"):
        await asyncio.wait_for(ticking, 1)
    assert original_get(task_id).state == TaskState.RUNNING
    assert runner.close_calls == 1


async def test_finish_failure_preserves_run_for_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)

    def broken_finish(*args: object, **kwargs: object) -> None:
        raise PersistenceError("finish failed")

    monkeypatch.setattr(tasks, "finish", broken_finish)
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(PersistenceError, match="finish failed"):
        await worker.tick()
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert tasks.runs(task_id)[0].result is None
    assert runner.close_calls == 1


async def test_persistence_failed_result_is_not_committed_as_success(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner()
    runner.persistence_failed = True
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(PersistenceError, match="journal persistence failed"):
        await worker.tick()
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert runner.close_calls == 1


async def test_serve_stop_cancels_and_waits_for_active_run(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    stop = asyncio.Event()
    serving = asyncio.create_task(worker.serve(stop))
    await asyncio.wait_for(runner.started.wait(), 1)
    stop.set()
    await asyncio.wait_for(runner.cancelled.wait(), 1)
    assert not serving.done()
    runner.release.set()
    await asyncio.wait_for(serving, 1)
    assert tasks.get(task_id).state == TaskState.CANCELLED
    assert runner.close_calls == 1


async def test_serve_idle_stop_closes_runner(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    runner = ControlledRunner()
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    stop = asyncio.Event()
    serving = asyncio.create_task(worker.serve(stop))
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(serving, 1)
    assert runner.close_calls == 1
    assert not runner.claims


async def test_cancelled_tick_drains_runner_before_releasing_lock(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    task_id = submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    ticking = asyncio.create_task(worker.tick())
    await asyncio.wait_for(runner.started.wait(), 1)
    ticking.cancel()
    await asyncio.wait_for(runner.cancelled.wait(), 1)
    ticking.cancel()
    with pytest.raises(ConfigurationError, match="Another worker"):
        TaskWorker(database, tasks, ControlledRunner())
    runner.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(ticking, 1)
    assert tasks.get(task_id).state == TaskState.RUNNING
    assert runner.close_calls == 1
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()


async def test_close_failure_is_reported_and_releases_lock(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    runner = ControlledRunner()
    runner.close_error = RuntimeError("close failed")
    worker = TaskWorker(database, tasks, runner)
    with pytest.raises(RuntimeError, match="close failed"):
        await worker.close()
    assert runner.close_calls == 1
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()


async def test_cancelled_close_waits_for_runner_and_releases_lock(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    submit(tasks)
    runner = ControlledRunner(gated=True)
    worker = TaskWorker(database, tasks, runner, poll_interval=0.01)
    ticking = asyncio.create_task(worker.tick())
    await asyncio.wait_for(runner.started.wait(), 1)
    closer = asyncio.create_task(worker.close())
    await asyncio.wait_for(runner.cancelled.wait(), 1)
    closer.cancel()
    await asyncio.sleep(0)
    assert not closer.done()
    with pytest.raises(ConfigurationError, match="Another worker"):
        TaskWorker(database, tasks, ControlledRunner())
    runner.release.set()
    await asyncio.wait_for(ticking, 1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closer, 1)
    assert runner.close_calls == 1
    successor = TaskWorker(database, tasks, ControlledRunner())
    await successor.close()


@pytest.mark.parametrize("poll_interval", [0, -1, float("inf"), float("nan")])
def test_invalid_polling_interval_is_rejected(tmp_path: Path, poll_interval: float) -> None:
    database, tasks = storage(tmp_path)
    with pytest.raises(ConfigurationError, match="positive and finite"):
        TaskWorker(database, tasks, ControlledRunner(), poll_interval=poll_interval)


async def test_os_lock_survives_other_process_and_releases_on_process_death(tmp_path: Path) -> None:
    database, tasks = storage(tmp_path)
    program = """
import sys
from pathlib import Path
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.worker import TaskWorker
database = KernelDatabase(Path(sys.argv[1]))
worker = TaskWorker(database, None, None)
print('locked', flush=True)
sys.stdin.read()
"""
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        program,
        str(database.path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        assert (await asyncio.wait_for(child.stdout.readline(), 5)).strip() == b"locked"
        with pytest.raises(ConfigurationError, match="Another worker"):
            TaskWorker(database, tasks, ControlledRunner())
        child.kill()
        await asyncio.wait_for(child.wait(), 5)
        successor = TaskWorker(database, tasks, ControlledRunner())
        await successor.close()
    finally:
        if child.returncode is None:
            child.kill()
            await asyncio.wait_for(child.wait(), 5)
        await child.communicate()
