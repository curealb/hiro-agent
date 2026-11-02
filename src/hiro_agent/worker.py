"""One process owns scheduling; execution and journal recovery stay in TaskRunner."""

from __future__ import annotations

import asyncio
import fcntl
import math
import os
from uuid import uuid4

from hiro_agent.contracts import RunMetrics
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    RunClaim,
    RunEvidence,
    TaskRecord,
    TaskRunner,
    TaskRunOutcome,
    TaskState,
    TaskStoreService,
)
from hiro_agent.kernel_db import KernelDatabase


class TaskWorker:
    """Hold the OS lock until all in-flight runner work has actually stopped.

    A runner exception deliberately leaves the claimed Run available for journal
    reconciliation. The scheduler never invents an execution result on failure.
    """

    def __init__(
        self,
        database: KernelDatabase,
        tasks: TaskStoreService,
        runner: TaskRunner,
        *,
        worker_id: str | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ConfigurationError("Worker polling interval must be positive and finite")
        if worker_id is not None and not worker_id.strip():
            raise ConfigurationError("Worker ID must not be empty")
        self.database = database
        self.tasks = tasks
        self.runner = runner
        self.worker_id = worker_id or uuid4().hex
        self.poll_interval = poll_interval
        self._operation_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._active: asyncio.Task[TaskRunOutcome] | None = None
        self._active_task_id: str | None = None
        self._stop: asyncio.Event | None = None
        self._recovered = False
        self._closing = False
        self._closed = False
        self._lock_fd: int | None = None
        try:
            descriptor = os.open(
                database.path.parent / "worker.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(descriptor)
                raise
            self._lock_fd = descriptor
        except BlockingIOError as exc:
            raise ConfigurationError("Another worker already owns this workspace") from exc
        except OSError as exc:
            raise PersistenceError("Worker process lock could not be acquired") from exc

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise ConfigurationError("Worker is closed")

    async def _recover_locked(self) -> list[TaskRecord]:
        recovered: list[TaskRecord] = []
        for task in self.tasks.list(states=(TaskState.RUNNING,)):
            run = next(
                (run for run in self.tasks.runs(task.id) if run.id == task.active_run_id), None
            )
            if run is None or run.session_id != task.session_id:
                raise PersistenceError("Running task has no matching active Run")
            evidence = await self.runner.recover(RunClaim(task=task, run=run))
            if evidence.run_id != run.id or evidence.session_id != run.session_id:
                raise PersistenceError("Runner recovery returned evidence for a different Run")
            recovered.append(self.tasks.reconcile(evidence))
        self._recovered = True
        return recovered

    async def recover(self) -> list[TaskRecord]:
        """Recover every interrupted Run before any new execution can start."""
        try:
            async with self._operation_lock:
                self._ensure_open()
                return await self._recover_locked()
        except BaseException:
            await self.close()
            raise

    async def _drain(self, execution: asyncio.Task[TaskRunOutcome]) -> None:
        # Repeated cancellation of the scheduler must not release the process
        # lock while its runner is still executing a side effect.
        while not execution.done():
            try:
                await asyncio.shield(execution)
            except asyncio.CancelledError:
                if not execution.done():
                    continue
        execution.result()

    async def _execute(self, claim: RunClaim) -> None:
        claim = RunClaim(task=claim.task, run=self.tasks.mark_started(claim.run.id))

        def checkpoint(metrics: RunMetrics) -> None:
            self.tasks.checkpoint(claim.run.id, metrics)

        execution = asyncio.create_task(self.runner.run(claim, checkpoint))
        self._active = execution
        self._active_task_id = claim.task.id
        try:
            cancellation_sent = False
            while not execution.done():
                await asyncio.wait((execution,), timeout=self.poll_interval)
                if execution.done():
                    break
                stop_requested = self._closing or (self._stop is not None and self._stop.is_set())
                if not cancellation_sent and (
                    stop_requested or self.tasks.get(claim.task.id).cancel_requested
                ):
                    self.runner.cancel(claim.task.id)
                    cancellation_sent = True
            outcome = execution.result()
            if outcome.result.persistence_failed:
                raise PersistenceError("Run journal persistence failed; reconciliation is required")
            if outcome.unknown_side_effect:
                self.tasks.reconcile(
                    RunEvidence(
                        run_id=claim.run.id,
                        session_id=claim.run.session_id,
                        started=True,
                        result=outcome.result,
                        metrics=outcome.result.metrics,
                        unknown_side_effect=True,
                        pending_operation=outcome.pending_operation,
                        interaction_id=outcome.interaction_id,
                        waiting_for=outcome.waiting_for,
                    )
                )
            else:
                self.tasks.finish(
                    claim.run.id,
                    outcome.result,
                    pending_operation=outcome.pending_operation,
                    interaction_id=outcome.interaction_id,
                    waiting_for=outcome.waiting_for,
                )
        except BaseException as error:
            if not execution.done():
                cleanup_errors: list[BaseException] = []
                try:
                    self.runner.cancel(claim.task.id)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                try:
                    await self._drain(execution)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                if cleanup_errors:
                    raise BaseExceptionGroup(
                        "Worker failed and its runner also failed during shutdown",
                        [error, *cleanup_errors],
                    ) from None
            raise
        finally:
            self._active = None
            self._active_task_id = None

    async def tick(self) -> bool:
        """Run at most one queued task, recovering old claims on first use."""
        try:
            async with self._operation_lock:
                self._ensure_open()
                if not self._recovered:
                    await self._recover_locked()
                claim = self.tasks.claim(self.worker_id)
                if claim is None:
                    return False
                await self._execute(claim)
                return True
        except BaseException:
            await self.close()
            raise

    async def serve(self, stop: asyncio.Event) -> None:
        """Poll until stop; cancel and settle an active runner before closing."""
        self._ensure_open()
        if self._stop is not None:
            raise ConfigurationError("Worker is already serving")
        self._stop = stop
        try:
            await self.recover()
            while not stop.is_set():
                if not await self.tick():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
                    except TimeoutError:
                        pass
        finally:
            await self.close()

    async def close(self) -> None:
        """Idempotently settle in-flight execution, close the runner, release flock."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._shutdown())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                if not self._close_task.done():
                    cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _shutdown(self) -> None:
        errors: list[BaseException] = []
        try:
            if self._active is not None and not self._active.done():
                assert self._active_task_id is not None
                try:
                    self.runner.cancel(self._active_task_id)
                except BaseException as error:
                    errors.append(error)
            async with self._operation_lock:
                try:
                    await self.runner.close()
                except BaseException as error:
                    errors.append(error)
        finally:
            # Never unlink the lock file: another process may already have
            # opened this inode while waiting for the current owner.
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Worker shutdown failed", errors)
