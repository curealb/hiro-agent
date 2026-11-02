"""Transactional task/run state; journal execution facts arrive as explicit evidence."""

from __future__ import annotations

import builtins
import sqlite3
from typing import Literal
from uuid import uuid4

from hiro_agent.bots import BotStore
from hiro_agent.chains import ChainStore
from hiro_agent.contracts import RunMetrics, RunResult, RunStatus
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    TERMINAL_TASK_STATES,
    WAITING_TASK_STATES,
    ChainRecord,
    DurableRunRecord,
    DurableRunState,
    OperationIntent,
    RunClaim,
    RunEvidence,
    TaskRecord,
    TaskRunOutcome,
    TaskState,
    TaskSubmission,
    canonical_arguments,
    utc_now,
)
from hiro_agent.kernel_db import KernelDatabase


class TaskStore:
    def __init__(self, database: KernelDatabase, bots: BotStore) -> None:
        self.database = database
        self.bots = bots
        database.migrate(
            "durable_tasks",
            1,
            (
                "CREATE TABLE durable_tasks (id TEXT PRIMARY KEY, "
                "bot_id TEXT NOT NULL REFERENCES bots(bot_id), session_id TEXT NOT NULL UNIQUE, "
                "idempotency_key TEXT, state TEXT NOT NULL, created_at TEXT NOT NULL, "
                "record_json TEXT NOT NULL, submission_json TEXT NOT NULL, "
                "UNIQUE(bot_id, idempotency_key))",
                "CREATE INDEX durable_tasks_queue ON durable_tasks(state, created_at, id)",
                "CREATE TABLE durable_runs (id TEXT PRIMARY KEY, "
                "task_id TEXT NOT NULL REFERENCES durable_tasks(id), state TEXT NOT NULL, "
                "record_json TEXT NOT NULL, outcome_json TEXT, evidence_json TEXT)",
                "CREATE UNIQUE INDEX durable_runs_active ON durable_runs(task_id) "
                "WHERE state IN ('claimed', 'started')",
                "CREATE TABLE durable_task_resolutions (id INTEGER PRIMARY KEY, "
                "task_id TEXT NOT NULL REFERENCES durable_tasks(id), run_id TEXT, "
                "action TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)",
            ),
        )
        self.chains = ChainStore(database)
        # Upgrade existing Tasks once, including already charged interrupted Runs.
        # Membership is the durable once-only backfill receipt.
        with database.transaction() as connection:
            rows = connection.execute(
                "SELECT record_json FROM durable_tasks ORDER BY created_at, id"
            ).fetchall()
            for row in rows:
                task = TaskRecord.model_validate_json(row[0])
                if task.chain_id is None:
                    task.chain_id = task.id
                    self._save_task(connection, task)
                self.chains.join(connection, task)

    def submit(self, request: TaskSubmission) -> TaskRecord:
        # Snapshot mutable input; canonical JSON also rejects NaN in caller metadata.
        submission = canonical_arguments(request.model_dump(mode="json"))
        request = TaskSubmission.model_validate_json(submission)
        bot = self.bots.get(request.bot_id)
        with self.database.transaction() as connection:
            if request.idempotency_key is not None:
                previous = connection.execute(
                    "SELECT record_json, submission_json FROM durable_tasks "
                    "WHERE bot_id=? AND idempotency_key=?",
                    (request.bot_id, request.idempotency_key),
                ).fetchone()
                if previous is not None:
                    previous_request = TaskSubmission.model_validate_json(
                        previous["submission_json"]
                    )
                    if canonical_arguments(previous_request.model_dump(mode="json")) != submission:
                        raise ConfigurationError("任务幂等键已用于不同的请求")
                    return TaskRecord.model_validate_json(previous["record_json"])
            task = TaskRecord(
                id=uuid4().hex,
                session_id=uuid4().hex,
                bot_id=request.bot_id,
                text=request.text,
                source=request.source,
                idempotency_key=request.idempotency_key,
                budget=request.budget or bot.config.budget,
                chain_id=request.chain_id,
                parent_task_id=request.parent_task_id,
                metadata=request.metadata,
            )
            parent = connection.execute(
                "SELECT record_json FROM durable_tasks WHERE id=?", (request.parent_task_id,)
            ).fetchone()
            if parent is not None:
                parent_task = TaskRecord.model_validate_json(parent[0])
                if request.chain_id is not None and request.chain_id != parent_task.chain_id:
                    raise ConfigurationError("Child Task cannot change its parent's chain")
                if request.chain_budget is not None or request.chain_max_hops is not None:
                    raise ConfigurationError("Only root Tasks may set chain limits")
                task.chain_id = parent_task.chain_id
            task.chain_id = task.chain_id or task.id
            connection.execute(
                "INSERT INTO durable_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.id,
                    task.bot_id,
                    task.session_id,
                    task.idempotency_key,
                    task.state,
                    task.created_at.isoformat(),
                    task.model_dump_json(),
                    submission,
                ),
            )
            self.chains.join(
                connection, task, budget=request.chain_budget, max_hops=request.chain_max_hops
            )
            return task

    def chain(self, task_id: str, *, bot_id: str | None = None) -> ChainRecord:
        with self.database.connect() as connection:
            task = self._task(connection, task_id)
            self._owner(task, bot_id)
            assert task.chain_id is not None
            return self.chains.get(connection, task.chain_id)

    def get(self, task_id: str, *, bot_id: str | None = None) -> TaskRecord:
        with self.database.connect() as connection:
            task = self._task(connection, task_id)
        self._owner(task, bot_id)
        return task

    def list(
        self, *, bot_id: str | None = None, states: tuple[TaskState, ...] = ()
    ) -> list[TaskRecord]:
        clauses: list[str] = []
        parameters: list[str] = []
        if bot_id is not None:
            clauses.append("bot_id=?")
            parameters.append(bot_id)
        if states:
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            parameters.extend(states)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM durable_tasks" + where + " ORDER BY created_at, id",
                parameters,
            ).fetchall()
        return [TaskRecord.model_validate_json(row["record_json"]) for row in rows]

    def cancel(self, task_id: str, *, bot_id: str | None = None) -> TaskRecord:
        with self.database.transaction() as connection:
            task = self._task(connection, task_id)
            self._owner(task, bot_id)
            if task.state in TERMINAL_TASK_STATES or task.cancel_requested:
                return task
            task.cancel_requested = True
            if task.state not in (TaskState.RUNNING, TaskState.NEEDS_RECONCILIATION):
                task.state = TaskState.CANCELLED
                task.reason = "任务已取消"
                task.pending_operation = None
                task.interaction_id = None
            self._save_task(connection, task)
            return task

    def claim(self, worker_id: str) -> RunClaim | None:
        if not worker_id.strip():
            raise ConfigurationError("worker ID 不能为空")
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT record_json FROM durable_tasks WHERE state=? ORDER BY created_at, id",
                (TaskState.QUEUED,),
            ).fetchall()
            for row in rows:
                task = TaskRecord.model_validate_json(row["record_json"])
                assert task.chain_id is not None
                exhausted = self._exhausted(task) or self.chains.exhausted(
                    self.chains.get(connection, task.chain_id)
                )
                if task.cancel_requested or exhausted:
                    task.state = TaskState.CANCELLED if task.cancel_requested else TaskState.FAILED
                    task.reason = "任务已取消" if task.cancel_requested else exhausted
                    self._save_task(connection, task)
                    continue
                run = DurableRunRecord(
                    id=uuid4().hex,
                    task_id=task.id,
                    session_id=task.session_id,
                    worker_id=worker_id,
                )
                connection.execute(
                    "INSERT INTO durable_runs(id, task_id, state, record_json) VALUES (?, ?, ?, ?)",
                    (run.id, task.id, run.state, run.model_dump_json()),
                )
                task.state = TaskState.RUNNING
                task.active_run_id = run.id
                self._save_task(connection, task)
                return RunClaim(task=task, run=run)
        return None

    def mark_started(self, run_id: str) -> DurableRunRecord:
        with self.database.transaction() as connection:
            run = self._run(connection, run_id)
            task = self._task(connection, run.task_id)
            self._active(task, run)
            if run.state == DurableRunState.STARTED:
                return run
            run.state = DurableRunState.STARTED
            run.started_at = utc_now()
            self._save_run(connection, run)
            return run

    def checkpoint(self, run_id: str, metrics: RunMetrics) -> TaskRecord:
        with self.database.transaction() as connection:
            run = self._run(connection, run_id)
            task = self._task(connection, run.task_id)
            self._active(task, run)
            if self._charge(connection, task, run, metrics):
                self._save_run(connection, run)
                self._save_task(connection, task)
            return task

    def finish(
        self,
        run_id: str,
        result: RunResult,
        *,
        pending_operation: OperationIntent | None = None,
        interaction_id: str | None = None,
        waiting_for: Literal["approval", "question"] | None = None,
    ) -> TaskRecord:
        outcome = TaskRunOutcome(
            result=result,
            pending_operation=pending_operation,
            interaction_id=interaction_id,
            waiting_for=waiting_for,
        )
        serialized = canonical_arguments(outcome.model_dump(mode="json"))
        with self.database.transaction() as connection:
            run = self._run(connection, run_id)
            task = self._task(connection, run.task_id)
            previous = connection.execute(
                "SELECT outcome_json FROM durable_runs WHERE id=?", (run_id,)
            ).fetchone()[0]
            if previous is not None:
                if previous != serialized:
                    raise ConfigurationError("Run 已用不同结果结算")
                return task
            self._active(task, run)
            self._settle(connection, task, run, outcome)
            connection.execute(
                "UPDATE durable_runs SET outcome_json=? WHERE id=?", (serialized, run_id)
            )
            return task

    def wake(self, task_id: str, interaction_id: str) -> TaskRecord:
        with self.database.transaction() as connection:
            task = self._task(connection, task_id)
            if (
                task.state not in WAITING_TASK_STATES
                or task.interaction_id != interaction_id
                or task.cancel_requested
            ):
                return task
            task.state = TaskState.QUEUED
            # Keep the exact intent/interaction until the new Run consumes them.
            self._save_task(connection, task)
            return task

    def reconcile(self, evidence: RunEvidence) -> TaskRecord:
        serialized = canonical_arguments(evidence.model_dump(mode="json"))
        with self.database.transaction() as connection:
            run = self._run(connection, evidence.run_id)
            task = self._task(connection, run.task_id)
            if evidence.session_id != run.session_id or task.session_id != run.session_id:
                raise ConfigurationError("恢复证据的 Session 与 Run 不匹配")
            previous = connection.execute(
                "SELECT evidence_json FROM durable_runs WHERE id=?", (run.id,)
            ).fetchone()[0]
            if previous is not None:
                if previous != serialized:
                    raise ConfigurationError("Run 已用不同恢复证据结算")
                return task
            self._validate_outcome(
                task,
                run,
                TaskRunOutcome(
                    result=evidence.result,
                    pending_operation=evidence.pending_operation,
                    interaction_id=evidence.interaction_id,
                    waiting_for=evidence.waiting_for,
                )
                if evidence.result is not None
                else None,
            )
            self._active(task, run)
            # A durable SQL started marker without journal evidence is never a
            # harmless empty claim. Likewise, counters imply execution occurred.
            missing_started = not evidence.started and (
                run.state == DurableRunState.STARTED
                or any(run.metrics.model_dump().values())
                or any(evidence.metrics.model_dump().values())
            )
            # Checkpointed counters remain a durable lower bound even if the
            # journal is missing or recovery can reconstruct only some fields.
            metrics = self._maximum(run.metrics, evidence.metrics)
            if evidence.result is not None:
                metrics = self._maximum(metrics, evidence.result.metrics)
            self._charge(connection, task, run, metrics)
            if evidence.started and run.started_at is None:
                run.started_at = utc_now()
            if evidence.unknown_side_effect or missing_started:
                run.state = DurableRunState.UNCERTAIN
                run.result = evidence.result
                run.finished_at = utc_now() if evidence.result is not None else None
                task.state = TaskState.NEEDS_RECONCILIATION
                task.reason = (
                    "工具副作用结果未知，需要人工核实"
                    if evidence.unknown_side_effect
                    else "已开始的 Run 缺少执行日志，需要人工核实"
                )
                task.active_run_id = None
                task.pending_operation = None
                task.interaction_id = None
                self._save_run(connection, run)
                self._save_task(connection, task)
            elif evidence.result is not None:
                # Evidence may reconstruct counters not yet present in run.ended.
                result = evidence.result.model_copy(update={"metrics": metrics})
                outcome = TaskRunOutcome(
                    result=result,
                    pending_operation=evidence.pending_operation,
                    interaction_id=evidence.interaction_id,
                    waiting_for=evidence.waiting_for,
                )
                self._settle(connection, task, run, outcome)
                connection.execute(
                    "UPDATE durable_runs SET outcome_json=? WHERE id=?",
                    (canonical_arguments(outcome.model_dump(mode="json")), run.id),
                )
            elif not evidence.started:
                run.state = DurableRunState.FINISHED
                run.finished_at = utc_now()
                task.active_run_id = None
                task.state = TaskState.CANCELLED if task.cancel_requested else TaskState.QUEUED
                task.reason = "任务已取消" if task.cancel_requested else "未开始的领取已回收"
                self._save_run(connection, run)
                self._save_task(connection, task)
            else:
                # The adapter must provide a validated recovered terminal result.
                # An unfinished started Run alone cannot prove replay is safe.
                run.state = DurableRunState.UNCERTAIN
                task.state = TaskState.NEEDS_RECONCILIATION
                task.reason = "已开始的 Run 缺少最终结果，需要人工核实"
                task.active_run_id = None
                task.pending_operation = None
                task.interaction_id = None
                self._save_run(connection, run)
                self._save_task(connection, task)
            connection.execute(
                "UPDATE durable_runs SET evidence_json=? WHERE id=?", (serialized, run.id)
            )
            return task

    def runs(self, task_id: str) -> builtins.list[DurableRunRecord]:
        with self.database.connect() as connection:
            self._task(connection, task_id)
            rows = connection.execute(
                "SELECT record_json FROM durable_runs WHERE task_id=? ORDER BY rowid", (task_id,)
            ).fetchall()
        return [DurableRunRecord.model_validate_json(row["record_json"]) for row in rows]

    def resolve(
        self, task_id: str, *, action: Literal["retry", "fail", "cancel"], note: str
    ) -> TaskRecord:
        if action not in ("retry", "fail", "cancel") or not note.strip():
            raise ConfigurationError("人工核实必须提供有效动作和非空说明")
        with self.database.transaction() as connection:
            task = self._task(connection, task_id)
            if task.state != TaskState.NEEDS_RECONCILIATION:
                raise ConfigurationError("只有待核实任务可以人工处理")
            if action == "retry" and task.cancel_requested:
                raise ConfigurationError("已请求取消的任务不能重新运行")
            task.state = {
                "retry": TaskState.QUEUED,
                "fail": TaskState.FAILED,
                "cancel": TaskState.CANCELLED,
            }[action]
            task.cancel_requested = task.cancel_requested or action == "cancel"
            task.reason = f"人工核实（{action}）: {note.strip()}"
            task.pending_operation = None
            task.interaction_id = None
            connection.execute(
                "INSERT INTO durable_task_resolutions(task_id, run_id, action, note, created_at) "
                "VALUES (?, (SELECT id FROM durable_runs WHERE task_id=? "
                "ORDER BY rowid DESC LIMIT 1), ?, ?, ?)",
                (task.id, task.id, action, note.strip(), utc_now().isoformat()),
            )
            self._save_task(connection, task)
            return task

    @staticmethod
    def _task(connection: sqlite3.Connection, task_id: str) -> TaskRecord:
        row = connection.execute(
            "SELECT record_json FROM durable_tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise ConfigurationError(f"任务不存在: {task_id}")
        return TaskRecord.model_validate_json(row["record_json"])

    @staticmethod
    def _run(connection: sqlite3.Connection, run_id: str) -> DurableRunRecord:
        row = connection.execute(
            "SELECT record_json FROM durable_runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise ConfigurationError(f"Run 不存在: {run_id}")
        return DurableRunRecord.model_validate_json(row["record_json"])

    @staticmethod
    def _owner(task: TaskRecord, bot_id: str | None) -> None:
        if bot_id is not None and task.bot_id != bot_id:
            raise ConfigurationError("任务不属于指定 Bot")

    @staticmethod
    def _active(task: TaskRecord, run: DurableRunRecord) -> None:
        if (
            task.state != TaskState.RUNNING
            or task.active_run_id != run.id
            or task.session_id != run.session_id
            or run.state not in (DurableRunState.CLAIMED, DurableRunState.STARTED)
        ):
            raise ConfigurationError("Run 不是任务的当前活动运行")

    @staticmethod
    def _save_task(connection: sqlite3.Connection, task: TaskRecord) -> None:
        previous_revision = task.revision
        task.revision += 1
        task.updated_at = utc_now()
        changed = connection.execute(
            "UPDATE durable_tasks SET state=?, record_json=? "
            "WHERE id=? AND json_extract(record_json, '$.revision')=?",
            (task.state, task.model_dump_json(), task.id, previous_revision),
        ).rowcount
        if changed != 1:
            raise PersistenceError("任务修订版本冲突")

    @staticmethod
    def _save_run(connection: sqlite3.Connection, run: DurableRunRecord) -> None:
        connection.execute(
            "UPDATE durable_runs SET state=?, record_json=? WHERE id=?",
            (run.state, run.model_dump_json(), run.id),
        )

    def _charge(
        self,
        connection: sqlite3.Connection,
        task: TaskRecord,
        run: DurableRunRecord,
        metrics: RunMetrics,
    ) -> bool:
        values = metrics.model_dump()
        previous = run.metrics.model_dump()
        if any(type(value) is not int or value < previous[name] for name, value in values.items()):
            raise ConfigurationError("Run 累计指标必须为非负整数且不能倒退")
        if values == previous:
            return False
        total = task.cumulative_metrics.model_dump()
        self.chains.charge(
            connection,
            task,
            RunMetrics.model_validate(
                {name: value - previous[name] for name, value in values.items()}
            ),
        )
        task.cumulative_metrics = RunMetrics.model_validate(
            {name: total[name] + value - previous[name] for name, value in values.items()}
        )
        run.metrics = metrics.model_copy(deep=True)
        return True

    @staticmethod
    def _maximum(left: RunMetrics, right: RunMetrics) -> RunMetrics:
        if any(
            type(value) is not int or value < 0
            for metrics in (left, right)
            for value in metrics.model_dump().values()
        ):
            raise ConfigurationError("恢复指标必须为非负整数")
        return RunMetrics.model_validate(
            {
                name: max(getattr(left, name), getattr(right, name))
                for name in RunMetrics.model_fields
            }
        )

    @staticmethod
    def _exhausted(task: TaskRecord) -> str:
        if task.cumulative_metrics.model_requests >= task.budget.max_model_requests:
            return "任务累计模型请求预算耗尽"
        if task.cumulative_metrics.tool_calls >= task.budget.max_tool_calls:
            return "任务累计工具调用预算耗尽"
        if task.cumulative_metrics.elapsed_ms >= task.budget.timeout_seconds * 1000:
            return "任务累计运行时间预算耗尽"
        return ""

    @staticmethod
    def _validate_outcome(
        task: TaskRecord, run: DurableRunRecord, outcome: TaskRunOutcome | None
    ) -> None:
        if outcome is None:
            return
        result = outcome.result
        if result.run_id != run.id or result.session_id != run.session_id:
            raise ConfigurationError("结果的 Run/Session 关联不匹配")
        if result.persistence_failed:
            raise PersistenceError("Run 日志持久化失败，必须恢复后对账")
        if result.status not in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELED,
            RunStatus.PAUSED,
        ):
            raise ConfigurationError("Run 结果不是可结算状态")
        operation = outcome.pending_operation
        if operation is not None and (
            operation.task_id != task.id
            or operation.bot_id != task.bot_id
            or operation.session_id != run.session_id
            or operation.run_id != run.id
        ):
            raise ConfigurationError("待执行操作的 Task/Bot/Session/Run 关联不匹配")
        if outcome.waiting_for is None:
            if outcome.interaction_id is not None or operation is not None:
                raise ConfigurationError("交互和待执行操作必须指定等待类型")
        elif (
            result.status != RunStatus.PAUSED
            or not outcome.interaction_id
            or (outcome.waiting_for == "approval" and operation is None)
        ):
            raise ConfigurationError("等待必须有 paused 结果和匹配的交互/审批操作")

    def _settle(
        self,
        connection: sqlite3.Connection,
        task: TaskRecord,
        run: DurableRunRecord,
        outcome: TaskRunOutcome,
    ) -> None:
        self._validate_outcome(task, run, outcome)
        result = outcome.result
        self._charge(connection, task, run, result.metrics)
        run.state = DurableRunState.FINISHED
        run.result = result.model_copy(deep=True)
        run.finished_at = utc_now()
        task.active_run_id = None
        task.output = result.output
        task.reason = result.reason
        task.pending_operation = None
        task.interaction_id = None
        if task.cancel_requested or result.status == RunStatus.CANCELED:
            task.state = TaskState.CANCELLED
        elif result.status == RunStatus.PAUSED and outcome.waiting_for is not None:
            task.state = (
                TaskState.WAITING_APPROVAL
                if outcome.waiting_for == "approval"
                else TaskState.WAITING_INPUT
            )
            task.pending_operation = outcome.pending_operation
            task.interaction_id = outcome.interaction_id
        else:
            task.state = (
                TaskState.SUCCEEDED if result.status == RunStatus.COMPLETED else TaskState.FAILED
            )
        self._save_run(connection, run)
        self._save_task(connection, task)
