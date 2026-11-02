"""Durable UTC schedules, dispatched exclusively through the public Task API."""

from __future__ import annotations

import builtins
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from hiro_agent.bots import BotStore
from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import DataModel
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    TaskStoreService,
    TaskSubmission,
    canonical_arguments,
    utc_now,
)
from hiro_agent.kernel_db import KernelDatabase


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Routine 时间必须包含明确时区")
    return value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds")


class RoutineCreate(DataModel):
    """An immutable schedule definition; interval=None means exactly one trigger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bot_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scheduled_at: datetime
    interval_seconds: int | None = Field(default=None, gt=0, strict=True)
    enabled: bool = Field(default=True, strict=True)
    budget: BudgetConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scheduled_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return _utc(value)


class RoutineRecord(DataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    definition: RoutineCreate
    enabled: bool
    next_scheduled_at: datetime | None
    last_scheduled_at: datetime | None = None
    last_task_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def bot_id(self) -> str:
        return self.definition.bot_id


class RoutineTrigger(DataModel):
    """task_id=None is an accepted, durable trigger awaiting acknowledgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    routine_id: str
    scheduled_at: datetime
    submission: TaskSubmission
    task_id: str | None = None


class RoutineStore:
    """Pause new triggers, retain backlog, and drain accepted triggers after restart.

    Each tick makes at most max_triggers submission attempts (default 100, max
    1000). Pending outbox entries take precedence; otherwise the earliest due
    schedule is selected. No missed interval is skipped. Concurrent ticks may
    submit the same immutable request; TaskStore supplies durable deduplication.
    """

    def __init__(self, database: KernelDatabase, bots: BotStore, tasks: TaskStoreService) -> None:
        self.database = database
        self.bots = bots
        self.tasks = tasks
        database.migrate(
            "routines",
            1,
            (
                "CREATE TABLE routines (id TEXT PRIMARY KEY, bot_id TEXT NOT NULL, "
                "enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)), next_scheduled_at TEXT, "
                "record_json TEXT NOT NULL)",
                "CREATE INDEX routines_due ON routines(enabled, next_scheduled_at, id)",
                "CREATE TABLE routine_triggers (routine_id TEXT NOT NULL "
                "REFERENCES routines(id), scheduled_at TEXT NOT NULL, "
                "submission_json TEXT NOT NULL, task_id TEXT, "
                "PRIMARY KEY(routine_id, scheduled_at))",
                "CREATE UNIQUE INDEX routine_one_pending ON routine_triggers(routine_id) "
                "WHERE task_id IS NULL",
                "CREATE INDEX routine_pending ON routine_triggers(scheduled_at, routine_id) "
                "WHERE task_id IS NULL",
            ),
        )

    def create(self, request: RoutineCreate, *, now: datetime | None = None) -> RoutineRecord:
        # Revalidate nested mutable values and reject non-finite JSON metadata.
        canonical_arguments(request.metadata)
        request = RoutineCreate.model_validate_json(
            canonical_arguments(request.model_dump(mode="json"))
        )
        self.bots.get(request.bot_id)
        created = self._now(now)
        if request.interval_seconds is not None:
            try:
                request.scheduled_at + timedelta(seconds=request.interval_seconds)
            except OverflowError as exc:
                raise ConfigurationError("Routine 间隔超出可表示时间范围") from exc
        routine = RoutineRecord(
            id=uuid4().hex,
            definition=request,
            enabled=request.enabled,
            next_scheduled_at=request.scheduled_at,
            created_at=created,
            updated_at=created,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO routines VALUES (?, ?, ?, ?, ?)",
                (
                    routine.id,
                    routine.bot_id,
                    routine.enabled,
                    _stamp(request.scheduled_at),
                    routine.model_dump_json(),
                ),
            )
        return routine

    def get(self, routine_id: str, *, bot_id: str | None = None) -> RoutineRecord:
        with self.database.connect() as connection:
            routine = self._get(connection, routine_id)
        self._owner(routine, bot_id)
        return routine

    def list(
        self, *, bot_id: str | None = None, enabled: bool | None = None
    ) -> list[RoutineRecord]:
        clauses: list[str] = []
        parameters: list[str | bool] = []
        if bot_id is not None:
            clauses.append("bot_id=?")
            parameters.append(bot_id)
        if enabled is not None:
            clauses.append("enabled=?")
            parameters.append(enabled)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM routines" + where + " ORDER BY id", parameters
            ).fetchall()
        return [RoutineRecord.model_validate_json(row["record_json"]) for row in rows]

    def set_enabled(
        self,
        routine_id: str,
        enabled: bool,
        *,
        bot_id: str | None = None,
        now: datetime | None = None,
    ) -> RoutineRecord:
        if type(enabled) is not bool:
            raise ConfigurationError("Routine enabled 必须为布尔值")
        current_time = self._now(now)
        with self.database.transaction() as connection:
            routine = self._get(connection, routine_id)
            self._owner(routine, bot_id)
            if enabled and routine.next_scheduled_at is None:
                raise ConfigurationError("已完成的 Routine 不能重新启用，请创建新计划")
            if routine.enabled == enabled:
                return routine
            routine = routine.model_copy(update={"enabled": enabled, "updated_at": current_time})
            self._save(connection, routine)
            return routine

    def enable(
        self, routine_id: str, *, bot_id: str | None = None, now: datetime | None = None
    ) -> RoutineRecord:
        return self.set_enabled(routine_id, True, bot_id=bot_id, now=now)

    def disable(
        self, routine_id: str, *, bot_id: str | None = None, now: datetime | None = None
    ) -> RoutineRecord:
        return self.set_enabled(routine_id, False, bot_id=bot_id, now=now)

    def triggers(
        self, routine_id: str, *, bot_id: str | None = None
    ) -> builtins.list[RoutineTrigger]:
        with self.database.connect() as connection:
            self._owner(self._get(connection, routine_id), bot_id)
            rows = connection.execute(
                "SELECT * FROM routine_triggers WHERE routine_id=? ORDER BY scheduled_at",
                (routine_id,),
            ).fetchall()
        return [self._trigger(row) for row in rows]

    def tick(
        self, *, now: datetime | None = None, max_triggers: int = 100
    ) -> builtins.list[RoutineTrigger]:
        """Return triggers acknowledged by this caller, with their Task IDs.

        Synchronous: call before awaiting the worker's next task. The fixed `now`
        cutoff excludes future schedules even when submission takes time. Errors
        propagate; previously acknowledged triggers remain committed. A repeated
        tick resumes pending work. Do not use this return list as a durable event
        stream: use triggers() or the Task API to query committed work.
        """
        if type(max_triggers) is not int or not 1 <= max_triggers <= 1000:
            raise ConfigurationError("每次 tick 的 max_triggers 必须是 1 到 1000 的整数")
        current_time = self._now(now)
        dispatched: list[RoutineTrigger] = []
        for _ in range(max_triggers):
            pending = self._reserve(current_time)
            if pending is None:
                break
            # Never hold our transaction while TaskStore starts its own transaction.
            task = self.tasks.submit(pending.submission)
            acknowledged = self._acknowledge(pending, task.id, current_time)
            if acknowledged is not None:
                dispatched.append(acknowledged)
        return dispatched

    def _reserve(self, now: datetime) -> RoutineTrigger | None:
        with self.database.transaction() as connection:
            pending = connection.execute(
                "SELECT * FROM routine_triggers WHERE task_id IS NULL "
                "ORDER BY scheduled_at, routine_id LIMIT 1"
            ).fetchone()
            if pending is not None:
                return self._trigger(pending)
            row = connection.execute(
                "SELECT record_json FROM routines WHERE enabled=1 "
                "AND next_scheduled_at<=? ORDER BY next_scheduled_at, id LIMIT 1",
                (_stamp(now),),
            ).fetchone()
            if row is None:
                return None
            routine = RoutineRecord.model_validate_json(row["record_json"])
            scheduled = routine.next_scheduled_at
            if scheduled is None:
                raise PersistenceError("Routine 的计划时间与索引不一致")
            submission = TaskSubmission(
                bot_id=routine.bot_id,
                text=routine.definition.text,
                source="routine",
                idempotency_key=f"routine:{routine.id}:{_stamp(scheduled)}",
                budget=routine.definition.budget,
                metadata={
                    "routine_id": routine.id,
                    "scheduled_at": _stamp(scheduled),
                    "routine_metadata": routine.definition.metadata,
                },
            )
            connection.execute(
                "INSERT INTO routine_triggers VALUES (?, ?, ?, NULL)",
                (
                    routine.id,
                    _stamp(scheduled),
                    canonical_arguments(submission.model_dump(mode="json")),
                ),
            )
            return RoutineTrigger(
                routine_id=routine.id, scheduled_at=scheduled, submission=submission
            )

    def _acknowledge(
        self, trigger: RoutineTrigger, task_id: str, now: datetime
    ) -> RoutineTrigger | None:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT task_id FROM routine_triggers WHERE routine_id=? AND scheduled_at=?",
                (trigger.routine_id, _stamp(trigger.scheduled_at)),
            ).fetchone()
            if row is None:
                raise PersistenceError("Routine 待提交记录不存在")
            if row["task_id"] is not None:
                if row["task_id"] != task_id:
                    raise PersistenceError("Routine 触发对应的 Task ID 冲突")
                return None
            routine = self._get(connection, trigger.routine_id)
            if routine.next_scheduled_at != trigger.scheduled_at:
                raise PersistenceError("Routine 待提交记录与当前计划不一致")
            next_time = None
            if routine.definition.interval_seconds is not None:
                try:
                    next_time = trigger.scheduled_at + timedelta(
                        seconds=routine.definition.interval_seconds
                    )
                except OverflowError:
                    # No further Python/ISO date exists after the last valid slot.
                    pass
            routine = routine.model_copy(
                update={
                    "enabled": routine.enabled and next_time is not None,
                    "next_scheduled_at": next_time,
                    "last_scheduled_at": trigger.scheduled_at,
                    "last_task_id": task_id,
                    "updated_at": now,
                }
            )
            connection.execute(
                "UPDATE routine_triggers SET task_id=? WHERE routine_id=? AND scheduled_at=?",
                (task_id, trigger.routine_id, _stamp(trigger.scheduled_at)),
            )
            self._save(connection, routine)
            return trigger.model_copy(update={"task_id": task_id})

    @staticmethod
    def _now(now: datetime | None) -> datetime:
        try:
            return _utc(now if now is not None else utc_now())
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

    @staticmethod
    def _get(connection: sqlite3.Connection, routine_id: str) -> RoutineRecord:
        row = connection.execute(
            "SELECT record_json FROM routines WHERE id=?", (routine_id,)
        ).fetchone()
        if row is None:
            raise ConfigurationError(f"Routine 不存在: {routine_id}")
        return RoutineRecord.model_validate_json(row["record_json"])

    @staticmethod
    def _save(connection: sqlite3.Connection, routine: RoutineRecord) -> None:
        connection.execute(
            "UPDATE routines SET enabled=?, next_scheduled_at=?, record_json=? WHERE id=?",
            (
                routine.enabled,
                _stamp(routine.next_scheduled_at) if routine.next_scheduled_at else None,
                routine.model_dump_json(),
                routine.id,
            ),
        )

    @staticmethod
    def _owner(routine: RoutineRecord, bot_id: str | None) -> None:
        if bot_id is not None and routine.bot_id != bot_id:
            raise ConfigurationError("Routine 不属于指定 Bot")

    @staticmethod
    def _trigger(row: sqlite3.Row) -> RoutineTrigger:
        return RoutineTrigger(
            routine_id=row["routine_id"],
            scheduled_at=datetime.fromisoformat(row["scheduled_at"]),
            submission=TaskSubmission.model_validate_json(row["submission_json"]),
            task_id=row["task_id"],
        )
