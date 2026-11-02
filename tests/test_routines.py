"""Real SQLite scheduling, bounded catch-up, competing ticks, and crash recovery."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from hiro_agent.bots import BotStore
from hiro_agent.config import BotConfig, BudgetConfig, Config
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import TaskRecord, TaskSubmission
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.routines import RoutineCreate, RoutineStore

START = datetime(2026, 11, 2, 1, 2, 3, 456789, tzinfo=UTC)


def reopen(path: Path) -> RoutineStore:
    database = KernelDatabase(path)
    bots = BotStore(database)
    return RoutineStore(database, bots, TaskStore(database, bots))


@pytest.fixture
def store(tmp_path: Path) -> RoutineStore:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    bots = BotStore(database)
    bots.save(Config(bot=BotConfig(id="writer")))
    bots.save(Config(bot=BotConfig(id="reader")))
    return RoutineStore(database, bots, TaskStore(database, bots))


def test_one_shot_only_submits_task_and_survives_restart(store: RoutineStore) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Do the report", scheduled_at=START), now=START
    )
    assert store.tick(now=START - timedelta(microseconds=1)) == []
    [trigger] = store.tick(now=START)
    assert trigger.scheduled_at == START
    assert trigger.task_id is not None
    task = store.tasks.get(trigger.task_id)
    assert task.bot_id == "writer" and task.text == "Do the report"
    assert task.source == "routine" and task.state == "queued"
    assert store.tasks.runs(task.id) == []
    assert task.idempotency_key == f"routine:{routine.id}:{START.isoformat()}"
    assert task.metadata == {
        "routine_id": routine.id,
        "scheduled_at": START.isoformat(),
        "routine_metadata": {},
    }
    restarted = reopen(store.database.path)
    assert restarted.tick(now=START + timedelta(days=30)) == []
    done = restarted.get(routine.id)
    assert not done.enabled and done.next_scheduled_at is None
    assert done.last_scheduled_at == START and done.last_task_id == task.id
    assert restarted.triggers(routine.id) == [trigger]
    assert len(restarted.tasks.list()) == 1
    with pytest.raises(ConfigurationError, match="已完成"):
        restarted.enable(routine.id)


def test_fixed_interval_retains_bounded_backlog_without_poll_drift(store: RoutineStore) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=3)
    )
    late = START + timedelta(seconds=10, microseconds=100)
    first = store.tick(now=late, max_triggers=2)
    assert [item.scheduled_at for item in first] == [START, START + timedelta(seconds=3)]
    assert store.get(routine.id).next_scheduled_at == START + timedelta(seconds=6)
    restarted = reopen(store.database.path)
    second = restarted.tick(now=late, max_triggers=2)
    assert [item.scheduled_at for item in second] == [
        START + timedelta(seconds=6),
        START + timedelta(seconds=9),
    ]
    assert restarted.tick(now=late) == []
    [third] = restarted.tick(now=START + timedelta(seconds=12, microseconds=100))
    assert third.scheduled_at == START + timedelta(seconds=12)
    assert restarted.get(routine.id).next_scheduled_at == START + timedelta(seconds=15)
    assert len(restarted.tasks.list()) == 5


def test_global_tick_cap_selects_earliest_due_across_bots(store: RoutineStore) -> None:
    late = store.create(
        RoutineCreate(bot_id="writer", text="Late", scheduled_at=START + timedelta(seconds=2))
    )
    early = store.create(RoutineCreate(bot_id="reader", text="Early", scheduled_at=START))
    middle = store.create(
        RoutineCreate(bot_id="writer", text="Middle", scheduled_at=START + timedelta(seconds=1))
    )
    cutoff = START + timedelta(days=1)
    assert [item.routine_id for item in store.tick(now=cutoff, max_triggers=2)] == [
        early.id,
        middle.id,
    ]
    assert [item.routine_id for item in store.tick(now=cutoff, max_triggers=2)] == [late.id]


def test_default_tick_limit_leaves_large_backlog_for_later_polls(store: RoutineStore) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=1)
    )
    late = START + timedelta(days=1000)
    triggers = store.tick(now=late)
    assert len(triggers) == 100
    assert triggers[-1].scheduled_at == START + timedelta(seconds=99)
    assert store.get(routine.id).next_scheduled_at == START + timedelta(seconds=100)
    assert len(store.tasks.list()) == 100


def test_timezone_normalization_and_metadata_budget_snapshot(store: RoutineStore) -> None:
    local = START.astimezone(timezone(timedelta(hours=8)))
    request = RoutineCreate(
        bot_id="writer",
        text="Check",
        scheduled_at=local,
        budget=BudgetConfig(max_model_requests=7),
        metadata={"labels": ["initial"], "routine_id": "user value"},
    )
    routine = store.create(request, now=local)
    request.metadata["labels"].append("changed")
    assert request.budget is not None
    request.budget.max_model_requests = 99
    routine.definition.metadata["labels"].append("also changed")
    assert routine.created_at == START
    assert routine.definition.scheduled_at.tzinfo == UTC
    [trigger] = store.tick(now=local)
    task = store.tasks.get(trigger.task_id or "")
    assert task.budget.max_model_requests == 7
    assert task.metadata["routine_metadata"] == {
        "labels": ["initial"],
        "routine_id": "user value",
    }
    assert task.metadata["routine_id"] == routine.id
    assert task.metadata["scheduled_at"] == START.isoformat()


@pytest.mark.parametrize("interval", [0, -1, True, 0.5, "3"])
def test_invalid_intervals_are_rejected(interval: object) -> None:
    with pytest.raises(ValidationError):
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=interval)


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_invalid_tick_limits_are_rejected(store: RoutineStore, limit: int) -> None:
    with pytest.raises(ConfigurationError, match="max_triggers"):
        store.tick(now=START, max_triggers=limit)


def test_naive_times_and_unrepresentable_intervals_are_rejected(store: RoutineStore) -> None:
    naive = START.replace(tzinfo=None)
    with pytest.raises(ValidationError, match="时区"):
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=naive)
    request = RoutineCreate(bot_id="writer", text="Check", scheduled_at=START)
    with pytest.raises(ConfigurationError, match="时区"):
        store.create(request, now=naive)
    routine = store.create(request)
    with pytest.raises(ConfigurationError, match="时区"):
        store.tick(now=naive)
    with pytest.raises(ConfigurationError, match="时区"):
        store.disable(routine.id, now=naive)
    with pytest.raises(ConfigurationError, match="时间范围"):
        store.create(request.model_copy(update={"interval_seconds": 10**30}))
    assert store.tasks.list() == []


def test_unknown_bot_non_json_metadata_and_wrong_owner_are_rejected(store: RoutineStore) -> None:
    with pytest.raises(ConfigurationError, match="Bot 不存在"):
        store.create(RoutineCreate(bot_id="missing", text="Check", scheduled_at=START))
    with pytest.raises(ValueError):
        store.create(
            RoutineCreate(
                bot_id="writer", text="Check", scheduled_at=START, metadata={"bad": float("nan")}
            )
        )
    routine = store.create(RoutineCreate(bot_id="writer", text="Check", scheduled_at=START))
    with pytest.raises(ConfigurationError, match="不属于"):
        store.get(routine.id, bot_id="reader")
    with pytest.raises(ConfigurationError, match="不属于"):
        store.disable(routine.id, bot_id="reader")
    with pytest.raises(ConfigurationError, match="不属于"):
        store.triggers(routine.id, bot_id="reader")
    with pytest.raises(ConfigurationError, match="不存在"):
        store.get("missing")
    assert store.list(bot_id="reader") == []
    assert store.list(bot_id="writer", enabled=True) == [routine]


@pytest.mark.parametrize("interval", [None, 10])
def test_disable_preserves_unaccepted_schedule_and_enable_catches_up(
    store: RoutineStore, interval: int | None
) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=interval)
    )
    disabled = store.disable(routine.id, now=START)
    assert store.disable(routine.id, now=START + timedelta(seconds=1)) == disabled
    assert store.tick(now=START + timedelta(seconds=30)) == []
    assert store.get(routine.id).next_scheduled_at == START
    assert store.list(enabled=False) == [disabled]
    restarted = reopen(store.database.path)
    restarted.enable(routine.id, bot_id="writer")
    triggers = restarted.tick(now=START + timedelta(seconds=20))
    assert len(triggers) == (1 if interval is None else 3)
    assert triggers[0].scheduled_at == START


def test_initially_disabled_and_invalid_enabled_flag(store: RoutineStore) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, enabled=False)
    )
    assert store.tick(now=START) == []
    with pytest.raises(ConfigurationError, match="布尔值"):
        store.set_enabled(routine.id, "yes")  # type: ignore[arg-type]
    store.enable(routine.id)
    assert len(store.tick(now=START)) == 1


def test_disable_does_not_cancel_already_accepted_pending_trigger(
    store: RoutineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=10)
    )

    def fail_submit(request: TaskSubmission) -> TaskRecord:
        raise PersistenceError("Injected before public Task submit")

    monkeypatch.setattr(store.tasks, "submit", fail_submit)
    with pytest.raises(PersistenceError, match="Injected"):
        store.tick(now=START)
    [pending] = store.triggers(routine.id)
    assert pending.task_id is None and store.tasks.list() == []
    store.disable(routine.id)
    restarted = reopen(store.database.path)
    # Accepted work is retried even if the wall clock moves backwards.
    [trigger] = restarted.tick(now=START - timedelta(seconds=1))
    assert trigger.submission == pending.submission and trigger.task_id is not None
    assert not restarted.get(routine.id).enabled
    assert restarted.get(routine.id).next_scheduled_at == START + timedelta(seconds=10)
    assert restarted.tick(now=START + timedelta(days=1)) == []
    restarted.enable(routine.id)
    [following] = restarted.tick(now=START + timedelta(seconds=10))
    assert following.scheduled_at == START + timedelta(seconds=10)


def test_disable_racing_with_submit_is_preserved(
    store: RoutineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=1)
    )
    submit = store.tasks.submit

    def disable_then_submit(request: TaskSubmission) -> TaskRecord:
        store.disable(routine.id)
        return submit(request)

    monkeypatch.setattr(store.tasks, "submit", disable_then_submit)
    assert len(store.tick(now=START + timedelta(seconds=10))) == 1
    paused = store.get(routine.id)
    assert not paused.enabled and paused.next_scheduled_at == START + timedelta(seconds=1)
    assert len(store.tasks.list()) == 1


@pytest.mark.parametrize("point", ["before_submit", "after_submit"])
def test_process_crash_recovers_pending_without_loss_or_duplicate(
    store: RoutineStore, point: str
) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=60)
    )
    script = """
import os
import sys
from datetime import datetime
from pathlib import Path

from hiro_agent.bots import BotStore
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.routines import RoutineStore


class CrashTaskStore(TaskStore):
    def submit(self, request):
        if sys.argv[3] == "before_submit":
            os._exit(71)
        super().submit(request)
        os._exit(72)


database = KernelDatabase(Path(sys.argv[1]))
bots = BotStore(database)
routines = RoutineStore(database, bots, CrashTaskStore(database, bots))
routines.tick(now=datetime.fromisoformat(sys.argv[2]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.database.path), START.isoformat(), point],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == (71 if point == "before_submit" else 72), result.stderr
    restarted = reopen(store.database.path)
    [pending] = restarted.triggers(routine.id)
    assert pending.task_id is None
    assert restarted.get(routine.id).next_scheduled_at == START
    before = restarted.tasks.list()
    assert len(before) == (0 if point == "before_submit" else 1)
    [trigger] = restarted.tick(now=START)
    assert trigger.submission == pending.submission
    if before:
        assert trigger.task_id == before[0].id
    assert len(restarted.tasks.list()) == 1
    assert restarted.get(routine.id).next_scheduled_at == START + timedelta(seconds=60)
    assert reopen(store.database.path).tick(now=START) == []


def test_acknowledgment_sql_failure_rolls_back_and_retries_existing_task(
    store: RoutineStore,
) -> None:
    routine = store.create(RoutineCreate(bot_id="writer", text="Check", scheduled_at=START))
    with store.database.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER reject_routine_advance BEFORE UPDATE ON routines "
            "BEGIN SELECT RAISE(ABORT, 'injected advance failure'); END"
        )
    with pytest.raises(PersistenceError):
        store.tick(now=START)
    [task] = store.tasks.list()
    assert store.get(routine.id).next_scheduled_at == START
    assert store.triggers(routine.id)[0].task_id is None
    with store.database.transaction() as connection:
        connection.execute("DROP TRIGGER reject_routine_advance")
    [trigger] = reopen(store.database.path).tick(now=START)
    assert trigger.task_id == task.id
    assert len(store.tasks.list()) == 1
    assert store.get(routine.id).next_scheduled_at is None


def test_concurrent_connections_submit_same_slot_and_advance_exactly_once(
    store: RoutineStore,
) -> None:
    routine = store.create(
        RoutineCreate(bot_id="writer", text="Check", scheduled_at=START, interval_seconds=10)
    )
    barrier = Barrier(6)

    class RacingTasks(TaskStore):
        def submit(self, request: TaskSubmission) -> TaskRecord:
            barrier.wait(timeout=10)
            return super().submit(request)

    def compete(_: int) -> list[str | None]:
        database = KernelDatabase(store.database.path)
        bots = BotStore(database)
        local = RoutineStore(database, bots, RacingTasks(database, bots))
        return [trigger.task_id for trigger in local.tick(now=START, max_triggers=1)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(compete, range(6)))
    returned = [task_id for ids in results for task_id in ids]
    [task] = store.tasks.list()
    assert returned == [task.id]
    assert store.get(routine.id).next_scheduled_at == START + timedelta(seconds=10)
    assert len(store.triggers(routine.id)) == 1
    assert store.tick(now=START) == []
    assert len(store.tick(now=START + timedelta(seconds=10))) == 1
    assert len(store.tasks.list()) == 2
