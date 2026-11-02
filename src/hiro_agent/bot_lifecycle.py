"""Atomic Bot history removal with locked journals and recoverable file staging.

The SQL commit is the decision point. Before it, staged journals are restored;
after it, they are removed. A process crash leaves ``history-cleanup`` in place,
which blocks new writers until this API is retried. No project or skill files
are removed, and stable session lock inodes are retained.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from hiro_agent.contracts import DelegationLink, SessionEvent
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import TERMINAL_TASK_STATES, TaskRecord, TaskState
from hiro_agent.kernel_db import KernelDatabase, maintenance_lock
from hiro_agent.routines import RoutineRecord, RoutineStore
from hiro_agent.session import _SESSION_ID, _read_events, read_session_snapshot


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _worker_guard(directory: Path) -> Iterator[None]:
    # Retain a shared worker lock when offline so a legacy worker cannot start
    # halfway through maintenance. Compatible workers retain their process lock
    # until they can acquire shared maintenance access during shutdown.
    descriptor = _regular(directory / "worker.lock", os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            compatible = False
            try:
                protocol = _regular(directory / "maintenance-worker.lock")
            except FileNotFoundError:
                pass
            else:
                try:
                    try:
                        fcntl.flock(protocol, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        compatible = True
                finally:
                    os.close(protocol)
            if not compatible:
                raise ConfigurationError(
                    "历史清理失败：正在运行的旧版 worker 不支持维护锁，请先重启 worker。"
                ) from None
        yield
    finally:
        os.close(descriptor)


def _regular(path: Path, flags: int = os.O_RDONLY) -> int:
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise PersistenceError("历史清理失败：日志或锁文件不是普通文件。")
    return descriptor


def _recover_staging(database: KernelDatabase) -> None:
    """Recover only our validated manifest; never overwrite a newer journal."""
    directory = database.path.parent
    staging = directory / "history-cleanup"
    if not staging.exists() and not staging.is_symlink():
        return
    if staging.is_symlink() or not staging.is_dir():
        raise PersistenceError("历史清理恢复失败：暂存目录不安全。")
    manifest = staging / "manifest.json"
    if not manifest.exists():
        # A crash before publishing the manifest cannot have moved any journals.
        entries = set(path.name for path in staging.iterdir())
        if entries - {"manifest.tmp"}:
            raise PersistenceError("历史清理恢复失败：暂存清单缺失。")
        if entries:
            (staging / "manifest.tmp").unlink()
        staging.rmdir()
        _sync(directory)
        return
    descriptor = _regular(manifest)
    try:
        with os.fdopen(descriptor) as stream:
            record = json.load(stream)
    except (ValueError, UnicodeError) as exc:
        raise PersistenceError("历史清理恢复失败：暂存清单无效。") from exc
    if not isinstance(record, dict):
        raise PersistenceError("历史清理恢复失败：暂存清单无效。")
    names = record.get("journals")
    if (
        not isinstance(record.get("id"), str)
        or not isinstance(names, list)
        or any(
            not isinstance(name, str)
            or not name.endswith(".jsonl")
            or not _SESSION_ID.fullmatch(name[:-6])
            for name in names
        )
        or len(names) != len(set(names))
        or set(path.name for path in staging.iterdir()) - {*names, "manifest.json"}
    ):
        raise PersistenceError("历史清理恢复失败：暂存清单无效。")
    with database.connect() as connection:
        committed = (
            connection.execute(
                "SELECT 1 FROM history_cleanup_commits WHERE id=?", (record["id"],)
            ).fetchone()
            is not None
        )
    sessions = directory / "sessions"
    if sessions.is_symlink():
        raise PersistenceError("历史清理恢复失败：会话目录不安全。")
    with ExitStack() as stack:
        for name in names:
            descriptor = _regular(sessions / (name[:-6] + ".lock"))
            stack.callback(os.close, descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PersistenceError("历史清理恢复失败：会话正在使用，请关闭后重试。") from None
        for name in names:
            saved = staging / name
            if not saved.exists() and not saved.is_symlink():
                continue
            descriptor = _regular(saved)
            os.close(descriptor)
            if committed:
                saved.unlink()
            else:
                destination = sessions / name
                if destination.exists() or destination.is_symlink():
                    raise PersistenceError("历史清理恢复失败：会话已被替换，不能覆盖。")
                saved.rename(destination)
        if names:
            _sync(sessions)
        manifest.unlink()
        _sync(staging)
        staging.rmdir()
        _sync(directory)
    with database.transaction() as connection:
        connection.execute("DELETE FROM history_cleanup_commits WHERE id=?", (record["id"],))


class _Journals:
    """Preflight a complete family union before the first filesystem mutation."""

    def __init__(self, workspace: Path, bot_id: str, task_sessions: set[str], stack: ExitStack):
        self.directory = workspace / ".hiro-agent" / "sessions"
        self.stack = stack
        self.opened: dict[Path, int] = {}
        self.journals: list[Path] = []
        self.events: dict[str, list[SessionEvent] | None] = {}
        if not self.directory.exists() and not self.directory.is_symlink():
            return
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise PersistenceError("历史清理失败：会话目录不安全。")
        roots = set(task_sessions)
        for path in sorted(self.directory.glob("*.jsonl")):
            if not _SESSION_ID.fullmatch(path.stem):
                continue
            snapshot = read_session_snapshot(workspace, path.stem)
            bindings = {
                event.data.get("bot_id") for event in snapshot if event.type == "session.bot"
            }
            if bot_id in bindings:
                if bindings != {bot_id}:
                    raise PersistenceError("历史清理失败：会话包含冲突的 Bot 绑定。")
                roots.add(path.stem)
        for session_id in sorted(roots):
            events = self.read(session_id)
            if events is None:
                continue
            if any(event.type == "subagent.created" for event in events):
                raise PersistenceError("历史清理失败：目标会话被其他主会话引用。")
            bindings = {event.data.get("bot_id") for event in events if event.type == "session.bot"}
            if bindings - {bot_id}:
                raise PersistenceError("历史清理失败：任务会话绑定到了其他 Bot。")
            self.children(events, bot_id, set(roots), set())
        for path, descriptor in self.opened.items():
            if not os.path.samestat(os.fstat(descriptor), path.stat(follow_symlinks=False)):
                raise PersistenceError("历史清理失败：日志或锁文件已变化，请重试。")

    def open(self, path: Path) -> int:
        descriptor = _regular(path)
        self.stack.callback(os.close, descriptor)
        self.opened[path] = descriptor
        return descriptor

    def read(self, session_id: str) -> list[SessionEvent] | None:
        if not _SESSION_ID.fullmatch(session_id):
            raise PersistenceError("历史清理失败：Session ID 不合法。")
        if session_id in self.events:
            return self.events[session_id]
        path = self.directory / f"{session_id}.jsonl"
        try:
            descriptor = self.open(path.with_suffix(".lock"))
        except FileNotFoundError:
            if path.exists() or path.is_symlink():
                raise PersistenceError("历史清理失败：会话锁缺失。") from None
            self.events[session_id] = None
            return None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PersistenceError(
                "历史清理失败：会话或关联子会话正在使用，请关闭后重试。"
            ) from None
        try:
            descriptor = self.open(path)
        except FileNotFoundError:
            self.events[session_id] = None
            return None
        events = _read_events(descriptor, session_id, repair=False)
        self.events[session_id] = events
        self.journals.append(path)
        return events

    def children(
        self, events: list[SessionEvent], bot_id: str, roots: set[str], seen: set[str]
    ) -> None:
        links: dict[str, DelegationLink] = {}
        for event in events:
            if event.type != "delegation.started":
                continue
            link = DelegationLink.model_validate(event.data)
            if links.setdefault(link.child_session_id, link) != link:
                raise PersistenceError("历史清理失败：子会话关联冲突。")
        for child_id, link in sorted(links.items()):
            if child_id in seen or child_id in roots:
                raise PersistenceError("历史清理失败：子会话关联冲突。")
            seen.add(child_id)
            children = self.read(child_id)
            if children is None:
                continue
            reciprocal = [event for event in children if event.type == "subagent.created"]
            if reciprocal:
                if (
                    len(reciprocal) != 1
                    or DelegationLink.model_validate(reciprocal[0].data) != link
                ):
                    raise PersistenceError("历史清理失败：父子会话关联不一致。")
            elif any(event.type != "session.created" for event in children):
                raise PersistenceError("历史清理失败：子会话缺少父任务关联。")
            if any(
                event.type == "session.bot" and event.data.get("bot_id") != bot_id
                for event in children
            ):
                raise PersistenceError("历史清理失败：子会话属于其他 Bot。")
            self.children(children, bot_id, roots, seen)


def _delete_records(
    connection: sqlite3.Connection, bot_id: str, delete_bot: bool
) -> dict[str, int]:
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    counts: dict[str, int] = {}

    def remove(table: str, condition: str, values: tuple[Any, ...] = ()) -> None:
        if table in tables:
            counts[table] = (
                counts.get(table, 0)
                + connection.execute(f"DELETE FROM {table} WHERE {condition}", values).rowcount
            )

    own = "SELECT id FROM durable_tasks WHERE bot_id=?"
    own_chains = []
    if "chain_members" in tables:
        own_chains = [
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT chain_id FROM chain_members WHERE task_id IN ({own})", (bot_id,)
            )
        ]
    if "durable_tasks" in tables:
        if "handoff_messages" in tables:
            condition = (
                f"source_bot_id=? OR target_bot_id=? OR source_task_id IN ({own}) "
                f"OR target_task_id IN ({own})"
            )
            values: tuple[str, ...] = (bot_id,) * 4
            if "handoff_results" in tables:
                condition += (
                    " OR id IN (SELECT message_id FROM handoff_results "
                    f"WHERE delivery_task_id IN ({own}))"
                )
                values += (bot_id,)
            message_ids = [
                row[0]
                for row in connection.execute(
                    f"SELECT id FROM handoff_messages WHERE {condition}", values
                )
            ]
            for message_id in message_ids:
                remove("handoff_results", "message_id=?", (message_id,))
                remove("handoff_messages", "id=?", (message_id,))
        for table in ("interactions", "memory_consumptions", "bot_memories"):
            remove(table, f"bot_id=? OR task_id IN ({own})", (bot_id, bot_id))
        for table in ("durable_task_resolutions", "durable_runs", "chain_members"):
            remove(table, f"task_id IN ({own})", (bot_id,))
    else:
        for table in ("interactions", "memory_consumptions", "bot_memories"):
            remove(table, "bot_id=?", (bot_id,))
    if "routines" in tables:
        if not delete_bot:
            for row in connection.execute(
                "SELECT id, record_json FROM routines WHERE bot_id=?", (bot_id,)
            ):
                routine = RoutineRecord.model_validate_json(row["record_json"])
                pending = connection.execute(
                    "SELECT 1 FROM routine_triggers WHERE routine_id=? AND task_id IS NULL",
                    (routine.id,),
                ).fetchone()
                next_time = routine.next_scheduled_at
                enabled = routine.enabled
                if pending is not None:
                    # An accepted slot is discarded once, including submit/ack crashes.
                    interval = routine.definition.interval_seconds
                    try:
                        next_time = (
                            next_time + timedelta(seconds=interval)
                            if (next_time is not None and interval is not None)
                            else None
                        )
                    except OverflowError:
                        next_time = None
                    enabled = enabled and next_time is not None
                routine = routine.model_copy(
                    update={
                        "last_task_id": None,
                        "last_scheduled_at": None,
                        "next_scheduled_at": next_time,
                        "enabled": enabled,
                    }
                )
                RoutineStore._save(connection, routine)
        remove(
            "routine_triggers", "routine_id IN (SELECT id FROM routines WHERE bot_id=?)", (bot_id,)
        )
        if delete_bot:
            remove("routines", "bot_id=?", (bot_id,))
    remove("durable_tasks", "bot_id=?", (bot_id,))
    for chain_id in own_chains:
        remove(
            "task_chains", "id=? AND id NOT IN (SELECT chain_id FROM chain_members)", (chain_id,)
        )
    if delete_bot:
        for table in ("skill_drafts", "skill_settings", "bots"):
            remove(table, "bot_id=?", (bot_id,))
    counts["tasks"] = counts.get("durable_tasks", 0)
    counts["memories"] = counts.get("bot_memories", 0)
    return counts


def _preflight(connection: sqlite3.Connection, bot_id: str) -> set[str]:
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "durable_tasks" not in tables:
        return set()
    tasks = [
        TaskRecord.model_validate_json(row[0])
        for row in connection.execute("SELECT record_json FROM durable_tasks")
    ]
    own = {task.id for task in tasks if task.bot_id == bot_id}
    chains = {task.chain_id for task in tasks if task.id in own}
    related = set(own)
    if "handoff_messages" in tables:
        for row in connection.execute(
            "SELECT m.*, r.delivery_task_id FROM handoff_messages m "
            "LEFT JOIN handoff_results r ON r.message_id=m.id"
        ):
            ids = {row["source_task_id"], row["target_task_id"], row["delivery_task_id"]} - {None}
            if row["source_bot_id"] == bot_id or row["target_bot_id"] == bot_id or ids & own:
                related.update(ids)
    for task in tasks:
        if task.id in own:
            if task.active_run_id is not None or task.state in (
                TaskState.RUNNING,
                TaskState.NEEDS_RECONCILIATION,
            ):
                raise ConfigurationError(
                    "历史清理失败：Bot 有正在执行或待核对的任务，请先结束或核对任务。"
                )
        elif (
            task.id in related or task.chain_id in chains or task.parent_task_id in own
        ) and task.state not in TERMINAL_TASK_STATES:
            raise ConfigurationError(
                "历史清理失败：关联的其他 Bot 任务尚未结束，请先结束关联任务。"
            )
    if (
        "durable_runs" in tables
        and connection.execute(
            "SELECT 1 FROM durable_runs WHERE state IN ('claimed', 'started') "
            "AND task_id IN (SELECT id FROM durable_tasks WHERE bot_id=?)",
            (bot_id,),
        ).fetchone()
    ):
        raise ConfigurationError("历史清理失败：Bot 的 Run 尚未结束。")
    return {task.session_id for task in tasks if task.id in own}


def clear_bot_history(workspace: Path, bot_id: str, *, delete_bot: bool = False) -> dict[str, Any]:
    """Clear all Bot context, or also remove its identity and owned definitions.

    Queued/waiting tasks may be discarded. Running or uncertain work, open
    journals, and unfinished peer tasks prevent *all* deletion. Repeating a
    clear is harmless; deleting an already absent Bot returns zero counts.
    """
    workspace = Path(workspace).resolve(strict=True)
    directory = workspace / ".hiro-agent"
    if not bot_id or directory.is_symlink() or (directory / "kernel.sqlite3").is_symlink():
        raise ConfigurationError("历史清理失败：Bot ID 或状态目录不合法。")
    if not (directory / "kernel.sqlite3").exists():
        raise ConfigurationError("Bot 不存在。")
    with maintenance_lock(directory, exclusive=True), _worker_guard(directory):
        database = KernelDatabase.for_workspace(workspace)
        database.migrate(
            "bot_lifecycle", 1, ("CREATE TABLE history_cleanup_commits (id TEXT PRIMARY KEY)",)
        )
        _recover_staging(database)
        with ExitStack() as stack:
            staging = directory / "history-cleanup"
            try:
                with database.transaction() as connection:
                    if (
                        connection.execute(
                            "SELECT 1 FROM bots WHERE bot_id=?", (bot_id,)
                        ).fetchone()
                        is None
                    ):
                        if delete_bot:
                            return {
                                "bot_id": bot_id,
                                "deleted": True,
                                "counts": {"sessions": 0, "tasks": 0, "memories": 0},
                            }
                        raise ConfigurationError("Bot 不存在。")
                    task_sessions = _preflight(connection, bot_id)
                    journals = _Journals(workspace, bot_id, task_sessions, stack)
                    operation_id = uuid4().hex
                    staging.mkdir(mode=0o700)
                    manifest = {
                        "id": operation_id,
                        "journals": [path.name for path in journals.journals],
                    }
                    with (staging / "manifest.tmp").open("x") as stream:
                        json.dump(manifest, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    (staging / "manifest.tmp").rename(staging / "manifest.json")
                    _sync(staging)
                    _sync(directory)
                    for path in journals.journals:
                        path.rename(staging / path.name)
                    if journals.journals:
                        _sync(journals.directory)
                    _sync(staging)
                    counts = _delete_records(connection, bot_id, delete_bot)
                    counts["sessions"] = len(journals.journals)
                    connection.execute(
                        "INSERT INTO history_cleanup_commits VALUES (?)", (operation_id,)
                    )
            except BaseException:
                # Release our locks before recovery reacquires the same inodes.
                stack.close()
                _recover_staging(database)
                raise
        _recover_staging(database)
        return {"bot_id": bot_id, "deleted": delete_bot, "counts": counts}
