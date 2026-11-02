"""SQLite authority for durable kernel state; JSONL remains an independent journal."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from hiro_agent.errors import PersistenceError

_maintenance_held: ContextVar[tuple[tuple[Path, bool], ...]] = ContextVar(
    "kernel_maintenance_held", default=()
)


@contextmanager
def maintenance_lock(
    directory: Path, *, exclusive: bool = False, allow_pending: bool = False
) -> Iterator[None]:
    """Coordinate short maintenance with live workers; nested shared use is safe.

    Exclusive acquisition never waits for running tasks. A leftover staging
    manifest blocks writers until the user retries the interrupted cleanup.
    """
    directory = directory.absolute()
    held = _maintenance_held.get()
    for path, owns_exclusive in held:
        if path == directory:
            if exclusive and not owns_exclusive:
                raise PersistenceError("历史清理失败：内核正在使用，请稍后重试。")
            yield
            return
    descriptor: int | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise PersistenceError("历史清理锁目录不安全。")
        descriptor = os.open(
            directory / "maintenance.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PersistenceError("历史清理锁不是普通文件。")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH)
        except BlockingIOError:
            raise PersistenceError("历史清理失败：内核正在执行任务，请待任务结束后重试。") from None
        if (
            not exclusive
            and not allow_pending
            and (
                (directory / "history-cleanup").exists()
                or (directory / "history-cleanup").is_symlink()
            )
        ):
            raise PersistenceError("上次历史清理尚未完成，请重试清理命令以恢复。")
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PersistenceError("历史清理锁无法访问。") from exc
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    token = _maintenance_held.set((*held, (directory, exclusive)))
    try:
        yield
    finally:
        _maintenance_held.reset(token)
        if descriptor is not None:
            os.close(descriptor)


class KernelDatabase:
    """Short, synchronous transactions. Never hold one across an await or JSONL I/O."""

    def __init__(self, path: Path) -> None:
        try:
            self.path = path.resolve()
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistenceError("内核数据库目录无法创建或访问") from exc
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS kernel_migrations ("
                "namespace TEXT NOT NULL, version INTEGER NOT NULL, checksum TEXT NOT NULL, "
                "PRIMARY KEY(namespace, version))"
            )

    @classmethod
    def for_workspace(cls, workspace: Path) -> KernelDatabase:
        return cls(workspace / ".hiro-agent" / "kernel.sqlite3")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA busy_timeout=5000")
                yield connection
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise PersistenceError("内核数据库读取或提交失败") from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with maintenance_lock(self.path.parent), self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def migrate(self, namespace: str, version: int, statements: Sequence[str]) -> None:
        """Apply one immutable, consecutive migration owned by its module."""
        if not namespace or version < 1:
            raise ValueError("Migration namespace and positive version are required")
        checksum = hashlib.sha256("\n".join(statements).encode()).hexdigest()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT checksum FROM kernel_migrations WHERE namespace=? AND version=?",
                (namespace, version),
            ).fetchone()
            if existing is not None:
                if existing["checksum"] != checksum:
                    raise PersistenceError("已应用的数据库迁移内容发生变化")
                return
            current = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM kernel_migrations WHERE namespace=?",
                (namespace,),
            ).fetchone()[0]
            if version != current + 1:
                raise PersistenceError("数据库迁移必须按版本顺序应用")
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO kernel_migrations(namespace, version, checksum) VALUES (?, ?, ?)",
                (namespace, version, checksum),
            )
