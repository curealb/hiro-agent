"""SQLite authority for durable kernel state; JSONL remains an independent journal."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from hiro_agent.errors import PersistenceError


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
        with self.connect() as connection:
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
