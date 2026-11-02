"""Chain accounting shares the Task transaction; no observer or callback ledger."""

from __future__ import annotations

import sqlite3

from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import RunMetrics
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import ChainRecord, TaskRecord
from hiro_agent.kernel_db import KernelDatabase


class ChainStore:
    def __init__(self, database: KernelDatabase) -> None:
        database.migrate(
            "task_chains",
            1,
            (
                "CREATE TABLE task_chains (id TEXT PRIMARY KEY, record_json TEXT NOT NULL)",
                "CREATE TABLE chain_members (task_id TEXT PRIMARY KEY "
                "REFERENCES durable_tasks(id), "
                "chain_id TEXT NOT NULL REFERENCES task_chains(id))",
            ),
        )

    @staticmethod
    def get(connection: sqlite3.Connection, chain_id: str) -> ChainRecord:
        row = connection.execute(
            "SELECT record_json FROM task_chains WHERE id=?", (chain_id,)
        ).fetchone()
        if row is None:
            raise ConfigurationError("Chain does not exist")
        return ChainRecord.model_validate_json(row[0])

    @staticmethod
    def save(connection: sqlite3.Connection, chain: ChainRecord) -> None:
        connection.execute(
            "UPDATE task_chains SET record_json=? WHERE id=?", (chain.model_dump_json(), chain.id)
        )

    @classmethod
    def join(
        cls,
        connection: sqlite3.Connection,
        task: TaskRecord,
        *,
        budget: BudgetConfig | None = None,
        max_hops: int | None = None,
    ) -> None:
        assert task.chain_id is not None
        row = connection.execute(
            "SELECT record_json FROM task_chains WHERE id=?", (task.chain_id,)
        ).fetchone()
        if row is None:
            chain = ChainRecord(
                id=task.chain_id, budget=budget or task.budget, max_hops=max_hops or 8
            )
            connection.execute(
                "INSERT INTO task_chains VALUES (?, ?)", (chain.id, chain.model_dump_json())
            )
        else:
            chain = ChainRecord.model_validate_json(row[0])
            if budget is not None or max_hops is not None:
                raise ConfigurationError("Only new root chains may set limits")
        inserted = connection.execute(
            "INSERT OR IGNORE INTO chain_members VALUES (?, ?)", (task.id, chain.id)
        ).rowcount
        if inserted:
            cls.charge(connection, task, task.cumulative_metrics)

    @classmethod
    def charge(cls, connection: sqlite3.Connection, task: TaskRecord, delta: RunMetrics) -> None:
        assert task.chain_id is not None
        chain = cls.get(connection, task.chain_id)
        chain.cumulative_metrics = RunMetrics.model_validate(
            {
                name: getattr(chain.cumulative_metrics, name) + value
                for name, value in delta.model_dump().items()
            }
        )
        cls.save(connection, chain)

    @staticmethod
    def exhausted(chain: ChainRecord) -> str:
        if chain.cumulative_metrics.model_requests >= chain.budget.max_model_requests:
            return "Chain cumulative model request budget exhausted"
        if chain.cumulative_metrics.tool_calls >= chain.budget.max_tool_calls:
            return "Chain cumulative tool call budget exhausted"
        if chain.cumulative_metrics.elapsed_ms >= chain.budget.timeout_seconds * 1000:
            return "Chain cumulative active time budget exhausted"
        return ""
