"""Durable Bot inboxes and once-only terminal result delivery through Task.submit."""

from __future__ import annotations

import builtins
from datetime import datetime
from uuid import uuid4

from pydantic import Field

from hiro_agent.bots import BotStore
from hiro_agent.chains import ChainStore
from hiro_agent.contracts import DataModel
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import (
    TERMINAL_TASK_STATES,
    TaskState,
    TaskSubmission,
    utc_now,
)
from hiro_agent.kernel_db import KernelDatabase


class HandoffMessage(DataModel):
    id: str
    source_bot_id: str
    source_task_id: str
    target_bot_id: str
    text: str
    idempotency_key: str
    chain_id: str
    hop: int
    target_task_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class HandoffResult(DataModel):
    message_id: str
    source_bot_id: str
    source_task_id: str
    target_bot_id: str
    target_task_id: str
    chain_id: str
    state: TaskState
    output: str
    reason: str
    delivery_task_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class HandoffStore:
    def __init__(self, database: KernelDatabase, bots: BotStore, tasks: TaskStore) -> None:
        self.database = database
        self.bots = bots
        self.tasks = tasks
        database.migrate(
            "handoffs",
            1,
            (
                "CREATE TABLE handoff_messages (id TEXT PRIMARY KEY, source_bot_id TEXT NOT NULL, "
                "source_task_id TEXT NOT NULL REFERENCES durable_tasks(id), "
                "target_bot_id TEXT NOT NULL REFERENCES bots(bot_id), "
                "idempotency_key TEXT NOT NULL, "
                "target_task_id TEXT REFERENCES durable_tasks(id), record_json TEXT NOT NULL, "
                "UNIQUE(source_task_id, idempotency_key))",
                "CREATE TABLE handoff_results (message_id TEXT PRIMARY KEY "
                "REFERENCES handoff_messages(id), source_bot_id TEXT NOT NULL, "
                "delivery_task_id TEXT REFERENCES durable_tasks(id), record_json TEXT NOT NULL)",
            ),
        )

    def send(
        self,
        bot_id: str,
        source_task_id: str,
        target_bot_id: str,
        text: str,
        *,
        idempotency_key: str,
    ) -> HandoffMessage:
        if not text.strip() or not idempotency_key.strip():
            raise ConfigurationError("Handoff text and idempotency key must not be empty")
        if len(text) > 32000 or len(idempotency_key) > 200:
            raise ConfigurationError("Handoff text or idempotency key is too long")
        self.bots.get(target_bot_id)
        with self.database.transaction() as connection:
            # BEGIN IMMEDIATE prevents a concurrent cancellation or permission
            # update from committing between this validation and acceptance.
            source = self.tasks.get(source_task_id, bot_id=bot_id)
            if not self.bots.authority(bot_id).allows("send_handoff"):
                raise ConfigurationError("Bot is not allowed to send handoffs")
            row = connection.execute(
                "SELECT record_json FROM handoff_messages "
                "WHERE source_task_id=? AND idempotency_key=?",
                (source.id, idempotency_key),
            ).fetchone()
            if row is not None:
                message = HandoffMessage.model_validate_json(row[0])
                if message.target_bot_id != target_bot_id or message.text != text:
                    raise ConfigurationError("Handoff idempotency key has a different payload")
                return message
            if source.cancel_requested or source.state in (
                TaskState.FAILED,
                TaskState.CANCELLED,
                TaskState.NEEDS_RECONCILIATION,
            ):
                raise ConfigurationError("Stopped or uncertain Tasks cannot send new handoffs")
            assert source.chain_id is not None
            chain = ChainStore.get(connection, source.chain_id)
            if chain.hops_used >= chain.max_hops:
                raise ConfigurationError("Chain handoff hop limit exhausted")
            chain.hops_used += 1
            ChainStore.save(connection, chain)
            message = HandoffMessage(
                id=uuid4().hex,
                source_bot_id=bot_id,
                source_task_id=source.id,
                target_bot_id=target_bot_id,
                text=text,
                idempotency_key=idempotency_key,
                chain_id=chain.id,
                hop=chain.hops_used,
            )
            connection.execute(
                "INSERT INTO handoff_messages VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    message.id,
                    bot_id,
                    source.id,
                    target_bot_id,
                    idempotency_key,
                    message.model_dump_json(),
                ),
            )
        return message

    def get(self, bot_id: str, message_id: str) -> HandoffMessage:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM handoff_messages WHERE id=?", (message_id,)
            ).fetchone()
        if row is None:
            raise ConfigurationError("Handoff does not exist")
        message = HandoffMessage.model_validate_json(row[0])
        if bot_id not in (message.source_bot_id, message.target_bot_id):
            raise ConfigurationError("Handoff does not belong to this Bot")
        return message

    def list(self, bot_id: str, *, outgoing: bool = False) -> list[HandoffMessage]:
        self.bots.get(bot_id)
        column = "source_bot_id" if outgoing else "target_bot_id"
        with self.database.connect() as connection:
            rows = connection.execute(
                f"SELECT record_json FROM handoff_messages WHERE {column}=? ORDER BY rowid",
                (bot_id,),
            ).fetchall()
        return [HandoffMessage.model_validate_json(row[0]) for row in rows]

    def result(self, bot_id: str, message_id: str) -> HandoffResult | None:
        message = self.get(bot_id, message_id)
        if message.source_bot_id != bot_id:
            raise ConfigurationError("Only the sender can read returned results")
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM handoff_results WHERE message_id=?", (message_id,)
            ).fetchone()
        return HandoffResult.model_validate_json(row[0]) if row is not None else None

    def results(self, bot_id: str) -> builtins.list[HandoffResult]:
        self.bots.get(bot_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM handoff_results WHERE source_bot_id=? ORDER BY rowid",
                (bot_id,),
            ).fetchall()
        return [HandoffResult.model_validate_json(row[0]) for row in rows]

    def _deliver_message(self, message: HandoffMessage) -> None:
        task = self.tasks.submit(
            TaskSubmission(
                bot_id=message.target_bot_id,
                text=message.text,
                idempotency_key=f"handoff:{message.id}",
                source="handoff",
                chain_id=message.chain_id,
                parent_task_id=message.source_task_id,
                metadata={
                    "message_id": message.id,
                    "source_bot_id": message.source_bot_id,
                    "source_task_id": message.source_task_id,
                },
            )
        )
        message.target_task_id = task.id
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE handoff_messages SET target_task_id=?, record_json=? "
                "WHERE id=? AND target_task_id IS NULL",
                (task.id, message.model_dump_json(), message.id),
            )

    def _capture_result(self, message: HandoffMessage) -> None:
        assert message.target_task_id is not None
        task = self.tasks.get(message.target_task_id, bot_id=message.target_bot_id)
        # Unknown effects are not a terminal result: manual reconciliation must
        # decide the outcome before another Bot receives a claimed completion.
        if task.state not in TERMINAL_TASK_STATES:
            return
        result = HandoffResult(
            message_id=message.id,
            source_bot_id=message.source_bot_id,
            source_task_id=message.source_task_id,
            target_bot_id=message.target_bot_id,
            target_task_id=task.id,
            chain_id=message.chain_id,
            state=task.state,
            output=task.output,
            reason=task.reason,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO handoff_results VALUES (?, ?, NULL, ?)",
                (message.id, message.source_bot_id, result.model_dump_json()),
            )

    def _deliver_result(self, result: HandoffResult) -> None:
        # Stable input excludes the delivery receipt, which is saved afterwards.
        text = (
            "A Bot handoff result has returned. Read it with read_handoff using message_id "
            f"{result.message_id}. Original source Task: {result.source_task_id}; "
            f"target Task: {result.target_task_id}; status: {result.state}.\n"
            f"Result: {result.output}\nReason: {result.reason}"
        )
        task = self.tasks.submit(
            TaskSubmission(
                bot_id=result.source_bot_id,
                text=text,
                idempotency_key=f"handoff-result:{result.message_id}",
                source="handoff_result",
                chain_id=result.chain_id,
                parent_task_id=result.target_task_id,
                metadata={
                    "message_id": result.message_id,
                    "source_task_id": result.source_task_id,
                    "target_task_id": result.target_task_id,
                },
            )
        )
        result.delivery_task_id = task.id
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE handoff_results SET delivery_task_id=?, record_json=? "
                "WHERE message_id=? AND delivery_task_id IS NULL",
                (task.id, result.model_dump_json(), result.message_id),
            )

    def sync(self) -> None:
        """Replay durable intents; Task idempotency keys close both submit/mark gaps."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM handoff_messages ORDER BY rowid"
            ).fetchall()
        for row in rows:
            message = HandoffMessage.model_validate_json(row[0])
            if message.target_task_id is None:
                self._deliver_message(message)
            self._capture_result(message)
        with self.database.connect() as connection:
            pending = connection.execute(
                "SELECT record_json FROM handoff_results WHERE delivery_task_id IS NULL "
                "ORDER BY rowid"
            ).fetchall()
        for row in pending:
            self._deliver_result(HandoffResult.model_validate_json(row[0]))
