"""Durable questions, approvals and task-bound, once-only authorization."""

from __future__ import annotations

import builtins
import sqlite3
from uuid import uuid4

from pydantic import ValidationError

from hiro_agent.bots import BotStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import (
    InteractionRecord,
    InteractionReply,
    InteractionRequest,
    InteractionState,
    OperationIntent,
    canonical_arguments,
    utc_now,
)
from hiro_agent.kernel_db import KernelDatabase


class InteractionStore:
    """Own only interaction state; the coordinator verifies Task ownership and wakes it."""

    def __init__(self, database: KernelDatabase, bots: BotStore) -> None:
        if database.path != bots.database.path:
            raise ConfigurationError("交互与 Bot 必须使用同一个内核数据库")
        self.database = database
        self.bots = bots
        database.migrate(
            "interactions",
            1,
            (
                "CREATE TABLE interactions ("
                "id TEXT PRIMARY KEY, task_id TEXT NOT NULL, bot_id TEXT NOT NULL, "
                "idempotency_key TEXT NOT NULL, request_json TEXT NOT NULL, "
                "state TEXT NOT NULL "
                "CHECK(state IN ('pending','answered','consumed','cancelled')), "
                "reply_json TEXT, consumed_by_run_id TEXT, created_at TEXT NOT NULL, "
                "answered_at TEXT, consumed_at TEXT, UNIQUE(task_id, idempotency_key), "
                "CHECK((reply_json IS NULL) = (answered_at IS NULL)), "
                "CHECK((state='pending' AND reply_json IS NULL AND consumed_at IS NULL "
                "AND consumed_by_run_id IS NULL) OR "
                "(state='answered' AND reply_json IS NOT NULL AND consumed_at IS NULL "
                "AND consumed_by_run_id IS NULL) OR "
                "(state='consumed' AND reply_json IS NOT NULL AND consumed_at IS NOT NULL "
                "AND consumed_by_run_id IS NOT NULL) OR "
                "(state='cancelled' AND consumed_at IS NULL AND consumed_by_run_id IS NULL)))",
                "CREATE INDEX interactions_ready ON interactions(state, created_at, id)",
                "CREATE INDEX interactions_owner ON interactions(bot_id, task_id)",
            ),
        )

    def create(self, request: InteractionRequest) -> InteractionRecord:
        # DTOs are mutable; do not let model_copy/model_construct bypass ownership checks.
        try:
            request = InteractionRequest.model_validate_json(request.model_dump_json())
        except ValueError as exc:
            raise ConfigurationError("交互请求无效") from exc
        payload = canonical_arguments(request.model_dump(mode="json"))
        with self.database.transaction() as connection:
            self.bots.get(request.bot_id)
            previous = connection.execute(
                "SELECT * FROM interactions WHERE task_id=? AND idempotency_key=?",
                (request.task_id, request.idempotency_key),
            ).fetchone()
            if previous is not None:
                if previous["request_json"] != payload:
                    raise ConfigurationError("交互创建键已用于不同请求")
                return self._record(previous)
            record = InteractionRecord(id=uuid4().hex, request=request)
            connection.execute(
                "INSERT INTO interactions "
                "(id, task_id, bot_id, idempotency_key, request_json, state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    request.task_id,
                    request.bot_id,
                    request.idempotency_key,
                    payload,
                    record.state.value,
                    record.created_at.isoformat(),
                ),
            )
            return record

    def get(self, interaction_id: str, *, bot_id: str | None = None) -> InteractionRecord:
        with self.database.connect() as connection:
            record = self._require(connection, interaction_id)
        if bot_id is not None and record.request.bot_id != bot_id:
            raise ConfigurationError("交互不属于指定 Bot")
        return record

    def list(
        self,
        *,
        bot_id: str | None = None,
        task_id: str | None = None,
        state: InteractionState | None = None,
    ) -> list[InteractionRecord]:
        clauses = []
        parameters = []
        for column, value in (("bot_id", bot_id), ("task_id", task_id), ("state", state)):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        query = "SELECT * FROM interactions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id"
        with self.database.connect() as connection:
            return [self._record(row) for row in connection.execute(query, parameters)]

    def reply(self, interaction_id: str, reply: InteractionReply) -> InteractionRecord:
        try:
            reply = InteractionReply.model_validate_json(reply.model_dump_json())
        except ValueError as exc:
            raise ConfigurationError("交互答复类型无效") from exc
        with self.database.transaction() as connection:
            record = self._require(connection, interaction_id)
            if record.state == InteractionState.CANCELLED:
                raise ConfigurationError("已取消的交互不能答复")
            self._validate_reply(record.request, reply)
            if record.reply is not None:
                if record.reply != reply:
                    raise ConfigurationError("交互已保存不同答复")
                return record
            connection.execute(
                "UPDATE interactions SET state='answered', reply_json=?, answered_at=? "
                "WHERE id=? AND state='pending'",
                (reply.model_dump_json(), utc_now().isoformat(), interaction_id),
            )
            return self._require(connection, interaction_id)

    def ready(self) -> builtins.list[InteractionRecord]:
        """A repeatable wake list: reading it never acknowledges or consumes a reply."""
        return self.list(state=InteractionState.ANSWERED)

    def consume_authorization(
        self, interaction_id: str, operation: OperationIntent, run_id: str
    ) -> bool:
        if not run_id.strip():
            return False
        try:
            operation = OperationIntent.model_validate_json(operation.model_dump_json())
        except ValueError:
            return False
        with self.database.transaction() as connection:
            record = self._find(connection, interaction_id)
            if (
                record is None
                or record.state != InteractionState.ANSWERED
                or record.request.kind != "approval"
                or record.reply is None
                or record.reply.approved is not True
                or record.request.operation is None
                or record.request.task_id != operation.task_id
                or record.request.bot_id != operation.bot_id
                or record.request.operation.binding != operation.binding
            ):
                return False
            # BEGIN IMMEDIATE excludes BotStore.save for this same database while its
            # public API reads authority on a separate connection. No Bot SQL coupling.
            try:
                authority = self.bots.authority(operation.bot_id)
            except ConfigurationError:
                return False
            if (
                authority.bot_id != operation.bot_id
                or authority.permission_version != operation.permission_version
                or authority.permission_digest != operation.permission_digest
                or not authority.allows(operation.tool_name, mutating=True)
            ):
                return False
            self._consume(connection, interaction_id, run_id)
            return True

    def consume_answer(self, interaction_id: str, task_id: str, run_id: str) -> str | None:
        if not run_id.strip():
            return None
        with self.database.transaction() as connection:
            record = self._find(connection, interaction_id)
            if (
                record is None
                or record.state != InteractionState.ANSWERED
                or record.request.kind != "question"
                or record.request.task_id != task_id
                or record.reply is None
                or record.reply.text is None
            ):
                return None
            self._consume(connection, interaction_id, run_id)
            return record.reply.text

    def cancel_for_task(self, task_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE interactions SET state='cancelled' "
                "WHERE task_id=? AND state IN ('pending', 'answered')",
                (task_id,),
            )

    @staticmethod
    def _validate_reply(request: InteractionRequest, reply: InteractionReply) -> None:
        if request.kind == "approval" and type(reply.approved) is not bool:
            raise ConfigurationError("审批答复必须提供布尔值 approved")
        if request.kind == "question" and (reply.text is None or not reply.text.strip()):
            raise ConfigurationError("提问答复必须提供非空文本")

    @staticmethod
    def _consume(connection: sqlite3.Connection, interaction_id: str, run_id: str) -> None:
        connection.execute(
            "UPDATE interactions SET state='consumed', consumed_by_run_id=?, consumed_at=? "
            "WHERE id=? AND state='answered'",
            (run_id, utc_now().isoformat(), interaction_id),
        )

    @classmethod
    def _find(cls, connection: sqlite3.Connection, interaction_id: str) -> InteractionRecord | None:
        row = connection.execute(
            "SELECT * FROM interactions WHERE id=?", (interaction_id,)
        ).fetchone()
        return cls._record(row) if row is not None else None

    @classmethod
    def _require(cls, connection: sqlite3.Connection, interaction_id: str) -> InteractionRecord:
        record = cls._find(connection, interaction_id)
        if record is None:
            raise ConfigurationError(f"交互不存在: {interaction_id}")
        return record

    @classmethod
    def _record(cls, row: sqlite3.Row) -> InteractionRecord:
        try:
            request = InteractionRequest.model_validate_json(row["request_json"])
            reply = (
                InteractionReply.model_validate_json(row["reply_json"])
                if row["reply_json"] is not None
                else None
            )
            record = InteractionRecord(
                id=row["id"],
                request=request,
                state=row["state"],
                reply=reply,
                consumed_by_run_id=row["consumed_by_run_id"],
                created_at=row["created_at"],
                answered_at=row["answered_at"],
                consumed_at=row["consumed_at"],
            )
        except ValidationError as exc:
            raise PersistenceError("已保存的交互数据无效") from exc
        if (request.task_id, request.bot_id, request.idempotency_key) != (
            row["task_id"],
            row["bot_id"],
            row["idempotency_key"],
        ):
            raise PersistenceError("交互归属与已保存的请求不一致")
        if reply is not None:
            try:
                cls._validate_reply(request, reply)
            except ConfigurationError as exc:
                raise PersistenceError("已保存的交互答复无效") from exc
        return record
