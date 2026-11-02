"""Real SQLite interaction persistence, contention and authorization boundaries."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from hiro_agent.bots import BotAuthority, BotStore
from hiro_agent.config import BotConfig, Config
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.interactions import InteractionStore
from hiro_agent.kernel_contracts import (
    InteractionRecord,
    InteractionReply,
    InteractionRequest,
    InteractionService,
    InteractionState,
    OperationIntent,
)
from hiro_agent.kernel_db import KernelDatabase


def make_config(*tools: str, bot_id: str = "writer", read_only: bool = False) -> Config:
    return Config(
        bot=BotConfig(id=bot_id, allowed_tools=tools),
        permissions="read-only" if read_only else "ask-write",
    )


@pytest.fixture
def store(tmp_path: Path) -> InteractionStore:
    database = KernelDatabase.for_workspace(tmp_path)
    bots = BotStore(database)
    bots.save(make_config("write_file", "read_file"))
    bots.save(make_config("write_file", bot_id="other"))
    return InteractionStore(database, bots)


def reopen(store: InteractionStore) -> InteractionStore:
    database = KernelDatabase(store.database.path)
    return InteractionStore(database, BotStore(database))


def intent(store: InteractionStore, **changes: Any) -> OperationIntent:
    authority = store.bots.authority("writer")
    values = {
        "task_id": "task-1",
        "bot_id": "writer",
        "session_id": "session-1",
        "run_id": "run-original",
        "call_id": "call-original",
        "tool_name": "write_file",
        "arguments_json": '{"path":"report.txt","content":"approved"}',
        "permission_version": authority.permission_version,
        "permission_digest": authority.permission_digest,
    }
    values.update(changes)
    return OperationIntent.model_validate(values)


def approval_request(
    store: InteractionStore, operation: OperationIntent | None = None, **changes: Any
) -> InteractionRequest:
    operation = operation or intent(store)
    values: dict[str, Any] = {
        "task_id": operation.task_id,
        "bot_id": operation.bot_id,
        "kind": "approval",
        "prompt": "Allow this write?",
        "idempotency_key": "approval-1",
        "operation": operation,
    }
    values.update(changes)
    return InteractionRequest.model_validate(values)


def question_request(**changes: Any) -> InteractionRequest:
    values: dict[str, Any] = {
        "task_id": "task-1",
        "bot_id": "writer",
        "kind": "question",
        "prompt": "Which branch?",
        "idempotency_key": "question-1",
    }
    values.update(changes)
    return InteractionRequest.model_validate(values)


def approved(
    store: InteractionStore, operation: OperationIntent | None = None
) -> InteractionRecord:
    record = store.create(approval_request(store, operation))
    return store.reply(record.id, InteractionReply(approved=True))


def test_migration_and_persistence_need_no_task_tables(store: InteractionStore) -> None:
    service: InteractionService = store
    record = service.create(approval_request(store))
    question = service.create(question_request())
    recovered = reopen(store)
    assert recovered.get(record.id) == record
    assert recovered.get(question.id) == question
    with recovered.database.connect() as connection:
        migrations = connection.execute(
            "SELECT namespace, version FROM kernel_migrations ORDER BY namespace"
        ).fetchall()
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert [tuple(row) for row in migrations] == [("bots", 1), ("interactions", 1)]
    assert {row[0] for row in tables} == {"bots", "interactions", "kernel_migrations"}


def test_different_authority_database_is_rejected(store: InteractionStore, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="同一个"):
        InteractionStore(KernelDatabase(tmp_path / "another.sqlite3"), store.bots)


def test_create_revalidates_mutable_request_and_bot_existence(store: InteractionStore) -> None:
    missing = question_request(bot_id="missing")
    with pytest.raises(ConfigurationError, match="Bot 不存在"):
        store.create(missing)
    forged = approval_request(store)
    forged.task_id = "other-task"
    with pytest.raises(ConfigurationError, match="请求无效"):
        store.create(forged)
    assert store.list() == []


def test_create_is_idempotent_in_every_state(store: InteractionStore) -> None:
    request = approval_request(store)
    record = store.create(request)
    assert store.create(request) == record
    answered = store.reply(record.id, InteractionReply(approved=True))
    assert reopen(store).create(request) == answered
    assert request.operation is not None
    assert store.consume_authorization(record.id, request.operation, "new-run")
    consumed = store.get(record.id)
    assert store.create(request) == consumed
    question = question_request()
    cancelled = store.create(question)
    store.cancel_for_task(question.task_id)
    assert store.create(question) == store.get(cancelled.id)
    assert store.get(record.id) == consumed


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": "Another prompt"},
        {"bot_id": "other"},
        {"kind": "approval"},
    ],
)
def test_creation_key_conflicts_do_not_replace_payload(
    store: InteractionStore, changes: dict[str, Any]
) -> None:
    request = question_request()
    first = store.create(request)
    changed = request.model_dump()
    changed.update(changes)
    if changed["kind"] == "approval":
        changed["operation"] = intent(store)
    with pytest.raises(ConfigurationError, match="不同请求"):
        store.create(InteractionRequest.model_validate(changed))
    assert store.get(first.id) == first


def test_creation_payload_compares_exact_intent_and_canonical_arguments(
    store: InteractionStore,
) -> None:
    operation = intent(store)
    request = approval_request(store, operation)
    record = store.create(request)
    reordered = operation.model_dump()
    reordered["arguments_json"] = '{ "content": "approved", "path": "report.txt" }'
    assert (
        store.create(approval_request(store, OperationIntent.model_validate(reordered))) == record
    )
    changed = operation.model_copy(update={"call_id": "different-origin"})
    with pytest.raises(ConfigurationError, match="不同请求"):
        store.create(approval_request(store, changed))


def test_creation_key_is_scoped_to_task_and_lookup_checks_bot(store: InteractionStore) -> None:
    first = store.create(question_request())
    second = store.create(question_request(task_id="task-2", bot_id="other"))
    assert first.id != second.id
    assert store.list(bot_id="writer") == [first]
    assert store.list(task_id="task-2") == [second]
    assert store.list(bot_id="writer", task_id="task-2") == []
    assert store.list(state=InteractionState.PENDING) == [first, second]
    assert store.get(first.id, bot_id="writer") == first
    with pytest.raises(ConfigurationError, match="不属于"):
        store.get(first.id, bot_id="other")
    with pytest.raises(ConfigurationError, match="不存在"):
        store.get("missing")
    with pytest.raises(ConfigurationError, match="不存在"):
        store.reply("missing", InteractionReply(text="answer"))


@pytest.mark.parametrize("reply", [InteractionReply(), InteractionReply(text="yes")])
def test_approval_requires_explicit_bool(store: InteractionStore, reply: InteractionReply) -> None:
    record = store.create(approval_request(store))
    with pytest.raises(ConfigurationError, match="布尔"):
        store.reply(record.id, reply)
    assert store.get(record.id) == record


@pytest.mark.parametrize("value", ["yes", 1, 0, "false"])
def test_reply_revalidates_bool_after_model_construct_bypass(
    store: InteractionStore, value: Any
) -> None:
    record = store.create(approval_request(store))
    forged = InteractionReply.model_construct(approved=value)
    with pytest.warns(UserWarning), pytest.raises(ConfigurationError, match="类型无效"):
        store.reply(record.id, forged)
    assert store.get(record.id) == record


@pytest.mark.parametrize("text", [None, "", " \t\n"])
def test_question_requires_nonblank_text(store: InteractionStore, text: str | None) -> None:
    record = store.create(question_request())
    with pytest.raises(ConfigurationError, match="非空文本"):
        store.reply(record.id, InteractionReply(text=text, approved=True))
    assert store.get(record.id) == record


@pytest.mark.parametrize("kind", ["approval", "question"])
def test_replies_are_durable_idempotent_and_conflicts_never_overwrite(
    store: InteractionStore, kind: str
) -> None:
    request = approval_request(store) if kind == "approval" else question_request()
    reply = (
        InteractionReply(approved=True, text="reviewed")
        if kind == "approval"
        else (InteractionReply(text="  feature/中文  "))
    )
    other = (
        InteractionReply(approved=False) if kind == "approval" else InteractionReply(text="main")
    )
    record = store.create(request)
    answered = store.reply(record.id, reply)
    assert answered.state == InteractionState.ANSWERED
    assert answered.answered_at is not None
    assert answered.consumed_at is None
    recovered = reopen(store)
    assert recovered.reply(record.id, reply) == answered
    assert recovered.ready() == recovered.ready() == [answered]
    with pytest.raises(ConfigurationError, match="不同答复"):
        recovered.reply(record.id, other)
    if kind == "approval":
        assert request.operation is not None
        assert recovered.consume_authorization(record.id, request.operation, "run-next")
    else:
        assert recovered.consume_answer(record.id, request.task_id, "run-next") == reply.text
    consumed = recovered.get(record.id)
    assert consumed.state == InteractionState.CONSUMED
    assert consumed.consumed_by_run_id == "run-next"
    assert consumed.consumed_at is not None
    assert store.reply(record.id, reply) == consumed
    assert store.ready() == []
    with pytest.raises(ConfigurationError, match="不同答复"):
        store.reply(record.id, other)
    assert store.get(record.id) == consumed


def test_denial_persists_without_issuing_authority(store: InteractionStore) -> None:
    request = approval_request(store)
    record = store.create(request)
    denied = store.reply(record.id, InteractionReply(approved=False))
    assert request.operation is not None
    assert not reopen(store).consume_authorization(record.id, request.operation, "run-next")
    assert store.ready() == [denied]
    assert store.reply(record.id, InteractionReply(approved=False)) == denied
    store.cancel_for_task(request.task_id)
    cancelled = store.get(record.id)
    assert cancelled.state == InteractionState.CANCELLED
    assert cancelled.reply == denied.reply
    assert cancelled.answered_at == denied.answered_at
    assert store.ready() == []


@pytest.mark.parametrize(
    "changes",
    [
        {"task_id": "task-2"},
        {"bot_id": "other"},
        {"tool_name": "read_file"},
        {"arguments_json": '{"path":"different.txt","content":"approved"}'},
        {"arguments_json": '{"path":"report.txt","content":"changed"}'},
        {"permission_version": 2},
        {"permission_digest": "forged"},
    ],
)
def test_authorization_rejects_every_binding_change_without_consuming(
    store: InteractionStore, changes: dict[str, Any]
) -> None:
    operation = intent(store)
    record = approved(store, operation)
    changed = OperationIntent.model_validate({**operation.model_dump(), **changes})
    assert not store.consume_authorization(record.id, changed, "run-next")
    assert store.get(record.id) == record
    assert store.consume_authorization(record.id, operation, "run-next")


def test_authorization_allows_new_run_and_call_but_cannot_be_consumed_twice(
    store: InteractionStore,
) -> None:
    operation = intent(store)
    record = approved(store, operation)
    resumed = operation.model_copy(update={"id": "new-intent", "run_id": "new", "call_id": "new"})
    assert store.consume_authorization(record.id, resumed, "run-next")
    assert not store.consume_authorization(record.id, operation, "run-next")
    assert not reopen(store).consume_authorization(record.id, resumed, "another-run")
    assert store.get(record.id).consumed_by_run_id == "run-next"


@pytest.mark.parametrize("change", ["revoked", "reverted", "read-only", "cosmetic"])
def test_authorization_checks_current_permission_policy(
    store: InteractionStore, change: str
) -> None:
    operation = intent(store)
    record = approved(store, operation)
    original = store.bots.get("writer").config
    changed = original.model_copy(deep=True)
    assert changed.bot is not None
    if change in {"revoked", "reverted"}:
        changed.bot.allowed_tools = ("read_file",)
    elif change == "read-only":
        changed.permissions = "read-only"
    else:
        changed.bot.name = "Renamed writer"
    store.bots.save(changed)
    if change == "reverted":
        restored = store.bots.save(original)
        assert restored.authority.permission_digest == operation.permission_digest
        assert restored.authority.permission_version != operation.permission_version
    assert store.consume_authorization(record.id, operation, "run-next") is (change == "cosmetic")
    if change != "cosmetic":
        assert store.get(record.id) == record


@pytest.mark.parametrize("flaw", ["digest", "version", "tool", "read-only"])
def test_saved_intent_itself_must_match_real_current_authority(
    store: InteractionStore, flaw: str
) -> None:
    changes: dict[str, Any] = {}
    if flaw == "digest":
        changes["permission_digest"] = "invented"
    elif flaw == "version":
        changes["permission_version"] = 99
    elif flaw == "tool":
        changes["tool_name"] = "run_command"
    else:
        store.bots.save(make_config("write_file", "read_file", read_only=True))
    operation = intent(store, **changes)
    record = approved(store, operation)
    assert not store.consume_authorization(record.id, operation, "run-next")
    assert store.get(record.id) == record


def test_permission_read_holds_same_database_write_lock(
    store: InteractionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation = intent(store)
    record = approved(store, operation)
    actual_authority = store.bots.authority
    observed = []

    def check_lock(bot_id: str) -> BotAuthority:
        # A distinct SQLite writer cannot change permission between check and consume.
        with sqlite3.connect(store.database.path, timeout=0, isolation_level=None) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")
        observed.append(bot_id)
        return actual_authority(bot_id)

    monkeypatch.setattr(store.bots, "authority", check_lock)
    assert store.consume_authorization(record.id, operation, "run-next")
    assert observed == ["writer"]


def test_answer_is_once_only_and_bound_to_task_and_question_kind(store: InteractionStore) -> None:
    question = store.create(question_request())
    assert store.consume_answer(question.id, "task-1", "run-next") is None
    answered = store.reply(question.id, InteractionReply(text="main"))
    assert store.consume_answer(question.id, "task-2", "run-next") is None
    assert store.consume_answer(question.id, "task-1", " ") is None
    assert store.get(question.id) == answered
    assert reopen(store).consume_answer(question.id, "task-1", "run-next") == "main"
    assert store.consume_answer(question.id, "task-1", "run-next") is None
    assert reopen(store).consume_answer(question.id, "task-1", "other-run") is None
    approval = approved(store)
    assert store.consume_answer(approval.id, "task-1", "run-next") is None


def test_authorization_rejects_pending_question_missing_and_blank_run(
    store: InteractionStore,
) -> None:
    operation = intent(store)
    pending = store.create(approval_request(store, operation))
    question = store.create(question_request())
    store.reply(question.id, InteractionReply(text="yes", approved=True))
    assert not store.consume_authorization(pending.id, operation, "run-next")
    assert not store.consume_authorization(question.id, operation, "run-next")
    assert not store.consume_authorization("missing", operation, "run-next")
    assert store.consume_answer("missing", "task-1", "run-next") is None
    store.reply(pending.id, InteractionReply(approved=True))
    assert not store.consume_authorization(pending.id, operation, "")
    assert store.consume_authorization(pending.id, operation, "run-next")


def test_cancel_only_affects_unconsumed_records_of_its_task(store: InteractionStore) -> None:
    operation = intent(store)
    consumed = approved(store, operation)
    assert store.consume_authorization(consumed.id, operation, "run-next")
    evidence = store.get(consumed.id)
    pending = store.create(question_request())
    answered = store.create(question_request(idempotency_key="question-2"))
    answer = store.reply(answered.id, InteractionReply(text="answer"))
    other = store.create(question_request(task_id="other-task"))
    store.cancel_for_task("task-1")
    store.cancel_for_task("task-1")
    assert reopen(store).get(consumed.id) == evidence
    assert store.get(other.id) == other
    for record in (pending, answered):
        assert store.get(record.id).state == InteractionState.CANCELLED
        assert store.consume_answer(record.id, "task-1", "run-next") is None
        with pytest.raises(ConfigurationError, match="已取消"):
            store.reply(record.id, InteractionReply(text="answer"))
    assert store.get(answered.id).reply == answer.reply
    assert store.get(answered.id).answered_at == answer.answered_at
    assert store.ready() == []


def test_concurrent_same_creation_returns_one_record(store: InteractionStore) -> None:
    contenders = [reopen(store) for _ in range(4)]
    barrier = Barrier(len(contenders))
    request = question_request()

    def create(contender: InteractionStore) -> InteractionRecord:
        barrier.wait(timeout=5)
        return contender.create(request)

    with ThreadPoolExecutor(max_workers=len(contenders)) as pool:
        records = list(pool.map(create, contenders))
    assert all(record == records[0] for record in records)
    assert store.list() == [records[0]]


@pytest.mark.parametrize("same_reply", [True, False])
def test_racing_replies_are_idempotent_or_conflict_without_losing_winner(
    store: InteractionStore, same_reply: bool
) -> None:
    record = store.create(question_request())
    contenders = [reopen(store), reopen(store)]
    barrier = Barrier(2)

    def reply(index: int) -> InteractionRecord | None:
        barrier.wait(timeout=5)
        text = "answer" if same_reply or index == 0 else "different"
        try:
            return contenders[index].reply(record.id, InteractionReply(text=text))
        except ConfigurationError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reply, range(2)))
    successes = [result for result in results if result is not None]
    assert len(successes) == (2 if same_reply else 1)
    assert all(result == successes[0] for result in successes)
    assert store.get(record.id) == successes[0]
    assert store.ready() == [successes[0]]


@pytest.mark.parametrize("kind", ["approval", "question"])
def test_racing_consumers_issue_exactly_one_authorization_or_answer(
    store: InteractionStore, kind: str
) -> None:
    operation = intent(store)
    if kind == "approval":
        record = approved(store, operation)
    else:
        record = store.create(question_request())
        store.reply(record.id, InteractionReply(text="answer"))
    contenders = [reopen(store) for _ in range(4)]
    barrier = Barrier(len(contenders))

    def consume(index: int) -> bool | str | None:
        barrier.wait(timeout=5)
        contender = contenders[index]
        if kind == "approval":
            return contender.consume_authorization(record.id, operation, f"run-{index}")
        return contender.consume_answer(record.id, "task-1", f"run-{index}")

    with ThreadPoolExecutor(max_workers=len(contenders)) as pool:
        results = list(pool.map(consume, range(len(contenders))))
    winners = [index for index, result in enumerate(results) if result]
    assert len(winners) == 1
    consumed = reopen(store).get(record.id)
    assert consumed.consumed_by_run_id == f"run-{winners[0]}"
    assert store.ready() == []


def test_persistence_errors_do_not_become_authorization_denials(store: InteractionStore) -> None:
    operation = intent(store)
    record = approved(store, operation)
    with store.database.transaction() as connection:
        connection.execute(
            "UPDATE bots SET permission_digest=? WHERE bot_id=?", ("corrupt", "writer")
        )
    with pytest.raises(PersistenceError, match="摘要"):
        store.consume_authorization(record.id, operation, "run-next")
    assert store.get(record.id) == record


@pytest.mark.parametrize("damage", ["owner", "reply"])
def test_corrupt_saved_interaction_is_not_usable(store: InteractionStore, damage: str) -> None:
    record = approved(store)
    with store.database.transaction() as connection:
        if damage == "owner":
            connection.execute("UPDATE interactions SET bot_id='other' WHERE id=?", (record.id,))
        else:
            connection.execute(
                "UPDATE interactions SET reply_json=? WHERE id=?", ('{"approved":"yes"}', record.id)
            )
    with pytest.raises(PersistenceError):
        store.get(record.id)
