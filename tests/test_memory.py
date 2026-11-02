"""Memory durability/isolation and bounded projections over real execution history."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hiro_agent.bots import BotStore
from hiro_agent.config import BotConfig, Config
from hiro_agent.contracts import (
    AssistantMessage,
    Message,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
)
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import RunClaim, TaskRecord, TaskSubmission
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.memory import (
    MAX_MEMORY_CHARS,
    SUMMARY_CHARS,
    ContextCapacityError,
    MemoryStore,
    context_size,
    prepare_context,
)
from hiro_agent.session import JsonlSession, recover_session


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    database = KernelDatabase.for_workspace(tmp_path)
    bots = BotStore(database)
    bots.save(Config(bot=BotConfig(id="writer")))
    bots.save(Config(bot=BotConfig(id="reader")))
    return MemoryStore(database, bots, TaskStore(database, bots))


def finish_task(
    store: MemoryStore,
    *,
    bot_id: str = "writer",
    text: str = "inspect repository",
    status: RunStatus = RunStatus.COMPLETED,
    output: str = "The package uses SQLite",
) -> tuple[TaskRecord, RunClaim]:
    task = store.tasks.submit(TaskSubmission(bot_id=bot_id, text=text))
    claim = store.tasks.claim("worker")
    assert claim is not None and claim.task.id == task.id
    store.tasks.mark_started(claim.run.id)
    result = RunResult(
        session_id=task.session_id,
        run_id=claim.run.id,
        status=status,
        reason="test result",
        output=output,
    )
    return store.tasks.finish(claim.run.id, result), claim


def test_crud_persists_across_instances_and_is_bot_isolated(store: MemoryStore) -> None:
    note = store.create("writer", "Use pytest; literal %_ is searchable")
    foreign = store.create("reader", "Reader private context")
    reopened = MemoryStore(KernelDatabase(store.database.path), store.bots, store.tasks)
    assert reopened.get("writer", note.id) == note
    assert reopened.list("writer", query="%_") == [note]
    assert reopened.list("writer", query="Reader") == []
    assert reopened.list("reader") == [foreign]
    for action in (
        lambda: reopened.get("reader", note.id),
        lambda: reopened.update("reader", note.id, "forged"),
        lambda: reopened.delete("reader", note.id),
    ):
        with pytest.raises(ConfigurationError, match="Bot"):
            action()
    updated = reopened.update("writer", note.id, "Use uv run pytest", expected_revision=1)
    assert updated.revision == 2 and updated.created_at == note.created_at
    with pytest.raises(ConfigurationError, match="版本"):
        reopened.update("writer", note.id, "stale", expected_revision=1)
    assert store.get("writer", note.id) == updated
    store.delete("writer", note.id)
    assert reopened.list("writer") == []
    with pytest.raises(ConfigurationError):
        reopened.get("writer", note.id)


@pytest.mark.parametrize("content", ["", " \n ", "x" * (MAX_MEMORY_CHARS + 1)])
def test_memory_content_boundaries(store: MemoryStore, content: str) -> None:
    with pytest.raises(ConfigurationError):
        store.create("writer", content)
    note = store.create("writer", "valid")
    with pytest.raises(ConfigurationError):
        store.update("writer", note.id, content)
    assert store.get("writer", note.id).content == "valid"


def test_unknown_bot_and_query_limits_reject(store: MemoryStore) -> None:
    note = store.create("writer", "x" * MAX_MEMORY_CHARS)
    assert len(note.content) == MAX_MEMORY_CHARS
    for action in (
        lambda: store.list("missing"),
        lambda: store.create("missing", "data"),
        lambda: store.get("missing", note.id),
        lambda: store.sync_terminal("missing"),
        lambda: store.list("writer", limit=0),
        lambda: store.list("writer", limit=1001),
    ):
        with pytest.raises(ConfigurationError):
            action()
    store.create("writer", "new")
    assert [item.content for item in store.list("writer", limit=1)] == ["new"]


def test_terminal_summary_replay_keeps_edits_and_deletions(store: MemoryStore) -> None:
    task, claim = finish_task(store)
    record = store.remember_task("writer", task.id)
    assert record is not None
    assert record.kind == "task_summary" and record.final_run_id == claim.run.id
    assert "succeeded" in record.content and task.output in record.content
    assert store.sync_terminal("writer") == [record]
    edited = store.update("writer", record.id, "Human corrected summary")
    assert store.remember_task("writer", task.id) == edited
    store.delete("writer", record.id)
    reopened = MemoryStore(KernelDatabase(store.database.path), store.bots, store.tasks)
    assert reopened.remember_task("writer", task.id) is None
    assert reopened.sync_terminal("writer") == []
    assert reopened.list("writer") == []
    next_task = reopened.tasks.submit(TaskSubmission(bot_id="writer", text="next task"))
    context = prepare_context(
        [Message(role="user", content=next_task.text)],
        bot_id="writer",
        current_task=next_task.text,
        memories=reopened.list("writer"),
    )
    assert all("Human corrected" not in (message.content or "") for message in context.messages)


def test_summaries_are_persistent_per_task_per_bot_and_bounded(store: MemoryStore) -> None:
    first, _ = finish_task(store, text="first", status=RunStatus.FAILED, output="x" * 20_000)
    second, _ = finish_task(store, text="second")
    other, _ = finish_task(store, bot_id="reader", text="private")
    records = store.sync_terminal("writer")
    assert {record.task_id for record in records} == {first.id, second.id}
    assert len(records[0].content) <= SUMMARY_CHARS and "failed" in records[0].content
    assert len(store.sync_terminal("reader")) == 1
    with pytest.raises(ConfigurationError, match="Bot"):
        store.remember_task("writer", other.id)
    fresh = MemoryStore(KernelDatabase(store.database.path), store.bots, store.tasks)
    context = prepare_context(
        [
            Message(role="system", content="Follow permissions"),
            Message(role="user", content="third"),
        ],
        bot_id="writer",
        current_task="third",
        memories=fresh.list("writer"),
    )
    text = "\n".join(message.content or "" for message in context.messages)
    assert "first" in text and "second" in text and "private" not in text
    assert context.omitted_memories == 0


def test_repeated_concurrent_consumption_is_atomic(store: MemoryStore) -> None:
    task, _ = finish_task(store)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda _: store.remember_task("writer", task.id), range(8)))
    assert len({record.id for record in records if record is not None}) == 1
    assert len(store.list("writer")) == 1


def test_no_run_cancellation_summary_and_nonterminal_rejection(store: MemoryStore) -> None:
    task = store.tasks.submit(TaskSubmission(bot_id="writer", text="never started"))
    with pytest.raises(ConfigurationError, match="终态"):
        store.remember_task("writer", task.id)
    store.tasks.cancel(task.id, bot_id="writer")
    record = store.remember_task("writer", task.id)
    assert record is not None and record.final_run_id is None
    assert "cancelled" in record.content
    assert store.remember_task("writer", task.id) == record


def batch(prefix: str, *, count: int = 2, content: str = "result") -> list[Message]:
    calls = [
        ToolCall(id=f"{prefix}-{index}", name="read_file", arguments="{}") for index in range(count)
    ]
    return [Message(role="assistant", tool_calls=calls)] + [
        Message(role="tool", tool_call_id=call.id, content=content) for call in reversed(calls)
    ]


def assert_paired(messages: list[Message]) -> None:
    pending: set[str] = set()
    for message in messages:
        if message.role == "tool":
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
        else:
            assert not pending
            pending.update(call.id for call in message.tool_calls)
    assert not pending


def test_compression_keeps_complete_multicall_batches_and_inputs() -> None:
    messages = [Message(role="system", content="required"), Message(role="user", content="work")]
    messages += batch("old", content="x" * 300)
    messages += [Message(role="assistant", content="old conclusion")]
    messages += batch("new")
    messages += [Message(role="user", content="approved exact destination", continuation=True)]
    prepared = prepare_context(messages, bot_id="writer", current_task="work", max_messages=6)
    assert prepared.messages == [*messages[:2], *messages[6:]]
    assert prepared.omitted_messages == 4
    assert_paired(prepared.messages)
    assert prepared.messages[-1].continuation
    assert prepared.char_count == context_size(prepared.messages)
    prepared.messages[2].tool_calls.clear()
    assert len(messages[6].tool_calls) == 2


def test_capacity_exact_boundary_and_protected_overflow() -> None:
    messages = [Message(role="system", content="必需指令"), Message(role="user", content="task")]
    size = context_size(messages)
    prepared = prepare_context(messages, bot_id="writer", current_task="task", max_chars=size)
    assert prepared.messages == messages and prepared.char_count == size
    with pytest.raises(ContextCapacityError):
        prepare_context(messages, bot_id="writer", current_task="task", max_chars=size - 1)
    with pytest.raises(ContextCapacityError):
        prepare_context(messages, bot_id="writer", current_task="task", max_messages=1)
    messages += batch("large", content="x" * 1_000)
    with pytest.raises(ContextCapacityError):
        prepare_context(messages, bot_id="writer", current_task="task", max_chars=500)


def test_missing_current_task_inserted_and_newest_answer_preserved() -> None:
    messages = [
        Message(role="system", content="safe"),
        Message(role="user", content="old task"),
        Message(role="assistant", content="old response"),
        Message(role="user", content="new answer", continuation=True),
    ]
    prepared = prepare_context(messages, bot_id="writer", current_task="real task", max_messages=3)
    assert [message.content for message in prepared.messages] == ["safe", "real task", "new answer"]
    assert prepared.omitted_messages == 2


def test_memory_context_filters_capacity_and_rejects_foreign_records(store: MemoryStore) -> None:
    first = store.create("writer", "first")
    second = store.create("writer", "second" * 100)
    messages = [Message(role="system", content="safe"), Message(role="user", content="task")]
    one = prepare_context(messages, bot_id="writer", current_task="task", memories=[first])
    limited = prepare_context(
        messages,
        bot_id="writer",
        current_task="task",
        memories=[first, second],
        max_chars=one.char_count,
    )
    assert limited.messages == one.messages and limited.omitted_memories == 1
    assert limited.messages[1].role == "user"
    minimal = prepare_context(
        messages,
        bot_id="writer",
        current_task="task",
        memories=[first],
        max_messages=2,
    )
    assert minimal.messages == messages and minimal.omitted_memories == 1
    with pytest.raises(ConfigurationError, match="Bot"):
        prepare_context(messages, bot_id="reader", current_task="task", memories=[first])


@pytest.mark.parametrize(
    "messages",
    [
        [Message(role="tool", tool_call_id="orphan", content="data")],
        batch("missing")[:-1],
        [batch("duplicate")[0], batch("duplicate")[1], batch("duplicate")[1]],
        [batch("interleaved")[0], Message(role="user", content="interrupt")],
        batch("reused") + batch("reused"),
        [Message(role="user", tool_calls=[ToolCall(id="bad", name="read", arguments="{}")])],
    ],
)
def test_malformed_tool_history_rejected_before_trimming(messages: list[Message]) -> None:
    with pytest.raises(ConfigurationError):
        prepare_context(messages, bot_id="writer", current_task="task", max_messages=1)


async def test_recovered_failed_and_resumed_tool_pairs_keep_jsonl_bytes(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "memory-history")
    await session.append("prompt.system", {"text": "must preserve"})
    await session.append("run.started", {}, "old-run")
    await session.append("task.received", {"text": "do work"}, "old-run")
    calls = [
        ToolCall(id="old-a", name="read_file", arguments="{}"),
        ToolCall(id="old-b", name="read_file", arguments="{}"),
    ]
    await session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=calls).model_dump(mode="json")},
        "old-run",
    )
    await session.append("tool.started", {"call": calls[0].model_dump(mode="json")}, "old-run")
    await recover_session(session)
    await session.append("run.started", {"continued": True}, "new-run")
    await session.append("task.continued", {"text": "approved answer"}, "new-run")
    resumed = ToolCall(id="new-call", name="read_file", arguments="{}")
    await session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=[resumed]).model_dump(mode="json")},
        "new-run",
    )
    failed = ToolResult(call_id=resumed.id, ok=False, error="read_failed", message="missing path")
    await session.append("tool.result", failed.model_dump(mode="json"), "new-run")
    original_bytes = session.path.read_bytes()
    original_events = session.events
    full = prepare_context(session.derive_messages(), bot_id="writer", current_task="do work")
    assert any("TOOL_OUTCOME_UNKNOWN" in (message.content or "") for message in full.messages)
    assert any("TOOL_NOT_STARTED" in (message.content or "") for message in full.messages)
    compressed = prepare_context(
        session.derive_messages(), bot_id="writer", current_task="do work", max_messages=5
    )
    assert compressed.omitted_messages == 3
    assert any("read_failed" in (message.content or "") for message in compressed.messages)
    assert any(message.content == "approved answer" for message in compressed.messages)
    assert_paired(full.messages)
    assert_paired(compressed.messages)
    assert session.events == original_events
    assert session.path.read_bytes() == original_bytes
    await session.close()
    reopened = await JsonlSession.open(tmp_path, "memory-history")
    assert reopened.path.read_bytes() == original_bytes
    assert reopened.derive_messages() == full.messages
    await reopened.close()
