"""Kernel persistence and authority at the actual execution boundaries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.bootstrap import create_harness
from hiro_agent.bots import BotStore
from hiro_agent.budget import Budget
from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig, SubagentsConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    ModelResponse,
    SessionEvent,
    ToolCall,
    ToolResult,
)
from hiro_agent.errors import ConfigurationError, HiroError, PersistenceError
from hiro_agent.kernel_contracts import OperationIntent
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.models import MockModel, ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession
from hiro_agent.subagents import SubagentManager
from hiro_agent.tools import ToolExecutor, build_file_tools


def bot_config(*tools: str, bot_id: str = "inspector") -> Config:
    return Config(model=ModelConfig(adapter="mock"), bot=BotConfig(id=bot_id, allowed_tools=tools))


def complete_config() -> Config:
    return Config.model_validate(
        {
            "bot": {
                "id": "inspector",
                "name": "Persistent inspector",
                "instructions": "Cite every finding.",
                "allowed_tools": ["read_file", "list_files", "delegate_task"],
            },
            "model": {
                "adapter": "mock",
                "model": "configured-model",
                "base_url": "https://model.example/v1",
                "api_key_env": "BOT_MODEL_KEY",
                "request_timeout_seconds": 31,
                "stream": False,
                "extra_body": {"temperature": 0.25},
            },
            "permissions": "read-only",
            "budget": {"max_model_requests": 8, "max_tool_calls": 9, "timeout_seconds": 40},
            "tools": {
                "max_output_chars": 2048,
                "max_file_bytes": 4096,
                "max_search_files": 10,
                "command_timeout_seconds": 12,
            },
            "subagents": {
                "enabled": True,
                "max_concurrency": 3,
                "max_model_requests": 4,
                "max_tool_calls": 5,
                "max_output_chars": 1024,
                "roles": {
                    "explorer": {
                        "instructions": "Inspect interfaces.",
                        "model": {"adapter": "mock", "model": "scout"},
                    },
                    "reviewer": {"instructions": "Check boundaries."},
                },
            },
        }
    )


def test_migration_failure_rolls_back_schema_and_version(tmp_path: Path) -> None:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    with pytest.raises(PersistenceError):
        database.migrate(
            "test",
            1,
            ("CREATE TABLE example (id INTEGER PRIMARY KEY)", "INSERT INTO missing VALUES (1)"),
        )
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM kernel_migrations").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='example'").fetchone()
            is None
        )
    database.migrate("test", 1, ("CREATE TABLE example (id INTEGER PRIMARY KEY)",))


def test_migration_reopen_is_idempotent_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "kernel.sqlite3"
    statements = ("CREATE TABLE example (id INTEGER PRIMARY KEY)",)
    database = KernelDatabase(path)
    database.migrate("test", 1, statements)
    with database.transaction() as connection:
        connection.execute("INSERT INTO example VALUES (7)")
    reopened = KernelDatabase(path)
    reopened.migrate("test", 1, statements)
    with pytest.raises(PersistenceError, match="迁移内容"):
        reopened.migrate("test", 1, ("CREATE TABLE changed (id INTEGER)",))
    with reopened.connect() as connection:
        assert connection.execute("SELECT id FROM example").fetchone()[0] == 7
        assert connection.execute("SELECT count(*) FROM kernel_migrations").fetchone()[0] == 1


def test_migrations_enforce_order_independently_for_each_namespace(tmp_path: Path) -> None:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    with pytest.raises(PersistenceError, match="顺序"):
        database.migrate("first", 2, ("CREATE TABLE skipped (id INTEGER)",))
    database.migrate("first", 1, ("CREATE TABLE first (id INTEGER)",))
    database.migrate("second", 1, ("CREATE TABLE second (id INTEGER)",))
    database.migrate("first", 2, ("ALTER TABLE first ADD COLUMN name TEXT",))
    with database.connect() as connection:
        versions = connection.execute(
            "SELECT namespace, version FROM kernel_migrations ORDER BY namespace, version"
        ).fetchall()
    assert [tuple(row) for row in versions] == [("first", 1), ("first", 2), ("second", 1)]


def test_transactions_roll_back_python_errors_and_enforce_foreign_keys(tmp_path: Path) -> None:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    database.migrate(
        "relations",
        1,
        (
            "CREATE TABLE parent (id INTEGER PRIMARY KEY)",
            "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id))",
        ),
    )
    with pytest.raises(RuntimeError, match="abort"):
        with database.transaction() as connection:
            connection.execute("INSERT INTO parent VALUES (1)")
            raise RuntimeError("abort")
    with pytest.raises(PersistenceError):
        with database.transaction() as connection:
            connection.execute("INSERT INTO parent VALUES (2)")
            connection.execute("INSERT INTO child VALUES (99)")
    with database.connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM parent").fetchone()[0] == 0
    with database.transaction() as connection:
        connection.execute("INSERT INTO parent VALUES (3)")
        connection.execute("INSERT INTO child VALUES (3)")


def test_database_directory_preparation_failure_is_persistence_error(tmp_path: Path) -> None:
    occupied_parent = tmp_path / "occupied"
    occupied_parent.write_text("existing file")
    with pytest.raises(PersistenceError) as caught:
        KernelDatabase(occupied_parent / "kernel.sqlite3")
    assert isinstance(caught.value.__cause__, OSError)
    assert occupied_parent.read_text() == "existing file"


@pytest.mark.parametrize("boundary", ["connect", "transaction"])
@pytest.mark.parametrize("damage", ["directory", "corrupt"])
def test_sqlite_entry_failures_preserve_persistence_error_boundary(
    tmp_path: Path, boundary: str, damage: str
) -> None:
    database = KernelDatabase(tmp_path / "kernel.sqlite3")
    database.path.unlink()
    if damage == "directory":
        database.path.mkdir()
    else:
        database.path.write_bytes(b"This is not a SQLite database.")
    with pytest.raises(PersistenceError) as caught:
        with getattr(database, boundary)() as connection:
            connection.execute("SELECT * FROM kernel_migrations")
    assert isinstance(caught.value.__cause__, sqlite3.Error)


def test_bot_identity_and_complete_configuration_survive_reopen(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    config = complete_config()
    saved = store.save(config)
    reopened = BotStore(KernelDatabase.for_workspace(tmp_path))
    restored = reopened.get("inspector")
    assert restored.model_dump(mode="json") == saved.model_dump(mode="json")
    assert restored.config.model_dump() == config.model_dump()
    assert restored.authority.permission_version == 1
    assert [record.bot_id for record in reopened.list()] == ["inspector"]
    store.save(bot_config("list_files", bot_id="other"))
    assert [record.bot_id for record in reopened.list()] == ["inspector", "other"]
    assert reopened.get("inspector").config == config


def test_only_permission_changes_increment_version_including_reverts(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    config = complete_config()
    original = store.save(config)
    changed = config.model_dump()
    changed["bot"].update(name="Renamed", instructions="New instructions")
    changed["bot"]["allowed_tools"] = ["read_file", "delegate_task", "list_files", "read_file"]
    changed["model"]["model"] = "new-model"
    changed["budget"]["max_tool_calls"] = 12
    cosmetic = store.save(Config.model_validate(changed), expected_version=1)
    assert cosmetic.authority == original.authority
    assert cosmetic.created_at == original.created_at
    assert cosmetic.config.bot is not None and cosmetic.config.bot.name == "Renamed"

    changed["bot"]["allowed_tools"] = ["list_files"]
    reduced = store.save(Config.model_validate(changed), expected_version=1)
    assert reduced.authority.permission_version == 2
    assert reduced.authority.permission_digest != original.authority.permission_digest
    restored = store.save(config, expected_version=2)
    assert restored.authority.permission_version == 3
    assert restored.authority.permission_digest == original.authority.permission_digest
    config.permissions = "ask-write"
    assert store.save(config, expected_version=3).authority.permission_version == 4


def test_stale_permission_version_cannot_overwrite_persisted_config(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(bot_config("list_files"), expected_version=0)
    authoritative = store.save(bot_config(), expected_version=1)
    with pytest.raises(ConfigurationError, match="版本"):
        store.save(bot_config("write_file"), expected_version=1)
    assert store.get("inspector") == authoritative
    with pytest.raises(ConfigurationError, match="稳定 ID"):
        store.save(Config())


async def execute(executor: ToolExecutor, call: ToolCall) -> tuple[ToolResult, Budget]:
    run_id = "foundation-run"
    await executor.session.append("run.started", {}, run_id)
    await executor.session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=[call]).model_dump(mode="json")},
        run_id,
    )
    budget = Budget(BudgetConfig())
    return await executor.execute(call, run_id, budget, Cancellation()), budget


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="foundation-call", name=name, arguments=json.dumps(arguments))


@pytest.mark.parametrize("name", ["read_file", "write_file", "run_command", "invented_tool"])
async def test_bot_whitelist_rejects_fabricated_calls_before_approval(
    tmp_path: Path, name: str
) -> None:
    (tmp_path / "private.txt").write_text("private evidence")
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(bot_config("list_files"))
    approvals = []

    async def approve(tool: str, _arguments: dict[str, Any]) -> bool:
        approvals.append(tool)
        return True

    executor = ToolExecutor(
        build_file_tools(tmp_path, Config().tools),
        MemorySession(),
        PermissionPolicy(callback=approve),
        authority=store.provider("inspector"),
    )
    assert {schema["function"]["name"] for schema in executor.schemas()} == {"list_files"}
    arguments: dict[str, Any] = {"path": "private.txt"}
    if name == "write_file":
        arguments["content"] = "overwritten"
    result, budget = await execute(executor, tool_call(name, **arguments))
    assert not result.ok and result.outcome == "not_started"
    assert result.error == ("unknown_tool" if name == "invented_tool" else "permission_denied")
    assert result.value is None
    assert approvals == []
    assert not any(event.type == "tool.started" for event in executor.session.events)
    assert (tmp_path / "private.txt").read_text() == "private evidence"
    assert budget.metrics.permission_denials == (0 if name == "invented_tool" else 1)


async def test_empty_whitelist_exposes_and_executes_no_tools(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(bot_config())
    executor = ToolExecutor(
        build_file_tools(tmp_path, Config().tools),
        MemorySession(),
        PermissionPolicy(),
        authority=store.provider("inspector"),
    )
    assert executor.schemas() == []
    result, _ = await execute(executor, tool_call("list_files"))
    assert result.error == "permission_denied" and result.outcome == "not_started"


@pytest.mark.parametrize("revoke_at", ["approval", "tool.started"])
async def test_fresh_revocation_prevents_effects_across_await_boundaries(
    tmp_path: Path, revoke_at: str
) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    original = store.save(bot_config("write_file"))

    class RevokingSession(MemorySession):
        async def append(
            self, event_type: str, data: dict[str, Any], run_id: str | None = None
        ) -> SessionEvent:
            event = await super().append(event_type, data, run_id)
            if event_type == revoke_at:
                store.save(bot_config(), expected_version=1)
            return event

    async def approve(_tool: str, _arguments: dict[str, Any]) -> bool:
        if revoke_at == "approval":
            store.save(bot_config(), expected_version=1)
        return True

    session = RevokingSession()
    executor = ToolExecutor(
        build_file_tools(tmp_path, Config().tools),
        session,
        PermissionPolicy(callback=approve),
        authority=store.provider("inspector"),
    )
    result, budget = await execute(executor, tool_call("write_file", path="out", content="x"))
    assert result.error == "permission_changed" and result.outcome == "not_started"
    assert not (tmp_path / "out").exists()
    assert budget.metrics.permission_denials == 1
    starts = [event for event in session.events if event.type == "tool.started"]
    assert len(starts) == (1 if revoke_at == "tool.started" else 0)
    if starts:
        assert starts[0].data["authority"] == original.authority.model_dump(mode="json")
    assert session.events[-1].type == "tool.result"
    assert session.events[-1].data["outcome"] == "not_started"
    assert executor.schemas() == []


async def test_bot_read_only_overrides_permissive_permission_callback(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    config = bot_config("write_file")
    config.permissions = "read-only"
    store.save(config)

    async def approve(_tool: str, _arguments: dict[str, Any]) -> bool:
        pytest.fail("The Bot permission boundary must reject before asking for approval")

    executor = ToolExecutor(
        build_file_tools(tmp_path, config.tools),
        MemorySession(),
        PermissionPolicy(callback=approve),
        authority=store.provider("inspector"),
    )
    assert executor.schemas() == []
    result, _ = await execute(executor, tool_call("write_file", path="out", content="x"))
    assert result.error == "permission_denied"
    assert not (tmp_path / "out").exists()


async def test_executor_without_bot_keeps_legacy_tools_and_approval(tmp_path: Path) -> None:
    approvals = []

    async def approve(tool: str, arguments: dict[str, Any]) -> bool:
        approvals.append((tool, arguments))
        return True

    registry = build_file_tools(tmp_path, Config().tools)
    executor = ToolExecutor(registry, MemorySession(), PermissionPolicy(callback=approve))
    assert executor.schemas() == registry.schemas()
    result, _ = await execute(executor, tool_call("write_file", path="out", content="legacy"))
    assert result.ok
    assert approvals[0][0] == "write_file"
    assert (tmp_path / "out").read_text() == "legacy"


async def test_child_inherits_context_authority_over_manager_defaults(tmp_path: Path) -> None:
    (tmp_path / "private.txt").write_text("private evidence")
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(bot_config("delegate_task", "list_files"))
    store.save(bot_config("list_files", "read_file", "search_text", bot_id="broad-manager"))
    model = ScriptedModel(
        [
            ModelResponse(
                assistant=AssistantMessage(tool_calls=[tool_call("read_file", path="private.txt")]),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                assistant=AssistantMessage(text=json.dumps({"summary": "Reading was denied."})),
                finish_reason="stop",
            ),
        ]
    )
    children = []

    async def sessions(session_id: str) -> MemorySession:
        child = MemorySession(session_id)
        children.append(child)
        return child

    registry = build_file_tools(tmp_path, Config().tools)
    manager = SubagentManager(
        model,
        registry,
        sessions,
        SubagentsConfig(enabled=True),
        authority=store.provider("broad-manager"),
    )
    manager.register(registry)
    executor = ToolExecutor(
        registry, MemorySession(), PermissionPolicy(), authority=store.provider("inspector")
    )
    try:
        result, _ = await execute(
            executor, tool_call("delegate_task", role="explorer", task="Read private.txt")
        )
        assert result.ok
        assert len(children) == 1
        assert {tool["function"]["name"] for tool in model.requests[0].tools} == {"list_files"}
        child_result = next(event for event in children[0].events if event.type == "tool.result")
        assert child_result.data["error"] == "permission_denied"
        assert child_result.data["outcome"] == "not_started"
        assert not any(event.type == "tool.started" for event in children[0].events)
        assert "private evidence" not in json.dumps(model.requests[-1].model_dump())
    finally:
        await manager.close()


async def test_bootstrap_binds_identity_and_resumes_complete_saved_config(tmp_path: Path) -> None:
    config = complete_config()
    harness = await create_harness(tmp_path, config)
    session_id = harness.session.session_id
    try:
        binding = [event for event in harness.session.events if event.type == "session.bot"]
        assert len(binding) == 1 and binding[0].data["bot_id"] == "inspector"
        assert harness.runtime.services["bot"].config == config
        assert "Cite every finding." in harness.session.derive_messages()[0].content
    finally:
        await harness.close()

    # Defaults select a credential-requiring model and disable subagents. A successful
    # resume must load the complete persisted configuration before building services.
    resumed = await create_harness(tmp_path, Config(), session_id)
    try:
        assert resumed.session.session_id == session_id
        assert resumed.runtime.services["bot"].config == config
        assert isinstance(resumed.runtime.services["model"], MockModel)
        assert resumed.runtime.services["agent"].budget_config == config.budget
        assert resumed.runtime.services["subagents"].config == config.subagents
        assert resumed.runtime.services["permission"].mode == "read-only"
        assert len([event for event in resumed.session.events if event.type == "session.bot"]) == 1
    finally:
        await resumed.close()


async def test_bootstrap_rejects_resuming_as_another_bot_and_releases_session(
    tmp_path: Path,
) -> None:
    harness = await create_harness(tmp_path, bot_config("list_files"))
    session_id = harness.session.session_id
    await harness.close()
    with pytest.raises(HiroError, match="其他 Bot"):
        await create_harness(tmp_path, bot_config("write_file", bot_id="intruder"), session_id)
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    assert [record.bot_id for record in store.list()] == ["inspector"]
    resumed = await create_harness(tmp_path, Config(), session_id)
    try:
        assert resumed.runtime.services["bot"].bot_id == "inspector"
    finally:
        await resumed.close()


def test_operation_binding_normalizes_arguments_and_invalidates_changed_authority() -> None:
    operation = OperationIntent(
        task_id="task",
        bot_id="inspector",
        session_id="session",
        run_id="old-run",
        call_id="old-call",
        tool_name="write_file",
        arguments_json='{"path":"out", "options":{"b":2,"a":1}}',
        permission_version=1,
        permission_digest="original-digest",
    )
    renewed = operation.model_dump()
    renewed.update(
        run_id="new-run",
        call_id="new-call",
        arguments_json='{"options": {"a":1, "b":2}, "path":"out"}',
    )
    assert OperationIntent.model_validate(renewed).binding == operation.binding
    for field, value in (
        ("task_id", "other-task"),
        ("bot_id", "other-bot"),
        ("tool_name", "other_tool"),
        ("arguments_json", '{"path":"different"}'),
        ("permission_version", 2),
        ("permission_digest", "changed-digest"),
    ):
        changed = {**renewed, field: value}
        assert OperationIntent.model_validate(changed).binding != operation.binding
