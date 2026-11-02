"""Persistent Bot creation inherits settings without extending approved authority."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hiro_agent.bots import BotStore, CreateAgentArguments
from hiro_agent.config import (
    BotConfig,
    BudgetConfig,
    Config,
    ModelConfig,
    SubagentRoleConfig,
    SubagentRolesConfig,
    SubagentsConfig,
    ToolsConfig,
)
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_db import KernelDatabase


@pytest.fixture
def store(tmp_path: Path) -> BotStore:
    bots = BotStore(KernelDatabase.for_workspace(tmp_path))
    bots.save(
        Config(
            bot=BotConfig(
                id="manager",
                name="Management Bot",
                instructions="Manage persistent Bots",
                allowed_tools=(
                    "create_agent",
                    "delegate_task",
                    "list_files",
                    "read_file",
                    "search_text",
                    "write_file",
                ),
            ),
            model=ModelConfig(
                base_url="https://example.invalid/v1",
                model="manager-model",
                api_key_env="MANAGER_API_KEY",
                request_timeout_seconds=93,
                stream=False,
                extra_body={"temperature": 0.4},
            ),
            budget=BudgetConfig(max_model_requests=23, max_tool_calls=47, timeout_seconds=321),
            tools=ToolsConfig(
                max_output_chars=4096,
                max_file_bytes=2048,
                max_search_files=50,
                command_timeout_seconds=17,
            ),
            subagents=SubagentsConfig(
                enabled=True,
                max_concurrency=3,
                max_model_requests=7,
                max_tool_calls=11,
                max_output_chars=1024,
                roles=SubagentRolesConfig(
                    explorer=SubagentRoleConfig(
                        model=ModelConfig(model="explorer-model", api_key_env="EXPLORER_API_KEY"),
                        instructions="Inspect carefully",
                    )
                ),
            ),
        )
    )
    return bots


def test_creation_inherits_settings_and_survives_reload(store: BotStore) -> None:
    creator = store.get("manager")
    arguments = CreateAgentArguments(
        id="researcher",
        name="  Research Assistant  ",
        instructions="Research the project",
        allowed_tools=("read_file", "list_files", "read_file"),
    )
    child = store.create_from_creator("manager", arguments, expected_authority=creator.authority)
    assert child.config.bot == BotConfig(
        id="researcher",
        name="Research Assistant",
        instructions="Research the project",
        allowed_tools=("list_files", "read_file"),
    )
    assert child.config.model == creator.config.model
    assert child.config.budget == creator.config.budget
    assert child.config.tools == creator.config.tools
    assert child.config.subagents == creator.config.subagents.model_copy(update={"enabled": False})
    assert child.config.permissions == creator.config.permissions
    assert child.authority.permission_version == 1
    assert child.created_at == child.updated_at
    assert child.authority.bot_id == "researcher"
    assert child.authority.allowed_tools == ("list_files", "read_file")
    assert store.get("manager") == creator

    reopened = BotStore(KernelDatabase(store.database.path))
    assert reopened.get("researcher") == child
    assert [bot.bot_id for bot in reopened.list()] == ["manager", "researcher"]


def test_permission_mode_and_tool_scope_can_be_reduced(store: BotStore) -> None:
    creator = store.get("manager")
    child = store.create_from_creator(
        "manager",
        CreateAgentArguments(id="reader", permissions="read-only", allowed_tools=("read_file",)),
        expected_authority=creator.authority,
    )
    assert child.authority.permission_mode == "read-only"
    assert child.authority.allows("read_file")
    assert not child.authority.allows("write_file", mutating=True)
    assert not child.authority.allows("create_agent", mutating=True)


@pytest.mark.parametrize("tools", [("delegate_task",), ("create_agent", "delegate_task"), ()])
def test_subagents_enabled_only_when_delegate_tool_is_granted(
    store: BotStore, tools: tuple[str, ...]
) -> None:
    child = store.create_from_creator(
        "manager",
        CreateAgentArguments(id="child", allowed_tools=tools),
        expected_authority=store.authority("manager"),
    )
    assert child.config.subagents.enabled == ("delegate_task" in tools)
    assert child.authority.allowed_tools == tools


@pytest.mark.parametrize("target", ["manager", "existing"])
def test_duplicate_ids_never_modify_existing_bots(store: BotStore, target: str) -> None:
    store.save(Config(bot=BotConfig(id="existing", name="Original", instructions="Keep me")))
    before = store.list()
    with pytest.raises(ConfigurationError, match=f"ID 已存在: {target}"):
        store.create_from_creator(
            "manager",
            CreateAgentArguments(id=target, name="Replacement", instructions="Overwrite"),
            expected_authority=store.authority("manager"),
        )
    assert store.list() == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": ""},
        {"id": "../escape"},
        {"id": "a" * 129},
        {"name": ""},
        {"name": " \t\n "},
        {"name": "a" * 201},
        {"instructions": "a" * 16001},
        {"allowed_tools": ["bad-tool"]},
        {"allowed_tools": [""]},
        {"permissions": "unrestricted"},
        {"model": {"model": "other-model"}},
        {"api_key": "a secret must not be accepted"},
        {"api_key_env": "OTHER_API_KEY"},
        {"budget": {"max_tool_calls": 10000}},
        {"config": {}},
        {"subagents": {"enabled": True}},
    ],
)
def test_invalid_creation_arguments_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CreateAgentArguments.model_validate({"id": "child", **overrides})


def test_creation_arguments_are_revalidated_after_mutation(store: BotStore) -> None:
    arguments = CreateAgentArguments(id="child")
    arguments.name = "   "
    with pytest.raises(ValidationError):
        store.create_from_creator(
            "manager", arguments, expected_authority=store.authority("manager")
        )
    assert [bot.bot_id for bot in store.list()] == ["manager"]


def test_instructions_maximum_is_accepted(store: BotStore) -> None:
    child = store.create_from_creator(
        "manager",
        CreateAgentArguments(id="child", instructions="x" * 16000),
        expected_authority=store.authority("manager"),
    )
    assert child.config.bot is not None and len(child.config.bot.instructions) == 16000


def test_ungranted_tools_are_rejected_without_inserting(store: BotStore) -> None:
    with pytest.raises(ConfigurationError, match="不能超出创建者权限: run_command"):
        store.create_from_creator(
            "manager",
            CreateAgentArguments(id="child", allowed_tools=("read_file", "run_command")),
            expected_authority=store.authority("manager"),
        )
    assert [bot.bot_id for bot in store.list()] == ["manager"]


@pytest.mark.parametrize(
    "changes",
    [
        {"bot_id": "other"},
        {"allowed_tools": ("create_agent",)},
        {"permission_mode": "read-only"},
        {"permission_version": 2},
        {"permission_digest": "wrong-digest"},
    ],
)
def test_entire_expected_authority_must_match(store: BotStore, changes: dict[str, Any]) -> None:
    mismatched = store.authority("manager").model_copy(update=changes)
    with pytest.raises(ConfigurationError, match="权限已变化"):
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=mismatched
        )
    assert [bot.bot_id for bot in store.list()] == ["manager"]


def test_revoke_and_restore_does_not_restore_old_creation_approval(store: BotStore) -> None:
    original = store.get("manager")
    revoked_config = original.config.model_copy(deep=True)
    assert revoked_config.bot is not None
    revoked_config.bot.allowed_tools = ("list_files", "read_file", "search_text")
    revoked = store.save(revoked_config)
    with pytest.raises(ConfigurationError, match="权限已变化"):
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=original.authority
        )
    with pytest.raises(ConfigurationError, match="没有 create_agent 写入权限"):
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=revoked.authority
        )
    restored = store.save(original.config)
    assert restored.authority.permission_digest == original.authority.permission_digest
    assert restored.authority.permission_version == 3
    with pytest.raises(ConfigurationError, match="权限已变化"):
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=original.authority
        )
    assert [bot.bot_id for bot in store.list()] == ["manager"]
    assert (
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=restored.authority
        ).bot_id
        == "child"
    )


def test_creator_latest_settings_are_inherited_without_authority_change(store: BotStore) -> None:
    original = store.get("manager")
    config = original.config.model_copy(deep=True)
    config.model.model = "updated-model"
    config.budget.max_model_requests = 5
    updated = store.save(config)
    assert updated.authority == original.authority
    child = store.create_from_creator(
        "manager", CreateAgentArguments(id="child"), expected_authority=original.authority
    )
    assert child.config.model.model == "updated-model"
    assert child.config.budget.max_model_requests == 5


def test_current_delegate_enablement_is_checked_even_when_authority_is_unchanged(
    store: BotStore,
) -> None:
    original = store.get("manager")
    config = original.config.model_copy(deep=True)
    config.subagents.enabled = False
    assert store.save(config).authority == original.authority
    with pytest.raises(ConfigurationError, match="不能授予 delegate_task"):
        store.create_from_creator(
            "manager",
            CreateAgentArguments(id="child", allowed_tools=("delegate_task",)),
            expected_authority=original.authority,
        )
    assert [bot.bot_id for bot in store.list()] == ["manager"]


@pytest.mark.parametrize("permissions", [None, "read-only", "ask-write"])
def test_read_only_creator_cannot_create_bots(store: BotStore, permissions: str | None) -> None:
    config = store.get("manager").config
    config.permissions = "read-only"
    creator = store.save(config)
    arguments = CreateAgentArguments.model_validate({"id": "child", "permissions": permissions})
    with pytest.raises(ConfigurationError, match="没有 create_agent 写入权限"):
        store.create_from_creator("manager", arguments, expected_authority=creator.authority)
    assert [bot.bot_id for bot in store.list()] == ["manager"]


def test_missing_creator_is_rejected(store: BotStore) -> None:
    with pytest.raises(ConfigurationError, match="Bot 不存在: missing"):
        store.create_from_creator(
            "missing",
            CreateAgentArguments(id="child"),
            expected_authority=store.authority("manager"),
        )


def test_corrupted_creator_authority_is_rejected(store: BotStore) -> None:
    authority = store.authority("manager")
    with store.database.transaction() as connection:
        connection.execute("UPDATE bots SET permission_digest='corrupted' WHERE bot_id='manager'")
    with pytest.raises(PersistenceError, match="配置与权限摘要不一致"):
        store.create_from_creator(
            "manager", CreateAgentArguments(id="child"), expected_authority=authority
        )
    with store.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1


def test_concurrent_duplicate_creation_has_one_winner(store: BotStore) -> None:
    authority = store.authority("manager")

    def create(name: str) -> str:
        try:
            store.create_from_creator(
                "manager",
                CreateAgentArguments(id="child", name=name),
                expected_authority=authority,
            )
        except ConfigurationError as error:
            assert "Bot ID 已存在: child" in str(error)
            return "duplicate"
        return name

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, ("First", "Second")))
    assert results.count("duplicate") == 1
    winner = next(name for name in results if name != "duplicate")
    child = store.get("child")
    assert child.config.bot is not None and child.config.bot.name == winner
    assert child.authority.permission_version == 1
