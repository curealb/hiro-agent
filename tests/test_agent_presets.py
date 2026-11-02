"""Default Agents are usable, missing-only, and committed as an atomic group."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hiro_agent.agent_management import MANAGER_INSTRUCTIONS
from hiro_agent.agent_presets import initialize_agents
from hiro_agent.bots import BotStore, permission_digest
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import TaskSubmission
from hiro_agent.kernel_db import KernelDatabase

IDS = ["default", "agent-manager", "code-writer"]
BASE_TOOLS = {
    "list_files",
    "read_file",
    "search_text",
    "write_file",
    "edit_file",
    "run_command",
    "list_agents",
    "send_message",
    "read_message",
}


@pytest.fixture
def store(tmp_path: Path) -> BotStore:
    return BotStore(KernelDatabase.for_workspace(tmp_path))


def source_config(bot_id: str | None = None, model: str = "source-model") -> Config:
    config = Config.model_validate(
        {
            "model": {
                "model": model,
                "base_url": "https://provider.example/v1",
                "api_key_env": "PRIVATE_PROVIDER_KEY",
                "request_timeout_seconds": 42,
                "stream": False,
                "extra_body": {"provider_metadata": {"private_token": "private-value"}},
            },
            "budget": {"max_model_requests": 9, "max_tool_calls": 18, "timeout_seconds": 80},
            "tools": {
                "max_output_chars": 2048,
                "max_file_bytes": 4096,
                "max_search_files": 17,
                "command_timeout_seconds": 23,
            },
            "subagents": {
                "max_concurrency": 3,
                "max_model_requests": 7,
                "max_tool_calls": 11,
                "max_output_chars": 1024,
                "roles": {"explorer": {"instructions": "Preserve role configuration"}},
            },
        }
    )
    if bot_id is not None:
        config.bot = BotConfig(
            id=bot_id,
            name=f"Existing {bot_id}",
            instructions="Preserve customized instructions",
            allowed_tools=("read_file",),
        )
    return config


@pytest.mark.parametrize("delegation", [False, True])
def test_exact_default_trio_inherits_runtime_settings(store: BotStore, delegation: bool) -> None:
    source = source_config("source")
    source.subagents.enabled = delegation
    original = source.model_dump_json()

    assert initialize_agents(store, source) == {"created": IDS, "existing": []}
    records = {bot.bot_id: bot for bot in store.list()}
    assert set(records) == set(IDS)
    assert {
        bot_id: record.config.bot.name
        for bot_id, record in records.items()
        if record.config.bot is not None
    } == {"default": "Hiro", "agent-manager": "Agent 管理员", "code-writer": "代码编写员"}
    for bot_id, record in records.items():
        config = record.config
        assert config.model == source.model
        assert config.budget == source.budget
        assert config.tools == source.tools
        assert config.subagents == source.subagents
        assert config.permissions == source.permissions
        expected = BASE_TOOLS | ({"delegate_task"} if delegation else set())
        if bot_id == "agent-manager":
            expected |= {"create_agent", "list_agents"}
        assert set(record.authority.allowed_tools) == expected
        assert record.authority.permission_version == 1
        assert record.authority.permission_digest == permission_digest(config)
        assert record.created_at == record.updated_at
    assert len({record.created_at for record in records.values()}) == 1
    manager = records["agent-manager"].config.bot
    writer = records["code-writer"].config.bot
    assert manager is not None and manager.instructions == MANAGER_INSTRUCTIONS
    assert writer is not None
    for responsibility in ("阅读", "修改", "验证", "测试", "审批", "如实报告"):
        assert responsibility in writer.instructions
    assert source.model_dump_json() == original


@pytest.mark.parametrize("missing", IDS)
def test_missing_only_preserves_existing_settings_and_history(
    store: BotStore, tmp_path: Path, missing: str
) -> None:
    existing_ids = [bot_id for bot_id in IDS if bot_id != missing]
    before = {}
    for bot_id in existing_ids:
        config = source_config(bot_id)
        store.save(config)
        assert config.bot is not None
        config.bot.allowed_tools = ("read_file", "search_text")
        before[bot_id] = store.save(config)
    tasks = TaskStore(store.database, store)
    task = tasks.submit(TaskSubmission(bot_id=existing_ids[0], text="Keep prior task"))
    journal = tmp_path / ".hiro-agent" / "sessions" / f"{task.session_id}.jsonl"
    journal.parent.mkdir(parents=True)
    history = b'{"prior": "history stays byte-for-byte"}\n'
    journal.write_bytes(history)

    assert initialize_agents(store) == {"created": [missing], "existing": existing_ids}
    all_records = store.list()
    assert initialize_agents(store) == {"created": [], "existing": IDS}
    assert store.list() == all_records
    assert {bot_id: store.get(bot_id) for bot_id in existing_ids} == before
    assert tasks.get(task.id) == task
    assert journal.read_bytes() == history


def test_explicit_source_only_changes_new_agents(store: BotStore) -> None:
    saved = store.save(source_config("default", "old-model"))
    source = source_config("explicit", "new-model")
    source.budget.max_model_requests = 31

    assert initialize_agents(store, source) == {
        "created": ["agent-manager", "code-writer"],
        "existing": ["default"],
    }
    assert store.get("default") == saved
    for bot_id in ("agent-manager", "code-writer"):
        assert store.get(bot_id).config.model == source.model
        assert store.get(bot_id).config.budget == source.budget


@pytest.mark.parametrize(
    ("saved_ids", "expected_source"),
    [
        (["z-source", "agent-manager", "default"], "default"),
        (["a-source", "agent-manager"], "agent-manager"),
        (["z-source", "a-source"], "a-source"),
    ],
)
def test_saved_source_priority(store: BotStore, saved_ids: list[str], expected_source: str) -> None:
    for bot_id in saved_ids:
        store.save(source_config(bot_id, f"model-for-{bot_id}"))

    result = initialize_agents(store)

    assert result["created"] == [bot_id for bot_id in IDS if bot_id not in saved_ids]
    assert result["existing"] == [bot_id for bot_id in IDS if bot_id in saved_ids]
    for bot_id in result["created"]:
        assert store.get(bot_id).config.model.model == f"model-for-{expected_source}"


def test_no_source_reports_configuration_options_without_writing(store: BotStore) -> None:
    with pytest.raises(ConfigurationError, match="--config.*HIRO_MODEL"):
        initialize_agents(store)
    assert store.list() == []


@pytest.mark.parametrize("model", ["", " \t\n"])
def test_blank_compatible_model_rejected_without_partial_creation(
    store: BotStore, model: str
) -> None:
    with pytest.raises(ConfigurationError, match="模型名称.*--config.*HIRO_MODEL"):
        initialize_agents(store, source_config(model=model))
    assert store.list() == []


def test_mock_bootstrap_does_not_need_model_name(store: BotStore) -> None:
    assert initialize_agents(store, Config(model=ModelConfig(adapter="mock"))) == {
        "created": IDS,
        "existing": [],
    }
    assert all(bot.config.model.adapter == "mock" for bot in store.list())


def test_read_only_source_rejects_missing_manager_without_creating_anything(
    store: BotStore,
) -> None:
    source = source_config()
    source.permissions = "read-only"
    with pytest.raises(ConfigurationError, match="管理员需要 ask-write"):
        initialize_agents(store, source)
    assert store.list() == []


def test_existing_manager_allows_read_only_defaults_without_overwriting_manager(
    store: BotStore,
) -> None:
    manager = store.save(source_config("agent-manager"))
    source = source_config()
    source.permissions = "read-only"

    assert initialize_agents(store, source) == {
        "created": ["default", "code-writer"],
        "existing": ["agent-manager"],
    }
    assert store.get("agent-manager") == manager
    for bot_id in ("default", "code-writer"):
        assert store.get(bot_id).config.permissions == "read-only"
        assert not store.authority(bot_id).allows("write_file", mutating=True)


def test_complete_trio_does_not_require_valid_source(store: BotStore) -> None:
    for bot_id in IDS:
        source = source_config(bot_id, "")
        source.permissions = "read-only"
        store.save(source)
    before = store.list()
    assert initialize_agents(store) == {"created": [], "existing": IDS}
    assert initialize_agents(store, Config()) == {"created": [], "existing": IDS}
    assert store.list() == before


def test_mutated_invalid_source_is_revalidated_without_leaking_config(store: BotStore) -> None:
    source = source_config()
    source.budget.max_model_requests = 0
    with pytest.raises(ConfigurationError, match="源配置字段不合法") as error:
        initialize_agents(store, source)
    assert "private-value" not in str(error.value)
    assert "PRIVATE_PROVIDER_KEY" not in str(error.value)
    assert store.list() == []


def test_database_insert_failure_rolls_back_entire_group(store: BotStore) -> None:
    with store.database.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER reject_manager BEFORE INSERT ON bots "
            "WHEN NEW.bot_id = 'agent-manager' "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(PersistenceError):
        initialize_agents(store, source_config())
    assert store.list() == []


def test_concurrent_initializers_create_group_once(store: BotStore) -> None:
    barrier = Barrier(2)

    def initialize(model: str) -> dict[str, list[str]]:
        barrier.wait(timeout=5)
        return initialize_agents(store, source_config(model=model))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(initialize, ("first-model", "second-model")))

    assert sorted(results, key=lambda result: len(result["created"])) == [
        {"created": [], "existing": IDS},
        {"created": IDS, "existing": []},
    ]
    records = store.list()
    assert len(records) == 3
    assert len({record.config.model.model for record in records}) == 1
    assert all(record.authority.permission_version == 1 for record in records)
