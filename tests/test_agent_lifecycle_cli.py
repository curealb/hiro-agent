"""User-facing commands clear context, remove identities, and rebuild presets."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hiro_agent import cli
from hiro_agent.bot_lifecycle import clear_bot_history
from hiro_agent.bot_tui import BotTUI
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.tui_backend import TuiBackend


def invoke(workspace: Path, *arguments: str, answer: str = ""):
    return CliRunner().invoke(
        cli.app, ["kernel", *arguments, "--workspace", str(workspace)], input=answer
    )


def seed(workspace: Path) -> KernelRuntime:
    runtime = KernelRuntime(workspace)
    runtime.bots.save(
        Config(model=ModelConfig(adapter="mock"), bot=BotConfig(id="default", name="Hiro"))
    )
    runtime.bots.save(Config(model=ModelConfig(adapter="mock"), bot=BotConfig(id="other")))
    runtime.tasks.submit(TaskSubmission(bot_id="default", text="discarded old task"))
    runtime.tasks.submit(TaskSubmission(bot_id="other", text="retained peer task"))
    runtime.memory.create("default", "old task-specific reference")
    runtime.memory.create("other", "peer reference")
    return runtime


@pytest.mark.parametrize("command", ["clear-history", "bot-delete"])
def test_destructive_cli_requires_confirmation_and_scopes_deletion(
    tmp_path: Path, command: str
) -> None:
    runtime = seed(tmp_path)
    original = runtime.bots.get("default")
    tasks = runtime.tasks.list()
    memories = runtime.memory.list("default")
    declined = invoke(tmp_path, command, "default", answer="n\n")
    assert declined.exit_code != 0
    assert runtime.bots.get("default") == original
    assert runtime.tasks.list() == tasks
    assert runtime.memory.list("default") == memories

    accepted = invoke(tmp_path, command, "default", "--yes")
    assert accepted.exit_code == 0, accepted.output
    result = json.loads(accepted.output)
    assert result["bot_id"] == "default"
    assert result["deleted"] == (command == "bot-delete")
    assert result["counts"]["tasks"] == result["counts"]["memories"] == 1
    assert runtime.tasks.list(bot_id="other") == [task for task in tasks if task.bot_id == "other"]
    assert runtime.memory.list("other")[0].content == "peer reference"
    if command == "clear-history":
        assert runtime.bots.get("default") == original
        assert runtime.tasks.list(bot_id="default") == []
        assert runtime.memory.list("default") == []
    else:
        with pytest.raises(ConfigurationError, match="不存在"):
            runtime.bots.get("default")


def test_confirmed_clear_and_initialize_preserve_existing_configuration(tmp_path: Path) -> None:
    runtime = seed(tmp_path)
    original = runtime.bots.get("default")
    cleared = invoke(tmp_path, "clear-history", "default", answer="y\n")
    assert cleared.exit_code == 0, cleared.output
    assert runtime.memory.list("default") == []

    initialized = invoke(tmp_path, "init-agents")
    assert initialized.exit_code == 0, initialized.output
    assert json.loads(initialized.output) == {
        "created": ["agent-manager", "code-writer"],
        "existing": ["default"],
    }
    assert runtime.bots.get("default") == original
    assert runtime.bots.get("code-writer").config.bot.name == "代码编写员"
    again = invoke(tmp_path, "init-agents")
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["created"] == []


def test_fresh_initialization_after_deleting_every_default(tmp_path: Path) -> None:
    first = invoke(tmp_path, "init-agents", "--mock")
    assert first.exit_code == 0, first.output
    ids = ["default", "agent-manager", "code-writer"]
    assert json.loads(first.output)["created"] == ids
    for bot_id in ids:
        deleted = invoke(tmp_path, "bot-delete", bot_id, "--yes")
        assert deleted.exit_code == 0, deleted.output
    assert KernelRuntime(tmp_path).bots.list() == []
    restored = invoke(tmp_path, "init-agents", "--mock")
    assert restored.exit_code == 0, restored.output
    assert json.loads(restored.output)["created"] == ids


def test_cli_can_retry_cleanup_when_a_crash_blocks_normal_kernel_startup(tmp_path: Path) -> None:
    runtime = seed(tmp_path)
    clear_bot_history(tmp_path, "other")
    staging = tmp_path / ".hiro-agent" / "history-cleanup"
    staging.mkdir()
    (staging / "manifest.json").write_text(
        json.dumps({"id": "interrupted-before-file-staging", "journals": []})
    )
    result = invoke(tmp_path, "clear-history", "default", "--yes")
    assert result.exit_code == 0, result.output
    assert not staging.exists()
    assert runtime.tasks.list(bot_id="default") == []
    assert runtime.memory.list("default") == []


def test_empty_tui_initializes_from_model_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HIRO_MODEL", "configured-model")
    monkeypatch.setenv("HIRO_BASE_URL", "https://provider.example/v1")
    backend = TuiBackend(tmp_path)
    assert backend.initialize_agents()["created"] == ["default", "agent-manager", "code-writer"]
    assert {bot.name for bot in backend.list_bots()} == {"Hiro", "Agent 管理员", "代码编写员"}
    assert all(bot.config.model.model == "configured-model" for bot in backend.kernel.bots.list())


def test_chat_roundtrip_after_tui_clear_starts_a_fresh_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            platform=sys.platform,
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    source = Config(model=ModelConfig(adapter="mock"), bot=BotConfig(id="default"))
    replies = iter(["/tui", "/exit"])
    opened = []
    create_harness = cli.create_harness

    async def read(*args, **kwargs):
        return next(replies)

    async def capture(*args, **kwargs):
        harness = await create_harness(*args, **kwargs)
        opened.append(harness.session.session_id)
        return harness

    async def clear_and_return(interface: BotTUI):
        result = await asyncio.to_thread(interface.backend.clear_history, "default")
        assert result["counts"]["sessions"] == 1
        interface.exit_action = "chat"

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(cli, "create_harness", capture)
    monkeypatch.setattr(BotTUI, "run", clear_and_return)
    asyncio.run(cli._run_interfaces(tmp_path, source, None, start_worker=False))
    assert len(opened) == 2 and opened[0] != opened[1]
    directory = tmp_path / ".hiro-agent" / "sessions"
    assert not (directory / f"{opened[0]}.jsonl").exists()
    assert (directory / f"{opened[1]}.jsonl").is_file()
