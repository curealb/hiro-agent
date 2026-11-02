"""Initialization resolves .env even when a saved Agent predates model setup."""

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from hiro_agent.bootstrap import create_harness
from hiro_agent.cli import app
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import TaskState, TaskSubmission
from hiro_agent.models import CompatibleModel
from hiro_agent.session import JsonlSession
from hiro_agent.tui_backend import TuiBackend


@pytest.fixture
def local_environment(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for name in ("HIRO_MODEL", "HIRO_BASE_URL", "HIRO_API_KEY"):
        # Track initially absent variables too: dotenv writes os.environ directly.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "HIRO_MODEL=env-model\nHIRO_BASE_URL=https://env.example/v1\nHIRO_API_KEY=test-secret\n"
    )
    return workspace


@pytest.mark.parametrize("entry", ["tui", "kernel"])
@pytest.mark.parametrize("saved_model", [None, "", "old-model"])
def test_init_reads_dotenv_and_preserves_saved_agent(
    local_environment: Path, entry: str, saved_model: str | None
) -> None:
    workspace = local_environment
    backend = TuiBackend(workspace)
    original = task = None
    if saved_model is not None:
        original = backend.kernel.bots.save(
            Config(
                model=ModelConfig(model=saved_model),
                bot=BotConfig(id="default", name="Custom Hiro", instructions="Keep me"),
            )
        )
        task = backend.kernel.tasks.submit(TaskSubmission(bot_id="default", text="Keep history"))
    if entry == "tui":
        result = backend.initialize_agents()
    else:
        response = CliRunner().invoke(app, ["kernel", "init-agents", "-w", str(workspace)])
        assert response.exit_code == 0, response.output
        result = json.loads(response.output)
    assert result["created"] == (
        ["default", "agent-manager", "code-writer"]
        if original is None
        else ["agent-manager", "code-writer"]
    )
    for bot_id in result["created"]:
        config = backend.kernel.bots.get(bot_id).config
        assert config.model.model == "env-model"
        assert config.model.base_url == "https://env.example/v1"
        assert "test-secret" not in config.model_dump_json()
    if original is not None:
        assert backend.kernel.bots.get("default") == original
        assert backend.kernel.tasks.get(task.id) == task
    before = backend.kernel.bots.list()
    assert backend.initialize_agents()["created"] == []
    assert backend.kernel.bots.list() == before


def test_init_prefers_exported_model(local_environment: Path, monkeypatch) -> None:
    monkeypatch.setenv("HIRO_MODEL", "shell-model")
    backend = TuiBackend(local_environment)
    backend.initialize_agents()
    assert all(bot.config.model.model == "shell-model" for bot in backend.kernel.bots.list())


def test_missing_model_mentions_dotenv_without_partial_creation(local_environment: Path) -> None:
    (local_environment / ".env").write_text("HIRO_API_KEY=test-secret\n")
    backend = TuiBackend(local_environment)
    original = backend.kernel.bots.save(Config(bot=BotConfig(id="default")))
    with pytest.raises(ConfigurationError, match=r"\.env.*HIRO_MODEL"):
        backend.initialize_agents()
    assert backend.kernel.bots.list() == [original]


@pytest.mark.parametrize("entry", ["worker", "resume"])
async def test_existing_agent_uses_dotenv_for_actual_requests(
    local_environment: Path, monkeypatch, entry: str
) -> None:
    backend = TuiBackend(local_environment)
    original = backend.kernel.bots.save(
        Config(model=ModelConfig(stream=False), bot=BotConfig(id="default", name="Hiro"))
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "hiro_agent.bootstrap.CompatibleModel", lambda config: CompatibleModel(config, client)
        )
        if entry == "worker":
            task = backend.kernel.tasks.submit(TaskSubmission(bot_id="default", text="hello"))
            worker = backend.kernel.worker(poll_interval=0.01)
            try:
                assert await worker.tick()
                result = backend.kernel.tasks.get(task.id)
                assert result.state == TaskState.SUCCEEDED
                assert result.output == "Hello!"
                assert backend.worker_running()
            finally:
                await worker.close()
        else:
            session = await JsonlSession.open(local_environment)
            await session.append("session.bot", original.authority.model_dump(mode="json"))
            session_id = session.session_id
            await session.close()
            harness = await create_harness(local_environment, Config(), session_id)
            try:
                result = await harness.agent.run("hello")
                assert result.output == "Hello!"
            finally:
                await harness.close()
    assert len(requests) == 1
    assert json.loads(requests[0].content)["model"] == "env-model"
    assert str(requests[0].url) == "https://env.example/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-secret"
    assert backend.list_bots()[0].model == "env-model"
    assert backend.kernel.bots.get("default") == original
