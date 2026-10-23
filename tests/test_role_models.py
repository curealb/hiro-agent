"""Role routing, owned clients, offline operation and credential isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import hiro_agent.bootstrap as bootstrap
from hiro_agent.bootstrap import create_harness
from hiro_agent.config import Config, ModelConfig, SubagentsConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Message,
    ModelRequest,
    ModelResponse,
    RunStatus,
    ToolCall,
)
from hiro_agent.errors import ConfigurationError
from hiro_agent.models import MockModel, ScriptedModel
from hiro_agent.session import JsonlSession


def response(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason="tool_calls" if calls else "stop",
    )


def delegate(role: str) -> ToolCall:
    return ToolCall(
        id=role,
        name="delegate_task",
        arguments=json.dumps({"role": role, "task": f"Inspect source as {role}"}),
    )


def test_defaults_and_role_models_are_complete_configs() -> None:
    old = Config.model_validate({"subagents": {"enabled": True}})
    assert old.subagents.max_concurrency == 2
    assert old.subagents.roles.explorer.model is None
    assert old.subagents.roles.reviewer.instructions == ""
    config = Config.model_validate(
        {
            "model": {
                "model": "main",
                "base_url": "https://main.example/v1",
                "api_key_env": "MAIN_KEY",
                "request_timeout_seconds": 17,
            },
            "subagents": {"roles": {"explorer": {"model": {"model": "explorer"}}}},
        }
    )
    role = config.subagents.roles.explorer.model
    assert role is not None
    assert role.base_url == "https://api.openai.com/v1"
    assert role.api_key_env == "HIRO_API_KEY"
    assert role.request_timeout_seconds == 120
    for invalid in (0, 9):
        with pytest.raises(ValidationError):
            SubagentsConfig(max_concurrency=invalid)
    with pytest.raises(ValidationError):
        Config.model_validate({"subagents": {"roles": {"writer": {}}}})


async def test_roles_route_clients_append_instructions_and_close_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = {}

    class TrackingModel:
        def __init__(self, config):
            self.name = config.model
            self.requests = []
            self.closed = 0
            models[self.name] = self

        async def complete(self, request):
            self.requests.append(request)
            if self.name == "main":
                return (
                    response(delegate("explorer"), delegate("reviewer"))
                    if len(self.requests) == 1
                    else response(text="Both reports received")
                )
            return response(text=json.dumps({"summary": f"Report from {self.name}"}))

        async def close(self):
            self.closed += 1

    monkeypatch.setattr(bootstrap, "CompatibleModel", TrackingModel)
    config = Config.model_validate(
        {
            "model": {"model": "main"},
            "subagents": {
                "enabled": True,
                "roles": {
                    "explorer": {"model": {"model": "scout"}, "instructions": "CUSTOM SCOUT"},
                    "reviewer": {"model": {"model": "audit"}, "instructions": "CUSTOM AUDIT"},
                },
            },
        }
    )
    harness = await create_harness(tmp_path, config)
    try:
        result = await harness.agent.run("Independent inspection")
        assert result.status == RunStatus.COMPLETED
        assert result.metrics.model_requests == 4
        assert len(models["main"].requests) == 2
        for name, custom in (("scout", "CUSTOM SCOUT"), ("audit", "CUSTOM AUDIT")):
            requests = models[name].requests
            assert len(requests) == 1
            system = requests[0].messages[0].content
            assert custom in system
            assert "不能修改文件" in system
            assert "最终回答必须是纯 JSON" in system
            assert {tool["function"]["name"] for tool in requests[0].tools} == {
                "list_files",
                "read_file",
                "search_text",
            }
        assert all(model.closed == 0 for model in models.values())
        started = [e for e in harness.session.events if e.type == "delegation.started"]
        assert {e.data["model_id"] for e in started} == {"compatible:scout", "compatible:audit"}
        assert "https://" not in json.dumps([e.data for e in started])
    finally:
        await harness.close()
    assert all(model.closed == 1 for model in models.values())


async def test_unconfigured_roles_share_main_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = []

    class TrackingModel:
        def __init__(self, _config):
            self.closed = 0
            instances.append(self)

        async def close(self):
            self.closed += 1

    monkeypatch.setattr(bootstrap, "CompatibleModel", TrackingModel)
    harness = await create_harness(
        tmp_path, Config(model=ModelConfig(model="main"), subagents=SubagentsConfig(enabled=True))
    )
    assert len(instances) == 1
    assert all(model is instances[0] for model in harness.runtime.services["role_models"].values())
    await harness.close()
    assert instances[0].closed == 1


async def test_partial_role_initialization_failure_cleans_clients_and_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = []

    class TrackingModel:
        def __init__(self, config):
            if config.model == "bad":
                raise ConfigurationError("invalid role model")
            self.closed = 0
            instances.append(self)

        async def close(self):
            self.closed += 1

    monkeypatch.setattr(bootstrap, "CompatibleModel", TrackingModel)
    config = Config.model_validate(
        {
            "model": {"model": "main"},
            "subagents": {
                "enabled": True,
                "roles": {
                    "explorer": {"model": {"model": "scout"}},
                    "reviewer": {"model": {"model": "bad"}},
                },
            },
        }
    )
    with pytest.raises(ConfigurationError, match="invalid role"):
        await create_harness(tmp_path, config)
    assert len(instances) == 2 and all(model.closed == 1 for model in instances)
    session_id = next((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl")).stem
    session = await JsonlSession.open(tmp_path, session_id)
    await session.close()


@pytest.mark.parametrize(
    "role_config",
    [
        {"model": ""},
        {"model": "valid", "base_url": "https://user:password@example.test/v1"},
        {"model": "valid", "api_key_env": "not a variable"},
        {"model": "valid", "api_key_env": "HIRO_MISSING_ROLE_KEY"},
    ],
)
async def test_enabled_role_invalid_config_fails_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role_config: dict
) -> None:
    monkeypatch.setenv("HIRO_API_KEY", "test-key")
    monkeypatch.delenv("HIRO_MISSING_ROLE_KEY", raising=False)
    config = Config.model_validate(
        {
            "model": {"model": "main"},
            "subagents": {"enabled": True, "roles": {"reviewer": {"model": role_config}}},
        }
    )
    with pytest.raises(ConfigurationError):
        await create_harness(tmp_path, config)


async def test_mock_forces_all_roles_offline_without_reading_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_compatible(_config):
        pytest.fail("Mock mode must not initialize a credential-reading adapter")

    monkeypatch.setattr(bootstrap, "CompatibleModel", forbidden_compatible)
    config = Config.model_validate(
        {
            "model": {"adapter": "mock"},
            "subagents": {
                "enabled": True,
                "roles": {
                    "explorer": {"model": {"model": "", "api_key_env": "invalid env"}},
                    "reviewer": {"model": {"base_url": "invalid-url"}},
                },
            },
        }
    )
    harness = await create_harness(tmp_path, config)
    assert all(
        isinstance(model, MockModel) for model in harness.runtime.services["role_models"].values()
    )
    assert harness.runtime.services["model_ids"] == {
        "explorer": "mock:mock",
        "reviewer": "mock:mock",
    }
    await harness.close()


async def test_all_configured_credentials_excluded_from_command_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("LANG", "LC_ALL", "PATH"):
        monkeypatch.setenv(variable, "credential-do-not-forward")
    config = Config.model_validate(
        {
            "model": {"adapter": "mock", "api_key_env": "LANG"},
            "subagents": {
                "roles": {
                    "explorer": {"model": {"api_key_env": "PATH"}},
                    "reviewer": {"model": {"api_key_env": "LC_ALL"}},
                },
            },
        }
    )

    async def approve(*_args):
        return True

    harness = await create_harness(tmp_path, config, approval=approve)
    try:
        harness.agent.model = ScriptedModel(
            [
                response(
                    ToolCall(
                        id="environment",
                        name="run_command",
                        arguments=json.dumps(
                            {
                                "executable": sys.executable,
                                "args": [
                                    "-c",
                                    "import os,json; print(json.dumps(dict(os.environ)))",
                                ],
                            }
                        ),
                    )
                ),
                response(text="Environment checked"),
            ]
        )
        result = await harness.agent.run("Check credential exclusion")
        assert result.status == RunStatus.COMPLETED
        tool_result = next(e for e in harness.session.events if e.type == "tool.result")
        assert tool_result.data["ok"]
        environment = json.loads(tool_result.data["value"]["stdout"])
        assert not {"LANG", "PATH", "LC_ALL"}.intersection(environment)
    finally:
        await harness.close()


async def test_mock_parallel_reports_are_combined_and_each_result_checked() -> None:
    command = ToolCall(id="command", name="run_command", arguments="{}")
    request = ModelRequest(
        messages=[
            Message(role="user", content="demo"),
            Message(role="assistant", tool_calls=[command]),
            Message(role="tool", tool_call_id="command", content=json.dumps({"ok": True})),
        ],
        tools=[{"function": {"name": "delegate_task"}}],
    )
    model = MockModel()
    parallel = await model.complete(request)
    calls = parallel.assistant.tool_calls
    assert [json.loads(call.arguments)["role"] for call in calls] == ["explorer", "reviewer"]
    request.messages.append(Message(role="assistant", tool_calls=calls))
    for index, call in enumerate(calls):
        request.messages.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                content=json.dumps(
                    {"ok": True, "value": {"report": {"summary": f"evidence{index}"}}}
                ),
            )
        )
    answer = (await model.complete(request)).assistant.text
    assert "explorer：evidence0" in answer and "reviewer：evidence1" in answer
    request.messages[-2].content = json.dumps({"ok": False})
    rejected = (await model.complete(request)).assistant.text
    assert "停止" in rejected and "演示完成" not in rejected


async def test_mock_resumes_old_initial_explorer_journals() -> None:
    explorer = delegate("explorer")
    request = ModelRequest(
        messages=[
            Message(role="user", content="old demo"),
            Message(role="assistant", tool_calls=[explorer]),
            Message(
                role="tool",
                tool_call_id=explorer.id,
                content=json.dumps({"ok": True, "value": {"report": {"summary": "old report"}}}),
            ),
        ],
        tools=[{"function": {"name": "delegate_task"}}],
    )
    resumed = await MockModel().complete(request)
    assert resumed.assistant.tool_calls[0].name == "read_file"
