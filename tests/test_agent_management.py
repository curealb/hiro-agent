"""Permanent Agent creation through real approval, storage, and sidebar paths."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hiro_agent.agent_management import manager_config
from hiro_agent.bootstrap import create_harness
from hiro_agent.cli import app
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.contracts import (
    AssistantMessage,
    ModelResponse,
    RunStatus,
    SessionEvent,
    ToolCall,
)
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import (
    InteractionReply,
    InteractionState,
    TaskState,
    TaskSubmission,
    canonical_arguments,
)
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.session import JsonlSession
from hiro_agent.tui_backend import TuiBackend


def response(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason="tool_calls" if calls else "stop",
    )


def call(name: str, arguments: dict, call_id: str = "call") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=canonical_arguments(arguments))


def creation_call() -> ToolCall:
    return call(
        "create_agent",
        {
            "id": "reviewer",
            "name": "代码审查员",
            "instructions": "Read source files and report concrete review findings.",
            "allowed_tools": ["read_file"],
            "permissions": "read-only",
        },
        "create-reviewer",
    )


def source_config() -> Config:
    return Config(
        model=ModelConfig(
            adapter="mock",
            model="inherited-model",
            api_key_env="PRIVATE_PROVIDER_KEY",
            extra_body={"provider_metadata": {"private_token": "private-provider-value"}},
        ),
        bot=BotConfig(
            id="default",
            name="Default",
            instructions="Private source instructions.",
            allowed_tools=("read_file", "write_file"),
        ),
    )


def kernel(workspace: Path, model: ScriptedModel) -> KernelRuntime:
    runtime = KernelRuntime(workspace, model_factory=lambda config: model)
    runtime.bots.save(manager_config(source_config()))
    return runtime


async def tick(runtime: KernelRuntime) -> bool:
    worker = runtime.worker(poll_interval=0.01)
    try:
        return await worker.tick()
    finally:
        await worker.close()


async def events(runtime: KernelRuntime, session_id: str) -> list[SessionEvent]:
    session = await JsonlSession.open(runtime.workspace, session_id, create=False)
    try:
        return session.events
    finally:
        await session.close()


async def test_approved_creation_survives_restart_and_new_agent_runs_from_sidebar(
    tmp_path: Path,
) -> None:
    create = creation_call()
    first_model = ScriptedModel([response(create)])
    runtime = kernel(tmp_path, first_model)
    task = runtime.tasks.submit(TaskSubmission(bot_id="agent-manager", text="创建代码审查员"))

    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.state == TaskState.WAITING_APPROVAL
    assert waiting.interaction_id is not None and waiting.pending_operation is not None
    assert waiting.pending_operation.tool_name == "create_agent"
    assert json.loads(waiting.pending_operation.arguments_json) == json.loads(create.arguments)
    assert [bot.bot_id for bot in runtime.bots.list()] == ["agent-manager"]
    assert len(first_model.requests) == 1
    assert not any(event.type == "tool.started" for event in await events(runtime, task.session_id))

    runtime.interactions.reply(waiting.interaction_id, InteractionReply(approved=True))
    resumed_model = ScriptedModel([response(text="代码审查员已创建")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: resumed_model)
    assert await tick(restarted)
    assert restarted.tasks.get(task.id).state == TaskState.SUCCEEDED
    assert restarted.interactions.get(waiting.interaction_id).state == InteractionState.CONSUMED
    created = restarted.bots.get("reviewer")
    assert created.config.model == source_config().model
    assert created.config.bot is not None
    assert created.config.bot.instructions == json.loads(create.arguments)["instructions"]
    assert created.config.bot.allowed_tools == ("read_file",)
    assert created.config.permissions == "read-only"
    assert restarted.tasks.list(bot_id="reviewer") == []
    successful_results = [
        event.data
        for event in await events(restarted, task.session_id)
        if event.type == "tool.result" and event.data["ok"]
    ]
    assert len(successful_results) == 1
    assert successful_results[0]["value"] == {
        "bot_id": "reviewer",
        "name": "代码审查员",
        "allowed_tools": ["read_file"],
        "permissions": "read-only",
    }

    (tmp_path / "source.txt").write_text("independent review evidence", encoding="utf-8")
    reviewer_model = ScriptedModel(
        [
            response(call("read_file", {"path": "source.txt"}, "review-read")),
            response(text="Review complete: source.txt was inspected."),
        ]
    )
    reopened = KernelRuntime(tmp_path, model_factory=lambda config: reviewer_model)
    backend = TuiBackend(tmp_path, kernel=reopened)
    sidebar = {bot.bot_id: bot for bot in backend.list_bots()}
    assert sidebar["reviewer"].name == "代码审查员"
    assert sidebar["reviewer"].status == "idle"
    review_task = backend.submit("reviewer", "Review source.txt")
    assert await tick(reopened)
    done = reopened.tasks.get(review_task.id)
    assert done.state == TaskState.SUCCEEDED
    assert done.output == "Review complete: source.txt was inspected."
    assert done.session_id != task.session_id
    assert backend.conversation("reviewer").error is None
    assert "Read source files" in reviewer_model.requests[0].messages[0].content
    assert any(
        message.role == "tool" and "independent review evidence" in (message.content or "")
        for message in reviewer_model.requests[1].messages
    )
    assert {schema["function"]["name"] for schema in reviewer_model.requests[0].tools} == {
        "read_file"
    }


@pytest.mark.parametrize("change", ["reject", "revoke_restore"])
async def test_rejected_or_stale_creation_approval_creates_nothing(
    tmp_path: Path, change: str
) -> None:
    runtime = kernel(tmp_path, ScriptedModel([response(creation_call())]))
    task = runtime.tasks.submit(TaskSubmission(bot_id="agent-manager", text="create reviewer"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    runtime.interactions.reply(
        waiting.interaction_id, InteractionReply(approved=change != "reject")
    )
    if change == "revoke_restore":
        original = runtime.bots.get("agent-manager").config
        revoked = original.model_copy(deep=True)
        assert revoked.bot is not None
        revoked.bot.allowed_tools = ("read_file", "list_agents")
        runtime.bots.save(revoked)
        runtime.bots.save(original)

    model = ScriptedModel([response(text="must not execute")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)
    assert await tick(restarted)
    assert restarted.tasks.get(task.id).state == TaskState.FAILED
    assert [bot.bot_id for bot in restarted.bots.list()] == ["agent-manager"]
    assert model.requests == []
    assert restarted.interactions.get(waiting.interaction_id).state == InteractionState.CANCELLED


@pytest.mark.parametrize("restriction", ["missing_whitelist", "read_only"])
async def test_creation_requires_creator_whitelist_and_write_permission(
    tmp_path: Path, restriction: str
) -> None:
    model = ScriptedModel([response(creation_call())])
    runtime = kernel(tmp_path, model)
    config = runtime.bots.get("agent-manager").config
    assert config.bot is not None
    if restriction == "missing_whitelist":
        config.bot.allowed_tools = ("read_file", "list_agents")
    else:
        config.permissions = "read-only"
    runtime.bots.save(config)
    task = runtime.tasks.submit(TaskSubmission(bot_id="agent-manager", text="create reviewer"))

    assert await tick(runtime)
    failed = runtime.tasks.get(task.id)
    assert failed.state == TaskState.FAILED
    assert failed.interaction_id is None
    assert "permission_denied" in failed.reason
    assert [bot.bot_id for bot in runtime.bots.list()] == ["agent-manager"]
    assert "create_agent" not in {schema["function"]["name"] for schema in model.requests[0].tools}


async def test_listing_agents_is_read_only_and_returns_no_private_config(tmp_path: Path) -> None:
    model = ScriptedModel([response(call("list_agents", {})), response(text="listed")])
    runtime = kernel(tmp_path, model)
    other = source_config()
    other.permissions = "read-only"
    runtime.bots.save(other)
    config = runtime.bots.get("agent-manager").config
    config.permissions = "read-only"
    runtime.bots.save(config)
    task = runtime.tasks.submit(TaskSubmission(bot_id="agent-manager", text="list agents"))

    assert await tick(runtime)
    done = runtime.tasks.get(task.id)
    assert done.state == TaskState.SUCCEEDED and done.interaction_id is None
    results = [
        event.data
        for event in await events(runtime, task.session_id)
        if event.type == "tool.result"
    ]
    assert len(results) == 1 and results[0]["ok"]
    summaries = results[0]["value"]
    assert {summary["bot_id"] for summary in summaries} == {"agent-manager", "default"}
    assert all(
        set(summary) == {"bot_id", "name", "allowed_tools", "permissions"} for summary in summaries
    )
    output = json.dumps(results, ensure_ascii=False)
    for private in (
        "PRIVATE_PROVIDER_KEY",
        "private-provider-value",
        "Private source instructions",
    ):
        assert private not in output
    assert all(summary["permissions"] == "read-only" for summary in summaries)


async def test_duplicate_creation_fails_without_overwriting_existing_agent(tmp_path: Path) -> None:
    runtime = kernel(tmp_path, ScriptedModel([response(creation_call())]))
    original = source_config()
    original.bot = BotConfig(
        id="reviewer", name="Existing specialist", allowed_tools=("write_file",)
    )
    saved = runtime.bots.save(original)
    task = runtime.tasks.submit(TaskSubmission(bot_id="agent-manager", text="create reviewer"))
    assert await tick(runtime)
    waiting = runtime.tasks.get(task.id)
    assert waiting.interaction_id is not None
    runtime.interactions.reply(waiting.interaction_id, InteractionReply(approved=True))
    model = ScriptedModel([response(text="must not report success")])
    restarted = KernelRuntime(tmp_path, model_factory=lambda config: model)

    assert await tick(restarted)
    failed = restarted.tasks.get(task.id)
    assert failed.state == TaskState.FAILED and "已存在" in failed.reason
    assert restarted.bots.get("reviewer") == saved
    assert model.requests == []


@pytest.mark.parametrize("identity", ["manager", "ordinary", "unbound"])
async def test_bound_chat_exposes_management_tools_only_when_authorized(
    tmp_path: Path, identity: str
) -> None:
    config = manager_config(source_config()) if identity == "manager" else source_config()
    if identity == "unbound":
        config.bot = None
    model = ScriptedModel([response(text="ready")])
    harness = await create_harness(tmp_path, config, model=model)
    try:
        assert (await harness.agent.run("available capabilities")).status == RunStatus.COMPLETED
        names = {schema["function"]["name"] for schema in model.requests[0].tools}
        assert ({"create_agent", "list_agents"} <= names) == (identity == "manager")
        if identity != "manager":
            assert names.isdisjoint({"create_agent", "list_agents"})
    finally:
        await harness.close()


def test_manager_cli_inherits_settings_and_never_replaces_existing_bot(tmp_path: Path) -> None:
    runtime = KernelRuntime(tmp_path)
    source = source_config()
    original = runtime.bots.save(source)
    cloned = manager_config(source)
    assert source == original.config
    assert cloned.model == source.model and cloned.model is not source.model
    assert [bot.bot_id for bot in runtime.bots.list()] == ["default"]

    args = [
        "kernel",
        "agent-manager",
        "--source",
        "default",
        "--bot-id",
        "manager",
        "--name",
        "团队管理员",
        "-w",
        str(tmp_path),
    ]
    runner = CliRunner()
    installed = runner.invoke(app, args)
    assert installed.exit_code == 0, installed.output
    summary = json.loads(installed.output)
    assert summary["bot_id"] == "manager" and summary["name"] == "团队管理员"
    manager = runtime.bots.get("manager")
    assert manager.config.model == source.model
    assert manager.config.bot is not None and source.bot is not None
    assert set(manager.config.bot.allowed_tools) == {
        *source.bot.allowed_tools,
        "create_agent",
        "list_agents",
        "send_message",
        "read_message",
    }
    assert "create_agent" in manager.config.bot.instructions
    assert runtime.bots.get("default") == original

    repeated = runner.invoke(app, args)
    assert repeated.exit_code == 1
    assert runtime.bots.get("manager") == manager


def test_read_only_source_cannot_become_manager() -> None:
    source = source_config()
    source.permissions = "read-only"
    with pytest.raises(ConfigurationError, match="ask-write"):
        manager_config(source)
