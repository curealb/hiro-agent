"""Offline providers, real message tools, SQLite, workers and reply wakeups."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hiro_agent.agent_communication import COMMUNICATION_TOOLS
from hiro_agent.agent_management import manager_config
from hiro_agent.bots import CreateAgentArguments
from hiro_agent.cli import app, with_default_bot
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.contracts import AssistantMessage, ModelRequest, ModelResponse, ToolCall
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_contracts import TaskState, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.session import read_session_events
from hiro_agent.tui_backend import TuiBackend

SENDER = "comm-test-sender"
RESPONDER = "comm-test-responder"


def response(name=None, arguments=None, *, text=None):
    return ModelResponse(
        assistant=AssistantMessage(
            text=text,
            tool_calls=(
                [ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments))]
                if name
                else []
            ),
        ),
        finish_reason="tool_calls" if name else "stop",
    )


def save_bot(runtime, bot_id, tools=COMMUNICATION_TOOLS):
    return runtime.bots.save(
        Config(
            model=ModelConfig(adapter="mock"),
            permissions="read-only",
            bot=BotConfig(id=bot_id, allowed_tools=tuple(tools)),
        )
    )


async def tick(runtime):
    worker = runtime.worker(poll_interval=0.01)
    try:
        await worker.recover()
        return await worker.tick()
    finally:
        await worker.close()


class ProbeSender(ScriptedModel):
    """Derive reads from actual tool receipts rather than a test-side relay."""

    def __init__(self):
        super().__init__([])

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        names = {tool["function"]["name"] for tool in request.tools}
        assert set(COMMUNICATION_TOOLS) <= names
        if len(self.requests) == 1:
            return response(
                "send_message",
                {
                    "target_agent_id": RESPONDER,
                    "correlation_id": "comm-001",
                    "payload": "ping",
                    "idempotency_key": "comm-001",
                },
            )
        result = json.loads(next(m.content for m in reversed(request.messages) if m.role == "tool"))
        assert result["ok"]
        value = result["value"]
        if len(self.requests) == 2:
            assert value["status"] == "accepted" and value["reply_pending"]
            return response("read_message", {"message_id": value["message_id"]})
        assert value["pending"] and value["result"] is None
        return response(text="消息已发送，等待实际回复；尚未核验通过。")


class ProbeResponder(ScriptedModel):
    def __init__(self, reply_id="comm-001", reply_payload="pong"):
        super().__init__([])
        self.reply_id = reply_id
        self.reply_payload = reply_payload

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text = next(m.content for m in request.messages if m.role == "user")
        envelope = json.loads(text.splitlines()[-1])
        assert envelope["source_agent_id"] == SENDER
        assert envelope["target_agent_id"] == RESPONDER
        assert envelope["correlation_id"] == "comm-001"
        assert envelope["payload"] == "ping"
        return response(
            text=json.dumps(
                {
                    "correlation_id": self.reply_id,
                    "payload": self.reply_payload,
                }
            )
        )


class ProbeVerifier(ScriptedModel):
    def __init__(self):
        super().__init__([])

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            text = next(m.content for m in request.messages if m.role == "user")
            message_id = re.search(r"message_id=([a-f0-9]+)", text)[1]
            return response("read_message", {"message_id": message_id})
        tool_result = json.loads(
            next(m.content for m in reversed(request.messages) if m.role == "tool")
        )
        assert tool_result["ok"]
        value = tool_result["value"]
        assert not value["pending"]
        assert value["source_agent_id"] == SENDER
        assert value["target_agent_id"] == RESPONDER
        assert value["correlation_id"] == "comm-001"
        result = value["result"]
        reply = json.loads(result["output"])
        passed = result["state"] == "succeeded" and reply == {
            "correlation_id": "comm-001",
            "payload": "pong",
        }
        return response(text="comm-001 通过" if passed else "comm-001 回复不匹配，未通过")


@pytest.mark.parametrize(
    "reply_id,payload,passed",
    [
        ("comm-001", "pong", True),
        ("comm-other", "pong", False),
        ("comm-001", "ping", False),
    ],
)
async def test_ping_pong_cross_restart_uses_actual_reply(
    tmp_path: Path,
    reply_id: str,
    payload: str,
    passed: bool,
):
    sender = ProbeSender()
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: sender)
    save_bot(runtime, SENDER)
    # Receiving and automatically replying does not require permission to initiate messages.
    save_bot(runtime, RESPONDER, ())
    task = TuiBackend(tmp_path, kernel=runtime).submit(SENDER, "向 responder 测试 comm-001")
    assert await tick(runtime)
    assert runtime.tasks.get(task.id).state == TaskState.SUCCEEDED
    assert "尚未核验" in runtime.tasks.get(task.id).output
    [message] = runtime.handoffs.list(SENDER, outgoing=True)
    assert message.correlation_id == "comm-001"
    assert runtime.handoffs.result(SENDER, message.id) is None

    responder = ProbeResponder(reply_id, payload)
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: responder)
    assert await tick(runtime)
    result = runtime.handoffs.result(SENDER, message.id)
    assert result is not None and result.delivery_task_id is not None
    assert json.loads(result.output) == {"correlation_id": reply_id, "payload": payload}

    verifier = ProbeVerifier()
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: verifier)
    assert await tick(runtime)
    delivered = runtime.tasks.get(result.delivery_task_id)
    assert delivered.output == ("comm-001 通过" if passed else "comm-001 回复不匹配，未通过")
    assert len(runtime.tasks.list()) == 3
    assert not await tick(runtime)  # Returning a reply never starts an echo loop.
    assert not runtime.interactions.list()
    assert runtime.tasks.chain(task.id).hops_used == 1
    events = read_session_events(tmp_path, task.session_id)
    assert {e.data["call"]["name"] for e in events if e.type == "tool.started"} == {
        "send_message",
        "read_message",
    }
    entries = TuiBackend(tmp_path, kernel=runtime).conversation(SENDER).entries
    assert any(e.text == delivered.output for e in entries)


def test_message_idempotency_authority_and_chain_limits(tmp_path: Path):
    runtime = KernelRuntime(tmp_path)
    save_bot(runtime, SENDER)
    save_bot(runtime, RESPONDER, ())
    source = runtime.tasks.submit(TaskSubmission(bot_id=SENDER, text="probe", chain_max_hops=1))
    args = (SENDER, source.id, RESPONDER, "ping")
    message = runtime.handoffs.send_message(*args, correlation_id="comm-001", idempotency_key="one")
    assert (
        runtime.handoffs.send_message(*args, correlation_id="comm-001", idempotency_key="one")
        == message
    )
    with pytest.raises(ConfigurationError, match="payload"):
        runtime.handoffs.send_message(*args, correlation_id="other", idempotency_key="one")
    with pytest.raises(ConfigurationError, match="hop limit"):
        runtime.handoffs.send_message(*args, correlation_id="other", idempotency_key="two")
    with pytest.raises(ConfigurationError, match="send_handoff"):
        runtime.handoffs.send(*args, idempotency_key="handoff")
    with pytest.raises(ConfigurationError, match="correlation"):
        runtime.handoffs.send_message(*args, correlation_id=" ", idempotency_key="blank")
    save_bot(runtime, "stranger", ("read_message",))
    with pytest.raises(ConfigurationError, match="belong"):
        runtime.handoffs.result("stranger", message.id)
    with pytest.raises(ConfigurationError, match="sender"):
        runtime.handoffs.result(RESPONDER, message.id)


async def test_unauthorized_tool_is_hidden_and_cannot_send(tmp_path: Path):
    model = ScriptedModel(
        [
            response(
                "send_message",
                {
                    "target_agent_id": RESPONDER,
                    "correlation_id": "comm-001",
                    "payload": "ping",
                    "idempotency_key": "one",
                },
            )
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: model)
    save_bot(runtime, SENDER, ())
    save_bot(runtime, RESPONDER, ())
    task = runtime.tasks.submit(TaskSubmission(bot_id=SENDER, text="probe"))
    assert await tick(runtime)
    assert not model.requests[0].tools
    assert runtime.tasks.get(task.id).state == TaskState.FAILED
    assert not runtime.handoffs.list(SENDER, outgoing=True)


def test_enable_existing_agents_and_manager_can_grant_communication(tmp_path: Path):
    runtime = KernelRuntime(tmp_path)
    original = save_bot(runtime, SENDER, ("read_file",))
    runner = CliRunner()
    args = ["kernel", "enable-communication", SENDER, "-w", str(tmp_path)]
    applied = runner.invoke(app, args)
    assert applied.exit_code == 0, applied.output
    assert set(json.loads(applied.output)) == {"bot_id", "name", "allowed_tools", "permissions"}
    updated = runtime.bots.get(SENDER)
    assert updated.config.bot.instructions == original.config.bot.instructions
    assert updated.config.model == original.config.model
    assert updated.config.permissions == "read-only"
    assert set(updated.authority.allowed_tools) == {"read_file", *COMMUNICATION_TOOLS}
    assert updated.authority.permission_version == original.authority.permission_version + 1
    assert runner.invoke(app, args).exit_code == 0
    assert runtime.bots.get(SENDER).authority == updated.authority

    source = Config(model=ModelConfig(adapter="mock"), bot=BotConfig(id="default"))
    manager = runtime.bots.save(manager_config(source))
    child = runtime.bots.create_from_creator(
        manager.bot_id,
        CreateAgentArguments(id="peer", allowed_tools=COMMUNICATION_TOOLS),
        expected_authority=manager.authority,
    )
    assert set(child.authority.allowed_tools) == set(COMMUNICATION_TOOLS)


def test_first_tui_identity_can_communicate_without_expanding_explicit_config():
    first = with_default_bot(Config())
    assert set(COMMUNICATION_TOOLS) <= set(first.bot.allowed_tools)
    explicit = Config(bot=BotConfig(id="reader", allowed_tools=("read_file",)))
    assert with_default_bot(explicit) == explicit
