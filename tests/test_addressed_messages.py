"""Slash targets address the current Agent's communication, not navigation."""

import json

import pytest
from test_agent_communication import response, save_bot, tick

from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.tui_backend import TuiBackend
from hiro_agent.ui_commands import addressed_agent

CHOICES = [("sender", "通信测试发送方"), ("responder", "通信测试应答方"), ("review", "Code Review")]


@pytest.mark.parametrize(
    "text,target,payload",
    [
        ("/responder 重新测试一下", "responder", "重新测试一下"),
        ("/ responder 重新测试一下", "responder", "重新测试一下"),
        ("/通信测试应答方 这个", "responder", "这个"),
        ("/Code Review 检查文件\n保留换行", "review", "检查文件\n保留换行"),
    ],
)
def test_address_parser_supports_id_name_and_slash_space(text, target, payload):
    result = addressed_agent(text, CHOICES, ("/agent",))
    assert [recipient.bot_id for recipient in result.targets] == [target]
    assert result.text == payload


def test_navigation_and_invalid_addresses_are_not_messages():
    assert addressed_agent("/agent responder", CHOICES, ("/agent",)) is None
    assert addressed_agent("normal message", CHOICES, ()) is None
    with pytest.raises(ConfigurationError, match="内容"):
        addressed_agent("/responder", CHOICES, ())
    with pytest.raises(ConfigurationError, match="未知"):
        addressed_agent("/unknown message", CHOICES, ())
    with pytest.raises(ConfigurationError, match="不唯一"):
        addressed_agent("/Same test", [("a", "Same"), ("b", "Same")], ())


@pytest.mark.parametrize(
    "text,targets,payload",
    [
        ("/responder /review 一起检查", ["responder", "review"], "一起检查"),
        ("/ 通信测试应答方 / Code Review 一起检查", ["responder", "review"], "一起检查"),
        ("/responder /通信测试应答方 /review 检查", ["responder", "review"], "检查"),
        ("/review -- /tmp/report.md", ["review"], "/tmp/report.md"),
        ("/responder 检查 /review 的回复", ["responder"], "检查 /review 的回复"),
    ],
)
def test_multiple_targets_deduplicate_and_preserve_shared_message(text, targets, payload):
    address = addressed_agent(text, CHOICES, ())
    assert [target.bot_id for target in address.targets] == targets
    assert address.text == payload


def test_unknown_second_target_prevents_partial_submission(tmp_path):
    runtime = KernelRuntime(tmp_path)
    save_bot(runtime, "sender")
    save_bot(runtime, "responder")
    with pytest.raises(ConfigurationError, match="未知"):
        TuiBackend(tmp_path, kernel=runtime).submit("sender", "/responder /missing test")
    assert not runtime.tasks.list()


def test_backend_assigns_target_and_id_without_exposing_transport_prompt(tmp_path):
    runtime = KernelRuntime(tmp_path)
    for bot_id in ("sender", "responder"):
        save_bot(runtime, bot_id)
    backend = TuiBackend(tmp_path, kernel=runtime)
    task = backend.submit("sender", "/ responder 重新测试一下", idempotency_key="click-001")
    assert task.bot_id == "sender"
    assert task.metadata["target_agent_id"] == "responder"
    correlation = task.metadata["correlation_id"]
    assert correlation.startswith("comm-") and correlation in task.text
    assert "不需要再要求用户提供唯一标识" in task.text
    assert "重新测试一下" in task.text
    retry = backend.submit("sender", "/ responder 重新测试一下", idempotency_key="click-001")
    assert retry.id == task.id
    another = backend.submit("sender", "/responder 重新测试一下", idempotency_key="click-002")
    assert another.metadata["correlation_id"] != correlation
    assert backend.conversation("sender").entries[0].text == "/ responder 重新测试一下"
    assert not runtime.handoffs.list("sender", outgoing=True)  # Model has not executed yet.


async def test_addressed_message_executes_real_tools_and_reply_without_requesting_id(tmp_path):
    runtime = KernelRuntime(tmp_path)
    save_bot(runtime, "sender")
    save_bot(runtime, "responder", ())
    backend = TuiBackend(tmp_path, kernel=runtime)
    task = backend.submit("sender", "/responder ping", idempotency_key="addressed-test")
    correlation = task.metadata["correlation_id"]
    sender = ScriptedModel(
        [
            response(
                "send_message",
                {
                    "target_agent_id": "responder",
                    "correlation_id": correlation,
                    "payload": "ping",
                    "idempotency_key": correlation,
                },
            ),
            response(text="已发送，等待回复"),
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: sender)
    assert await tick(runtime)
    assert correlation in sender.requests[0].messages[-1].content
    [message] = runtime.handoffs.list("sender", outgoing=True)
    assert message.correlation_id == correlation and message.source_task_id == task.id
    responder = ScriptedModel(
        [
            response(
                text=json.dumps(
                    {
                        "correlation_id": correlation,
                        "payload": "pong",
                    }
                )
            )
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: responder)
    assert await tick(runtime)
    verifier = ScriptedModel(
        [
            response("read_message", {"message_id": message.id}),
            response(text="pong 核验通过"),
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: verifier)
    assert await tick(runtime)
    assert "pong" in verifier.requests[-1].messages[-1].content
    result = runtime.handoffs.result("sender", message.id)
    assert runtime.tasks.get(result.delivery_task_id).output == "pong 核验通过"


def test_addressed_submit_rejects_missing_authority_without_granting_tools(tmp_path):
    runtime = KernelRuntime(tmp_path)
    original = save_bot(runtime, "sender", ())
    save_bot(runtime, "responder")
    with pytest.raises(ConfigurationError, match="send_message"):
        TuiBackend(tmp_path, kernel=runtime).submit("sender", "/responder test")
    assert runtime.bots.get("sender").authority == original.authority
    assert not runtime.tasks.list()


@pytest.mark.parametrize("multiple", [False, True])
async def test_traditional_chat_routes_without_switching_or_running_local_model(
    tmp_path,
    monkeypatch,
    multiple,
):
    from hiro_agent.cli import Terminal, chat_loop

    runtime = KernelRuntime(tmp_path)
    sender = save_bot(runtime, "sender")
    save_bot(runtime, "responder")
    save_bot(runtime, "review")
    terminal = Terminal()
    message = "/ responder / review 一起检查" if multiple else "/ responder 重新测试一下"
    lines = iter([message, "/exit"])

    async def read(*args, **kwargs):
        return next(lines)

    starts = []
    monkeypatch.setattr(terminal, "read", read)
    monkeypatch.setattr(TuiBackend, "start_worker", lambda self: starts.append(True))
    assert await chat_loop(tmp_path, sender.config, None, terminal=terminal) is None
    [task] = runtime.tasks.list()
    assert task.bot_id == "sender"
    expected_targets = ["responder", "review"] if multiple else ["responder"]
    assert task.metadata["target_agent_ids"] == expected_targets
    assert starts == [True]
    assert not runtime.tasks.runs(task.id)


def batch_calls(name, arguments):
    return ModelResponse(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(id=f"{name}-{i}", name=name, arguments=json.dumps(args))
                for i, args in enumerate(arguments)
            ]
        ),
        finish_reason="tool_calls",
    )


class BatchVerifier(ScriptedModel):
    def __init__(self):
        super().__init__([])

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            text = next(m.content for m in request.messages if m.role == "user")
            inventory = json.loads(text.splitlines()[-1])
            assert inventory["requested_targets"] == ["responder", "review"]
            return batch_calls(
                "read_message",
                [{"message_id": m["message_id"]} for m in inventory["sent_messages"]],
            )
        values = [json.loads(m.content)["value"] for m in request.messages if m.role == "tool"]
        assert len(values) == 2
        assert {v["target_agent_id"] for v in values} == {"responder", "review"}
        assert all(not v["pending"] for v in values)
        failures = [v["target_agent_id"] for v in values if v["result"]["state"] != "succeeded"]
        return response(text="汇总：review 失败" if failures else "汇总：两位 Agent 均已回复")


@pytest.mark.parametrize("failed_recipient", [False, True])
async def test_multi_agent_real_tools_route_every_target_and_verify_all_results(
    tmp_path,
    failed_recipient,
):
    runtime = KernelRuntime(tmp_path)
    for bot_id in ("sender", "responder", "review"):
        save_bot(runtime, bot_id)
    backend = TuiBackend(tmp_path, kernel=runtime)
    original = "/responder /review 一起检查这个问题"
    task = backend.submit("sender", original, idempotency_key="multi-click")
    assert task.bot_id == "sender" and task.metadata["target_agent_ids"] == ["responder", "review"]
    correlations = task.metadata["target_correlations"]
    assert len(set(correlations.values())) == 2
    assert backend.submit("sender", original, idempotency_key="multi-click").id == task.id
    assert backend.conversation("sender").entries[0].text == original
    sender = ScriptedModel(
        [
            batch_calls(
                "send_message",
                [
                    {
                        "target_agent_id": target,
                        "correlation_id": correlation,
                        "payload": "一起检查这个问题",
                        "idempotency_key": correlation,
                    }
                    for target, correlation in correlations.items()
                ],
            ),
            response(text="已联系两位，等待分别回复"),
        ]
    )
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: sender)
    assert await tick(runtime)
    messages = runtime.handoffs.list("sender", outgoing=True)
    assert {m.target_bot_id for m in messages} == {"responder", "review"}
    assert all(m.source_task_id == task.id for m in messages)
    assert len({m.correlation_id for m in messages}) == 2
    reviewer = ScriptedModel(
        [
            RuntimeError("recipient unavailable")
            if failed_recipient
            else response(text="review 回复"),
        ]
    )
    responder = ScriptedModel([response(text="responder 回复")])
    runtime = KernelRuntime(
        tmp_path,
        model_factory=lambda config: responder if config.bot.id == "responder" else reviewer,
    )
    assert await tick(runtime)
    first = runtime.handoffs.result("sender", messages[0].id)
    assert first is not None
    assert runtime.handoffs.result("sender", messages[1].id) is None
    callback = runtime.tasks.get(first.delivery_task_id)
    assert all(m.id in callback.text for m in messages)
    assert "部分结果" in callback.text
    assert await tick(runtime)
    runtime = KernelRuntime(tmp_path, model_factory=lambda config: BatchVerifier())
    for _ in messages:
        assert await tick(runtime)
    expected = "汇总：review 失败" if failed_recipient else "汇总：两位 Agent 均已回复"
    for message in messages:
        result = runtime.handoffs.result("sender", message.id)
        assert runtime.tasks.get(result.delivery_task_id).output == expected
    assert not await tick(runtime)
