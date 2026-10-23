"""Provider boundary checks do not require network access or real credentials."""

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from hiro_agent.config import ModelConfig
from hiro_agent.contracts import AssistantMessage, Message, ModelRequest, ModelResponse, ToolCall
from hiro_agent.errors import ConfigurationError, ContextLimitError, ModelError
from hiro_agent.models import CompatibleModel, MockModel, ScriptedModel

SECRET = "test-secret-never-in-errors"


@pytest.fixture(autouse=True)
def model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIRO_TEST_KEY", SECRET)


def config(**kwargs: Any) -> ModelConfig:
    return ModelConfig(model="test-model", api_key_env="HIRO_TEST_KEY", **kwargs)


def completion(
    *, calls: list[dict[str, Any]] | None = None, finish_reason: str = "stop"
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": "hello", "tool_calls": calls or []},
            }
        ],
    }


def wire_call(call_id: str = "call-1", arguments: str = '{"path":"demo.txt"}') -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read_file", "arguments": arguments},
    }


async def test_compatible_preserves_text_calls_usage_and_api_prefix() -> None:
    calls = [wire_call("first"), wire_call("second", "{invalid json")]
    payload = completion(calls=calls, finish_reason="tool_calls")
    payload["choices"][0]["message"]["reasoning_content"] = "private provider detail"
    payload["usage"] = {"prompt_tokens": 14, "completion_tokens": 7, "total_tokens": 21}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    schema = {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object"}},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = CompatibleModel(
            config(
                base_url="https://example.test/custom/v1/",
                request_timeout_seconds=7,
                extra_body={"temperature": 0.2},
            ),
            client,
        )
        result = await adapter.complete(
            ModelRequest(
                messages=[
                    Message(role="system", content="system"),
                    Message(role="user", content="task"),
                ],
                tools=[schema],
            )
        )
        assert result.assistant.text == "hello"
        assert [call.id for call in result.assistant.tool_calls] == ["first", "second"]
        assert result.assistant.tool_calls[1].arguments == "{invalid json"
        assert result.assistant.provider_metadata == {
            "reasoning_content": "private provider detail"
        }
        assert result.finish_reason == "tool_calls"
        assert result.usage is not None
        assert (result.usage.input_tokens, result.usage.output_tokens) == (14, 7)
        assert str(requests[0].url) == "https://example.test/custom/v1/chat/completions"
        assert requests[0].headers["authorization"] == f"Bearer {SECRET}"
        assert requests[0].extensions["timeout"]["read"] == 7
        assert json.loads(requests[0].content) == {
            "model": "test-model",
            "temperature": 0.2,
            "stream": False,
            "n": 1,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            "tools": [schema],
        }
        await adapter.complete(
            ModelRequest(
                messages=[
                    Message(
                        role="assistant",
                        content=result.assistant.text,
                        tool_calls=result.assistant.tool_calls,
                    ),
                    Message(role="tool", content='{"ok":true}', tool_call_id="first"),
                    Message(role="tool", content='{"ok":false}', tool_call_id="second"),
                ]
            )
        )
        replay = json.loads(requests[1].content)["messages"]
        assert replay[0] == {"role": "assistant", "content": "hello", "tool_calls": calls}
        assert replay[1]["tool_call_id"] == "first"
        assert "reasoning_content" not in requests[1].content.decode()
        await adapter.close()
        assert not client.is_closed


async def test_length_keeps_incomplete_arguments_and_missing_usage() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json=completion(calls=[wire_call(arguments='{"path":')], finish_reason="length"),
            )
        )
    ) as client:
        result = await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))
    assert result.finish_reason == "length"
    assert result.assistant.tool_calls[0].arguments == '{"path":'
    assert result.usage is None


async def test_continuation_metadata_stays_out_of_provider_protocol() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await CompatibleModel(config(), client).complete(
            ModelRequest(
                messages=[
                    Message(role="user", content="continue", continuation=True),
                ]
            )
        )
    assert json.loads(requests[0].content)["messages"] == [{"role": "user", "content": "continue"}]


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.test/v1",
        "https:///v1",
        "https://user:secret@example.test/v1",
        "https://example.test/v1?key=secret",
        "https://example.test/v1#fragment",
        "https://example.test:invalid/v1",
        "https://example.test:70000/v1",
        "https://exam ple.test/v1",
        "https://example.test/v1?",
        "https://example.test/v1#",
    ],
)
def test_rejects_invalid_base_url_without_echoing_it(base_url: str) -> None:
    with pytest.raises(ConfigurationError) as error:
        CompatibleModel(config(base_url=base_url))
    assert base_url not in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("base_url", ["http://localhost:8000/v1", "https://example.test/v1"])
def test_key_is_required_for_all_endpoints(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIRO_TEST_KEY")
    with pytest.raises(ConfigurationError):
        CompatibleModel(config(base_url=base_url))


def test_validates_model_env_name_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError):
        CompatibleModel(ModelConfig(model="  ", api_key_env="HIRO_TEST_KEY"))
    with pytest.raises(ConfigurationError):
        CompatibleModel(ModelConfig(model="test", api_key_env="not a variable"))
    monkeypatch.setenv("HIRO_TEST_KEY", "secret\nheader")
    with pytest.raises(ConfigurationError) as error:
        CompatibleModel(config())
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("field", ["messages", "tools", "stream", "model", "n", "api_key"])
def test_reserved_extra_fields_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        config(extra_body={field: "override"})


@pytest.mark.parametrize("status", [307, 400, 401, 429, 500])
async def test_http_failures_do_not_leak_or_retry(status: int) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            status, json={"error": {"message": SECRET}}, headers={"location": "https://other.test"}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(ModelError) as error:
            await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))
    assert str(status) in str(error.value)
    assert SECRET not in str(error.value)
    assert requests == 1


@pytest.mark.parametrize("field", ["code", "type"])
async def test_context_limit_mapping(field: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                400, json={"error": {field: "context_length_exceeded", "message": SECRET}}
            )
        )
    ) as client:
        with pytest.raises(ContextLimitError) as error:
            await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_failures_do_not_leak_or_retry(failure: type[httpx.HTTPError]) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise failure(SECRET, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as error:
            await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__
    assert requests == 1


@pytest.mark.parametrize(
    "defect",
    [
        "no_choices",
        "multiple_choices",
        "wrong_role",
        "bad_content",
        "bad_finish",
        "calls_object",
        "bad_call_type",
        "empty_id",
        "duplicate_id",
        "parsed_arguments",
        "negative_usage",
        "bool_usage",
        "missing_usage_field",
    ],
)
async def test_protocol_errors_are_rejected(defect: str) -> None:
    payload = completion(calls=[wire_call()])
    message = payload["choices"][0]["message"]
    if defect == "no_choices":
        payload["choices"] = []
    elif defect == "multiple_choices":
        payload["choices"] *= 2
    elif defect == "wrong_role":
        message["role"] = "user"
    elif defect == "bad_content":
        message["content"] = {"secret": SECRET}
    elif defect == "bad_finish":
        payload["choices"][0]["finish_reason"] = None
    elif defect == "calls_object":
        message["tool_calls"] = {}
    elif defect == "bad_call_type":
        message["tool_calls"][0]["type"] = "custom"
    elif defect == "empty_id":
        message["tool_calls"][0]["id"] = " "
    elif defect == "duplicate_id":
        message["tool_calls"] *= 2
    elif defect == "parsed_arguments":
        message["tool_calls"][0]["function"]["arguments"] = {"path": "file"}
    elif defect == "negative_usage":
        payload["usage"] = {"prompt_tokens": -1, "completion_tokens": 1}
    elif defect == "bool_usage":
        payload["usage"] = {"prompt_tokens": True, "completion_tokens": 1}
    else:
        payload["usage"] = {"total_tokens": 1}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ModelError) as error:
            await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))
    assert SECRET not in str(error.value)


async def test_non_json_success_and_response_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    for body in [b"not json", b"x" * 33]:
        monkeypatch.setattr("hiro_agent.models._MAX_RESPONSE_BYTES", 32)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _, content=body: httpx.Response(200, content=content)
            )
        ) as client:
            with pytest.raises(ModelError):
                await CompatibleModel(config(), client).complete(ModelRequest(messages=[]))


@pytest.mark.parametrize(
    "message",
    [
        Message(role="tool", content="result"),
        Message(role="user", content="user", tool_call_id="unrelated"),
        Message(
            role="user", content="user", tool_calls=[ToolCall(id="one", name="f", arguments="{}")]
        ),
        Message(role="assistant", tool_calls=[ToolCall(id="one", name="f", arguments="{}")] * 2),
    ],
)
async def test_invalid_outgoing_protocol_does_not_send(message: Message) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("invalid request reached transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError):
            await CompatibleModel(config(), client).complete(ModelRequest(messages=[message]))


async def test_owned_client_is_closed() -> None:
    model = CompatibleModel(config())
    await model.close()
    assert model._client.is_closed


async def test_scripted_model_records_snapshots_raises_and_exhausts() -> None:
    response = ModelResponse(assistant=AssistantMessage(text="first"), finish_reason="stop")
    model = ScriptedModel([response, ModelError("scripted error")])
    request = ModelRequest(messages=[Message(role="user", content="original")])
    returned = await model.complete(request)
    returned.assistant.text = "changed"
    request.messages[0].content = "changed"
    assert model.requests[0].messages[0].content == "original"
    assert response.assistant.text == "first"
    with pytest.raises(ModelError, match="scripted error"):
        await model.complete(request)
    with pytest.raises(ModelError, match="耗尽"):
        await model.complete(request)
    assert len(model.requests) == 3
    await model.close()


def append_result(request: ModelRequest, response: ModelResponse, **result: Any) -> None:
    call = response.assistant.tool_calls[0]
    request.messages.extend(
        [
            Message(role="assistant", content=response.assistant.text, tool_calls=[call]),
            Message(role="tool", tool_call_id=call.id, content=json.dumps(result)),
        ]
    )


async def test_mock_demo_uses_history_and_checks_before_claiming_success() -> None:
    model = MockModel()
    request = ModelRequest(messages=[Message(role="user", content="demo")])
    read = await model.complete(request)
    assert read.assistant.tool_calls[0].name == "read_file"
    assert json.loads(read.assistant.tool_calls[0].arguments) == {"path": "hiro-demo.txt"}
    append_result(request, read, ok=False, error="tool_error")
    # A new instance must pick up the same stage from persisted messages.
    write = await MockModel().complete(request)
    assert write.assistant.tool_calls[0].name == "write_file"
    assert json.loads(write.assistant.tool_calls[0].arguments)["content"] == "Hello from Hiro!\n"
    append_result(request, write, ok=True, value={})
    request.messages.append(Message(role="user", content="continue", continuation=True))
    check = await model.complete(request)
    arguments = json.loads(check.assistant.tool_calls[0].arguments)
    assert check.assistant.tool_calls[0].name == "run_command"
    assert arguments["executable"] == "python3"
    assert arguments["args"][0] == "-c"
    assert "assert Path('hiro-demo.txt')" in arguments["args"][1]
    append_result(request, check, ok=True, value={"exit_code": 0})
    answer = await model.complete(request)
    assert answer.finish_reason == "stop"
    assert "演示完成" in (answer.assistant.text or "")
    assert (
        len(
            {
                read.assistant.tool_calls[0].id,
                write.assistant.tool_calls[0].id,
                check.assistant.tool_calls[0].id,
            }
        )
        == 3
    )
    request.messages.append(Message(role="user", content="next turn"))
    assert (await model.complete(request)).assistant.tool_calls[0].name == "read_file"
    await model.close()


@pytest.mark.parametrize(
    "tool,error",
    [
        ("read_file", "permission_denied"),
        ("write_file", "permission_denied"),
        ("run_command", "PERMISSION_DENIED"),
        ("write_file", "tool_error"),
        ("run_command", "tool_error"),
    ],
)
async def test_mock_does_not_claim_success_after_denial_or_failure(tool: str, error: str) -> None:
    request = ModelRequest(messages=[Message(role="user", content="demo")])
    response = ModelResponse(
        assistant=AssistantMessage(tool_calls=[ToolCall(id="previous", name=tool, arguments="{}")]),
        finish_reason="tool_calls",
    )
    append_result(request, response, ok=False, error=error)
    answer = await MockModel().complete(request)
    assert not answer.assistant.tool_calls
    assert "演示完成" not in (answer.assistant.text or "")
    assert "停止" in (answer.assistant.text or "")
