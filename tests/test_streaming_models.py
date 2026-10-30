"""Streaming boundaries: incremental display, complete tools, usage and cleanup."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from hiro_agent.config import ModelConfig
from hiro_agent.contracts import ModelRequest
from hiro_agent.errors import ContextLimitError, ModelError, RunCancelled
from hiro_agent.models import CompatibleModel


@pytest.fixture(autouse=True)
def model_key(monkeypatch):
    monkeypatch.setenv("HIRO_STREAM_TEST_KEY", "test-secret")


def config(**changes):
    return ModelConfig(model="test", api_key_env="HIRO_STREAM_TEST_KEY", **changes)


def event(payload):
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def delta(content=None, *, finish=None, **fields):
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, **fields},
                "finish_reason": finish,
            }
        ]
    }


DONE = b"data: [DONE]\n\n"
USAGE = {
    "choices": [],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 12},
    },
}


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def adapter_for(stream):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        yield CompatibleModel(config(), client), requests


async def test_text_is_delivered_before_later_chunks_and_usage_trailer():
    first_delta = asyncio.Event()
    release = asyncio.Event()

    class GatedStream(ByteStream):
        async def __aiter__(self):
            yield event(delta("你好", role="assistant", reasoning_content="private reasoning"))
            await release.wait()
            yield event(delta("！", finish="stop", reasoning_content=" more"))
            yield event(USAGE)
            yield DONE

    stream = GatedStream([])
    received = []

    def on_delta(text):
        received.append(text)
        first_delta.set()

    async with adapter_for(stream) as (adapter, requests):
        task = asyncio.create_task(adapter.complete_stream(ModelRequest(messages=[]), on_delta))
        try:
            await asyncio.wait_for(first_delta.wait(), 2)
            assert received == ["你好"]
            assert not task.done()
            release.set()
            response = await asyncio.wait_for(task, 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert response.assistant.text == "你好！"
    assert received == ["你好", "！"]
    assert response.assistant.provider_metadata == {"reasoning_content": "private reasoning more"}
    assert response.usage.cached_input_tokens == 12
    assert response.usage.input_tokens == 20
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert len(requests) == 1
    assert stream.closed


async def test_fragmented_utf8_crlf_comments_and_multiline_sse_data():
    payload = json.dumps(delta("中文🙂", finish="stop"), ensure_ascii=False, indent=2)
    wire = (
        "\ufeff: heartbeat\r\nevent: message\r\nid: 1\r\n"
        + "\r\n".join("data: " + line for line in payload.splitlines())
        + "\r\n\r\ndata: [DONE]\r\n\r\n"
    ).encode()
    stream = ByteStream([wire[index : index + 1] for index in range(len(wire))])
    received = []
    async with adapter_for(stream) as (adapter, _):
        response = await adapter.complete_stream(ModelRequest(messages=[]), received.append)
    assert received == ["中文🙂"]
    assert response.assistant.text == "中文🙂"
    assert response.usage is None


async def test_interleaved_tools_assemble_by_index_and_preserve_arguments():
    stream = ByteStream(
        [
            event(
                delta(
                    tool_calls=[
                        {
                            "index": 1,
                            "id": "second",
                            "type": "function",
                            "function": {"name": "search_", "arguments": '{"query":'},
                        },
                        {
                            "index": 0,
                            "id": "first",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        },
                    ]
                )
            ),
            event(
                delta(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": None,
                            "type": None,
                            "function": {"name": None, "arguments": '"a.py"}'},
                        },
                        {"index": 1, "function": {"name": "text", "arguments": '"你好"}'}},
                    ],
                    finish="tool_calls",
                )
            ),
            event(USAGE),
            DONE,
        ]
    )
    received = []
    async with adapter_for(stream) as (adapter, _):
        response = await adapter.complete_stream(ModelRequest(messages=[]), received.append)
    assert received == []
    assert [(call.id, call.name, call.arguments) for call in response.assistant.tool_calls] == [
        ("first", "read_file", '{"path":"a.py"}'),
        ("second", "search_text", '{"query":"你好"}'),
    ]
    assert response.finish_reason == "tool_calls"


async def test_length_retains_incomplete_tool_arguments():
    stream = ByteStream(
        [
            event(
                delta(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "one",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        }
                    ],
                    finish="length",
                )
            ),
            DONE,
        ]
    )
    async with adapter_for(stream) as (adapter, _):
        response = await adapter.complete(ModelRequest(messages=[]))
    assert response.finish_reason == "length"
    assert response.assistant.tool_calls[0].arguments == '{"path":'


@pytest.mark.parametrize(
    "chunks",
    [
        [event(delta("partial"))],
        [event(delta("partial", finish="stop"))],
        [event(delta("partial")), DONE],
        [DONE],
        [b"data: {bad json}\n\n", DONE],
        [b"data: \xff\n\n", DONE],
    ],
)
async def test_incomplete_or_invalid_streams_fail_without_retry(chunks):
    stream = ByteStream(chunks)
    async with adapter_for(stream) as (adapter, requests):
        with pytest.raises(ModelError):
            await adapter.complete(ModelRequest(messages=[]))
    assert len(requests) == 1
    assert stream.closed


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [{"index": True, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [{"index": 0, "delta": [], "finish_reason": "stop"}]},
        delta({"private": "secret"}, finish="stop"),
        delta("text", finish=True),
        delta(role="user", finish="stop"),
        delta(tool_calls={}),
        delta(tool_calls=[{"index": True}]),
        delta(tool_calls=[{"index": -1}]),
        delta(tool_calls=[{"index": 0, "type": "not-function"}]),
        delta(tool_calls=[{"index": 0, "id": 3}]),
        delta(tool_calls=[{"index": 0, "function": "bad"}]),
        delta(tool_calls=[{"index": 0, "function": {"name": 3}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": {}}}]),
        {**delta("text", finish="stop"), "usage": "bad"},
    ],
)
async def test_malformed_stream_fields_are_rejected(payload):
    async with adapter_for(ByteStream([event(payload), DONE])) as (adapter, _):
        with pytest.raises(ModelError):
            await adapter.complete(ModelRequest(messages=[]))


@pytest.mark.parametrize("context_limit", [False, True])
async def test_provider_errors_inside_stream_are_sanitized(context_limit):
    failure = {"message": "test-secret"}
    if context_limit:
        failure["code"] = "context_length_exceeded"
    stream = ByteStream([event({"error": failure})])
    async with adapter_for(stream) as (adapter, requests):
        with pytest.raises(ContextLimitError if context_limit else ModelError) as error:
            await adapter.complete(ModelRequest(messages=[]))
    assert "test-secret" not in str(error.value)
    assert len(requests) == 1
    assert stream.closed


async def test_response_limit_counts_all_events(monkeypatch):
    chunk = b": heartbeat\n\n"
    monkeypatch.setattr("hiro_agent.models._MAX_RESPONSE_BYTES", len(chunk) * 2)
    stream = ByteStream([chunk, chunk, chunk])
    async with adapter_for(stream) as (adapter, _):
        with pytest.raises(ModelError, match="大小"):
            await adapter.complete(ModelRequest(messages=[]))
    assert stream.closed


@pytest.mark.parametrize("exception", [RunCancelled("cancel"), ValueError("callback error")])
async def test_callback_failure_propagates_and_closes_stream(exception):
    def on_delta(text):
        raise exception

    stream = ByteStream([event(delta("text")), DONE])
    async with adapter_for(stream) as (adapter, _):
        with pytest.raises(type(exception)) as error:
            await adapter.complete_stream(ModelRequest(messages=[]), on_delta)
    assert error.value is exception
    assert stream.closed


async def test_cancellation_closes_response_while_waiting_for_next_chunk():
    seen = asyncio.Event()

    class WaitingStream(ByteStream):
        async def __aiter__(self):
            yield event(delta("partial"))
            await asyncio.Event().wait()

    stream = WaitingStream([])
    async with adapter_for(stream) as (adapter, _):
        task = asyncio.create_task(
            adapter.complete_stream(ModelRequest(messages=[]), lambda _: seen.set())
        )
        try:
            await asyncio.wait_for(seen.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert stream.closed


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.RemoteProtocolError])
async def test_network_failure_after_partial_text_closes_without_retry(failure):
    class FailedStream(ByteStream):
        async def __aiter__(self):
            yield event(delta("partial"))
            raise failure("test-secret")

    stream = FailedStream([])
    received = []
    async with adapter_for(stream) as (adapter, requests):
        with pytest.raises(ModelError) as error:
            await adapter.complete_stream(ModelRequest(messages=[]), received.append)
    assert received == ["partial"]
    assert "test-secret" not in str(error.value)
    assert len(requests) == 1
    assert stream.closed


async def test_shared_adapter_keeps_concurrent_callbacks_isolated():
    arrivals = 0
    both_arrived = asyncio.Event()

    class ConcurrentStream(ByteStream):
        def __init__(self, number):
            super().__init__([])
            self.number = number

        async def __aiter__(self):
            nonlocal arrivals
            yield event(delta(str(self.number)))
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await both_arrived.wait()
            yield event(delta("done", finish="stop"))
            yield DONE

    streams = []

    def handler(request):
        stream = ConcurrentStream(len(streams))
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    first, second = [], []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = CompatibleModel(config(), client)
        results = await asyncio.wait_for(
            asyncio.gather(
                adapter.complete_stream(ModelRequest(messages=[]), first.append),
                adapter.complete_stream(ModelRequest(messages=[]), second.append),
            ),
            2,
        )
    assert first == ["0", "done"]
    assert second == ["1", "done"]
    assert [result.assistant.text for result in results] == ["0done", "1done"]
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("stream_enabled", [True, False])
async def test_json_responses_and_explicit_stream_opt_out(stream_enabled):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "plain"}, "finish_reason": "stop"}
                ]
            },
        )

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await CompatibleModel(config(stream=stream_enabled), client).complete_stream(
            ModelRequest(messages=[]), received.append
        )
    assert response.assistant.text == "plain"
    assert received == []
    assert len(requests) == 1
    assert requests[0]["stream"] is stream_enabled
    assert ("stream_options" in requests[0]) is stream_enabled
