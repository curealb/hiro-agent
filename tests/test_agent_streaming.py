"""Streaming previews must not become durable messages or tools before completion."""

import asyncio
import io
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, ModelConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelService,
    RunStatus,
    ToolCall,
    Usage,
)
from hiro_agent.core import EventBus
from hiro_agent.errors import ModelError
from hiro_agent.models import CompatibleModel, ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.run_display import RunDisplay
from hiro_agent.session import MemorySession
from hiro_agent.tools import ToolExecutor, build_file_tools


def response(
    text: str,
    *,
    calls: list[ToolCall] | None = None,
    finish: str = "stop",
    usage: Usage | None = None,
) -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=calls or []),
        finish_reason=finish,
        usage=usage,
    )


def write_call() -> ToolCall:
    return ToolCall(
        id="write-1",
        name="write_file",
        arguments=json.dumps({"path": "streamed.txt", "content": "complete"}),
    )


class GatedStreamingModel:
    def __init__(self, result: ModelResponse | Exception) -> None:
        self.result = result
        self.partial_sent = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned_up = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("Streaming-capable models should use complete_stream")

    async def complete_stream(
        self, request: ModelRequest, on_delta: Callable[[str], None]
    ) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) > 1:
            on_delta("done")
            return response(
                "done", usage=Usage(input_tokens=10, output_tokens=2, cached_input_tokens=0)
            )
        try:
            on_delta("partial ")
            self.partial_sent.set()
            await self.release.wait()
            if isinstance(self.result, Exception):
                raise self.result
            on_delta("answer")
            return self.result
        finally:
            self.cleaned_up.set()

    async def close(self) -> None:
        pass


def harness(
    tmp_path: Path,
    model: ModelService,
    *,
    config: BudgetConfig | None = None,
    events: EventBus | None = None,
) -> tuple[Agent, MemorySession, list[tuple[str, dict[str, Any]]]]:
    session = MemorySession()
    bus = events or EventBus()
    observed: list[tuple[str, dict[str, Any]]] = []
    bus.subscribe(lambda name, data: observed.append((name, data)))

    async def allow(tool: str, arguments: dict[str, Any]) -> bool:
        return True

    tools = ToolExecutor(
        build_file_tools(tmp_path, ToolsConfig()),
        session,
        PermissionPolicy("ask-write", allow),
        bus.emit,
    )
    agent = Agent(
        session,
        model,
        tools,
        PassthroughContext(),
        IdentityPreprocessor(),
        config or BudgetConfig(),
        bus,
    )
    return agent, session, observed


async def test_deltas_arrive_before_complete_response_and_tool_execution(tmp_path: Path) -> None:
    model = GatedStreamingModel(
        response(
            "partial answer",
            calls=[write_call()],
            finish="tool_calls",
            usage=Usage(input_tokens=20, output_tokens=4, cached_input_tokens=5),
        )
    )
    agent, session, observed = harness(tmp_path, model)
    durable_when_delta: list[list[str]] = []
    agent.events.subscribe(
        lambda name, data: (
            durable_when_delta.append([event.type for event in session.events])
            if name == "assistant.delta"
            else None
        )
    )
    task = asyncio.create_task(agent.run("write after the response is complete"))
    try:
        await asyncio.wait_for(model.partial_sent.wait(), 2)
        assert not task.done()
        assert [data["text"] for name, data in observed if name == "assistant.delta"] == [
            "partial "
        ]
        assert "model.requested" in durable_when_delta[0]
        assert not {"assistant.message", "tool.started"}.intersection(durable_when_delta[0])
        assert not any(name == "model.finished" for name, _ in observed)
        assert not (tmp_path / "streamed.txt").exists()
        assert [message.role for message in session.derive_messages()] == ["user"]

        model.release.set()
        result = await asyncio.wait_for(task, 2)
        assert result.status == RunStatus.COMPLETED
        assert result.output == "done"
        assert (tmp_path / "streamed.txt").read_text() == "complete"
        assistant_events = [event for event in session.events if event.type == "assistant.message"]
        assert [event.data["message"]["text"] for event in assistant_events] == [
            "partial answer",
            "done",
        ]
        assert not any(event.type == "assistant.delta" for event in session.events)
        assert len([name for name, _ in observed if name == "model.finished"]) == 2
        assert result.metrics.worker_steps == 2
        assert (result.metrics.input_tokens, result.metrics.output_tokens) == (30, 6)
        assert result.metrics.cached_input_tokens == 5
        assert result.metrics.tool_calls == 1
        requests = [data for name, data in observed if name == "model.requested"]
        assert len(requests) == 2
        assert len({item["attempt_id"] for item in requests}) == 2
        assert all(item["session_id"] == session.session_id for item in requests)
        assert all(item["run_id"] == result.run_id for item in requests)
        assert {data["attempt_id"] for name, data in observed if name == "assistant.delta"} == {
            item["attempt_id"] for item in requests
        }
    finally:
        model.release.set()
        agent.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


@pytest.mark.parametrize(
    "outcome,status,event_name",
    [
        ("cancel", RunStatus.CANCELED, "model.interrupted"),
        ("timeout", RunStatus.PAUSED, "model.interrupted"),
        ("failure", RunStatus.FAILED, "model.failed"),
    ],
)
async def test_interrupted_stream_discards_partial_history_and_never_runs_tools(
    tmp_path: Path, outcome: str, status: RunStatus, event_name: str
) -> None:
    model = GatedStreamingModel(
        ModelError("stream disconnected")
        if outcome == "failure"
        else response("partial answer", calls=[write_call()], finish="tool_calls")
    )
    budget = BudgetConfig(timeout_seconds=0.2 if outcome == "timeout" else 600)
    agent, session, observed = harness(tmp_path, model, config=budget)
    task = asyncio.create_task(agent.run("test interrupted streaming"))
    try:
        await asyncio.wait_for(model.partial_sent.wait(), 2)
        assert any(name == "assistant.delta" for name, _ in observed)
        if outcome == "cancel":
            agent.cancel()
        elif outcome == "failure":
            model.release.set()
        result = await asyncio.wait_for(task, 2)
        assert result.status == status
        assert result.output == ""
        assert model.cleaned_up.is_set()
        assert len(model.requests) == 1
        assert result.metrics.worker_steps == 0
        assert result.metrics.tool_calls == 0
        assert not (tmp_path / "streamed.txt").exists()
        assert not any(event.type == "assistant.message" for event in session.events)
        assert not any(message.role == "assistant" for message in session.derive_messages())
        interrupted = [event for event in session.events if event.type == event_name]
        assert len(interrupted) == 1
        assert [data["attempt_id"] for name, data in observed if name == event_name] == [
            interrupted[0].data["attempt_id"]
        ]
        assert not any(name == "model.finished" for name, _ in observed)
    finally:
        model.release.set()
        agent.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.close()


async def test_length_limited_stream_pauses_without_running_truncated_tool(tmp_path: Path) -> None:
    call = write_call().model_copy(update={"arguments": '{"path":"streamed.txt","content":'})
    model = GatedStreamingModel(response("partial answer", calls=[call], finish="length"))
    model.release.set()
    agent, session, observed = harness(tmp_path, model)
    try:
        result = await agent.run("test length limited stream")
        assert result.status == RunStatus.PAUSED
        assert result.output == "partial answer"
        assert result.metrics.worker_steps == 1
        assert result.metrics.tool_calls == 0
        assert not (tmp_path / "streamed.txt").exists()
        assert len([name for name, _ in observed if name == "model.finished"]) == 1
        assert len([event for event in session.events if event.type == "assistant.message"]) == 1
        assert json.loads(session.derive_messages()[-1].content)["error"] == "TOOL_NOT_STARTED"
    finally:
        await agent.close()


async def test_complete_only_model_remains_usable(tmp_path: Path) -> None:
    model = ScriptedModel([response("legacy answer")])
    agent, session, observed = harness(tmp_path, model)
    try:
        result = await agent.run("legacy custom model")
        assert result.status == RunStatus.COMPLETED
        assert result.output == "legacy answer"
        assert len(model.requests) == 1
        assert not any(name == "assistant.delta" for name, _ in observed)
        assert len([event for event in session.events if event.type == "assistant.message"]) == 1
    finally:
        await agent.close()


async def test_shared_streaming_model_keeps_parallel_agent_callbacks_isolated(
    tmp_path: Path,
) -> None:
    class SharedStreamingModel:
        def __init__(self) -> None:
            self.waiting = 0
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError("Expected streaming path")

        async def complete_stream(
            self, request: ModelRequest, on_delta: Callable[[str], None]
        ) -> ModelResponse:
            label = request.messages[-1].content
            on_delta(f"{label}:first ")
            self.waiting += 1
            if self.waiting == 2:
                self.both_started.set()
            await self.release.wait()
            on_delta(f"{label}:last")
            return response(f"{label}:first {label}:last")

        async def close(self) -> None:
            pass

    model = SharedStreamingModel()
    bus = EventBus()
    first, first_session, observed = harness(tmp_path, model, events=bus)
    second, second_session, _ = harness(tmp_path, model, events=bus)
    tasks = [asyncio.create_task(first.run("alpha")), asyncio.create_task(second.run("beta"))]
    try:
        await asyncio.wait_for(model.both_started.wait(), 2)
        model.release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        for label, session, result in zip(
            ("alpha", "beta"), (first_session, second_session), results, strict=True
        ):
            deltas = [
                data
                for name, data in observed
                if name == "assistant.delta" and data["session_id"] == session.session_id
            ]
            assert [data["text"] for data in deltas] == [f"{label}:first ", f"{label}:last"]
            assert {data["run_id"] for data in deltas} == {result.run_id}
            assert len({data["attempt_id"] for data in deltas}) == 1
            assert result.output == f"{label}:first {label}:last"
    finally:
        model.release.set()
        first.cancel()
        second.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await first.close()
        await second.close()


async def test_provider_sse_stream_reaches_display_and_waits_for_done_before_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIRO_STREAM_TEST_KEY", "offline-test-key")
    before_done = asyncio.Event()
    allow_done = asyncio.Event()
    closed = asyncio.Event()
    requests: list[dict[str, Any]] = []

    def frame(delta: dict[str, Any], finish: str | None = None) -> bytes:
        return event({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def event(data: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode()

    def usage(inputs: int, outputs: int, cached: int) -> bytes:
        return event(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": inputs,
                    "completion_tokens": outputs,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )

    class ToolStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield frame(
                {
                    "role": "assistant",
                    "content": "writing ",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "write-1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"path":"stream'},
                        }
                    ],
                }
            )
            yield frame(
                {
                    "content": "file",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'ed.txt","content":"complete"}'},
                        }
                    ],
                }
            )
            yield frame({}, "tool_calls")
            yield usage(20, 4, 5)
            before_done.set()
            await allow_done.wait()
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            closed.set()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        headers = {"content-type": "text/event-stream"}
        if len(requests) == 1:
            return httpx.Response(200, headers=headers, stream=ToolStream())
        return httpx.Response(
            200,
            headers=headers,
            content=frame({"role": "assistant", "content": "done"}, "stop")
            + usage(10, 2, 0)
            + b"data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = CompatibleModel(
            ModelConfig(model="stream-test", api_key_env="HIRO_STREAM_TEST_KEY"), client
        )
        agent, session, observed = harness(tmp_path, model)
        output = io.StringIO()
        display = RunDisplay(Console(file=output, width=100))
        agent.events.subscribe(display.observe)
        display.start()
        task = asyncio.create_task(agent.run("write only after the complete SSE response"))
        try:
            await asyncio.wait_for(before_done.wait(), 2)
            display.console.print(display.render())
            assert "writing file" in output.getvalue()
            assert not task.done()
            assert not (tmp_path / "streamed.txt").exists()
            assert not any(event.type == "assistant.message" for event in session.events)
            assert display.metrics.worker_steps == 0
            assert not any(name == "model.finished" for name, _ in observed)

            allow_done.set()
            result = await asyncio.wait_for(task, 2)
            assert result.status == RunStatus.COMPLETED
            assert result.output == "done"
            assert closed.is_set()
            assert (tmp_path / "streamed.txt").read_text() == "complete"
            assert len(requests) == 2
            assert all(body["stream"] is True for body in requests)
            assert all(body["stream_options"] == {"include_usage": True} for body in requests)
            assistant = [event for event in session.events if event.type == "assistant.message"]
            assert len(assistant) == 2
            assert json.loads(assistant[0].data["message"]["tool_calls"][0]["arguments"]) == {
                "path": "streamed.txt",
                "content": "complete",
            }
            assert result.metrics.worker_steps == 2
            assert result.metrics.tool_calls == 1
            assert (result.metrics.input_tokens, result.metrics.output_tokens) == (30, 6)
            assert result.metrics.cached_input_tokens == 5
            assert result.metrics.cache_reported_input_tokens == 30
            assert display.metrics == result.metrics
        finally:
            allow_done.set()
            agent.cancel()
            await asyncio.gather(task, return_exceptions=True)
            display.stop()
            await agent.close()
            await model.close()
