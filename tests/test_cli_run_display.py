"""The running indicator shares terminal ownership with prompts and final output."""

import asyncio
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from hiro_agent import bootstrap, cli
from hiro_agent.agent import Agent
from hiro_agent.config import Config
from hiro_agent.contracts import (
    AgentService,
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    RunResult,
    Usage,
)
from hiro_agent.errors import ModelError, RunCancelled
from hiro_agent.models import MockModel
from hiro_agent.run_display import RunDisplay


@pytest.fixture
def terminal_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    monkeypatch.setenv("TERM", "xterm-256color")
    console = Console(file=io.StringIO(), force_terminal=True, width=100, highlight=False)
    monkeypatch.setattr(cli, "console", console)
    return console


def mock_config() -> Config:
    config = Config()
    config.model.adapter = "mock"
    return config


async def test_running_display_yields_to_approvals_and_final_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_console: Console
) -> None:
    answers = iter(["demo", "y", "y", "/exit"])
    prompts: list[bool] = []
    requests = 0
    results: list[RunResult] = []
    complete = MockModel.complete
    show_result = cli.show_result

    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        assert not terminal_console._live_stack, "A prompt must have sole terminal ownership"
        prompts.append(approval)
        return next(answers)

    async def checked_complete(self: MockModel, request: ModelRequest) -> ModelResponse:
        nonlocal requests
        assert len(terminal_console._live_stack) == 1, "Show progress while awaiting the model"
        requests += 1
        return await complete(self, request)

    def checked_result(result: RunResult) -> None:
        assert not terminal_console._live_stack, "Remove the indicator before the final answer"
        results.append(result)
        show_result(result)

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(MockModel, "complete", checked_complete)
    monkeypatch.setattr(cli, "show_result", checked_result)
    await cli.chat_loop(tmp_path, mock_config(), None)

    assert prompts == [False, True, True, False]
    assert requests == 4
    assert len(results) == 1
    assert results[0].status == "completed"
    assert (tmp_path / "hiro-demo.txt").read_text() == "Hello from Hiro!\n"
    assert not terminal_console._live_stack


async def test_cannot_continue_stops_indicator_before_returning_to_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_console: Console
) -> None:
    answers = iter(["/continue", "/exit"])
    attempted = False
    continue_run = Agent.continue_run

    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        assert not terminal_console._live_stack
        return next(answers)

    async def checked_continue(self: Agent) -> RunResult:
        nonlocal attempted
        attempted = True
        assert terminal_console._live_stack
        return await continue_run(self)

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(Agent, "continue_run", checked_continue)
    await cli.chat_loop(tmp_path, mock_config(), None)

    assert attempted
    assert "没有可继续的未完成任务" in terminal_console.file.getvalue()
    assert not terminal_console._live_stack


async def test_unexpected_failure_removes_running_indicator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_console: Console
) -> None:
    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        assert not terminal_console._live_stack
        return "demo"

    async def fail(self: Agent, task: str) -> RunResult:
        assert terminal_console._live_stack
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(Agent, "run", fail)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        await cli.chat_loop(tmp_path, mock_config(), None)

    assert not terminal_console._live_stack


class CancelTarget:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.parametrize(
    "keys,approved",
    [("y\r", True), ("n\r", False), ("\r", False), ("\x04", False), ("\x03", None)],
)
async def test_approval_keyboard_takes_terminal_from_live_display(
    monkeypatch: pytest.MonkeyPatch,
    terminal_console: Console,
    keys: str,
    approved: bool | None,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    target = CancelTarget()
    display = RunDisplay(terminal_console)
    with create_pipe_input() as pipe:
        terminal = cli.Terminal()
        terminal.agent = cast(AgentService, target)
        terminal.run_display = display
        terminal._prompt = PromptSession(input=pipe, output=DummyOutput())
        rendered = asyncio.Event()
        live_during_prompt: list[bool] = []

        def on_render(_) -> None:
            live_during_prompt.append(bool(terminal_console._live_stack))
            rendered.set()

        terminal._prompt.app.before_render += on_render
        display.start()
        assert terminal_console._live_stack
        task = asyncio.create_task(terminal.approve("write_file", {"path": "demo.txt"}))
        try:
            await asyncio.wait_for(rendered.wait(), 2)
            assert not terminal_console._live_stack
            pipe.send_text(keys)
            if approved is None:
                with pytest.raises(RunCancelled):
                    await asyncio.wait_for(task, 2)
            else:
                assert await asyncio.wait_for(task, 2) is approved
            assert live_during_prompt and not any(live_during_prompt)
            assert target.cancelled is (approved is None)
            assert terminal_console._live_stack
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            display.stop()
            terminal.close()
        assert not terminal_console._live_stack


async def test_cancellation_during_approval_cleans_up_display_before_next_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_console: Console
) -> None:
    normal_answers = iter(["demo", "/exit"])
    approvals = 0

    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        nonlocal approvals
        assert not terminal_console._live_stack
        if approval:
            approvals += 1
            raise KeyboardInterrupt
        return next(normal_answers)

    monkeypatch.setattr(cli.Terminal, "read", read)
    await cli.chat_loop(tmp_path, mock_config(), None)

    assert approvals == 1
    assert not terminal_console._live_stack
    assert not (tmp_path / "hiro-demo.txt").exists()
    journal = next((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["data"]["status"] for event in events if event["type"] == "run.ended"] == [
        "canceled"
    ]


class PreviewStreamingModel:
    def __init__(self, *, fail: bool = False, chunks: tuple[str, str] = ("alpha", "omega")) -> None:
        self.fail = fail
        self.chunks = chunks
        self.first_delta = asyncio.Event()
        self.second_delta = asyncio.Event()
        self.allow_second = asyncio.Event()
        self.finish = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("Expected the streaming request path")

    async def complete_stream(
        self, request: ModelRequest, on_delta: Callable[[str], None]
    ) -> ModelResponse:
        on_delta(self.chunks[0])
        self.first_delta.set()
        await self.allow_second.wait()
        on_delta(self.chunks[1])
        self.second_delta.set()
        await self.finish.wait()
        if self.fail:
            raise ModelError("connection lost")
        return ModelResponse(
            assistant=AssistantMessage(text="".join(self.chunks)),
            finish_reason="stop",
            usage=Usage(input_tokens=12, output_tokens=4, cached_input_tokens=3),
        )

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("fail", [False, True])
async def test_cli_stream_preview_progresses_before_one_permanent_markdown_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_console: Console, fail: bool
) -> None:
    model = PreviewStreamingModel(fail=fail, chunks=("### alpha\n\n**ome", "ga**"))
    monkeypatch.setattr(bootstrap, "MockModel", lambda: model)
    answers = iter(["demo", "/exit"])
    terminals: list[cli.Terminal] = []
    permanent_messages: list[str] = []
    rendered_messages: list[str] = []
    show_assistant_message = cli.show_assistant_message

    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        terminals.append(self)
        return next(answers)

    def show_message(console: Console, text: str) -> None:
        permanent_messages.append(text)
        start = console.file.tell()
        show_assistant_message(console, text)
        rendered_messages.append(Text.from_ansi(console.file.getvalue()[start:]).plain)

    def preview() -> str:
        output = io.StringIO()
        display = terminals[0].run_display
        assert display is not None
        Console(file=output, width=100).print(display.render())
        return output.getvalue()

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(cli, "show_assistant_message", show_message)
    task = asyncio.create_task(cli.chat_loop(tmp_path, mock_config(), None))
    try:
        await asyncio.wait_for(model.first_delta.wait(), 2)
        assert "alpha" in preview()
        assert "omega" not in preview()
        assert terminal_console._live_stack
        assert permanent_messages == []

        model.allow_second.set()
        await asyncio.wait_for(model.second_delta.wait(), 2)
        assert "alpha" in preview() and "omega" in preview()
        assert "###" not in preview() and "**" not in preview()
        assert not task.done()
        assert permanent_messages == []
        journal = next((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        assert not any(event["type"] == "assistant.message" for event in events)

        model.finish.set()
        await asyncio.wait_for(task, 2)
        assert permanent_messages == ["".join(model.chunks)]
        assert len(rendered_messages) == 1
        assert rendered_messages[0].count("alpha") == rendered_messages[0].count("omega") == 1
        assert "###" not in rendered_messages[0] and "**" not in rendered_messages[0]
        assert ("输出未完成" in terminal_console.file.getvalue()) is fail
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        messages = [event for event in events if event["type"] == "assistant.message"]
        assert len(messages) == (0 if fail else 1)
        if messages:
            assert messages[0]["data"]["message"]["text"] == "".join(model.chunks)
        assert not terminal_console._live_stack
    finally:
        model.allow_second.set()
        model.finish.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("role", ["main", "explorer"])
def test_intermediate_assistant_event_renders_markdown_with_separate_child_prefix(
    terminal_console: Console, role: str
) -> None:
    cli.observe(
        "assistant.message",
        {
            "role": role,
            "task_id": "task-1",
            "session_id": "child-1",
            "text": "### 检查结果\n\n**通过**",
        },
    )
    rendered = Text.from_ansi(terminal_console.file.getvalue()).plain
    assert rendered.count("检查结果") == rendered.count("通过") == 1
    assert "### 检查结果" not in rendered and "**通过**" not in rendered
    assert ("[任务 task-1 · explorer · 子会话 child-1]" in rendered) is (role == "explorer")


@pytest.mark.parametrize("fail", [False, True])
async def test_streamed_plain_output_is_printed_once_and_partial_failure_is_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    monkeypatch.setattr(cli, "console", console)
    model = PreviewStreamingModel(fail=fail)
    model.allow_second.set()
    model.finish.set()
    monkeypatch.setattr(bootstrap, "MockModel", lambda: model)
    answers = iter(["demo", "/exit"])

    async def read(self: cli.Terminal, prompt: str, *, approval: bool = False) -> str:
        return next(answers)

    monkeypatch.setattr(cli.Terminal, "read", read)
    await cli.chat_loop(tmp_path, mock_config(), None)

    transcript = output.getvalue()
    assert transcript.count("alphaomega") == 1
    assert "\x1b" not in transcript
    assert ("输出未完成" in transcript) is fail
    assert ("[failed]" in transcript) is fail
    journal = next((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assistant_messages = [event for event in events if event["type"] == "assistant.message"]
    assert len(assistant_messages) == (0 if fail else 1)
