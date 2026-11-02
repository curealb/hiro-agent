"""No-keypress refresh and real worker streaming across the journal boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from prompt_toolkit.input import create_pipe_input
from test_agent_streaming import harness
from test_bot_tui import ALPHA, BackendFake, TerminalOutput
from test_tui_acceptance import Terminal

from hiro_agent.bot_tui import BotTUI
from hiro_agent.budget import Budget
from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    ModelRequest,
    ModelResponse,
    RunScope,
    RunStatus,
    Usage,
)
from hiro_agent.errors import ModelError, PersistenceError
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.session import (
    JsonlSession,
    MemorySession,
    read_session_events,
    read_session_snapshot,
    recover_session,
)
from hiro_agent.tui_backend import ChatEntry, TuiBackend


async def wait_screen(
    terminal: Terminal,
    predicate: Callable[[], bool],
    action: Callable[[], object] = lambda: None,
) -> None:
    """Wait for actual rendering; never inject keys, redraws, or polling timers."""
    rendered = asyncio.Event()

    def after_render(_: object) -> None:
        if predicate():
            rendered.set()

    terminal.ui.app.after_render += after_render
    try:
        action()
        if predicate():
            return
        await asyncio.wait_for(rendered.wait(), 1.5)
    finally:
        terminal.ui.app.after_render -= after_render


@asynccontextmanager
async def running_terminal(backend: TuiBackend) -> AsyncIterator[Terminal]:
    with create_pipe_input() as pipe:
        output = TerminalOutput()
        ui = BotTUI(
            backend,
            input=pipe,
            output=output,
            auto_start_worker=False,
            refresh_interval=0.02,
        )
        terminal = Terminal(ui, pipe.send_text, output)
        # Resize polling must not accidentally supply the refresh heartbeat.
        ui.app.terminal_size_polling_interval = None
        task = asyncio.create_task(ui.run())
        try:
            await wait_screen(terminal, lambda: ui.details is not None)
            yield terminal
        finally:
            if ui.app.is_running:
                ui.app.exit()
            await asyncio.wait_for(task, 3)


async def test_polling_repaints_without_input_even_when_thread_wakeup_is_delayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BackendFake(tmp_path)
    async with running_terminal(cast(TuiBackend, backend)) as terminal:
        # Model the reported environment: executor callbacks are ready, but
        # socket notifications do not wake the terminal's selector.
        with monkeypatch.context() as patch:
            patch.setattr(asyncio.get_running_loop(), "_write_to_self", lambda: None)
            snapshot = backend.conversations[ALPHA]
            backend.conversations[ALPHA] = replace(
                snapshot,
                entries=(
                    *snapshot.entries,
                    ChatEntry(
                        "assistant",
                        "Background output without Enter",
                        snapshot.tasks[0].id,
                    ),
                ),
            )
            await wait_screen(
                terminal, lambda: "Background output without Enter" in terminal.screen
            )


class StreamingModel:
    def __init__(self, outcome: str = "success") -> None:
        self.next_chunk = asyncio.Event()
        self.finish = asyncio.Event()
        self.closed = asyncio.Event()
        self.outcome = outcome

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("must stream")

    async def complete_stream(
        self,
        request: ModelRequest,
        on_delta: Callable[[str], None],
    ) -> ModelResponse:
        try:
            on_delta("FIRST_CHUNK")
            await self.next_chunk.wait()
            on_delta(" SECOND_CHUNK")
            await self.finish.wait()
            if self.outcome == "failure":
                raise ModelError("connection lost")
            return ModelResponse(
                assistant=AssistantMessage(text="FIRST_CHUNK SECOND_CHUNK"),
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5, cached_input_tokens=5),
            )
        finally:
            self.closed.set()

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_real_worker_streams_without_enter_and_settles_preview_once(
    tmp_path: Path,
    outcome: str,
) -> None:
    model = StreamingModel(outcome)
    kernel = KernelRuntime(tmp_path, model_factory=lambda _: model)
    kernel.bots.save(
        Config(bot=BotConfig(id="writer", name="Writer"), model=ModelConfig(adapter="mock"))
    )
    backend = TuiBackend(tmp_path, kernel=kernel)
    task = backend.submit("writer", "stream the response")
    worker = kernel.worker(poll_interval=0.01)
    execution = asyncio.create_task(worker.tick())
    try:
        async with running_terminal(backend) as terminal:
            await wait_screen(terminal, lambda: "FIRST_CHUNK" in terminal.screen)
            assert "生成中" in terminal.screen
            assert not execution.done()
            events = read_session_snapshot(tmp_path, task.session_id)
            assert any(event.type == "assistant.delta" for event in events)
            assert not any(event.type == "assistant.message" for event in events)

            await wait_screen(
                terminal,
                lambda: "SECOND_CHUNK" in terminal.screen,
                model.next_chunk.set,
            )
            assert not execution.done()
            if outcome == "cancel":
                backend.cancel("writer", task.id)
            else:
                model.finish.set()
            assert await asyncio.wait_for(execution, 3)
            await wait_screen(terminal, lambda: "生成中" not in terminal.screen)
            assert terminal.screen.count("FIRST_CHUNK") == 1
            if outcome == "success":
                assert "未完成" not in terminal.screen
                assert "Token 15" in terminal.screen
            else:
                assert "输出中断（未完成）" in terminal.screen
            assert model.closed.is_set()
        events = read_session_events(tmp_path, task.session_id)
        completed = [event for event in events if event.type == "assistant.message"]
        assert len(completed) == (1 if outcome == "success" else 0)
        assert not any(event.type == "tool.started" for event in events)
        session = await JsonlSession.open(tmp_path, task.session_id, create=False)
        try:
            assistant_messages = [
                message for message in session.derive_messages() if message.role == "assistant"
            ]
            assert len(assistant_messages) == (1 if outcome == "success" else 0)
        finally:
            await session.close()
    finally:
        model.next_chunk.set()
        model.finish.set()
        await worker.close()
        await asyncio.gather(execution, return_exceptions=True)


async def test_preview_write_failure_closes_provider_and_stops_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StreamingModel()
    agent, session, _ = harness(tmp_path, model)
    append = session.append

    async def fail_preview(event_type, data, run_id=None):
        if event_type == "assistant.delta":
            raise PersistenceError("disk full")
        return await append(event_type, data, run_id)

    monkeypatch.setattr(session, "append", fail_preview)
    try:
        result = await asyncio.wait_for(
            agent.run(
                "stream",
                scope=RunScope(
                    run_id="run",
                    budget=Budget(BudgetConfig()),
                    cancellation=Cancellation(),
                    root_session_id=session.session_id,
                    root_run_id="run",
                    persistent_task_id="task",
                ),
            ),
            2,
        )
        assert result.status == RunStatus.FAILED
        assert result.persistence_failed
        assert model.closed.is_set()
        assert not any(
            event.type in {"assistant.message", "tool.started"} for event in session.events
        )
    finally:
        await agent.close()


@pytest.mark.parametrize("invalid", ["unknown", "wrong_run", "closed", "non_text", "ended_run"])
async def test_stream_preview_rejects_invalid_or_finished_attempts(invalid: str) -> None:
    session = MemorySession()
    await session.append("run.started", {}, "run")
    await session.append("run.started", {}, "other-run")
    await session.append("model.requested", {"attempt_id": "attempt"}, "run")
    data = {"attempt_id": "attempt", "text": "preview"}
    run_id = "run"
    if invalid == "unknown":
        data["attempt_id"] = "unknown"
    elif invalid == "wrong_run":
        run_id = "other-run"
    elif invalid == "closed":
        await session.append("model.interrupted", {"attempt_id": "attempt"}, "run")
    elif invalid == "non_text":
        data["text"] = None
    else:
        await session.append(
            "run.ended", {"status": "paused", "reason": "stop", "metrics": {}}, "run"
        )
    with pytest.raises(PersistenceError, match="streaming delta"):
        await session.append("assistant.delta", data, run_id)


async def test_recovery_keeps_partial_preview_out_of_context_and_usage(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "interrupted-stream")
    try:
        await session.append("run.started", {}, "run")
        await session.append("task.received", {"text": "task"}, "run")
        await session.append("model.requested", {"attempt_id": "attempt"}, "run")
        await session.append(
            "assistant.delta", {"attempt_id": "attempt", "text": "unfinished"}, "run"
        )
    finally:
        await session.close()
    session = await JsonlSession.open(tmp_path, "interrupted-stream", create=False)
    try:
        await recover_session(session)
        assert [message.role for message in session.derive_messages()] == ["user"]
        assert any(event.type == "model.interrupted" for event in session.events)
        ended = next(event for event in session.events if event.type == "run.ended")
        assert ended.data["metrics"]["worker_steps"] == 0
        assert ended.data["metrics"]["input_tokens"] == 0
    finally:
        await session.close()
