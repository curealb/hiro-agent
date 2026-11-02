"""Independent acceptance probes using real durable stores and terminal input."""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from hiro_agent.bot_tui import BotTUI
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.contracts import RunResult, RunStatus
from hiro_agent.kernel_contracts import (
    InteractionReply,
    InteractionRequest,
    InteractionState,
    TaskState,
)
from hiro_agent.session import JsonlSession
from hiro_agent.tui_backend import TuiBackend

F2 = "\x1bOQ"
F3 = "\x1bOR"
ALT_DOWN = "\x1b\x1b[B"


class ResizableOutput(DummyOutput):
    columns = 110
    rows = 30

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


class Terminal:
    def __init__(self, ui: BotTUI, send: Callable[[str], None], output: ResizableOutput):
        self.ui = ui
        self.send = send
        self.output = output

    @property
    def screen(self) -> str:
        screen = self.ui.app.renderer.last_rendered_screen
        if screen is None:
            return ""
        return "\n".join(
            "".join(screen.data_buffer[y][x].char for x in range(self.output.columns))
            for y in range(screen.height)
        )

    async def until(self, predicate: Callable[[], bool]) -> None:
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()
        handle: asyncio.TimerHandle | None = None

        def check() -> None:
            nonlocal handle
            if predicate():
                ready.set()
            else:
                handle = loop.call_later(0.01, check)

        check()
        try:
            await asyncio.wait_for(ready.wait(), 3)
        finally:
            if handle is not None:
                handle.cancel()

    async def keys(self, value: str, predicate: Callable[[], bool]) -> None:
        self.send(value)
        await self.until(predicate)

    async def redraw(self) -> None:
        drawn = asyncio.Event()

        def rendered(_: object) -> None:
            drawn.set()

        self.ui.app.after_render += rendered
        try:
            self.ui.app.invalidate()
            await asyncio.wait_for(drawn.wait(), 3)
        finally:
            self.ui.app.after_render -= rendered


@pytest.fixture
def backend(tmp_path: Path) -> TuiBackend:
    backend = TuiBackend(tmp_path)
    for bot_id in ("alpha", "beta"):
        backend.kernel.bots.save(
            Config(bot=BotConfig(id=bot_id, name=bot_id.title()), model=ModelConfig(adapter="mock"))
        )
    return backend


@pytest.fixture
async def terminal(backend: TuiBackend) -> AsyncIterator[Terminal]:
    with create_pipe_input() as pipe:
        output = ResizableOutput()
        ui = BotTUI(
            backend, input=pipe, output=output, auto_start_worker=False, refresh_interval=60
        )
        terminal = Terminal(ui, pipe.send_text, output)
        task = asyncio.create_task(ui.run())
        try:
            await terminal.until(lambda: ui.details is not None)
            yield terminal
        finally:
            if ui.app.is_running:
                ui.app.exit()
            await asyncio.wait_for(task, 3)


def pending_question(backend: TuiBackend, bot_id: str):
    task = backend.submit(bot_id, f"question task for {bot_id}")
    claimed = backend.kernel.tasks.claim("acceptance")
    assert claimed is not None and claimed.task.id == task.id
    request = backend.kernel.interactions.create(
        InteractionRequest(
            task_id=task.id,
            bot_id=bot_id,
            kind="question",
            prompt=f"Select a branch for {bot_id}",
            idempotency_key=f"question:{task.id}",
        )
    )
    backend.kernel.tasks.finish(
        claimed.run.id,
        RunResult(
            session_id=task.session_id,
            run_id=claimed.run.id,
            status=RunStatus.PAUSED,
            reason="await question",
        ),
        interaction_id=request.id,
        waiting_for="question",
    )
    return request


async def test_real_store_reply_and_submit_stay_with_owner_after_switch(
    backend: TuiBackend, terminal: Terminal
) -> None:
    alpha = pending_question(backend, "alpha")
    beta = pending_question(backend, "beta")
    await terminal.ui.refresh()
    await terminal.keys(F3, lambda: terminal.ui.answer is not None)
    await terminal.keys(
        "alpha-branch\t\r" + ALT_DOWN, lambda: terminal.ui.selected_bot_id == "beta"
    )
    await terminal.until(
        lambda: backend.kernel.interactions.get(alpha.id).state == InteractionState.ANSWERED
    )
    answer = backend.kernel.interactions.get(alpha.id)
    assert answer.reply is not None and answer.reply.text == "alpha-branch"
    assert backend.kernel.interactions.get(beta.id).state == InteractionState.PENDING
    await terminal.keys("beta task\r", lambda: len(backend.kernel.tasks.list(bot_id="beta")) == 2)
    assert backend.kernel.tasks.list(bot_id="beta")[-1].text == "beta task"
    assert len(backend.kernel.tasks.list(bot_id="alpha")) == 1


async def test_real_live_journal_read_during_switch_does_not_repair_or_leak(
    backend: TuiBackend, terminal: Terminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = backend.submit("alpha", "alpha private task")
    assert backend.kernel.tasks.claim("acceptance") is not None
    session = await JsonlSession.open(backend.workspace, task.session_id)
    gate = threading.Event()
    entered = threading.Event()
    original = backend.conversation

    def delayed(bot_id: str):
        result = original(bot_id)
        if bot_id == "alpha":
            entered.set()
            assert gate.wait(3)
        return result

    try:
        await session.append("assistant.message", {"message": {"text": "alpha private answer"}})
        with session.path.open("ab") as stream:
            stream.write(b'{"version":1,"unfinished":')
        before = session.path.read_bytes()
        monkeypatch.setattr(backend, "conversation", delayed)
        refresh = asyncio.create_task(terminal.ui.refresh())
        try:
            await terminal.until(entered.is_set)
            await terminal.keys(ALT_DOWN, lambda: terminal.ui.selected_bot_id == "beta")
        finally:
            gate.set()
            await asyncio.wait_for(refresh, 3)
        await terminal.until(lambda: terminal.ui.details is not None)
        await terminal.redraw()
        assert terminal.ui.details is not None and terminal.ui.details.bot_id == "beta"
        assert "alpha private answer" not in terminal.screen
        assert session.path.read_bytes() == before
        assert backend.kernel.tasks.get(task.id).state == TaskState.RUNNING
        assert len(backend.kernel.tasks.runs(task.id)) == 1
    finally:
        gate.set()
        await session.close()


async def test_resize_keeps_visible_pane_and_draft_usable(terminal: Terminal) -> None:
    await terminal.keys("retained draft", lambda: terminal.ui.composer.text == "retained draft")
    await terminal.redraw()
    row = next(i for i, line in enumerate(terminal.screen.splitlines()) if "● Alpha" in line)
    await terminal.keys(
        f"\x1b[<0;5;{row + 1}M\x1b[<0;5;{row + 1}m",
        lambda: terminal.ui.app.layout.has_focus(terminal.ui.sidebar_control),
    )
    terminal.output.columns = 42
    terminal.output.rows = 12
    await terminal.redraw()
    assert "AGENTS" in terminal.screen and "Alpha" in terminal.screen, terminal.screen
    await terminal.keys("\x1b[B", lambda: terminal.ui.selected_bot_id == "beta")
    await terminal.redraw()
    assert "Beta" in terminal.screen, terminal.screen
    await terminal.keys("\x1b[A", lambda: terminal.ui.selected_bot_id == "alpha")
    await terminal.keys("\r", lambda: terminal.ui.app.layout.has_focus(terminal.ui.composer))
    await terminal.redraw()
    assert "retained draft" in terminal.screen
    terminal.output.columns = 110
    terminal.output.rows = 30
    await terminal.redraw()
    assert "Alpha" in terminal.screen and "Beta" in terminal.screen
    assert terminal.ui.composer.text == "retained draft"


@pytest.mark.parametrize("size", [(80, 24), (42, 12)])
async def test_question_dialog_remains_usable_at_small_terminal_sizes(
    backend: TuiBackend, terminal: Terminal, size: tuple[int, int]
) -> None:
    record = pending_question(backend, "alpha")
    await terminal.ui.refresh()
    terminal.output.columns, terminal.output.rows = size
    await terminal.keys(F3, lambda: terminal.ui.answer is not None)
    await terminal.redraw()
    assert "Window too small" not in terminal.screen, terminal.screen
    assert "提交答复" in terminal.screen, terminal.screen
    await terminal.keys("small-terminal-answer\t\r", lambda: terminal.ui.dialog is None)
    await terminal.until(
        lambda: backend.kernel.interactions.get(record.id).state == InteractionState.ANSWERED
    )


async def test_modal_background_click_cannot_change_bot_or_submit(
    backend: TuiBackend, terminal: Terminal
) -> None:
    record = pending_question(backend, "alpha")
    await terminal.ui.refresh()
    await terminal.keys(F3, lambda: terminal.ui.answer is not None)
    await terminal.redraw()
    row = next(i for i, line in enumerate(terminal.screen.splitlines()) if " › " in line)
    terminal.send(f"\x1b[<0;40;{row + 1}M\x1b[<0;40;{row + 1}m" + ALT_DOWN)
    await terminal.redraw()
    await terminal.keys("safe draft", lambda: terminal.ui.answer.text == "safe draft")
    assert terminal.ui.selected_bot_id == "alpha"
    assert terminal.ui.app.layout.has_focus(terminal.ui.answer)
    await terminal.keys("\x1b", lambda: terminal.ui.dialog is None)
    assert backend.kernel.interactions.get(record.id).state == InteractionState.PENDING
    assert len(backend.kernel.tasks.list()) == 1


async def test_unread_activity_is_cleared_when_its_bot_is_selected(
    backend: TuiBackend, terminal: Terminal
) -> None:
    backend.submit("beta", "new activity on background Bot")
    await terminal.ui.refresh()
    await terminal.redraw()
    assert terminal.ui.unread == {"beta"}
    assert "Beta •" in terminal.screen
    await terminal.keys(ALT_DOWN, lambda: terminal.ui.selected_bot_id == "beta")
    await terminal.ui.refresh()
    await terminal.redraw()
    assert not terminal.ui.unread
    assert "Beta •" not in terminal.screen
    assert "new activity on background Bot" in terminal.screen


@pytest.mark.skipif(sys.platform != "linux", reason="TUI supports Linux")
def test_real_worker_executes_new_task_after_tui_process_exits(tmp_path: Path) -> None:
    (tmp_path / "hiro-demo.txt").write_text("pre-existing read-only fixture\n")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
    attributes = termios.tcgetattr(slave)
    attributes[0] &= ~(termios.IXON | termios.IXOFF)
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    process = subprocess.Popen(
        [sys.executable, "-m", "hiro_agent.cli", "tui", "--workspace", str(tmp_path), "--demo"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        start_new_session=True,
        env={**os.environ, "TERM": "xterm-256color", "PROMPT_TOOLKIT_NO_CPR": "1"},
    )
    os.close(slave)
    worker_pid: int | None = None
    transcript = bytearray()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert process.poll() is None
            children = [
                child
                for task_directory in Path(f"/proc/{process.pid}/task").iterdir()
                for child in (task_directory / "children").read_text().split()
            ]
            if children:
                worker_pid = int(children[0])
            if select.select([master], [], [], 0.05)[0]:
                transcript.extend(os.read(master, 65536))
            if worker_pid is not None and "worker 已连接".encode() in transcript:
                break
        assert worker_pid is not None, transcript.decode(errors="replace")
        backend = TuiBackend(tmp_path)
        assert backend.worker_running()
        assert os.getsid(worker_pid) == worker_pid
        os.write(master, b"\x11")
        assert process.wait(timeout=5) == 0
        assert backend.worker_running()
        task = backend.submit("demo-b", "work submitted after UI exited")
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and backend.kernel.tasks.get(task.id).state != TaskState.SUCCEEDED
        ):
            for record in backend.conversation("demo-b").interactions:
                assert record.request.kind == "approval"
                backend.reply("demo-b", record.id, InteractionReply(approved=True))
            time.sleep(0.05)
        assert backend.kernel.tasks.get(task.id).state == TaskState.SUCCEEDED
        assert backend.worker_running()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and TuiBackend(tmp_path).worker_running():
                time.sleep(0.05)
        os.close(master)
