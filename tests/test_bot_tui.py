"""Bot chat behavior through real prompt-toolkit keyboard and mouse input."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from hiro_agent.bot_tui import BotTUI
from hiro_agent.kernel_contracts import (
    InteractionRecord,
    InteractionReply,
    InteractionRequest,
    InteractionState,
    OperationIntent,
    TaskRecord,
    TaskState,
)
from hiro_agent.tui_backend import BotSummary, ChatEntry, Conversation, TuiBackend

ALPHA = "bot-alpha"
BETA = "bot-beta"
ALT_DOWN = "\x1b\x1b[B"
ALT_UP = "\x1b\x1b[A"
F2 = "\x1bOQ"
F3 = "\x1bOR"
F8 = "\x1b[19~"


class TerminalOutput(DummyOutput):
    columns = 110
    rows = 30

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


class WorkerProbe:
    """A detached worker must survive leaving the terminal application."""

    def __init__(self) -> None:
        self.stopped = False

    def poll(self) -> int | None:
        return 0 if self.stopped else None

    def terminate(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        self.stopped = True


class BackendFake:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.summaries = [
            BotSummary(ALPHA, "Alpha Agent", "model-a", "running", "alpha preview", 0, 1),
            BotSummary(BETA, "Beta Agent", "model-b", "queued", "beta preview", 0, 1),
        ]
        self.conversations = {
            bot_id: Conversation(
                bot_id,
                (ChatEntry("assistant", f"History belonging to {bot_id}", f"task-{bot_id}"),),
                (self.task(bot_id),),
                (),
            )
            for bot_id in (ALPHA, BETA)
        }
        self.submissions: list[tuple[str, str]] = []
        self.replies: list[tuple[str, str, InteractionReply]] = []
        self.cancellations: list[tuple[str, str]] = []
        self.worker = WorkerProbe()
        self.worker_starts = 0
        self.closed = False
        self.submit_gate: threading.Event | None = None
        self.submit_returned = threading.Event()
        self.reply_gate: threading.Event | None = None
        self.reply_returned = threading.Event()
        self.detail_gate: threading.Event | None = None
        self.detail_entered = threading.Event()

    @staticmethod
    def task(bot_id: str, *, suffix: str = "") -> TaskRecord:
        return TaskRecord(
            id=f"task-{bot_id}{suffix}",
            bot_id=bot_id,
            session_id=f"session-{bot_id}{suffix}",
            text=f"Work belonging to {bot_id}{suffix}",
            state=TaskState.RUNNING,
        )

    def list_bots(self) -> list[BotSummary]:
        return list(self.summaries)

    def conversation(self, bot_id: str) -> Conversation:
        result = self.conversations[bot_id]
        if bot_id == ALPHA and self.detail_gate is not None:
            self.detail_entered.set()
            assert self.detail_gate.wait(5), "test did not release delayed conversation"
        return result

    def submit(self, bot_id: str, text: str, *, idempotency_key: str | None = None) -> TaskRecord:
        self.submissions.append((bot_id, text))
        if self.submit_gate is not None:
            assert self.submit_gate.wait(5), "test did not release delayed submission"
        self.submit_returned.set()
        return self.task(bot_id).model_copy(update={"text": text, "state": TaskState.QUEUED})

    def reply(self, bot_id: str, interaction_id: str, reply: InteractionReply) -> InteractionRecord:
        self.replies.append((bot_id, interaction_id, reply))
        if self.reply_gate is not None:
            assert self.reply_gate.wait(5), "test did not release delayed reply"
        snapshot = self.conversations[bot_id]
        record = next(item for item in snapshot.interactions if item.id == interaction_id)
        self.conversations[bot_id] = replace(snapshot, interactions=())
        self.reply_returned.set()
        return record.model_copy(update={"state": InteractionState.ANSWERED, "reply": reply})

    def cancel(self, bot_id: str, task_id: str) -> TaskRecord:
        self.cancellations.append((bot_id, task_id))
        return next(task for task in self.conversations[bot_id].tasks if task.id == task_id)

    def worker_running(self) -> bool:
        return not self.worker.stopped

    def start_worker(self) -> WorkerProbe:
        self.worker_starts += 1
        return self.worker

    def close(self) -> None:
        self.closed = True

    def pending(self, kind: Literal["approval", "question"]) -> InteractionRecord:
        task = self.task(ALPHA)
        operation = (
            OperationIntent(
                task_id=task.id,
                bot_id=ALPHA,
                session_id=task.session_id,
                run_id="run-alpha",
                call_id="call-alpha",
                tool_name="write_file",
                arguments_json='{"path":"alpha.txt","content":"alpha"}',
                permission_version=1,
                permission_digest="digest-alpha",
            )
            if kind == "approval"
            else None
        )
        record = InteractionRecord(
            id=f"interaction-{kind}",
            request=InteractionRequest(
                task_id=task.id,
                bot_id=ALPHA,
                kind=kind,
                prompt="Approve alpha write?" if kind == "approval" else "Which alpha target?",
                idempotency_key=f"pending-{kind}",
                operation=operation,
            ),
        )
        task = task.model_copy(
            update={
                "state": (
                    TaskState.WAITING_APPROVAL if kind == "approval" else TaskState.WAITING_INPUT
                ),
                "interaction_id": record.id,
                "pending_operation": operation,
            }
        )
        self.conversations[ALPHA] = replace(
            self.conversations[ALPHA], tasks=(task,), interactions=(record,)
        )
        self.summaries[0] = replace(self.summaries[0], pending_count=1, status=task.state.value)
        return record


class Keyboard:
    def __init__(self, pipe: PipeInput, workspace: Path) -> None:
        self.pipe = pipe
        self.output = TerminalOutput()
        self.backend = BackendFake(workspace)
        self.ui = BotTUI(
            cast(TuiBackend, self.backend),
            auto_start_worker=False,
            input=pipe,
            output=self.output,
            refresh_interval=60,
        )
        self.tasks: list[asyncio.Task[None]] = []

    @property
    def lines(self) -> list[str]:
        screen = self.ui.app.renderer.last_rendered_screen
        assert screen is not None
        return [
            "".join(screen.data_buffer[y][x].char for x in range(self.output.columns))
            for y in range(screen.height)
        ]

    @property
    def screen(self) -> str:
        return "\n".join(self.lines)

    async def rendered(self, action: Callable[[], object], ready: Callable[[], bool]) -> None:
        event = asyncio.Event()

        def after_render(_: object) -> None:
            if ready():
                event.set()

        self.ui.app.after_render += after_render
        try:
            action()
            await asyncio.wait_for(event.wait(), 3)
        finally:
            self.ui.app.after_render -= after_render

    async def start(self) -> asyncio.Task[None]:
        await self.rendered(
            lambda: self.tasks.append(asyncio.create_task(self.ui.run())),
            lambda: self.ui.app.is_running and self.ui.details is not None,
        )
        return self.tasks[-1]

    async def send(self, keys: str, ready: Callable[[], bool]) -> None:
        await self.rendered(lambda: self.pipe.send_text(keys), ready)

    async def focus_composer(self) -> None:
        await self.rendered(
            lambda: (self.ui.app.layout.focus(self.ui.composer), self.ui.app.invalidate()),
            lambda: self.ui.app.layout.has_focus(self.ui.composer),
        )

    async def finish_pending(self, gate: threading.Event, returned: threading.Event) -> None:
        await self.rendered(lambda: (gate.set(), self.ui.app.invalidate()), returned.is_set)


@pytest.fixture
async def keyboard(tmp_path: Path) -> AsyncIterator[Keyboard]:
    with create_pipe_input() as pipe:
        keyboard = Keyboard(pipe, tmp_path)
        try:
            yield keyboard
        finally:
            for gate in (
                keyboard.backend.submit_gate,
                keyboard.backend.reply_gate,
                keyboard.backend.detail_gate,
            ):
                if gate is not None:
                    gate.set()
            if keyboard.ui.app.is_running:
                keyboard.ui.app.exit()
            for task in keyboard.tasks:
                if not task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(task), 3)
                    except TimeoutError:
                        task.cancel()
            await asyncio.gather(*keyboard.tasks, return_exceptions=True)


async def test_sidebar_keyboard_switching_preserves_each_bot_draft(keyboard: Keyboard) -> None:
    await keyboard.start()
    await keyboard.focus_composer()
    await keyboard.send("alpha draft", lambda: keyboard.ui.composer.text == "alpha draft")
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.ui.composer.text == ""
    await keyboard.send("beta draft", lambda: keyboard.ui.composer.text == "beta draft")
    await keyboard.send(ALT_UP, lambda: keyboard.ui.selected_bot_id == ALPHA)
    assert keyboard.ui.composer.text == "alpha draft"
    assert keyboard.ui.drafts[BETA] == "beta draft"
    await keyboard.send(F2, lambda: keyboard.ui.app.layout.has_focus(keyboard.ui.sidebar_control))
    await keyboard.send("\x1b[B\r", lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.ui.composer.text == "beta draft"
    assert "Beta Agent" in keyboard.screen
    assert not keyboard.backend.submissions


async def test_mouse_click_selects_the_agent_row(keyboard: Keyboard) -> None:
    await keyboard.start()
    row = next(index for index, line in enumerate(keyboard.lines) if "Beta Agent" in line)
    column = keyboard.lines[row].index("Beta Agent") + 1
    # SGR mouse coordinates are one based. Both press and release pass through
    # prompt-toolkit's terminal parser and the rendered row's mouse handler.
    await keyboard.send(
        f"\x1b[<0;{column};{row + 1}M\x1b[<0;{column};{row + 1}m",
        lambda: keyboard.ui.selected_bot_id == BETA,
    )
    assert keyboard.ui.details is None or keyboard.ui.details.bot_id == BETA


async def test_filter_reduces_sidebar_and_keyboard_selects_matching_bot(keyboard: Keyboard) -> None:
    await keyboard.start()
    keyboard.ui.app.layout.focus(keyboard.ui.search)
    await keyboard.send("Beta", lambda: keyboard.ui.search.text == "Beta")
    assert "Beta Agent" in keyboard.screen
    await keyboard.send(F2 + "\r", lambda: keyboard.ui.selected_bot_id == BETA)
    assert "Alpha Agent" not in keyboard.screen
    assert not keyboard.backend.submissions


async def test_submit_captures_bot_before_switch_and_preserves_new_draft(
    keyboard: Keyboard,
) -> None:
    gate = threading.Event()
    keyboard.backend.submit_gate = gate
    await keyboard.start()
    await keyboard.focus_composer()
    await keyboard.send("alpha request\r", lambda: keyboard.ui.composer.text == "")
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.send("beta unsent", lambda: keyboard.ui.composer.text == "beta unsent")
    await keyboard.finish_pending(gate, keyboard.backend.submit_returned)
    assert keyboard.backend.submissions == [(ALPHA, "alpha request")]
    assert keyboard.ui.selected_bot_id == BETA
    assert keyboard.ui.composer.text == "beta unsent"
    assert not keyboard.backend.cancellations


async def test_refresh_cannot_replace_current_bot_with_delayed_old_details(
    keyboard: Keyboard,
) -> None:
    await keyboard.start()
    keyboard.backend.detail_gate = gate = threading.Event()
    refresh = asyncio.create_task(keyboard.ui.refresh())
    keyboard.tasks.append(refresh)
    assert await asyncio.to_thread(keyboard.backend.detail_entered.wait, 2)
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    gate.set()
    await asyncio.wait_for(refresh, 3)
    assert keyboard.ui.selected_bot_id == BETA
    assert keyboard.ui.details is None or keyboard.ui.details.bot_id == BETA
    await keyboard.ui.refresh()
    await keyboard.rendered(keyboard.ui.app.invalidate, lambda: keyboard.ui.details is not None)
    assert keyboard.ui.details is not None
    assert keyboard.ui.details.bot_id == BETA
    assert "History belonging to bot-alpha" not in keyboard.screen
    assert "History belonging to bot-beta" in keyboard.screen


async def test_refresh_updates_history_without_losing_unsent_text(keyboard: Keyboard) -> None:
    await keyboard.start()
    await keyboard.focus_composer()
    await keyboard.send("keep my draft", lambda: keyboard.ui.composer.text == "keep my draft")
    snapshot = keyboard.backend.conversations[ALPHA]
    keyboard.backend.conversations[ALPHA] = replace(
        snapshot,
        entries=(*snapshot.entries, ChatEntry("assistant", "Worker completed alpha", "task-alpha")),
    )
    await keyboard.ui.refresh()
    await keyboard.rendered(
        keyboard.ui.app.invalidate, lambda: "Worker completed alpha" in keyboard.screen
    )
    assert keyboard.ui.composer.text == "keep my draft"


@pytest.mark.parametrize("kind", ["approval", "question"])
async def test_pending_modal_blocks_switching_and_escape_does_not_reply(
    keyboard: Keyboard, kind: Literal["approval", "question"]
) -> None:
    record = keyboard.backend.pending(kind)
    await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.dialog is not None)
    assert record.request.prompt in keyboard.screen
    if kind == "approval":
        assert "批准一次操作" in keyboard.screen
        assert "write_file" in keyboard.screen
        assert "alpha.txt" in keyboard.screen
    else:
        assert "提交答复" in keyboard.screen
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.dialog is not None)
    assert keyboard.ui.selected_bot_id == ALPHA
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    assert not keyboard.backend.replies
    assert not keyboard.backend.cancellations


@pytest.mark.parametrize("kind", ["approval", "question"])
async def test_modal_reply_keeps_captured_identity_after_switch(
    keyboard: Keyboard, kind: Literal["approval", "question"]
) -> None:
    record = keyboard.backend.pending(kind)
    keyboard.backend.reply_gate = gate = threading.Event()
    await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.dialog is not None)
    if kind == "question":
        assert keyboard.ui.answer is not None
        await keyboard.send("alpha target", lambda: keyboard.ui.answer.text == "alpha target")
        submit = "\t\r"
    else:
        submit = "\r"
    await keyboard.send(submit, lambda: keyboard.ui.dialog is None)
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.finish_pending(gate, keyboard.backend.reply_returned)
    assert len(keyboard.backend.replies) == 1
    bot_id, interaction_id, reply = keyboard.backend.replies[0]
    assert (bot_id, interaction_id) == (ALPHA, record.id)
    assert reply == (
        InteractionReply(approved=True)
        if kind == "approval"
        else InteractionReply(text="alpha target")
    )
    assert keyboard.ui.selected_bot_id == BETA


async def test_cancel_dialog_cancels_only_selected_task_and_escape_is_safe(
    keyboard: Keyboard,
) -> None:
    snapshot = keyboard.backend.conversations[ALPHA]
    second = keyboard.backend.task(ALPHA, suffix="-second")
    keyboard.backend.conversations[ALPHA] = replace(snapshot, tasks=(*snapshot.tasks, second))
    await keyboard.start()
    await keyboard.send(F8, lambda: keyboard.ui.dialog is not None)
    assert "取消所选任务" in keyboard.screen
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    assert not keyboard.backend.cancellations
    await keyboard.send(F8, lambda: keyboard.ui.dialog is not None)
    # The newest task is listed first; explicitly choose the older task.
    await keyboard.send("\x1b[B \t\r", lambda: keyboard.ui.dialog is None)
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.backend.cancellations == [(ALPHA, snapshot.tasks[0].id)]


async def test_question_draft_survives_closing_and_reopening_dialog(keyboard: Keyboard) -> None:
    keyboard.backend.pending("question")
    await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.answer is not None)
    await keyboard.send("partial answer", lambda: keyboard.ui.answer.text == "partial answer")
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    await keyboard.send(F3, lambda: keyboard.ui.answer is not None)
    assert keyboard.ui.answer.text == "partial answer"
    assert not keyboard.backend.replies


@pytest.mark.parametrize("columns,rows", [(110, 30), (42, 12)])
async def test_narrow_layout_can_compose_multiline_and_switch_bot(
    keyboard: Keyboard, columns: int, rows: int
) -> None:
    keyboard.output.columns = columns
    keyboard.output.rows = rows
    await keyboard.start()
    await keyboard.focus_composer()
    await keyboard.send("line one\x0aline two", lambda: "line two" in keyboard.ui.composer.text)
    assert keyboard.ui.composer.text == "line one\nline two"
    assert "line two" in keyboard.screen
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.send(ALT_UP, lambda: keyboard.ui.selected_bot_id == ALPHA)
    assert keyboard.ui.composer.text == "line one\nline two"
    assert not keyboard.backend.submissions


async def test_ctrl_c_clears_composer_and_quit_preserves_worker_and_tasks(
    keyboard: Keyboard,
) -> None:
    keyboard.ui.auto_start_worker = True
    task = await keyboard.start()
    await keyboard.rendered(keyboard.ui.app.invalidate, lambda: keyboard.backend.worker_starts == 1)
    await keyboard.focus_composer()
    await keyboard.send("discard this", lambda: keyboard.ui.composer.text == "discard this")
    await keyboard.send("\x03", lambda: keyboard.ui.composer.text == "")
    assert not task.done()
    assert not keyboard.backend.cancellations
    keyboard.pipe.send_text("\x11")
    await asyncio.wait_for(task, 3)
    assert not keyboard.backend.closed
    assert not keyboard.backend.worker.stopped
    assert not keyboard.backend.cancellations
    assert keyboard.backend.conversations[ALPHA].tasks[0].state == TaskState.RUNNING
    assert keyboard.backend.conversations[BETA].tasks[0].state == TaskState.RUNNING


async def test_narrow_sidebar_can_open_select_and_return_to_composer(keyboard: Keyboard) -> None:
    keyboard.output.columns = 42
    keyboard.output.rows = 12
    await keyboard.start()
    await keyboard.send(F2, lambda: keyboard.ui.app.layout.has_focus(keyboard.ui.sidebar_control))
    assert "Alpha Agent" in keyboard.screen
    await keyboard.send("\x1b[B", lambda: keyboard.ui.selected_bot_id == BETA)
    assert "Beta Agent" in keyboard.screen
    await keyboard.send("\r", lambda: keyboard.ui.app.layout.has_focus(keyboard.ui.composer))
    await keyboard.send("narrow draft", lambda: keyboard.ui.composer.text == "narrow draft")
    assert "narrow draft" in keyboard.screen
