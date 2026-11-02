"""Terminal selection mode through real prompt-toolkit keyboard input."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from test_bot_tui import ALPHA, ALT_DOWN, BETA, F2, F3, F8, Keyboard

from hiro_agent.tui_backend import ChatEntry

F4 = "\x1bOS"
PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"


@pytest.fixture
async def keyboard(tmp_path: Path) -> AsyncIterator[Keyboard]:
    with create_pipe_input() as pipe:
        keyboard = Keyboard(pipe, tmp_path)
        try:
            yield keyboard
        finally:
            if keyboard.backend.detail_gate is not None:
                keyboard.backend.detail_gate.set()
            if keyboard.ui.app.is_running:
                keyboard.ui.app.exit()
            await asyncio.wait_for(asyncio.gather(*keyboard.tasks, return_exceptions=True), 3)


async def test_f4_releases_terminal_mouse_and_restores_draft_and_sidebar(
    keyboard: Keyboard, monkeypatch: pytest.MonkeyPatch
) -> None:
    mouse_events: list[str] = []
    monkeypatch.setattr(
        keyboard.output, "enable_mouse_support", lambda: mouse_events.append("enable")
    )
    monkeypatch.setattr(
        keyboard.output, "disable_mouse_support", lambda: mouse_events.append("disable")
    )
    await keyboard.start()
    await keyboard.send("keep this draft", lambda: keyboard.ui.composer.text == "keep this draft")
    assert keyboard.ui.app.mouse_support()
    assert "AGENTS" in keyboard.screen
    assert mouse_events[-1] == "enable"

    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    assert not keyboard.ui.app.mouse_support()
    assert mouse_events[-1] == "disable"
    assert "AGENTS" not in keyboard.screen
    assert "History belonging to bot-alpha" in keyboard.screen
    assert "Ctrl+Shift+C" in keyboard.screen
    assert keyboard.ui.composer.text == "keep this draft"
    assert keyboard.ui.drafts[ALPHA] == "keep this draft"

    await keyboard.send(F4, lambda: not keyboard.ui.selection_mode)
    assert keyboard.ui.app.mouse_support()
    assert mouse_events[-1] == "enable"
    assert "AGENTS" in keyboard.screen
    assert keyboard.ui.app.layout.has_focus(keyboard.ui.composer)
    assert keyboard.ui.composer.text == "keep this draft"
    assert not keyboard.backend.submissions


async def test_selection_blocks_editing_submitting_and_agent_actions(keyboard: Keyboard) -> None:
    keyboard.backend.pending("question")
    await keyboard.start()
    await keyboard.send("unsent message", lambda: keyboard.ui.composer.text == "unsent message")
    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    keyboard.ui.send_current()
    assert keyboard.ui.composer.text == "unsent message"
    assert not keyboard.ui._submitting

    await keyboard.send(
        "ignored text\r\t\x03" + ALT_DOWN + F2 + F3 + F8 + F4,
        lambda: not keyboard.ui.selection_mode,
    )
    assert keyboard.ui.composer.text == "unsent message"
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.dialog is None
    assert not keyboard.backend.submissions
    assert not keyboard.backend.replies
    assert not keyboard.backend.cancellations


async def test_escape_resumes_updates_after_a_frozen_selection(keyboard: Keyboard) -> None:
    await keyboard.start()
    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    frozen_detail = keyboard.ui.details
    frozen_screen = keyboard.screen
    snapshot = keyboard.backend.conversations[ALPHA]
    keyboard.backend.conversations[ALPHA] = replace(
        snapshot,
        entries=(*snapshot.entries, ChatEntry("assistant", "New worker output", "task-alpha")),
    )
    keyboard.backend.summaries[0] = replace(keyboard.backend.summaries[0], status="succeeded")

    await keyboard.ui.refresh()
    assert keyboard.ui.details is frozen_detail
    assert keyboard.screen == frozen_screen
    assert keyboard.ui.selected_bot is not None
    assert keyboard.ui.selected_bot.status == "running"
    assert not keyboard.backend.worker.stopped

    await keyboard.send("\x1b", lambda: "New worker output" in keyboard.screen)
    assert not keyboard.ui.selection_mode
    assert keyboard.ui.app.mouse_support()
    assert "AGENTS" in keyboard.screen
    assert keyboard.ui.selected_bot.status == "succeeded"
    assert keyboard.ui.app.layout.has_focus(keyboard.ui.composer)


async def test_refresh_started_before_selection_cannot_replace_frozen_history(
    keyboard: Keyboard,
) -> None:
    await keyboard.start()
    await keyboard.ui.refresh()
    frozen_detail = keyboard.ui.details
    snapshot = keyboard.backend.conversations[ALPHA]
    keyboard.backend.conversations[ALPHA] = replace(
        snapshot,
        entries=(*snapshot.entries, ChatEntry("assistant", "Delayed worker output", "task-alpha")),
    )
    keyboard.backend.detail_gate = gate = threading.Event()
    refresh = asyncio.create_task(keyboard.ui.refresh())
    keyboard.tasks.append(refresh)
    assert await asyncio.to_thread(keyboard.backend.detail_entered.wait, 2)
    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    gate.set()
    await asyncio.wait_for(refresh, 3)

    assert keyboard.ui.details is frozen_detail
    assert "Delayed worker output" not in keyboard.screen
    await keyboard.send("\x1b", lambda: "Delayed worker output" in keyboard.screen)
    assert not keyboard.ui.selection_mode


async def test_select_command_completion_and_help_stay_local(keyboard: Keyboard) -> None:
    keyboard.ui.drafts[BETA] = "other agent draft"
    await keyboard.start()
    await keyboard.send("/sel", lambda: keyboard.ui.composer.buffer.complete_state is not None)
    assert "/select" in keyboard.screen
    await keyboard.send("\t", lambda: keyboard.ui.composer.text == "/select")
    await keyboard.send("\r", lambda: keyboard.ui.composer.buffer.complete_state is None)
    assert not keyboard.ui.selection_mode
    await keyboard.send("\r", lambda: keyboard.ui.selection_mode)
    assert keyboard.ui.composer.text == ""
    assert keyboard.ui.drafts[BETA] == "other agent draft"
    await keyboard.send(F4, lambda: not keyboard.ui.selection_mode)
    await keyboard.send("/help\r", lambda: keyboard.ui.dialog is not None)
    assert "/select" in keyboard.screen
    assert "F4" in keyboard.screen
    assert not keyboard.backend.submissions


async def test_f4_cannot_replace_a_pending_question_modal(keyboard: Keyboard) -> None:
    keyboard.backend.pending("question")
    await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.answer is not None)
    await keyboard.send("partial reply", lambda: keyboard.ui.answer.text == "partial reply")
    await keyboard.send(F4, lambda: keyboard.ui.dialog is not None)
    assert not keyboard.ui.selection_mode
    assert keyboard.ui.app.mouse_support()
    assert keyboard.ui.answer is not None
    assert keyboard.ui.answer.text == "partial reply"
    assert not keyboard.backend.replies


async def test_selection_mode_keeps_keyboard_history_paging(keyboard: Keyboard) -> None:
    snapshot = keyboard.backend.conversations[ALPHA]
    keyboard.backend.conversations[ALPHA] = replace(
        snapshot,
        entries=tuple(
            ChatEntry("assistant", f"History message {index}", f"task-{index}")
            for index in range(40)
        ),
    )
    await keyboard.start()
    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    latest_anchor = keyboard.ui.transcript_control.anchor
    await keyboard.send(PAGE_UP, lambda: keyboard.ui.transcript_control.anchor < latest_anchor)
    previous_anchor = keyboard.ui.transcript_control.anchor
    assert not keyboard.ui.transcript_control.follow
    await keyboard.send(PAGE_DOWN, lambda: keyboard.ui.transcript_control.anchor > previous_anchor)
    assert keyboard.ui.selection_mode
    assert not keyboard.ui.app.mouse_support()
