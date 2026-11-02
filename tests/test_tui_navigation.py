"""Slash navigation uses real keyboard input without creating worker tasks."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from test_bot_tui import ALPHA, BETA, F3, Keyboard


@pytest.fixture
async def keyboard(tmp_path: Path) -> AsyncIterator[Keyboard]:
    with create_pipe_input() as pipe:
        keyboard = Keyboard(pipe, tmp_path)
        try:
            yield keyboard
        finally:
            for gate in (keyboard.backend.submit_gate, keyboard.backend.reply_gate):
                if gate is not None:
                    gate.set()
            if keyboard.ui.app.is_running:
                keyboard.ui.app.exit()
            await asyncio.wait_for(asyncio.gather(*keyboard.tasks, return_exceptions=True), 3)


async def test_completion_selects_command_and_agent_without_submitting(keyboard: Keyboard) -> None:
    keyboard.ui.drafts[BETA] = "beta draft"
    keyboard.ui.unread.update({BETA, "another-agent"})
    await keyboard.start()
    await keyboard.send("/ag", lambda: keyboard.ui.composer.buffer.complete_state is not None)
    assert "/agent" in keyboard.screen
    await keyboard.send("\t", lambda: keyboard.ui.composer.text == "/agent")
    await keyboard.send("\r", lambda: keyboard.ui.composer.buffer.complete_state is None)
    assert keyboard.ui.app.layout.has_focus(keyboard.ui.composer)
    await keyboard.send(" Beta", lambda: keyboard.ui.composer.buffer.complete_state is not None)
    assert "Beta Agent" in keyboard.screen
    await keyboard.send("\t", lambda: keyboard.ui.composer.text == f"/agent {BETA}")
    await keyboard.send("\r", lambda: keyboard.ui.composer.buffer.complete_state is None)
    assert keyboard.ui.selected_bot_id == ALPHA
    await keyboard.send("\r", lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.ui.composer.text == "beta draft"
    assert keyboard.ui.drafts[ALPHA] == ""
    assert keyboard.ui.unread == {"another-agent"}
    assert not keyboard.backend.submissions


async def test_completion_arrow_keys_and_escape_keep_navigation_local(keyboard: Keyboard) -> None:
    await keyboard.start()
    await keyboard.send("/", lambda: keyboard.ui.composer.buffer.complete_state is not None)
    await keyboard.send("\x1b[B", lambda: keyboard.ui.composer.text == "/agent")
    await keyboard.send("\x1b[B", lambda: keyboard.ui.composer.text == "/tui")
    await keyboard.send("\x1b[A", lambda: keyboard.ui.composer.text == "/agent")
    await keyboard.send("\x1b", lambda: keyboard.ui.composer.buffer.complete_state is None)
    assert keyboard.ui.composer.text == "/"
    assert keyboard.ui.selected_bot_id == ALPHA
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("target", [BETA, "Beta Agent", "bEtA aGeNt"])
async def test_agent_command_matches_id_or_complete_name(keyboard: Keyboard, target: str) -> None:
    await keyboard.start()
    await keyboard.send(f"/agent {target}\r", lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.ui.composer.text == ""
    assert not keyboard.backend.submissions


async def test_agent_exact_id_wins_over_another_agent_name(keyboard: Keyboard) -> None:
    keyboard.backend.summaries[1] = replace(keyboard.backend.summaries[1], name=ALPHA)
    await keyboard.start()
    await keyboard.send(f"/agent {ALPHA}\r", lambda: "已切换" in keyboard.ui.notice)
    assert keyboard.ui.selected_bot_id == ALPHA
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_bad_agent_target_stays_put_and_never_submits(
    keyboard: Keyboard, ambiguous: bool
) -> None:
    if ambiguous:
        keyboard.backend.summaries = [
            replace(bot, name="Same Name") for bot in keyboard.backend.summaries
        ]
    command = "/agent Same Name" if ambiguous else "/agent missing"
    await keyboard.start()
    await keyboard.send(
        command + "\r",
        lambda: ("不唯一" if ambiguous else "找不到") in keyboard.ui.notice,
    )
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.composer.text == command
    if ambiguous:
        assert ALPHA in keyboard.ui.notice
        assert BETA in keyboard.ui.notice
    else:
        assert "/agent" in keyboard.ui.notice
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("columns", [42, 110])
async def test_agent_without_target_opens_unfiltered_list(keyboard: Keyboard, columns: int) -> None:
    keyboard.output.columns = columns
    keyboard.ui.search.text = "Alpha"
    await keyboard.start()
    await keyboard.send(
        "/agent\r", lambda: keyboard.ui.app.layout.has_focus(keyboard.ui.sidebar_control)
    )
    assert keyboard.ui.composer.text == ""
    assert keyboard.ui.search.text == ""
    assert "Beta Agent" in keyboard.screen
    await keyboard.send("\x1b[B\r", lambda: keyboard.ui.selected_bot_id == BETA)
    assert keyboard.ui.app.layout.has_focus(keyboard.ui.composer)
    assert not keyboard.backend.submissions


@pytest.mark.parametrize(
    ("command", "message"),
    [("/tui", "当前已在"), ("/unknown", "未知命令"), ("/chat extra", "用法")],
)
async def test_local_command_notices_never_create_a_task(
    keyboard: Keyboard, command: str, message: str
) -> None:
    task = await keyboard.start()
    await keyboard.send(command + "\r", lambda: message in keyboard.ui.notice)
    assert not task.done()
    assert keyboard.ui.exit_action is None
    assert keyboard.ui.composer.text == ("" if command == "/tui" else command)
    assert not keyboard.backend.submissions


async def test_help_is_modal_and_preserves_agent_selection(keyboard: Keyboard) -> None:
    await keyboard.start()
    await keyboard.send("/help\r", lambda: keyboard.ui.dialog is not None)
    assert "/agent ID 或名称" in keyboard.screen
    assert "/chat" in keyboard.screen
    await keyboard.send("\x1b\x1b[B", lambda: keyboard.ui.dialog is not None)
    assert keyboard.ui.selected_bot_id == ALPHA
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    assert keyboard.ui.app.layout.has_focus(keyboard.ui.composer)
    assert not keyboard.backend.submissions


async def test_question_answer_slash_text_cannot_navigate(keyboard: Keyboard) -> None:
    keyboard.backend.pending("question")
    task = await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.answer is not None)
    await keyboard.send(
        "/chat\r/agent Beta Agent",
        lambda: keyboard.ui.answer is not None and "Beta Agent" in keyboard.ui.answer.text,
    )
    assert keyboard.ui.answer is not None
    assert keyboard.ui.answer.text == "/chat\n/agent Beta Agent"
    assert keyboard.ui.dialog is not None
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.exit_action is None
    assert not task.done()
    assert not keyboard.backend.submissions
    assert not keyboard.backend.replies


async def test_chat_exit_and_reuse_preserve_state_and_reset_exit_action(keyboard: Keyboard) -> None:
    keyboard.ui.drafts[BETA] = "unsent beta"
    keyboard.ui.unread.add(BETA)
    keyboard.ui.answer_drafts["pending-question"] = "partial answer"
    first = await keyboard.start()
    keyboard.pipe.send_text("/chat\r")
    assert await asyncio.wait_for(first, 3) is None
    assert keyboard.ui.exit_action == "chat"
    assert keyboard.ui.composer.text == ""
    second = await keyboard.start()
    assert keyboard.ui.exit_action is None
    assert keyboard.ui.drafts[BETA] == "unsent beta"
    assert BETA in keyboard.ui.unread
    assert keyboard.ui.answer_drafts["pending-question"] == "partial answer"
    keyboard.pipe.send_text("/exit\r")
    assert await asyncio.wait_for(second, 3) is None
    assert keyboard.ui.exit_action is None
    assert not keyboard.backend.closed
    assert not keyboard.backend.worker.stopped
    assert not keyboard.backend.submissions
    assert not keyboard.backend.cancellations


async def test_chat_is_available_without_any_agents(keyboard: Keyboard) -> None:
    keyboard.backend.summaries.clear()
    await keyboard.rendered(
        lambda: keyboard.tasks.append(asyncio.create_task(keyboard.ui.run())),
        lambda: keyboard.ui.app.is_running,
    )
    keyboard.pipe.send_text("/chat\r")
    await asyncio.wait_for(keyboard.tasks[-1], 3)
    assert keyboard.ui.exit_action == "chat"
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("command", ["/chat", "/exit"])
async def test_interface_exit_waits_for_submission_persistence(
    keyboard: Keyboard, command: str
) -> None:
    gate = threading.Event()
    keyboard.backend.submit_gate = gate
    task = await keyboard.start()
    await keyboard.send("save this\r", lambda: bool(keyboard.backend.submissions))
    await keyboard.send(command + "\r", lambda: "正在保存" in keyboard.ui.notice)
    assert keyboard.ui.exit_action is None
    assert keyboard.ui.composer.text == command
    assert not task.done()
    gate.set()
    await keyboard.rendered(keyboard.ui.app.invalidate, lambda: not keyboard.ui._submitting)
    keyboard.pipe.send_text("\r")
    await asyncio.wait_for(task, 3)
    assert keyboard.ui.exit_action == ("chat" if command == "/chat" else None)
    assert keyboard.backend.submissions == [(ALPHA, "save this")]


async def test_interface_exit_waits_for_captured_modal_reply(keyboard: Keyboard) -> None:
    record = keyboard.backend.pending("approval")
    keyboard.backend.reply_gate = gate = threading.Event()
    task = await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.dialog is not None)
    await keyboard.send("\r", lambda: keyboard.ui.dialog is None)
    await keyboard.send(f"/agent {BETA}\r", lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.send("/chat\r", lambda: "正在保存" in keyboard.ui.notice)
    assert not task.done()
    gate.set()
    await keyboard.rendered(keyboard.ui.app.invalidate, lambda: not keyboard.ui._replying)
    keyboard.pipe.send_text("\r")
    await asyncio.wait_for(task, 3)
    assert keyboard.ui.exit_action == "chat"
    assert keyboard.backend.replies[0][:2] == (ALPHA, record.id)
    assert not keyboard.backend.submissions
