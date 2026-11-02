"""Agent lifecycle commands through real prompt-toolkit keyboard input."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from test_bot_tui import ALPHA, ALT_DOWN, BETA, F3, BackendFake, Keyboard
from test_tui_selection import F4

from hiro_agent.errors import ConfigurationError
from hiro_agent.tui_backend import BotSummary, Conversation, TuiBackend


class LifecycleBackendFake(BackendFake):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self.lifecycle_calls: list[tuple[str, str | None]] = []
        self.lifecycle_gate: threading.Event | None = None
        self.lifecycle_error: str | None = None

    def list_bots(self) -> list[BotSummary]:
        return sorted(super().list_bots(), key=lambda bot: bot.bot_id)

    def _operation(self, action: str, bot_id: str | None = None) -> None:
        self.lifecycle_calls.append((action, bot_id))
        if self.lifecycle_gate is not None:
            assert self.lifecycle_gate.wait(5), "test did not release lifecycle operation"
        if self.lifecycle_error:
            raise ConfigurationError(self.lifecycle_error)
        if bot_id is not None and bot_id not in self.conversations:
            raise ConfigurationError("Agent 已不存在")

    def clear_history(self, bot_id: str) -> dict[str, object]:
        self._operation("clear", bot_id)
        self.conversations[bot_id] = Conversation(bot_id, (), (), ())
        self.summaries = [
            replace(bot, status="idle", preview="", pending_count=0, task_count=0)
            if bot.bot_id == bot_id
            else bot
            for bot in self.summaries
        ]
        return {"bot_id": bot_id}

    def delete_bot(self, bot_id: str) -> dict[str, object]:
        self._operation("delete", bot_id)
        self.summaries = [bot for bot in self.summaries if bot.bot_id != bot_id]
        self.conversations.pop(bot_id)
        return {"bot_id": bot_id}

    def initialize_agents(self) -> dict[str, object]:
        self._operation("initialize")
        existing = {bot.bot_id for bot in self.summaries}
        created = []
        for bot_id, name in (
            ("default", "Hiro"),
            ("agent-manager", "Agent 管理员"),
            ("code-writer", "代码编写员"),
        ):
            if bot_id not in existing:
                self.summaries.append(BotSummary(bot_id, name, "current-model", "idle", "", 0, 0))
                self.conversations[bot_id] = Conversation(bot_id, (), (), ())
                created.append(bot_id)
        return {"created": created}


class LifecycleKeyboard(Keyboard):
    backend: LifecycleBackendFake

    def __init__(self, pipe: PipeInput, workspace: Path) -> None:
        super().__init__(pipe, workspace)
        self.backend = LifecycleBackendFake(workspace)
        self.ui.backend = cast(TuiBackend, self.backend)


@pytest.fixture
async def keyboard(tmp_path: Path) -> AsyncIterator[LifecycleKeyboard]:
    with create_pipe_input() as pipe:
        keyboard = LifecycleKeyboard(pipe, tmp_path)
        try:
            yield keyboard
        finally:
            if keyboard.backend.lifecycle_gate is not None:
                keyboard.backend.lifecycle_gate.set()
            if keyboard.ui.app.is_running:
                keyboard.ui.app.exit()
            await asyncio.wait_for(asyncio.gather(*keyboard.tasks, return_exceptions=True), 3)


@pytest.mark.parametrize("command", ["/clear", "/delete-agent"])
async def test_cleanup_requires_confirmation_and_cancel_is_local(
    keyboard: LifecycleKeyboard, command: str
) -> None:
    await keyboard.start()
    await keyboard.send(command + "\r", lambda: keyboard.ui.dialog is not None)
    assert "Alpha Agent" in keyboard.screen
    assert ALPHA in keyboard.screen
    assert "无法撤销" in keyboard.screen
    if command == "/clear":
        assert "保留 Agent 身份和配置" in keyboard.screen
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.dialog is not None)
    assert keyboard.ui.selected_bot_id == ALPHA
    await keyboard.send("\r", lambda: keyboard.ui.dialog is None)
    assert not keyboard.backend.lifecycle_calls
    await keyboard.send(command + "\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    assert not keyboard.backend.lifecycle_calls
    assert not keyboard.backend.submissions
    assert keyboard.ui.details is not None and keyboard.ui.details.entries


async def test_clear_removes_current_history_and_local_drafts(keyboard: LifecycleKeyboard) -> None:
    request = keyboard.backend.pending("question")
    keyboard.ui.drafts[BETA] = "keep beta draft"
    await keyboard.start()
    await keyboard.send(F3, lambda: keyboard.ui.answer is not None)
    await keyboard.send("old answer", lambda: keyboard.ui.answer.text == "old answer")
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    assert keyboard.ui.answer_drafts[request.id] == "old answer"
    await keyboard.send("/clear\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send(
        "\t\r", lambda: not keyboard.ui._managing_agents and "已清空" in keyboard.ui.notice
    )
    assert keyboard.backend.lifecycle_calls == [("clear", ALPHA)]
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.details == Conversation(ALPHA, (), (), ())
    assert ALPHA not in keyboard.ui.drafts
    assert request.id not in keyboard.ui.answer_drafts
    assert keyboard.ui.drafts[BETA] == "keep beta draft"
    assert "Alpha Agent 已就绪" in keyboard.screen
    assert "History belonging to bot-alpha" not in keyboard.screen
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("only_agent", [False, True])
async def test_delete_selects_remaining_agent_or_empty_welcome(
    keyboard: LifecycleKeyboard, only_agent: bool
) -> None:
    if only_agent:
        keyboard.backend.summaries = keyboard.backend.summaries[:1]
    await keyboard.start()
    keyboard.ui.unread.add(ALPHA)
    keyboard.ui._pending_index[ALPHA] = 1
    await keyboard.send("/delete-agent\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send(
        "\t\r", lambda: not keyboard.ui._managing_agents and "已删除" in keyboard.ui.notice
    )
    assert keyboard.backend.lifecycle_calls == [("delete", ALPHA)]
    assert keyboard.ui.selected_bot_id == (None if only_agent else BETA)
    assert ALPHA not in keyboard.ui.drafts
    assert ALPHA not in keyboard.ui.unread
    assert ALPHA not in keyboard.ui._pending_index
    assert "History belonging to bot-alpha" not in keyboard.screen
    if only_agent:
        assert keyboard.ui.details is None
        assert not keyboard.ui.bot_summaries
        assert "欢迎来到 Hiro" in keyboard.screen
        assert "/init-agents" in keyboard.screen
    else:
        assert keyboard.ui.details is not None and keyboard.ui.details.bot_id == BETA
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("command, action", [("/clear", "clear"), ("/delete-agent", "delete")])
async def test_confirmed_operation_keeps_target_after_agent_switch(
    keyboard: LifecycleKeyboard, command: str, action: str
) -> None:
    keyboard.backend.pending("question")
    await keyboard.start()
    keyboard.backend.lifecycle_gate = gate = threading.Event()
    await keyboard.send(command + "\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send("\t\r", lambda: keyboard.ui._managing_agents)
    await keyboard.send(F3, lambda: keyboard.ui.dialog is None)
    assert keyboard.ui.answer is None
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.send("beta draft", lambda: keyboard.ui.composer.text == "beta draft")
    await keyboard.rendered(
        lambda: (gate.set(), keyboard.ui.app.invalidate()),
        lambda: not keyboard.ui._managing_agents,
    )
    assert keyboard.backend.lifecycle_calls == [(action, ALPHA)]
    assert keyboard.ui.selected_bot_id == BETA
    assert keyboard.ui.composer.text == "beta draft"
    assert keyboard.ui.details is not None and keyboard.ui.details.bot_id == BETA
    assert keyboard.backend.conversations[BETA].entries


async def test_confirmation_target_survives_external_removal_while_modal_open(
    keyboard: LifecycleKeyboard,
) -> None:
    await keyboard.start()
    await keyboard.send("/clear\r", lambda: keyboard.ui.dialog is not None)
    keyboard.backend.summaries = keyboard.backend.summaries[1:]
    keyboard.backend.conversations.pop(ALPHA)
    await keyboard.ui.refresh()
    assert keyboard.ui.selected_bot_id == BETA
    assert keyboard.ui.dialog is not None
    await keyboard.send("\t\r", lambda: "失败" in keyboard.ui.notice)
    assert keyboard.backend.lifecycle_calls == [("clear", ALPHA)]
    assert keyboard.backend.conversations[BETA].entries
    assert "Agent 已不存在" in keyboard.ui.notice


async def test_external_deletion_waits_for_selection_mode_to_end(
    keyboard: LifecycleKeyboard,
) -> None:
    await keyboard.start()
    await keyboard.send("keep alpha draft", lambda: keyboard.ui.composer.text == "keep alpha draft")
    await keyboard.send(F4, lambda: keyboard.ui.selection_mode)
    frozen = keyboard.ui.details
    keyboard.backend.summaries = []
    keyboard.backend.conversations.clear()
    await keyboard.ui.refresh()
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.details is frozen
    await keyboard.send(F4, lambda: keyboard.ui.selected_bot_id is None)
    assert keyboard.ui.details is None
    assert keyboard.ui.composer.text == ""
    assert ALPHA not in keyboard.ui.drafts
    assert "欢迎来到 Hiro" in keyboard.screen


async def test_initialize_empty_sidebar_and_preserve_existing_agents(
    keyboard: LifecycleKeyboard,
) -> None:
    keyboard.backend.summaries = []
    keyboard.backend.conversations = {}
    await keyboard.rendered(
        lambda: keyboard.tasks.append(asyncio.create_task(keyboard.ui.run())),
        lambda: keyboard.ui.app.is_running,
    )
    await keyboard.send("/init", lambda: keyboard.ui.composer.text == "/init")
    await keyboard.ui.refresh()
    assert keyboard.ui.composer.text == "/init"
    await keyboard.send(
        "-agents\r",
        lambda: not keyboard.ui._managing_agents and len(keyboard.ui.bot_summaries) == 3,
    )
    assert [(bot.bot_id, bot.name) for bot in keyboard.ui.bot_summaries] == [
        ("agent-manager", "Agent 管理员"),
        ("code-writer", "代码编写员"),
        ("default", "Hiro"),
    ]
    assert keyboard.ui.selected_bot_id == "default"
    assert keyboard.ui.dialog is None
    custom = replace(keyboard.backend.summaries[0], name="My Hiro", model="custom-model")
    keyboard.backend.summaries[0] = custom
    await keyboard.send(
        "/init-agents\r",
        lambda: not keyboard.ui._managing_agents and "新建 0 个" in keyboard.ui.notice,
    )
    assert keyboard.ui.selected_bot == custom
    assert len(keyboard.ui.bot_summaries) == 3
    await keyboard.send(
        "/agent code-writer\r", lambda: keyboard.ui.selected_bot_id == "code-writer"
    )
    await keyboard.send(
        "/init-agents\r",
        lambda: not keyboard.ui._managing_agents and len(keyboard.backend.lifecycle_calls) == 3,
    )
    assert keyboard.ui.selected_bot_id == "code-writer"
    assert not keyboard.backend.submissions


@pytest.mark.parametrize("command", ["/clear", "/delete-agent", "/init-agents"])
async def test_lifecycle_rejects_arguments_without_submitting(
    keyboard: LifecycleKeyboard, command: str
) -> None:
    await keyboard.start()
    await keyboard.send(command + " accidental-id\r", lambda: "不接受参数" in keyboard.ui.notice)
    assert keyboard.ui.dialog is None
    assert not keyboard.backend.lifecycle_calls
    assert not keyboard.backend.submissions


async def test_lifecycle_failure_keeps_history_and_configuration(
    keyboard: LifecycleKeyboard,
) -> None:
    await keyboard.start()
    snapshot = keyboard.ui.details
    keyboard.backend.lifecycle_error = "当前 Agent 有正在运行的任务"
    await keyboard.send("/delete-agent\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send("\t\r", lambda: "失败" in keyboard.ui.notice)
    assert keyboard.ui.selected_bot_id == ALPHA
    assert keyboard.ui.details == snapshot
    assert len(keyboard.ui.bot_summaries) == 2
    assert "正在运行的任务" in keyboard.ui.notice
    assert not keyboard.ui._managing_agents


async def test_lifecycle_completion_and_help_are_local(keyboard: LifecycleKeyboard) -> None:
    await keyboard.start()
    await keyboard.send("/del", lambda: keyboard.ui.composer.buffer.complete_state is not None)
    assert "/delete-agent" in keyboard.screen
    await keyboard.send("\t", lambda: keyboard.ui.composer.text == "/delete-agent")
    await keyboard.send("\r", lambda: keyboard.ui.composer.buffer.complete_state is None)
    assert keyboard.ui.dialog is None
    await keyboard.send("\r", lambda: keyboard.ui.dialog is not None)
    await keyboard.send("\x1b", lambda: keyboard.ui.dialog is None)
    await keyboard.send("/help\r", lambda: keyboard.ui.dialog is not None)
    for command in ("/clear", "/delete-agent", "/init-agents"):
        assert command in keyboard.screen
    assert not keyboard.backend.lifecycle_calls
    assert not keyboard.backend.submissions
