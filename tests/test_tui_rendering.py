"""Regression coverage for the pixels actually painted by prompt-toolkit."""

from __future__ import annotations

import io
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.styles import DEFAULT_ATTRS, Attrs
from prompt_toolkit.utils import get_cwidth
from test_bot_tui import ALPHA, BETA, BackendFake

from hiro_agent import bot_tui
from hiro_agent.bot_tui import BotTUI
from hiro_agent.tui_backend import BotSummary, ChatEntry, Conversation, TuiBackend


@dataclass(frozen=True)
class PaintedCell:
    char: str = " "
    background: str = ""
    foreground: str = ""


class PaintedOutput(DummyOutput):
    """Track terminal writes, including blank cells the renderer leaves untouched.

    Looking up resolved styles in ``last_rendered_screen`` misses the regression:
    its transparent blank cells resolve to the default background, but the screen
    diff may never emit them to the terminal. Only output operations paint here.
    """

    def __init__(self, *, columns: int = 110, rows: int = 30) -> None:
        self.columns = columns
        self.rows = rows
        self.x = self.y = 0
        self.attrs = DEFAULT_ATTRS
        self.cells: dict[tuple[int, int], PaintedCell] = {}

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)

    def get_default_color_depth(self) -> ColorDepth:
        return ColorDepth.DEPTH_24_BIT

    def set_attributes(self, attrs: Attrs, color_depth: ColorDepth) -> None:
        self.attrs = attrs

    def reset_attributes(self) -> None:
        self.attrs = DEFAULT_ATTRS

    def write(self, data: str) -> None:
        for char in data:
            if char == "\r":
                self.x = 0
            elif char == "\n":
                self.y = min(self.rows - 1, self.y + 1)
            else:
                width = get_cwidth(char)
                if not width:
                    continue
                self.cells[self.x, self.y] = PaintedCell(char, self.attrs.bgcolor, self.attrs.color)
                for offset in range(1, width):
                    self.cells[self.x + offset, self.y] = PaintedCell(
                        "", self.attrs.bgcolor, self.attrs.color
                    )
                self.x = min(self.columns - 1, self.x + width)

    def cursor_goto(self, row: int = 0, column: int = 0) -> None:
        self.x, self.y = column, row

    def cursor_up(self, amount: int) -> None:
        self.y = max(0, self.y - amount)

    def cursor_down(self, amount: int) -> None:
        self.y = min(self.rows - 1, self.y + amount)

    def cursor_forward(self, amount: int) -> None:
        self.x = min(self.columns - 1, self.x + amount)

    def cursor_backward(self, amount: int) -> None:
        self.x = max(0, self.x - amount)

    def erase_end_of_line(self) -> None:
        for x in range(self.x, self.columns):
            self.cells[x, self.y] = PaintedCell(background=self.attrs.bgcolor)

    def erase_down(self) -> None:
        self.erase_end_of_line()
        for y in range(self.y + 1, self.rows):
            for x in range(self.columns):
                self.cells[x, y] = PaintedCell(background=self.attrs.bgcolor)


class RenderedTUI:
    def __init__(self, workspace: Path, *, columns: int = 110) -> None:
        self.output = PaintedOutput(columns=columns)
        self.backend = BackendFake(workspace)
        self.ui = BotTUI(
            cast(TuiBackend, self.backend),
            auto_start_worker=False,
            input=DummyInput(),
            output=self.output,
        )
        self.ui.bot_summaries = self.backend.list_bots()
        self.ui.select_bot(ALPHA)

    def render(self, text: str) -> list[list[PaintedCell]]:
        bot_id = self.ui.selected_bot_id
        assert bot_id is not None
        self.ui.details = replace(
            self.backend.conversations[bot_id], entries=(ChatEntry("assistant", text, "task"),)
        )
        return self.render_current()

    def render_current(self) -> list[list[PaintedCell]]:
        # Exercise the production Renderer directly, without worker or refresh
        # timing. Buffer history loading still requires a running event loop.
        with set_app(self.ui.app):
            self.ui.app.render_counter += 1
            self.ui.app.before_render.fire()
            self.ui.app.renderer.render(self.ui.app, self.ui.app.layout)
        window = next(
            window
            for window in self.ui.app.layout.visible_windows
            if window.content is self.ui.transcript_control
        )
        info = window.render_info
        assert info is not None
        return [
            [
                self.output.cells.get((x, y), PaintedCell())
                for x in range(info._x_offset, info._x_offset + info.window_width)
            ]
            for y in range(info._y_offset, info._y_offset + info.window_height)
        ]


def painted_text(rows: list[list[PaintedCell]]) -> str:
    return "\n".join("".join(cell.char for cell in row) for row in rows)


@pytest.fixture
async def terminal_factory(tmp_path: Path) -> AsyncIterator[Callable[[int], RenderedTUI]]:
    terminals: list[RenderedTUI] = []

    def create(columns: int) -> RenderedTUI:
        terminal = RenderedTUI(tmp_path, columns=columns)
        terminals.append(terminal)
        return terminal

    try:
        yield create
    finally:
        for terminal in terminals:
            await terminal.ui.app.cancel_and_wait_for_background_tasks()


@pytest.mark.parametrize("source,label", [("handoff", "消息"), ("handoff_result", "回传")])
async def test_agent_sender_name_and_color_differ_from_user_in_terminal(
    terminal_factory: Callable[[int], RenderedTUI],
    source: str,
    label: str,
) -> None:
    terminal = terminal_factory(110)
    snapshot = terminal.backend.conversations[ALPHA]
    task = snapshot.tasks[0].model_copy(update={"source": source})
    terminal.ui.details = replace(
        snapshot,
        tasks=(task,),
        entries=(
            ChatEntry("user", "Human request", "human"),
            ChatEntry(
                "agent",
                "Agent reply",
                task.id,
                sender_bot_id="comm-test-sender",
                sender_name="通信测试发送方",
            ),
        ),
    )
    rows = terminal.render_current()
    title = f"通信测试发送方 · {label}"
    assert title in painted_text(rows)
    user_row = next(row for row in rows if "".join(c.char for c in row).strip() == "你")
    agent_row = next(row for row in rows if title in "".join(c.char for c in row))
    user_colors = {c.foreground for c in user_row if c.char.strip()}
    agent_colors = {c.foreground for c in agent_row if c.char.strip()}
    assert user_colors == {"e2b8d9"}
    assert agent_colors == {"9ccfd8"}
    assert {c.background for row in rows for c in row} == {"232136"}


@pytest.mark.parametrize("columns", [42, 110, 180])
async def test_transcript_paints_blank_rows_and_short_message_tails(
    terminal_factory: Callable[[int], RenderedTUI], columns: int
) -> None:
    terminal = terminal_factory(columns)
    rows = terminal.render("你好，世界。\n\nShort reply.")
    text = painted_text(rows)
    assert "你好，世界。" in text
    assert "Short reply." in text
    assert any(all(cell.char == " " for cell in row) for row in rows)
    assert all(cell.char == " " for cell in rows[-1])
    assert {cell.background for row in rows for cell in row} == {"232136"}


@pytest.mark.parametrize("columns", [42, 110])
async def test_transcript_clears_shortened_messages_and_previous_bot(
    terminal_factory: Callable[[int], RenderedTUI], columns: int
) -> None:
    terminal = terminal_factory(columns)
    rows = terminal.render(("OLD_ALPHA " * 8 + "\n\n") * 4)
    assert "OLD_ALPHA" in painted_text(rows)

    rows = terminal.render("短答。")
    assert "短答。" in painted_text(rows)
    assert "OLD_ALPHA" not in painted_text(rows)
    assert {cell.background for row in rows for cell in row} == {"232136"}
    assert rows == terminal_factory(columns).render("短答。")

    terminal.ui.select_bot(BETA)
    rows = terminal.render("Beta only.")
    assert "Beta only." in painted_text(rows)
    assert "Alpha Agent" not in painted_text(rows)
    assert "短答。" not in painted_text(rows)
    assert {cell.background for row in rows for cell in row} == {"232136"}
    fresh = terminal_factory(columns)
    fresh.ui.select_bot(BETA)
    assert rows == fresh.render("Beta only.")


async def test_new_agent_empty_conversation_replaces_loading_after_refresh(
    terminal_factory: Callable[[int], RenderedTUI],
) -> None:
    terminal = terminal_factory(110)
    terminal.render("Existing conversation")
    bot_id = "new-reviewer"
    terminal.backend.summaries.append(BotSummary(bot_id, "代码审查员", "model-a", "idle", "", 0, 0))
    terminal.backend.conversations[bot_id] = Conversation(bot_id, (), (), ())
    await terminal.ui.refresh()
    terminal.ui.select_bot(bot_id)
    assert "正在读取对话" in painted_text(terminal.render_current())

    await terminal.ui.refresh()
    text = painted_text(terminal.render_current())
    assert "代码审查员 已就绪" in text
    assert "在下方发送第一条消息" in text
    assert "正在读取对话" not in text
    assert "Existing conversation" not in text

    # An unchanged empty snapshot stays ready on subsequent polling refreshes.
    await terminal.ui.refresh()
    assert painted_text(terminal.render_current()) == text


async def test_transcript_paints_background_after_narrow_and_wide_resize(
    terminal_factory: Callable[[int], RenderedTUI],
) -> None:
    terminal = terminal_factory(180)
    for columns in (180, 42, 110):
        terminal.output.columns = columns
        rows = terminal.render("中文内容，resized reply.")
        assert "中文内容，resized reply." in painted_text(rows)
        assert {cell.background for row in rows for cell in row} == {"232136"}


async def test_inline_code_keeps_its_background_inside_uniform_transcript(
    terminal_factory: Callable[[int], RenderedTUI],
) -> None:
    terminal = terminal_factory(110)
    rows = terminal.render("Before `inline_code` after.")
    assert "Before inline_code after." in painted_text(rows)
    assert {cell.background for row in rows for cell in row} == {"232136", "332e49"}
    assert (
        "".join(cell.char for row in rows for cell in row if cell.background == "332e49")
        == "inline_code"
    )
    assert all(cell.background == "232136" for cell in rows[-1])


@pytest.fixture(autouse=True)
def clean_color_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("COLORTERM", "TERM", "NO_COLOR", "PROMPT_TOOLKIT_COLOR_DEPTH"):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, None),
        ({"TERM": "xterm-256color"}, None),
        ({"COLORTERM": "truecolor"}, ColorDepth.DEPTH_24_BIT),
        ({"COLORTERM": "24bit"}, ColorDepth.DEPTH_24_BIT),
        ({"TERM": "xterm-direct"}, ColorDepth.DEPTH_24_BIT),
        ({"TERM": "xterm-truecolor"}, ColorDepth.DEPTH_24_BIT),
        ({"COLORTERM": "truecolor", "NO_COLOR": "1"}, ColorDepth.DEPTH_1_BIT),
        (
            {"COLORTERM": "truecolor", "PROMPT_TOOLKIT_COLOR_DEPTH": "DEPTH_8_BIT"},
            ColorDepth.DEPTH_8_BIT,
        ),
        (
            {"NO_COLOR": "1", "PROMPT_TOOLKIT_COLOR_DEPTH": "DEPTH_24_BIT"},
            ColorDepth.DEPTH_1_BIT,
        ),
    ],
)
def test_terminal_color_depth_honors_explicit_settings_before_capabilities(
    clean_color_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: ColorDepth | None,
) -> None:
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    assert bot_tui._terminal_color_depth() == expected


@pytest.mark.parametrize("override", [None, "DEPTH_8_BIT", "NO_COLOR"])
async def test_advertised_truecolor_reaches_terminal_escape_sequences(
    tmp_path: Path,
    clean_color_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    override: str | None,
) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    if override == "NO_COLOR":
        monkeypatch.setenv("NO_COLOR", "1")
    elif override is not None:
        monkeypatch.setenv("PROMPT_TOOLKIT_COLOR_DEPTH", override)
    stream = io.StringIO()
    # Many terminals advertise truecolor via COLORTERM while TERM and the
    # output backend still default to xterm's 256-color palette.
    output = Vt100_Output(
        stream,
        lambda: Size(rows=30, columns=110),
        term="xterm-256color",
        default_color_depth=ColorDepth.DEPTH_8_BIT,
        enable_cpr=False,
    )
    ui = BotTUI(
        cast(TuiBackend, BackendFake(tmp_path)),
        auto_start_worker=False,
        input=DummyInput(),
        output=output,
    )
    try:
        with set_app(ui.app):
            ui.app.render_counter += 1
            ui.app.renderer.render(ui.app, ui.app.layout)
        emitted = stream.getvalue()
        if override == "NO_COLOR":
            assert not re.search(r"[34]8;(?:2|5);", emitted)
        elif override == "DEPTH_8_BIT":
            assert "48;5;" in emitted
            assert "48;2;" not in emitted
        else:
            assert "48;2;35;33;54" in emitted
    finally:
        await ui.app.cancel_and_wait_for_background_tasks()
