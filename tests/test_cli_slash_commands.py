"""Slash menus exercised through real prompt-toolkit keyboard input."""

import asyncio
import io
import json
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.layout.screen import Screen
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from hiro_agent import cli
from hiro_agent.config import Config


class TerminalOutput(DummyOutput):
    columns = 80

    def get_size(self) -> Size:
        return Size(rows=40, columns=self.columns)


class Keyboard:
    def __init__(self, pipe: PipeInput) -> None:
        self.pipe = pipe
        self.terminal = cli.Terminal()
        self.output = TerminalOutput()
        self.prompt: PromptSession[str] = PromptSession(input=pipe, output=self.output)
        self.terminal._prompt = self.prompt
        self.tasks: list[asyncio.Task] = []

    @property
    def choices(self) -> list[str]:
        interface = self.terminal._interface
        if interface is None or not interface.menu_visible:
            return []
        state = self.prompt.default_buffer.complete_state
        return [item.text for item in state.completions] if state else []

    @property
    def screen(self) -> Screen:
        screen = self.prompt.app.renderer.last_rendered_screen
        assert screen is not None
        return screen

    @property
    def screen_lines(self) -> list[str]:
        return [
            "".join(self.screen.data_buffer[y][x].char for x in range(self.output.columns))
            for y in range(self.screen.height)
        ]

    async def rendered(self, action: Callable[[], object], ready: Callable[[], bool]) -> None:
        event = asyncio.Event()

        def after_render(_) -> None:
            if ready():
                event.set()

        self.prompt.app.after_render += after_render
        try:
            action()
            await asyncio.wait_for(event.wait(), 2)
        finally:
            self.prompt.app.after_render -= after_render

    async def start(self, *, approval: bool = False, selection: bool = False) -> asyncio.Task[str]:
        def begin() -> None:
            self.tasks.append(
                asyncio.create_task(
                    self.terminal.read(
                        "选择会话：" if selection else "hiro> ",
                        approval=approval,
                        selection=selection,
                    )
                )
            )

        await self.rendered(begin, lambda: self.prompt.app.is_running)
        return self.tasks[-1]

    async def send(self, text: str, ready: Callable[[], bool]) -> None:
        await self.rendered(lambda: self.pipe.send_text(text), ready)


@pytest.fixture
async def keyboard(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Keyboard]:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe:
        keyboard = Keyboard(pipe)
        try:
            yield keyboard
        finally:
            for task in keyboard.tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*keyboard.tasks, return_exceptions=True)
            keyboard.terminal.close()


async def test_slash_opens_described_menu_and_prefix_filters(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    assert keyboard.choices == ["/help", "/tasks", "/continue", "/resume", "/exit"]
    state = keyboard.prompt.default_buffer.complete_state
    assert state is not None
    assert all(item.display_meta_text for item in state.completions)

    await keyboard.send("co", lambda: keyboard.choices == ["/continue"])
    keyboard.pipe.send_text("\t\r")
    assert await asyncio.wait_for(task, 2) == "/continue"


@pytest.mark.parametrize(
    "keys,command",
    [
        ("", "/help"),
        ("\x1b[B", "/tasks"),
        ("\t", "/help"),
        ("\x1b[B\x1b[B", "/continue"),
        ("\x1b[B\x1b[B\x1b[A", "/tasks"),
        ("\x1b[B\x1b[B\x1b[B", "/resume"),
        ("\x1b[A", "/exit"),
    ],
)
async def test_selection_submits_with_one_enter(
    keyboard: Keyboard, keys: str, command: str
) -> None:
    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    keyboard.pipe.send_text(keys + "\r")
    assert await asyncio.wait_for(task, 2) == command


async def test_selection_preserves_filter_and_typing_resets_highlight(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    await keyboard.send("\x1b[B", lambda: keyboard.terminal._interface.selected_index == 1)
    assert keyboard.prompt.default_buffer.text == "/"
    assert keyboard.choices == ["/help", "/tasks", "/continue", "/resume", "/exit"]

    await keyboard.send("co", lambda: keyboard.choices == ["/continue"])
    assert keyboard.prompt.default_buffer.text == "/co"
    assert keyboard.terminal._interface.selected_index == 0
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "/continue"


async def test_tab_completes_selected_command_without_submitting(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    await keyboard.send("\x1b[B\t", lambda: keyboard.prompt.default_buffer.text == "/tasks")
    assert not task.done()
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "/tasks"


async def test_escape_dismisses_menu_and_further_typing_reopens_it(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    await keyboard.send("\x1b", lambda: not keyboard.choices)
    assert keyboard.prompt.default_buffer.text == "/"

    await keyboard.send("h", lambda: keyboard.choices == ["/help"])
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "/help"


async def test_ctrl_c_discards_menu_and_next_prompt_is_empty(keyboard: Keyboard) -> None:
    async def read_until_interrupted() -> None:
        with pytest.raises(KeyboardInterrupt):
            await keyboard.terminal.read("hiro> ")

    def begin() -> None:
        keyboard.tasks.append(asyncio.create_task(read_until_interrupted()))

    await keyboard.rendered(begin, lambda: keyboard.prompt.app.is_running)
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    keyboard.pipe.send_text("\x03")
    await asyncio.wait_for(keyboard.tasks[-1], 2)

    task = await keyboard.start()
    assert keyboard.prompt.default_buffer.text == ""
    assert not keyboard.choices
    await keyboard.send("/ex", lambda: keyboard.choices == ["/exit"])
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "/exit"


async def test_ctrl_d_at_empty_prompt_ends_input(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    keyboard.pipe.send_text("\x04")
    with pytest.raises(EOFError):
        await asyncio.wait_for(task, 2)


async def test_ctrl_r_searches_history_and_returns_to_composer(keyboard: Keyboard) -> None:
    previous = await keyboard.start()
    keyboard.pipe.send_text("review parser\r")
    assert await asyncio.wait_for(previous, 2) == "review parser"

    task = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    await keyboard.send(
        "\x12parser",
        lambda: (
            keyboard.prompt.app.layout.has_focus(keyboard.prompt.search_buffer)
            and keyboard.prompt.search_buffer.text == "parser"
        ),
    )
    assert not keyboard.choices
    assert any("review parser" in line for line in keyboard.screen_lines)
    await keyboard.send(
        "\r",
        lambda: (
            keyboard.prompt.app.layout.has_focus(keyboard.prompt.default_buffer)
            and keyboard.prompt.default_buffer.text == "review parser"
        ),
    )
    assert not task.done()
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "review parser"


@pytest.mark.parametrize("columns", [30, 80])
async def test_menu_renders_above_composer_with_full_width_selected_row(
    keyboard: Keyboard, columns: int
) -> None:
    keyboard.output.columns = columns
    await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    screen = keyboard.screen
    menu_rows = [
        y
        for y, line in enumerate(keyboard.screen_lines)
        if any(command in line for command in keyboard.choices)
    ]
    composer_rows = [
        y
        for y in range(screen.height)
        if any("composer.prefix" in screen.data_buffer[y][x].style for x in range(columns))
    ]
    footer_rows = [
        y
        for y in range(screen.height)
        if any("footer" in screen.data_buffer[y][x].style for x in range(columns))
    ]
    composer_background_rows = [
        y
        for y in range(screen.height)
        if any("class:composer" in screen.data_buffer[y][x].style.split() for x in range(columns))
    ]
    assert len(menu_rows) == 5
    assert len(composer_background_rows) == 3
    assert composer_rows and footer_rows
    assert max(menu_rows) < min(composer_rows) < min(footer_rows)
    selected_row = menu_rows[0]
    # Wide characters own both cells; their empty continuation cells are not
    # independently painted by prompt-toolkit's renderer.
    selected_cells = [
        screen.data_buffer[selected_row][x]
        for x in range(columns)
        if screen.data_buffer[selected_row][x].char
    ]
    assert sum(cell.width for cell in selected_cells) == columns
    assert all("menu.selected" in cell.style for cell in selected_cells)
    assert "查看帮助" in keyboard.screen_lines[selected_row]


async def test_long_chinese_input_wraps_in_narrow_composer(keyboard: Keyboard) -> None:
    keyboard.output.columns = 30
    task = await keyboard.start()
    text = "测试输入" * 20
    await keyboard.send(text, lambda: keyboard.prompt.default_buffer.text == text)
    input_lines = [line for line in keyboard.screen_lines if "测" in line or "入" in line]
    assert len(input_lines) > 2
    assert "".join(line.strip().removeprefix("›").strip() for line in input_lines) == text
    assert not keyboard.choices
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == text


@pytest.mark.parametrize("text", ["explain /help", "/help argument", "/unknown", "/tmp/file"])
async def test_non_command_input_has_no_menu(keyboard: Keyboard, text: str) -> None:
    task = await keyboard.start()
    await keyboard.send(
        text,
        lambda: keyboard.prompt.default_buffer.text == text and not keyboard.choices,
    )
    keyboard.pipe.send_text("\t\r")
    assert await asyncio.wait_for(task, 2) == text


async def test_tab_does_not_replace_text_when_cursor_is_inside_command(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    await keyboard.send("/help", lambda: keyboard.prompt.default_buffer.text == "/help")
    keyboard.pipe.send_text("\x1b[D\t\r")
    assert await asyncio.wait_for(task, 2) == "/help"


async def test_prompt_reuse_disables_menu_during_approval_then_restores_it(
    keyboard: Keyboard,
) -> None:
    normal = await keyboard.start()
    await keyboard.send("/", lambda: len(keyboard.choices) == 5)
    keyboard.pipe.send_text("\t\r")
    assert await asyncio.wait_for(normal, 2) == "/help"

    approval = await keyboard.start(approval=True)
    await keyboard.send(
        "/", lambda: keyboard.prompt.default_buffer.text == "/" and not keyboard.choices
    )
    keyboard.pipe.send_text("\t\r")
    assert await asyncio.wait_for(approval, 2) == "/"

    normal = await keyboard.start()
    await keyboard.send("/ta", lambda: keyboard.choices == ["/tasks"])
    keyboard.pipe.send_text("\t\r")
    assert await asyncio.wait_for(normal, 2) == "/tasks"


async def test_selected_command_executes_in_chat_without_starting_model_run(
    keyboard: Keyboard, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=120))
    monkeypatch.setattr(cli, "Terminal", lambda: keyboard.terminal)
    config = Config()
    config.model.adapter = "mock"

    def begin() -> None:
        keyboard.tasks.append(asyncio.create_task(cli.chat_loop(tmp_path, config, None)))

    await keyboard.rendered(begin, lambda: keyboard.prompt.app.is_running)
    await keyboard.send("/ta", lambda: keyboard.choices == ["/tasks"])
    await keyboard.send("\t\r", lambda: "没有子任务。" in output.getvalue())
    keyboard.pipe.send_text("/exit\r")
    await asyncio.wait_for(keyboard.tasks[-1], 2)

    assert "没有子任务。" in output.getvalue()
    logs = list((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert not any(event["type"] in {"task.received", "run.started"} for event in events)


async def test_session_selector_has_own_prompt_and_returns_to_command_menu(
    keyboard: Keyboard,
) -> None:
    selection = await keyboard.start(selection=True)
    assert any("选择会话" in line for line in keyboard.screen_lines)
    assert any("Session ID" in line for line in keyboard.screen_lines)
    await keyboard.send(
        "/", lambda: keyboard.prompt.default_buffer.text == "/" and not keyboard.choices
    )
    keyboard.pipe.send_text("\x15" + "1\r")
    assert await asyncio.wait_for(selection, 2) == "1"

    task = await keyboard.start()
    await keyboard.send("/res", lambda: keyboard.choices == ["/resume"])
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "/resume"


async def test_ctrl_c_cancels_session_selection_without_starting_chat(
    keyboard: Keyboard, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "list_sessions", lambda _: [{"session_id": "saved", "task_count": 1}])
    monkeypatch.setattr(cli, "console", Console(file=io.StringIO()))
    rendered = asyncio.Event()
    picker_class = cli.SessionPicker

    def picker(*args, **kwargs):
        instance = picker_class(*args, **kwargs)
        instance.app.after_render += lambda _: rendered.set()
        return instance

    monkeypatch.setattr(cli, "SessionPicker", picker)
    keyboard.tasks.append(asyncio.create_task(cli.select_session(keyboard.terminal, tmp_path)))
    await asyncio.wait_for(rendered.wait(), 2)
    keyboard.pipe.send_text("\x03")
    assert await asyncio.wait_for(keyboard.tasks[-1], 2) is None
    assert not list(tmp_path.iterdir())

    # The picker shares input with the chat prompt; cancellation must not leak
    # into the next prompt or leave its bindings installed.
    next_prompt = await keyboard.start()
    await keyboard.send("/res", lambda: keyboard.choices == ["/resume"])
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(next_prompt, 2) == "/resume"
