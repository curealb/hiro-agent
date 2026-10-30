"""History selection and deletion through real prompt-toolkit key input."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from hiro_agent.errors import PersistenceError
from hiro_agent.session_picker import SessionPicker


class TerminalOutput(DummyOutput):
    columns = 80
    rows = 24

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


def items(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "session_id": f"session-{index:02d}",
            "title": f"第 {index} 个任务 [bold]原样显示[/bold]",
            "status": "completed",
            "updated_at": "2026-10-31T10:24:02+00:00",
        }
        for index in range(count)
    ]


class Keyboard:
    def __init__(self, pipe: PipeInput) -> None:
        self.pipe = pipe
        self.output = TerminalOutput()
        self.deleted: list[str] = []
        self.original = items()
        self.picker = SessionPicker(
            self.original, self.deleted.append, input=pipe, output=self.output
        )
        self.tasks: list[asyncio.Task[str | None]] = []

    @property
    def screen_lines(self) -> list[str]:
        screen = self.picker.app.renderer.last_rendered_screen
        assert screen is not None
        return [
            "".join(screen.data_buffer[y][x].char for x in range(self.output.columns))
            for y in range(screen.height)
        ]

    async def rendered(self, action: Callable[[], object], ready: Callable[[], bool]) -> None:
        event = asyncio.Event()

        def after_render(_: object) -> None:
            if ready():
                event.set()

        self.picker.app.after_render += after_render
        try:
            action()
            await asyncio.wait_for(event.wait(), 2)
        finally:
            self.picker.app.after_render -= after_render

    async def start(self) -> asyncio.Task[str | None]:
        def begin() -> None:
            self.tasks.append(asyncio.create_task(self.picker.run()))

        await self.rendered(begin, lambda: self.picker.app.is_running)
        return self.tasks[-1]

    async def send(self, keys: str, ready: Callable[[], bool]) -> None:
        await self.rendered(lambda: self.pipe.send_text(keys), ready)


@pytest.fixture
async def keyboard() -> AsyncIterator[Keyboard]:
    with create_pipe_input() as pipe:
        keyboard = Keyboard(pipe)
        try:
            yield keyboard
        finally:
            for task in keyboard.tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*keyboard.tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "keys,selected",
    [
        ("", "session-00"),
        ("\x1b[B", "session-01"),
        ("\x1b[B\x1b[B\x1b[A", "session-01"),
        ("\x1b[A", "session-00"),
        ("\x1b[F", "session-02"),
        ("\x1b[F\x1b[H", "session-00"),
    ],
)
async def test_arrow_selection_and_enter_restore(
    keyboard: Keyboard, keys: str, selected: str
) -> None:
    task = await keyboard.start()
    keyboard.pipe.send_text(keys + "\r")
    assert await asyncio.wait_for(task, 2) == selected
    assert not keyboard.deleted


@pytest.mark.parametrize("keys", ["\x1b", "\x03", "\x04"])
async def test_cancel_returns_without_deleting(keyboard: Keyboard, keys: str) -> None:
    task = await keyboard.start()
    keyboard.pipe.send_text(keys)
    assert await asyncio.wait_for(task, 2) is None
    assert not keyboard.deleted


@pytest.mark.parametrize("cancel", ["\r", "n", "N", "\x1b"])
async def test_deletion_requires_explicit_yes_and_cancel_preserves_selection(
    keyboard: Keyboard, cancel: str
) -> None:
    task = await keyboard.start()
    await keyboard.send("\x1b[Bd", lambda: keyboard.picker.confirming_delete)
    assert not keyboard.deleted
    assert "session-01" in "\n".join(keyboard.screen_lines)
    assert "确认删除" in "\n".join(keyboard.screen_lines)
    await keyboard.send(cancel, lambda: not keyboard.picker.confirming_delete)
    assert len(keyboard.picker.items) == 3
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "session-01"
    assert not keyboard.deleted


@pytest.mark.parametrize("delete_key,confirm", [("d", "y"), ("\x1b[3~", "Y")])
async def test_confirmed_delete_refreshes_list_and_clamps_selection(
    keyboard: Keyboard, delete_key: str, confirm: str
) -> None:
    task = await keyboard.start()
    await keyboard.send("\x1b[F" + delete_key, lambda: keyboard.picker.confirming_delete)
    await keyboard.send(confirm, lambda: len(keyboard.picker.items) == 2)
    assert keyboard.deleted == ["session-02"]
    assert keyboard.picker.selected_index == 1
    assert len(keyboard.original) == 3
    assert "已删除会话 session-02" in "\n".join(keyboard.screen_lines)
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "session-01"


async def test_delete_last_session_keeps_empty_picker_until_exit(keyboard: Keyboard) -> None:
    keyboard.picker.items = items(1)
    task = await keyboard.start()
    await keyboard.send("dy", lambda: not keyboard.picker.items)
    assert not task.done()
    assert "没有历史会话" in "\n".join(keyboard.screen_lines)
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) is None
    assert keyboard.deleted == ["session-00"]


async def test_delete_failure_preserves_item_and_allows_restore(keyboard: Keyboard) -> None:
    def fail(session_id: str) -> None:
        raise PersistenceError("会话正在使用，无法删除。")

    keyboard.picker._delete = fail
    task = await keyboard.start()
    await keyboard.send("dy", lambda: "删除失败" in keyboard.picker.status)
    assert not keyboard.picker.confirming_delete
    assert len(keyboard.picker.items) == 3
    assert "会话正在使用" in "\n".join(keyboard.screen_lines)
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "session-00"


@pytest.mark.parametrize("columns,rows", [(80, 24), (30, 8)])
async def test_large_list_scrolls_selected_session_into_view(
    keyboard: Keyboard, columns: int, rows: int
) -> None:
    keyboard.output.columns = columns
    keyboard.output.rows = rows
    keyboard.picker.items = items(40)
    task = await keyboard.start()
    await keyboard.send("\x1b[F", lambda: keyboard.picker.selected_index == 39)
    assert any("session-39" in line for line in keyboard.screen_lines)
    assert keyboard.picker.list_window.vertical_scroll > 0
    await keyboard.send("\x1b[5~", lambda: keyboard.picker.selected_index < 39)
    previous = keyboard.picker.selected_index
    await keyboard.send("\x1b[6~", lambda: keyboard.picker.selected_index > previous)
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(task, 2) == "session-39"


async def test_selected_rows_highlight_full_width_and_titles_are_literal(
    keyboard: Keyboard,
) -> None:
    keyboard.picker.items[0]["session_id"] = "0123456789abcdef0123456789abcdef"
    await keyboard.start()
    screen = keyboard.picker.app.renderer.last_rendered_screen
    assert screen is not None
    assert any("0123456789abcdef0123456789abcdef" in line for line in keyboard.screen_lines)
    assert any("[bold]原样显示[/bold]" in line for line in keyboard.screen_lines)
    for text in ("0123456789abcdef", "[bold]原样显示", "completed"):
        row = next(i for i, line in enumerate(keyboard.screen_lines) if text in line)
        cells = [
            screen.data_buffer[row][x]
            for x in range(keyboard.output.columns)
            if screen.data_buffer[row][x].char
        ]
        assert sum(cell.width for cell in cells) == keyboard.output.columns
        assert all("menu.selected" in cell.style for cell in cells)


async def test_picker_can_run_again_without_pending_confirmation(keyboard: Keyboard) -> None:
    first = await keyboard.start()
    await keyboard.send("d", lambda: keyboard.picker.confirming_delete)
    keyboard.pipe.send_text("\x03")
    assert await asyncio.wait_for(first, 2) is None
    second = await keyboard.start()
    assert not keyboard.picker.confirming_delete
    keyboard.pipe.send_text("\r")
    assert await asyncio.wait_for(second, 2) == "session-00"
    assert not keyboard.deleted


async def test_closed_input_returns_without_deleting(keyboard: Keyboard) -> None:
    task = await keyboard.start()
    keyboard.pipe.close()
    assert await asyncio.wait_for(task, 2) is None
    assert not keyboard.deleted
