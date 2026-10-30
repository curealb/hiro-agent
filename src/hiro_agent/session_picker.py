"""Keyboard navigation and confirmed deletion for local conversation history."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (
    FormattedTextControl,
    HSplit,
    Layout,
    ScrollOffsets,
    UIContent,
    UIControl,
    Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.utils import get_cwidth

from hiro_agent.errors import HiroError
from hiro_agent.terminal_ui import TERMINAL_STYLE, clip_text


def _plain(value: object) -> str:
    """Keep journal text on one row, without embedded terminal control codes."""
    return " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())


class _SessionList(UIControl):
    def __init__(self, picker: SessionPicker) -> None:
        self.picker = picker

    def is_focusable(self) -> bool:
        return True

    def preferred_height(
        self, width: int, max_available_height: int, wrap_lines: bool, get_line_prefix: object
    ) -> int:
        return min(max(1, len(self.picker.items) * 3), max_available_height)

    def create_content(self, width: int, height: int) -> UIContent:
        lines: list[StyleAndTextTuples] = []
        for index, item in enumerate(self.picker.items):
            selected = index == self.picker.selected_index
            fields = (
                (" › " if selected else "   ") + _plain(item["session_id"]),
                "   " + _plain(item.get("title") or "（无任务记录）"),
                f"   {_plain(item.get('status', 'unknown'))} · "
                + _plain(item.get("updated_at", "时间未知")),
            )
            for row, text in enumerate(fields):
                style = (
                    "class:menu.selected"
                    if selected
                    else "class:menu.description"
                    if row == 2
                    else "class:menu.command"
                )
                clipped = clip_text(text, width)
                lines.append([(style, clipped + " " * max(0, width - get_cwidth(clipped)))])
        if not lines:
            lines = [[("class:menu.description", "没有历史会话。Enter 或 Esc 返回。")]]
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=Point(x=0, y=self.picker.selected_index * 3),
            show_cursor=False,
        )


class SessionPicker:
    def __init__(
        self,
        items: list[dict[str, Any]],
        delete: Callable[[str], None],
        *,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.items = list(items)
        self._delete = delete
        self.selected_index = 0
        self.confirming_delete = False
        self.status = ""
        self._error = False
        control = _SessionList(self)
        self.list_window = Window(
            control,
            height=Dimension(min=1, max=12),
            dont_extend_height=True,
            scroll_offsets=ScrollOffsets(top=0, bottom=2),
            style="class:menu",
        )
        layout = Layout(
            HSplit(
                [
                    Window(FormattedTextControl(self._heading), height=1),
                    self.list_window,
                    Window(
                        FormattedTextControl(self._message),
                        height=Dimension(min=1, max=3),
                        wrap_lines=True,
                        dont_extend_height=True,
                    ),
                    Window(
                        FormattedTextControl(self._footer),
                        height=Dimension(min=1, max=2),
                        wrap_lines=True,
                        dont_extend_height=True,
                        style="class:footer",
                    ),
                ]
            ),
            focused_element=control,
        )
        self.app: Application[str | None] = Application(
            layout=layout,
            key_bindings=self._bindings(),
            style=merge_styles(
                [
                    TERMINAL_STYLE,
                    Style.from_dict(
                        {"picker.heading": "#dce5ef bold", "picker.warning": "#e8bb77"}
                    ),
                ]
            ),
            full_screen=False,
            erase_when_done=True,
            input=input,
            output=output,
        )

    def _heading(self) -> StyleAndTextTuples:
        position = self.selected_index + 1 if self.items else 0
        return [("class:picker.heading", f"历史会话 · Session  {position}/{len(self.items)}")]

    def _message(self) -> StyleAndTextTuples:
        if self.confirming_delete:
            item = self.items[self.selected_index]
            return [
                (
                    "class:picker.warning",
                    f"删除会话 {_plain(item['session_id'])}？\n"
                    + _plain(item.get("title") or "（无任务记录）")
                    + "\n会话及其子会话记录将被删除。",
                )
            ]
        return [("class:picker.warning" if self._error else "class:footer", self.status)]

    def _footer(self) -> str:
        if self.confirming_delete:
            return " y 确认删除 · Enter / n / Esc 取消"
        return " ↑/↓ 选择 · Enter 恢复 · Delete / d 删除 · Esc 返回"

    def _move(self, offset: int) -> None:
        if self.items:
            self.selected_index = max(0, min(len(self.items) - 1, self.selected_index + offset))
        self.status = ""
        self._error = False

    def _cancel_deletion(self) -> None:
        self.confirming_delete = False
        self.status = "已取消删除。"
        self._error = False

    def _confirm_deletion(self) -> None:
        session_id = str(self.items[self.selected_index]["session_id"])
        self.confirming_delete = False
        try:
            self._delete(session_id)
        except (HiroError, OSError, ValueError) as exc:
            self.status = "删除失败：" + _plain(exc)
            self._error = True
            return
        self.items.pop(self.selected_index)
        self.selected_index = min(self.selected_index, max(0, len(self.items) - 1))
        self.status = f"已删除会话 {session_id}。"
        self._error = False

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        navigating = Condition(lambda: not self.confirming_delete)
        confirming = Condition(lambda: self.confirming_delete)

        @bindings.add("down", filter=navigating)
        @bindings.add("c-n", filter=navigating)
        def next_session(event: KeyPressEvent) -> None:
            self._move(1)

        @bindings.add("up", filter=navigating)
        @bindings.add("c-p", filter=navigating)
        def previous_session(event: KeyPressEvent) -> None:
            self._move(-1)

        @bindings.add("home", filter=navigating)
        def first_session(event: KeyPressEvent) -> None:
            self._move(-len(self.items))

        @bindings.add("end", filter=navigating)
        def last_session(event: KeyPressEvent) -> None:
            self._move(len(self.items))

        @bindings.add("pageup", filter=navigating)
        def previous_page(event: KeyPressEvent) -> None:
            self._move(-self._page_size())

        @bindings.add("pagedown", filter=navigating)
        def next_page(event: KeyPressEvent) -> None:
            self._move(self._page_size())

        @bindings.add("enter", filter=navigating)
        def restore(event: KeyPressEvent) -> None:
            event.app.exit(
                result=str(self.items[self.selected_index]["session_id"]) if self.items else None
            )

        @bindings.add("d", filter=navigating)
        @bindings.add("delete", filter=navigating)
        def request_delete(event: KeyPressEvent) -> None:
            if self.items:
                self.confirming_delete = True
                self.status = ""

        @bindings.add("y", filter=confirming)
        @bindings.add("Y", filter=confirming)
        def confirm_delete(event: KeyPressEvent) -> None:
            self._confirm_deletion()

        @bindings.add("n", filter=confirming)
        @bindings.add("N", filter=confirming)
        @bindings.add("enter", filter=confirming)
        def decline_delete(event: KeyPressEvent) -> None:
            self._cancel_deletion()

        @bindings.add("escape")
        def escape(event: KeyPressEvent) -> None:
            if self.confirming_delete:
                self._cancel_deletion()
            else:
                event.app.exit(result=None)

        @bindings.add("c-c")
        @bindings.add("c-d")
        def cancel(event: KeyPressEvent) -> None:
            event.app.exit(result=None)

        return bindings

    def _page_size(self) -> int:
        info = self.list_window.render_info
        return max(1, info.window_height // 3) if info else 4

    async def run(self) -> str | None:
        self.confirming_delete = False
        try:
            return await self.app.run_async()
        except (EOFError, KeyboardInterrupt):
            return None
