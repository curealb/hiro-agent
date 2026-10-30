"""Inline terminal composer and command menu, backed by PromptSession."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Always, Condition, is_done
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import (
    BufferControl,
    ConditionalContainer,
    FloatContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    UIContent,
    UIControl,
    Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

TERMINAL_STYLE = Style.from_dict(
    {
        "menu": "bg:#171a1e #b8c2cf",
        "menu.command": "#c5cfdb",
        "menu.description": "#8993a1",
        "menu.selected": "bg:#68a7f7 #101820",
        "composer": "bg:#30343b #e2e8f0",
        "composer.prefix": "#b69aff bold",
        "composer.placeholder": "#949caa",
        "composer.approval": "#e8bb77",
        "footer": "#8b95a5",
    }
)


def clip_text(text: str, width: int) -> str:
    """Clip by terminal cells, including double-width Chinese characters."""
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    result = ""
    cells = 0
    for character in text:
        cells += get_cwidth(character)
        if cells > width - 1:
            break
        result += character
    return result + "…"


class CommandMenuControl(UIControl):
    def __init__(self, interface: TerminalInterface) -> None:
        self.interface = interface

    def preferred_height(
        self, width: int, max_available_height: int, wrap_lines: bool, get_line_prefix: object
    ) -> int:
        state = self.interface.session.default_buffer.complete_state
        return min(len(state.completions), max_available_height) if state else 0

    def create_content(self, width: int, height: int) -> UIContent:
        state = self.interface.session.default_buffer.complete_state
        completions = state.completions if state else []
        lines: list[StyleAndTextTuples] = []
        command_width = max((get_cwidth(item.text) for item in completions), default=0) + 2
        for index, completion in enumerate(completions):
            selected = index == self.interface.selected_index
            command_style = "class:menu.selected" if selected else "class:menu.command"
            description_style = "class:menu.selected" if selected else "class:menu.description"
            command = (" › " if selected else "   ") + completion.text.ljust(command_width)
            command = clip_text(command, width)
            description = clip_text(completion.display_meta_text, width - get_cwidth(command))
            padding = " " * (width - get_cwidth(command) - get_cwidth(description))
            lines.append([(command_style, command), (description_style, description + padding)])
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=Point(x=0, y=self.interface.selected_index),
            show_cursor=False,
        )


class TerminalInterface:
    def __init__(self, session: PromptSession[str]) -> None:
        self.session = session
        self.approval = False
        self.selection = False
        self.selected_index = 0
        session.default_buffer.on_text_changed += self._reset_selection

        # Keep PromptSession's input/search controls, processors and bindings.
        # Only its floating completion menus are replaced by the inline list.
        original_layout = session.layout
        for container in original_layout.walk():
            if isinstance(container, Window) and isinstance(container.content, BufferControl):
                # Grow with wrapped input, not with the terminal's empty space.
                container.dont_extend_height = Always()
            if isinstance(container, FloatContainer):
                container.floats = [
                    item
                    for item in container.floats
                    if not isinstance(item.content, CompletionsMenu | MultiColumnCompletionsMenu)
                ]

        menu = ConditionalContainer(
            Window(
                CommandMenuControl(self),
                height=Dimension(min=1, max=8),
                dont_extend_height=True,
                style="class:menu",
            ),
            filter=Condition(lambda: self.menu_visible),
        )
        composer = HSplit(
            [
                ConditionalContainer(Window(height=1), filter=~is_done),
                original_layout.container,
                ConditionalContainer(Window(height=1), filter=~is_done),
            ],
            style="class:composer",
        )
        footer = ConditionalContainer(
            Window(FormattedTextControl(self._footer), height=1, style="class:footer"),
            filter=~is_done,
        )
        layout = Layout(HSplit([menu, composer, footer]), original_layout.current_window)
        session.layout = session.app.layout = layout
        bindings = self._key_bindings()
        session.key_bindings = (
            merge_key_bindings([session.key_bindings, bindings])
            if session.key_bindings is not None
            else bindings
        )

    @property
    def menu_visible(self) -> bool:
        state = self.session.default_buffer.complete_state
        return (
            not self.approval
            and not self.selection
            and not self.session.app.is_done
            and self.session.app.layout.has_focus(self.session.default_buffer)
            and state is not None
            and bool(state.completions)
        )

    def _reset_selection(self, _: object) -> None:
        self.selected_index = 0

    def _footer(self) -> str:
        if self.approval:
            return "  y / yes 允许一次   Enter 拒绝   Ctrl+C 取消任务"
        if self.selection:
            return "  输入序号或 Session ID   Enter 确认   空输入 / Ctrl+C 取消"
        if self.menu_visible:
            return "  ↑/↓ 选择   Enter 执行   Tab 补全   Esc 关闭"
        return "  / 命令   Enter 发送   Ctrl+C 清空   Ctrl+D 退出"

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        in_menu = Condition(lambda: self.menu_visible)

        @bindings.add("down", filter=in_menu)
        @bindings.add("c-n", filter=in_menu)
        def next_command(event: KeyPressEvent) -> None:
            self._move_selection(1)

        @bindings.add("up", filter=in_menu)
        @bindings.add("c-p", filter=in_menu)
        def previous_command(event: KeyPressEvent) -> None:
            self._move_selection(-1)

        @bindings.add("tab", filter=in_menu)
        def complete_command(event: KeyPressEvent) -> None:
            self._apply_selection()

        @bindings.add("enter", filter=in_menu)
        def execute_command(event: KeyPressEvent) -> None:
            self._apply_selection()
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", filter=in_menu)
        def close_menu(event: KeyPressEvent) -> None:
            event.current_buffer.cancel_completion()

        return bindings

    def _move_selection(self, delta: int) -> None:
        state = self.session.default_buffer.complete_state
        if state and state.completions:
            self.selected_index = (self.selected_index + delta) % len(state.completions)

    def _apply_selection(self) -> None:
        buffer = self.session.default_buffer
        state = buffer.complete_state
        if state and state.completions:
            buffer.apply_completion(state.completions[self.selected_index])
