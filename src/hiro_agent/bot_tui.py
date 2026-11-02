"""Full-screen Bot conversations; the independent worker owns all execution."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Coroutine
from typing import Any, Literal

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, split_lines, to_formatted_text
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    ScrollOffsets,
    UIContent,
    UIControl,
    VSplit,
    Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Button, Dialog, Label, RadioList, TextArea
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from hiro_agent.config import load_config
from hiro_agent.errors import ConfigurationError, HiroError
from hiro_agent.kernel_contracts import InteractionRecord, InteractionReply, TaskState
from hiro_agent.presentation import assistant_markdown
from hiro_agent.terminal_ui import clip_text
from hiro_agent.tui_backend import BotSummary, Conversation, TuiBackend
from hiro_agent.ui_commands import NAVIGATION_COMMANDS, AgentCommandCompleter, resolve_agent

TUI_COMMANDS = {
    **NAVIGATION_COMMANDS,
    "/help": "显示命令帮助",
    "/exit": "退出界面（后台任务继续运行）",
}

STATUS_LABELS = {
    "idle": "就绪",
    "queued": "排队中",
    "running": "执行中",
    "waiting_approval": "待批准",
    "waiting_input": "待答复",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "needs_reconciliation": "待核实",
}

TUI_STYLE = Style.from_dict(
    {
        "": "bg:#232136 #ece8f5",
        "brand": "bg:#201e31 #c4a7ed bold",
        "sidebar": "bg:#29263d #d5cfe3",
        "sidebar.title": "bg:#29263d #aaa2bf bold",
        "sidebar.row": "bg:#29263d #ece8f5",
        "sidebar.selected": "bg:#44385e #f5efff bold",
        "sidebar.preview": "bg:#29263d #aaa2bf",
        "sidebar.selected.preview": "bg:#44385e #d5c9e8",
        "sidebar.badge": "#e7c28f bold",
        "search": "bg:#332e49 #e0d9ed",
        "header": "bg:#201e31 #ece8f5 bold",
        "muted": "#aaa2bf",
        "border": "#514866",
        "accent": "#c4a7ed",
        "user": "#e2b8d9 bold",
        "assistant": "#c4a7ed bold",
        "system": "#b6abc9",
        "tool": "#aaa2bf",
        "warning": "#e7c28f",
        "error": "#ee9ba9",
        "pending": "bg:#393043 #e7c28f",
        "composer": "bg:#2e2a45 #ece8f5",
        "footer": "bg:#201e31 #aaa2bf",
        "button": "bg:#44385e #e0d2f2",
        "button.focused": "bg:#c4a7ed #232136 bold",
        "dialog": "bg:#2e2a45 #ece8f5",
        "dialog.body": "bg:#2e2a45 #ece8f5",
        "dialog frame.label": "#c4a7ed bold",
        "dialog text-area": "bg:#201e31 #ece8f5",
        "radio-selected": "#c4a7ed bold",
        "completion-menu.completion": "bg:#332e49 #e0d9ed",
        "completion-menu.completion.current": "bg:#44385e #f5efff bold",
        "completion-menu.meta.completion": "bg:#332e49 #aaa2bf",
        "completion-menu.meta.completion.current": "bg:#44385e #d5c9e8",
        "scrollbar.background": "bg:#29263d",
        "scrollbar.button": "bg:#74648f",
    }
)

# Rich emits explicit ANSI colors, so Markdown needs the same local palette.
TUI_MARKDOWN_THEME = Theme(
    {
        "markdown.code": "#e0c9fa on #332e49",
        "markdown.code_block": "#ece8f5 on #2e2a45",
        "markdown.block_quote": "#b6abc9",
        "markdown.link": "#c4a7ed",
        "markdown.link_url": "underline #c4a7ed",
        "markdown.h1": "bold #e0c9fa",
        "markdown.h2": "bold #c4a7ed",
        "markdown.h3": "bold #c4a7ed",
        "markdown.h4": "bold #c4a7ed",
        "markdown.h5": "bold #c4a7ed",
        "markdown.h6": "bold #c4a7ed",
        "markdown.list": "#ece8f5",
        "markdown.item.bullet": "#c4a7ed",
        "markdown.item.number": "#c4a7ed",
        "markdown.hr": "#74648f",
        "markdown.table.border": "#74648f",
        "markdown.table.header": "bold #e0c9fa",
    }
)


def _safe(text: str) -> str:
    """Treat Bot names, model output and errors as text, never terminal commands."""
    return "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) < 160))


def _line(text: str, width: int) -> str:
    from prompt_toolkit.utils import get_cwidth

    clipped = clip_text(_safe(text).replace("\n", " ").replace("\t", " "), width)
    return clipped + " " * max(0, width - get_cwidth(clipped))


class _Sidebar(UIControl):
    def __init__(self, owner: BotTUI) -> None:
        self.owner = owner

    def is_focusable(self) -> bool:
        return True

    def create_content(self, width: int, height: int) -> UIContent:
        bots = self.owner.filtered_bots
        lines: list[StyleAndTextTuples] = []
        selected_row = 0
        for index, bot in enumerate(bots):
            selected = bot.bot_id == self.owner.selected_bot_id
            if selected:
                selected_row = index * 3
            style = "class:sidebar.selected" if selected else "class:sidebar.row"
            preview_style = style + ".preview" if selected else "class:sidebar.preview"
            marker = "▌" if selected else " "
            unread = " •" if bot.bot_id in self.owner.unread else ""
            badge = f"  [{bot.pending_count}]" if bot.pending_count else ""
            lines.append([(style, _line(f"{marker} ● {bot.name}{unread}{badge}", width))])
            preview = STATUS_LABELS.get(bot.status, bot.status) + " · " + (bot.preview or bot.model)
            lines.append([(preview_style, _line("    " + preview, width))])
            lines.append([("class:sidebar", " " * width)])
        if not lines:
            lines = [[("class:muted", _line("  没有匹配的 Agent", width))]]
        return UIContent(
            get_line=lambda i: lines[i],
            line_count=len(lines),
            cursor_position=Point(x=0, y=selected_row),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if self.owner.dialog is not None:
            return NotImplemented
        if mouse_event.event_type in {MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN}:
            self.owner.move_bot(-1 if mouse_event.event_type == MouseEventType.SCROLL_UP else 1)
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            bots = self.owner.filtered_bots
            index = mouse_event.position.y // 3
            if 0 <= index < len(bots):
                self.owner.select_bot(bots[index].bot_id)
                if self.owner.narrow:
                    self.owner.focus_composer()
                else:
                    self.owner.app.layout.focus(self)
            return None
        return NotImplemented


class _Transcript(UIControl):
    def __init__(self, owner: BotTUI) -> None:
        self.owner = owner
        self.follow = True
        self.anchor = 0
        self.height = 20
        self._key: object = None
        self._lines: list[StyleAndTextTuples] = [[("", "")]]

    def is_focusable(self) -> bool:
        return True

    def reset(self) -> None:
        self.follow = True
        self.anchor = 0
        self._key = None

    def scroll(self, amount: int) -> None:
        self.follow = False
        self.anchor = max(0, min(len(self._lines) - 1, self.anchor + amount))
        self.owner.app.invalidate()

    def create_content(self, width: int, height: int) -> UIContent:
        self.height = height
        detail = self.owner.details
        bot = self.owner.selected_bot
        key = (width, bot, detail.entries if detail else (), detail.error if detail else None)
        if key != self._key:
            self._key = key
            self._lines = self._render(width)
        if self.follow:
            self.anchor = len(self._lines) - 1
        self.anchor = max(0, min(self.anchor, len(self._lines) - 1))
        return UIContent(
            get_line=lambda i: self._lines[i],
            line_count=len(self._lines),
            cursor_position=Point(x=0, y=self.anchor),
            show_cursor=False,
        )

    def _render(self, width: int) -> list[StyleAndTextTuples]:
        detail = self.owner.details
        bot = self.owner.selected_bot
        if bot is None:
            return [
                [("", "")],
                [("class:assistant", "   欢迎来到 Hiro")],
                [("", "")],
                [("class:muted", "   从左侧选择 Agent，开始一项任务。")],
                [("class:muted", "   还没有 Agent？按 F7 导入 Bot 配置。")],
                [("class:muted", "   也可用 hiro-agent tui --demo 体验离线演示。")],
            ]
        if detail is None:
            return [[("class:muted", "  正在读取对话…")]]
        lines: list[StyleAndTextTuples] = [[("", "")]]
        if detail.error:
            lines.append([("class:error", _line("  " + detail.error, width))])
            lines.append([("", "")])
        if not detail.entries:
            lines.extend(
                [
                    [("class:assistant", _line(f"   {bot.name} 已就绪", width))],
                    [("", "")],
                    [("class:muted", "   在下方发送第一条消息。")],
                    [("class:muted", "   任务在后台执行，可以随时切换 Agent。")],
                ]
            )
        for entry in detail.entries:
            role = entry.role if entry.role in {"user", "assistant", "tool"} else "system"
            title = {"user": "你", "assistant": bot.name, "tool": "工具", "system": "任务"}[role]
            lines.append([(f"class:{role}", _line("  " + title, width))])
            buffer = io.StringIO()
            console = Console(
                file=buffer,
                width=max(8, width - 5),
                force_terminal=True,
                color_system="truecolor",
                markup=False,
                highlight=False,
                theme=TUI_MARKDOWN_THEME,
            )
            text = _safe(entry.text)
            if role == "assistant":
                markdown = assistant_markdown(text)
                markdown.code_theme = "dracula"
                console.print(markdown)
            else:
                console.print(Text(text))
            for fragments in split_lines(to_formatted_text(ANSI(buffer.getvalue().rstrip("\n")))):
                lines.append([("", "  "), *fragments])
            lines.append([("", "")])
        return lines

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.owner.app.layout.focus(self)
            return None
        return NotImplemented


class BotTUI:
    def __init__(
        self,
        backend: TuiBackend,
        *,
        auto_start_worker: bool = True,
        input: Input | None = None,
        output: Output | None = None,
        refresh_interval: float = 0.5,
    ) -> None:
        self.backend = backend
        self.auto_start_worker = auto_start_worker
        self.refresh_interval = refresh_interval
        self.bot_summaries: list[BotSummary] = []
        self.selected_bot_id: str | None = None
        self.details: Conversation | None = None
        self.drafts: dict[str, str] = {}
        self.answer_drafts: dict[str, str] = {}
        self.unread: set[str] = set()
        self.exit_action: Literal["chat"] | None = None
        self.worker_online = False
        self.notice = "/agent 切换 · /chat 聊天 · /help 帮助"
        self.dialog: Dialog | None = None
        self.answer: TextArea | None = None
        self._answer_id: str | None = None
        self._refresh_lock = asyncio.Lock()
        self._starting_worker = False
        self._submitting: set[str] = set()
        self._replying: set[str] = set()
        self._pending_index: dict[str, int] = {}
        self._compact_pane = "chat"
        self.search = TextArea(
            height=1,
            multiline=False,
            prompt=" 搜索 ",
            style="class:search",
            focus_on_click=True,
        )
        self.composer = TextArea(
            height=Dimension(min=2, preferred=3, max=5),
            multiline=True,
            style="class:composer",
            focus_on_click=True,
            wrap_lines=True,
            prompt=" › ",
            completer=AgentCommandCompleter(
                TUI_COMMANDS,
                lambda: ((bot.bot_id, bot.name) for bot in self.bot_summaries),
            ),
            complete_while_typing=True,
        )
        self.composer.buffer.on_text_changed += self._save_draft
        self.search.buffer.on_text_changed += lambda _: self.app.invalidate()
        self.sidebar_control = _Sidebar(self)
        self.transcript_control = _Transcript(self)
        self._sidebar = HSplit(
            [
                Window(FormattedTextControl(self._sidebar_title), height=2),
                self.search,
                Window(height=1),
                Window(
                    self.sidebar_control,
                    scroll_offsets=ScrollOffsets(top=1, bottom=2),
                    right_margins=[ScrollbarMargin()],
                ),
                VSplit(
                    [Window(width=1), Button("+ 导入 Agent", self.open_import, width=20), Window()],
                    height=1,
                ),
                Window(FormattedTextControl(self._worker_label), height=2),
            ],
            style="class:sidebar",
        )
        pending = ConditionalContainer(
            HSplit(
                [
                    Window(FormattedTextControl(self._pending_label), height=1),
                    VSplit(
                        [
                            Button("查看并处理 F3", handler=self.open_interaction, width=18),
                            Window(width=1),
                            Button("下一条", handler=self.next_interaction, width=10),
                            Window(),
                        ],
                        height=1,
                    ),
                ],
                style="class:pending",
            ),
            filter=Condition(lambda: bool(self.details and self.details.interactions)),
        )
        self._body = HSplit(
            [
                Window(FormattedTextControl(self._header), height=2, style="class:header"),
                Window(height=1, char="─", style="class:border"),
                Window(
                    self.transcript_control,
                    wrap_lines=False,
                    right_margins=[ScrollbarMargin(display_arrows=True)],
                ),
                pending,
                Window(height=1, char="─", style="class:border"),
                Window(
                    FormattedTextControl(self._composer_label), height=1, style="class:composer"
                ),
                self.composer,
            ]
        )
        self._wide = VSplit(
            [
                Box(self._sidebar, width=Dimension(min=24, preferred=30, max=34), padding=0),
                Window(width=1, char="│", style="class:border"),
                self._body,
            ]
        )
        content = HSplit(
            [
                Window(
                    FormattedTextControl(lambda: "  HIRO  /  Agent 工作台"),
                    height=1,
                    style="class:brand",
                ),
                DynamicContainer(self._main_layout),
                Window(FormattedTextControl(self._footer), height=1, style="class:footer"),
            ]
        )
        self._completion_float = Float(
            xcursor=True,
            ycursor=True,
            content=CompletionsMenu(
                max_height=8,
                scroll_offset=1,
                extra_filter=Condition(
                    lambda: self.dialog is None and self.app.layout.has_focus(self.composer)
                ),
            ),
        )
        self._container = FloatContainer(content, floats=[self._completion_float])
        self.app: Application[None] = Application(
            layout=Layout(self._container, focused_element=self.composer),
            key_bindings=self._bindings(),
            style=TUI_STYLE,
            full_screen=True,
            mouse_support=True,
            input=input,
            output=output,
        )
        self.app.before_render += self._synchronize_pane

    def _synchronize_pane(self, _: Application[None]) -> None:
        # Mouse focus can change in wide mode before a terminal resize. Keep the
        # focused control visible when the two panes become one compact pane.
        if self.dialog is None:
            if self.app.layout.has_focus(self._sidebar):
                self._compact_pane = "agents"
            elif self.app.layout.has_focus(self._body):
                self._compact_pane = "chat"

    @property
    def narrow(self) -> bool:
        return hasattr(self, "app") and self.app.output.get_size().columns < 78

    def _main_layout(self) -> Any:
        if not self.narrow:
            return self._wide
        if self._compact_pane == "agents":
            return self._sidebar
        return self._body

    def focus_composer(self) -> None:
        self._compact_pane = "chat"
        self.app.layout.focus(self.composer)

    def focus_agents(self) -> None:
        self._compact_pane = "agents"
        self.app.layout.focus(self.sidebar_control)

    @property
    def filtered_bots(self) -> list[BotSummary]:
        query = self.search.text.strip().casefold()
        return [
            bot
            for bot in self.bot_summaries
            if query in f"{bot.name} {bot.bot_id} {bot.model}".casefold()
        ]

    @property
    def selected_bot(self) -> BotSummary | None:
        return next((b for b in self.bot_summaries if b.bot_id == self.selected_bot_id), None)

    def _sidebar_title(self) -> StyleAndTextTuples:
        return [("class:sidebar.title", f"\n  AGENTS  {len(self.bot_summaries)}")]

    def _worker_label(self) -> StyleAndTextTuples:
        label = "● worker 在线" if self.worker_online else "○ worker 离线 · F6 启动"
        return [("class:accent" if self.worker_online else "class:warning", "\n  " + label)]

    def _header(self) -> StyleAndTextTuples:
        bot = self.selected_bot
        if bot is None:
            return [("class:header", "  对话\n"), ("class:muted", "  选择或导入一个 Agent")]
        status = STATUS_LABELS.get(bot.status, bot.status)
        return [
            ("class:header", f"  {_safe(bot.name)}  "),
            ("class:accent", f"{status}\n"),
            ("class:muted", f"  {_safe(bot.model)} · {bot.task_count} 项任务"),
        ]

    def _composer_label(self) -> str:
        if self.composer.text.lstrip().startswith("/"):
            return "  /agent ID 或名称 · /tui · /chat · /help · Tab 补全"
        name = self.selected_bot.name if self.selected_bot else "Agent"
        return f"  给 {_safe(name)} 发消息   Enter 发送 · Alt+Enter 换行 · / 命令"

    def _footer(self) -> str:
        from prompt_toolkit.utils import get_cwidth

        hint = "F2 Agent · F3 请求 · F8 取消 · PgUp/PgDn 历史 · Ctrl+Q 退出"
        width = self.app.output.get_size().columns
        if width < 100:
            hint = "F2 Agent · F3 请求 · Ctrl+Q 退出"
        if width < 60:
            hint = "F2 Agent · Ctrl+Q 退出"
        space = max(0, width - get_cwidth(hint) - 5)
        return _line(" " + _line(self.notice, space) + " | " + hint, width)

    def _pending(self) -> InteractionRecord | None:
        if not self.details or not self.details.interactions:
            return None
        index = self._pending_index.get(self.details.bot_id, 0) % len(self.details.interactions)
        return self.details.interactions[index]

    def _pending_label(self) -> str:
        request = self._pending()
        if request is None:
            return ""
        kind = "批准操作" if request.request.kind == "approval" else "回答问题"
        return "  待处理 · " + kind + "  " + _safe(request.request.prompt).replace("\n", " ")

    def _save_draft(self, _: object) -> None:
        if self.selected_bot_id is not None:
            self.drafts[self.selected_bot_id] = self.composer.text

    def select_bot(self, bot_id: str) -> None:
        if self.dialog is not None or bot_id == self.selected_bot_id:
            return
        if not any(bot.bot_id == bot_id for bot in self.bot_summaries):
            return
        self._save_draft(None)
        self.selected_bot_id = bot_id
        self.details = None
        self.unread.discard(bot_id)
        draft = self.drafts.get(bot_id, "")
        self.composer.buffer.set_document(Document(draft, len(draft)), bypass_readonly=True)
        self.transcript_control.reset()
        self.app.invalidate()
        if self.app.is_running:
            self._spawn(self.refresh())

    def move_bot(self, delta: int) -> None:
        bots = self.filtered_bots
        if not bots:
            return
        index = next((i for i, b in enumerate(bots) if b.bot_id == self.selected_bot_id), -1)
        index = max(0, min(len(bots) - 1, index + delta))
        self.select_bot(bots[index].bot_id)

    async def refresh(self) -> None:
        async with self._refresh_lock:
            try:
                bots = await asyncio.to_thread(self.backend.list_bots)
                old = {b.bot_id: b for b in self.bot_summaries}
                for bot in bots:
                    if bot.bot_id in old and old[bot.bot_id] != bot:
                        if bot.bot_id != self.selected_bot_id:
                            self.unread.add(bot.bot_id)
                self.bot_summaries = bots
                if self.selected_bot_id is None and bots:
                    self.select_bot(bots[0].bot_id)
                bot_id = self.selected_bot_id
                if bot_id is not None:
                    detail = await asyncio.to_thread(self.backend.conversation, bot_id)
                    if self.selected_bot_id == bot_id:
                        self.details = detail
                self.worker_online = await asyncio.to_thread(self.backend.worker_running)
            except (HiroError, OSError, ValueError) as exc:
                self.notice = "读取失败：" + _safe(str(exc))
            self.app.invalidate()

    def _spawn(self, coroutine: Coroutine[Any, Any, None]) -> None:
        self.app.create_background_task(coroutine)

    def send_current(self) -> None:
        text = self.composer.text.strip()
        if self.dialog is not None or not text:
            return
        if text.startswith("/"):
            self._command(text)
            return
        bot_id = self.selected_bot_id
        if bot_id is None or bot_id in self._submitting:
            return
        self._submitting.add(bot_id)
        self.composer.text = ""
        self.drafts[bot_id] = ""
        self.transcript_control.follow = True
        self._spawn(self._submit(bot_id, text))

    def _command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0].casefold()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if command not in TUI_COMMANDS:
            self.notice = "未知命令：" + _safe(parts[0]) + "；输入 /help 查看帮助"
        elif command != "/agent" and argument:
            self.notice = f"用法：{command}（不接受参数）；输入 /help 查看帮助"
        elif command in {"/chat", "/exit"} and (self._submitting or self._replying):
            self.notice = f"操作正在保存，请稍后重试 {command}"
        elif command == "/agent":
            if not argument:
                self.composer.text = ""
                self.search.text = ""
                self.focus_agents()
                self.notice = "↑/↓ 选择 Agent · Enter 返回对话 · F7 导入配置"
            else:
                try:
                    bot_id = resolve_agent(
                        argument, ((bot.bot_id, bot.name) for bot in self.bot_summaries)
                    )
                except ConfigurationError as exc:
                    self.notice = _safe(str(exc))
                else:
                    self.composer.text = ""
                    self.search.text = ""
                    self.select_bot(bot_id)
                    self.focus_composer()
                    self.notice = f"已切换 Agent：{_safe(bot_id)}"
        else:
            self.composer.text = ""
            if command == "/tui":
                self.notice = "当前已在全屏工作台"
            elif command == "/help":
                close = Button("返回", handler=self.close_dialog)
                dialog = Dialog(
                    title="命令帮助",
                    body=Label(
                        "/agent              选择 Agent\n"
                        "/agent ID 或名称    切换到指定 Agent\n"
                        "/tui                全屏工作台\n"
                        "/chat               传统聊天\n"
                        "/help               显示帮助\n"
                        "/exit               退出界面\n\n"
                        "Tab / ↑↓ 补全，Enter 确认。后台任务继续运行。"
                    ),
                    buttons=[close],
                    width=Dimension(min=28, preferred=64, max=80),
                )
                self._show_dialog(dialog, close)
            else:
                if command == "/chat":
                    self.exit_action = "chat"
                self.app.exit()
        self.app.invalidate()

    async def _submit(self, bot_id: str, text: str) -> None:
        from uuid import uuid4

        try:
            await asyncio.to_thread(
                self.backend.submit, bot_id, text, idempotency_key="tui:" + uuid4().hex
            )
            self.notice = "消息已提交，后台执行"
            if self.auto_start_worker:
                await self.ensure_worker()
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "提交失败：" + _safe(str(exc))
            if not self.drafts.get(bot_id):
                self.drafts[bot_id] = text
                if self.selected_bot_id == bot_id and not self.composer.text:
                    self.composer.text = text
        finally:
            self._submitting.discard(bot_id)
        await self.refresh()

    async def ensure_worker(self) -> None:
        if self._starting_worker:
            return
        self._starting_worker = True
        try:
            child = await asyncio.to_thread(self.backend.start_worker)
            for _ in range(40):
                if await asyncio.to_thread(self.backend.worker_running):
                    self.worker_online = True
                    self.notice = "worker 已连接 · 退出界面后仍继续运行"
                    break
                if child is not None and child.poll() is not None:
                    raise ConfigurationError("worker 启动失败，请检查 .hiro-agent/worker.log")
                await asyncio.sleep(0.1)
            else:
                self.notice = "worker 尚未就绪，任务已保存在队列中"
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "worker 未连接：" + _safe(str(exc))
        finally:
            self._starting_worker = False
            self.app.invalidate()

    def _show_dialog(self, dialog: Dialog, focus: Any) -> None:
        self.composer.buffer.complete_state = None
        self.dialog = dialog
        self._container.floats[:] = [self._completion_float, Float(content=dialog)]
        self.app.layout.focus(focus)
        self.app.invalidate()

    def close_dialog(self) -> None:
        if self.answer is not None and self._answer_id is not None:
            self.answer_drafts[self._answer_id] = self.answer.text
        self.dialog = None
        self.answer = None
        self._answer_id = None
        self._container.floats[:] = [self._completion_float]
        self.focus_composer()
        self.app.invalidate()

    def next_interaction(self) -> None:
        if self.selected_bot_id is not None:
            self._pending_index[self.selected_bot_id] = (
                self._pending_index.get(self.selected_bot_id, 0) + 1
            )
        self.app.invalidate()

    def open_interaction(self) -> None:
        request = self._pending()
        bot_id = self.selected_bot_id
        if self.dialog is not None or request is None or bot_id is None:
            return
        interaction_id = request.id

        def reply(value: InteractionReply) -> None:
            self.close_dialog()
            self._replying.add(interaction_id)
            self._spawn(self._reply(bot_id, interaction_id, value))

        title = "批准一次操作" if request.request.kind == "approval" else "回答 Agent 的问题"
        body = _safe(request.request.prompt)
        if request.request.operation is not None:
            operation = request.request.operation
            arguments = json.dumps(
                json.loads(operation.arguments_json), ensure_ascii=False, indent=2
            )
            body += f"\n\n工具：{operation.tool_name}\n{arguments}"
        prompt = TextArea(
            text=body,
            read_only=True,
            scrollbar=True,
            height=Dimension(min=3, preferred=8, max=12),
            wrap_lines=True,
        )
        if request.request.kind == "approval":
            buttons = [
                Button("批准一次", handler=lambda: reply(InteractionReply(approved=True))),
                Button("拒绝", handler=lambda: reply(InteractionReply(approved=False))),
                Button("返回", handler=self.close_dialog),
            ]
            dialog = Dialog(
                title=title,
                body=prompt,
                buttons=buttons,
                width=Dimension(min=28, preferred=80, max=90),
            )
            self._show_dialog(dialog, buttons[0])
        else:
            self._answer_id = interaction_id
            self.answer = TextArea(
                text=self.answer_drafts.get(interaction_id, ""),
                height=3,
                multiline=True,
                focus_on_click=True,
            )

            def submit_answer() -> None:
                assert self.answer is not None
                if self.answer.text.strip():
                    reply(InteractionReply(text=self.answer.text))

            dialog = Dialog(
                title=title,
                body=HSplit([prompt, Label("你的答复："), self.answer]),
                buttons=[
                    Button("提交答复", handler=submit_answer),
                    Button("返回", self.close_dialog),
                ],
                width=Dimension(min=28, preferred=80, max=90),
            )
            self._show_dialog(dialog, self.answer)

    async def _reply(self, bot_id: str, interaction_id: str, reply: InteractionReply) -> None:
        try:
            await asyncio.to_thread(self.backend.reply, bot_id, interaction_id, reply)
            self.answer_drafts.pop(interaction_id, None)
            self.notice = "答复已保存"
            if self.auto_start_worker:
                await self.ensure_worker()
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "答复失败：" + _safe(str(exc))
        finally:
            self._replying.discard(interaction_id)
        await self.refresh()

    def open_cancel(self) -> None:
        if self.dialog is not None or self.details is None or self.selected_bot_id is None:
            return
        bot_id = self.selected_bot_id
        states = {
            TaskState.RUNNING,
            TaskState.QUEUED,
            TaskState.WAITING_APPROVAL,
            TaskState.WAITING_INPUT,
        }
        tasks = [task for task in reversed(self.details.tasks) if task.state in states]
        if not tasks:
            self.notice = "当前 Agent 没有可取消的任务"
            return
        choices = RadioList(
            [(task.id, f"{STATUS_LABELS[task.state]} · {_safe(task.text)[:65]}") for task in tasks]
        )

        def cancel() -> None:
            task_id = choices.current_value
            self.close_dialog()
            self._spawn(self._cancel(bot_id, task_id))

        dialog = Dialog(
            title="取消任务",
            body=choices,
            buttons=[Button("取消所选任务", handler=cancel), Button("返回", self.close_dialog)],
            width=Dimension(min=28, preferred=80, max=90),
        )
        self._show_dialog(dialog, choices)

    async def _cancel(self, bot_id: str, task_id: str) -> None:
        try:
            await asyncio.to_thread(self.backend.cancel, bot_id, task_id)
            self.notice = "取消请求已保存"
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "取消失败：" + _safe(str(exc))
        await self.refresh()

    def open_import(self) -> None:
        if self.dialog is not None:
            return
        path_input = TextArea(height=1, multiline=False)

        def accept() -> None:
            path = path_input.text.strip()
            if path:
                self.close_dialog()
                self._spawn(self._import(path))

        dialog = Dialog(
            title="导入 / 更新 Agent",
            body=HSplit([Label("Bot JSON 配置路径（需要 bot.id）："), path_input]),
            buttons=[Button("导入", handler=accept), Button("返回", self.close_dialog)],
            width=Dimension(min=28, preferred=72, max=85),
        )
        self._show_dialog(dialog, path_input)

    async def _import(self, path: str) -> None:
        from pathlib import Path

        try:
            config = await asyncio.to_thread(load_config, Path(path).expanduser())
            if config.bot is None:
                raise ConfigurationError("配置需要 bot.id 和 bot.name")
            await asyncio.to_thread(self.backend.kernel.bots.save, config)
            await self.refresh()
            self.select_bot(config.bot.id)
            self.notice = "Agent 已导入"
            if self.auto_start_worker:
                await self.ensure_worker()
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "导入失败：" + _safe(str(exc))
        self.app.invalidate()

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        normal = Condition(lambda: self.dialog is None)
        in_composer = normal & Condition(lambda: self.app.layout.has_focus(self.composer))
        in_sidebar = normal & Condition(lambda: self.app.layout.has_focus(self.sidebar_control))
        command_input = in_composer & Condition(
            lambda: self.composer.text.lstrip().startswith("/") and "\n" not in self.composer.text
        )

        @bindings.add("c-q")
        def quit_app(event: KeyPressEvent) -> None:
            event.app.exit()

        @bindings.add("c-d", filter=in_composer & Condition(lambda: not self.composer.text))
        def eof(event: KeyPressEvent) -> None:
            event.app.exit()

        @bindings.add("f2", filter=normal)
        @bindings.add("c-b", filter=normal)
        def agents(event: KeyPressEvent) -> None:
            self.focus_agents()

        @bindings.add("up", filter=in_sidebar)
        def previous(event: KeyPressEvent) -> None:
            self.move_bot(-1)

        @bindings.add("down", filter=in_sidebar)
        def following(event: KeyPressEvent) -> None:
            self.move_bot(1)

        @bindings.add("enter", filter=in_sidebar)
        def enter_agent(event: KeyPressEvent) -> None:
            bots = self.filtered_bots
            if bots and not any(b.bot_id == self.selected_bot_id for b in bots):
                self.select_bot(bots[0].bot_id)
            self.focus_composer()

        @bindings.add("escape", "up", filter=normal)
        def previous_anywhere(event: KeyPressEvent) -> None:
            self.move_bot(-1)

        @bindings.add("escape", "down", filter=normal)
        def next_anywhere(event: KeyPressEvent) -> None:
            self.move_bot(1)

        @bindings.add("escape", "up", filter=~normal)
        @bindings.add("escape", "down", filter=~normal)
        def keep_modal(event: KeyPressEvent) -> None:
            pass

        @bindings.add("enter", filter=in_composer)
        def send(event: KeyPressEvent) -> None:
            buffer = self.composer.buffer
            if buffer.complete_state and buffer.complete_state.current_completion:
                buffer.complete_state = None
            else:
                self.send_current()

        @bindings.add("tab", filter=command_input)
        @bindings.add("down", filter=command_input)
        def complete_next(event: KeyPressEvent) -> None:
            buffer = self.composer.buffer
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        @bindings.add("s-tab", filter=command_input)
        @bindings.add("up", filter=command_input)
        def complete_previous(event: KeyPressEvent) -> None:
            buffer = self.composer.buffer
            if buffer.complete_state:
                buffer.complete_previous()
            else:
                buffer.start_completion(select_last=True)

        @bindings.add("escape", "enter", filter=in_composer)
        @bindings.add("c-j", filter=in_composer)
        def newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c", filter=in_composer)
        def clear(event: KeyPressEvent) -> None:
            self.composer.text = ""

        @bindings.add("tab", filter=~command_input)
        def next_focus(event: KeyPressEvent) -> None:
            event.app.layout.focus_next()

        @bindings.add("s-tab", filter=~command_input)
        def previous_focus(event: KeyPressEvent) -> None:
            event.app.layout.focus_previous()

        @bindings.add("f3", filter=normal)
        def interaction(event: KeyPressEvent) -> None:
            self.open_interaction()

        @bindings.add("f6", filter=normal)
        def start_worker(event: KeyPressEvent) -> None:
            self._spawn(self.ensure_worker())

        @bindings.add("f7", filter=normal)
        def import_agent(event: KeyPressEvent) -> None:
            self.open_import()

        @bindings.add("f8", filter=normal)
        def cancel_task(event: KeyPressEvent) -> None:
            self.open_cancel()

        @bindings.add("pageup", filter=normal)
        def page_up(event: KeyPressEvent) -> None:
            self.transcript_control.scroll(-max(3, self.transcript_control.height - 2))

        @bindings.add("pagedown", filter=normal)
        def page_down(event: KeyPressEvent) -> None:
            self.transcript_control.scroll(max(3, self.transcript_control.height - 2))

        @bindings.add("c-end", filter=normal)
        def latest(event: KeyPressEvent) -> None:
            self.transcript_control.follow = True

        @bindings.add("escape")
        def escape(event: KeyPressEvent) -> None:
            if self.dialog is not None:
                self.close_dialog()
            elif self.composer.buffer.complete_state is not None:
                self.composer.buffer.cancel_completion()
            else:
                self.focus_composer()

        return bindings

    async def _poll(self) -> None:
        await self.refresh()
        if self.auto_start_worker and self.bot_summaries:
            self._spawn(self.ensure_worker())
        while True:
            await asyncio.sleep(self.refresh_interval)
            await self.refresh()

    async def run(self) -> None:
        # The TUI owns display tasks only. Quitting never closes the detached worker.
        self.exit_action = None
        await self.app.run_async(pre_run=lambda: self._spawn(self._poll()))
