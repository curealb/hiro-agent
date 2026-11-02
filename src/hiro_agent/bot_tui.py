"""Full-screen Bot conversations; the independent worker owns all execution."""

from __future__ import annotations

import asyncio
import io
import json
import os
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
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Button, CheckboxList, Dialog, Label, TextArea
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from hiro_agent.config import load_config
from hiro_agent.errors import ConfigurationError, HiroError
from hiro_agent.kernel_contracts import InteractionRecord, InteractionReply, TaskRecord, TaskState
from hiro_agent.presentation import assistant_markdown
from hiro_agent.run_display import usage_summary
from hiro_agent.terminal_ui import clip_text
from hiro_agent.tui_backend import BotSummary, ChatEntry, Conversation, TuiBackend
from hiro_agent.ui_commands import (
    NAVIGATION_COMMANDS,
    AgentCommandCompleter,
    AgentCompletion,
    AgentTargetCompletion,
    addressed_agent,
    resolve_agent,
)

TUI_COMMANDS = {
    **NAVIGATION_COMMANDS,
    "/clear": "清空当前 Agent 的历史和上下文（保留配置）",
    "/delete-agent": "删除当前 Agent 及其历史",
    "/init-agents": "补齐 Hiro、Agent 管理员、代码编写员",
    "/select": "选择文字：拖选后用终端快捷键复制，F4 返回",
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
        "surface": "bg:#232136 #ece8f5",
        "transcript": "bg:#232136 #ece8f5",
        "transcript.meta": "#aaa2bf",
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
        "agent": "#9ccfd8 bold",
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


def _terminal_color_depth() -> ColorDepth | None:
    # prompt-toolkit otherwise defaults to 256 colors even when COLORTERM
    # advertises RGB support, which reduces our dark purple to plain gray.
    override = ColorDepth.from_env()
    if override is not None:
        return override
    if os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"} or os.environ.get(
        "TERM", ""
    ).lower().endswith(("-direct", "-truecolor")):
        return ColorDepth.DEPTH_24_BIT
    return None


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
        self._expanded_tools: set[str] = set()
        self._tool_rows: dict[int, str] = {}

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

    def toggle_tools(self) -> None:
        detail = self.owner.details
        keys = {entry.entry_id for entry in detail.entries if entry.entry_id} if detail else set()
        if keys <= self._expanded_tools:
            self._expanded_tools.difference_update(keys)
        else:
            self._expanded_tools.update(keys)
        self.owner.app.invalidate()

    def create_content(self, width: int, height: int) -> UIContent:
        self.height = height
        detail = self.owner.details
        bot = self.owner.selected_bot
        key = (
            width,
            bot,
            detail is not None,
            detail.entries if detail else (),
            detail.error if detail else None,
            tuple(
                (task.id, task.state, task.cumulative_metrics.model_dump_json())
                for task in detail.tasks
            )
            if detail
            else (),
            frozenset(self._expanded_tools),
        )
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
        self._tool_rows = {}
        detail = self.owner.details
        bot = self.owner.selected_bot
        padding = " " * (3 if width >= 16 else 1)
        content_width = max(1, min(100, width - 2 * len(padding)))
        if bot is None:
            return [
                [("", "")],
                [("class:assistant", "   欢迎来到 Hiro")],
                [("", "")],
                [("class:muted", "   从左侧选择 Agent，开始一项任务。")],
                [("class:muted", "   输入 /init-agents 创建默认 Agent，或按 F7 导入配置。")],
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
        tasks = {task.id: task for task in detail.tasks}
        last_entries = {entry.task_id: index for index, entry in enumerate(detail.entries)}
        entries: list[ChatEntry] = []
        for index, entry in enumerate(detail.entries):
            entries.append(entry)
            task = tasks.get(entry.task_id)
            if task is not None and index == last_entries[entry.task_id]:
                entries.append(
                    ChatEntry(
                        "usage",
                        usage_summary(
                            task.cumulative_metrics,
                            pending=task.state
                            in {
                                TaskState.QUEUED,
                                TaskState.RUNNING,
                                TaskState.WAITING_APPROVAL,
                                TaskState.WAITING_INPUT,
                            },
                        ),
                        task.id,
                    )
                )
        for entry in entries:
            role = entry.role if entry.role in {"user", "agent", "assistant", "tool"} else "system"
            if role == "agent":
                sender = entry.sender_name or entry.sender_bot_id or "Agent（来源不可用）"
                task = tasks.get(entry.task_id)
                kind = "回传" if task is not None and task.source == "handoff_result" else "消息"
                title = f"{sender} · {kind}"
            else:
                title = {"user": "你", "assistant": bot.name, "tool": "工具", "system": "任务"}[
                    role
                ]
            if entry.preview is not None:
                title += " · 生成中" if entry.preview == "streaming" else " · 输出中断（未完成）"
            text = _safe(entry.text)
            if role == "tool" and entry.entry_id:
                summary, _, text = text.partition("\n")
                expanded = entry.entry_id in self._expanded_tools
                marker, action = ("▾", "收起") if expanded else ("▸", "展开")
                self._tool_rows[len(lines)] = entry.entry_id
                lines.append(
                    [("class:tool", _line(f"{padding}{marker} {summary} · {action}", width))]
                )
                if not expanded:
                    lines.append([("", "")])
                    continue
            elif entry.role != "usage":
                lines.append([(f"class:{role}", _line(padding + title, width))])
            buffer = io.StringIO()
            console = Console(
                file=buffer,
                width=content_width,
                force_terminal=True,
                color_system="truecolor",
                markup=False,
                highlight=False,
                theme=TUI_MARKDOWN_THEME,
            )
            if role == "assistant":
                markdown = assistant_markdown(text)
                markdown.code_theme = "dracula"
                console.print(markdown)
            else:
                console.print(Text(text))
            for fragments in split_lines(to_formatted_text(ANSI(buffer.getvalue().rstrip("\n")))):
                body_style = "class:transcript.meta" if role in {"tool", "system"} else ""
                lines.append(
                    [
                        ("", padding),
                        *((f"{body_style} {part[0]}", part[1]) for part in fragments),
                    ]
                )
            lines.append([("", "")])
            if role in {"user", "agent", "assistant"}:
                lines.append([("", "")])
        return lines

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if self.owner.dialog is not None or self.owner.selection_mode:
            return NotImplemented
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.owner.app.layout.focus(self)
            key = self._tool_rows.get(mouse_event.position.y)
            if key is not None and mouse_event.button == MouseButton.LEFT:
                self._expanded_tools.symmetric_difference_update({key})
                self.follow = False
                self.anchor = mouse_event.position.y
                self.owner.app.invalidate()
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
        self._answer_bot_ids: dict[str, str] = {}
        self.unread: set[str] = set()
        self.exit_action: Literal["chat"] | None = None
        self.worker_online = False
        self.selection_mode = False
        self.notice = "/agent 切换 · /chat 聊天 · /help 帮助"
        self.dialog: Dialog | None = None
        self.answer: TextArea | None = None
        self._answer_id: str | None = None
        self._refresh_lock = asyncio.Lock()
        self._starting_worker = False
        self._submitting: set[str] = set()
        self._replying: set[str] = set()
        self._managing_agents = False
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
                    style="class:transcript",
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
            ],
            # A named style materializes blank cells; the default palette alone
            # leaves untouched cells transparent to the terminal's own background.
            style="class:surface",
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
            color_depth=_terminal_color_depth,
            full_screen=True,
            # Keep an independent redraw timer alive even while a snapshot is
            # waiting for an executor callback; updates must not need a keypress.
            refresh_interval=self.refresh_interval,
            mouse_support=Condition(lambda: not self.selection_mode),
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
        if self.selection_mode:
            return self._body
        if not self.narrow:
            return self._wide
        if self._compact_pane == "agents":
            return self._sidebar
        return self._body

    def focus_composer(self) -> None:
        self._compact_pane = "chat"
        self.app.layout.focus(self.composer)

    def toggle_selection_mode(self) -> None:
        if self.dialog is not None:
            return
        if self._managing_agents:
            self.notice = "Agent 操作正在保存，请稍后进入选择模式"
            self.app.invalidate()
            return
        self.selection_mode = not self.selection_mode
        self.composer.buffer.complete_state = None
        if self.selection_mode:
            self.app.layout.focus(self.transcript_control)
        else:
            self.focus_composer()
            if self.app.is_running:
                self._spawn(self.refresh())
        self.app.invalidate()

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
        if self.selection_mode:
            return "  选择模式 · 拖选文字后使用终端复制快捷键 · F4 / Esc 返回"
        if self.composer.text.lstrip().startswith("/"):
            return "  /agent ID 或名称 · /tui · /chat · /select · /help · Tab 补全"
        name = self.selected_bot.name if self.selected_bot else "Agent"
        return f"  给 {_safe(name)} 发消息   Enter 发送 · Alt+Enter 换行 · / 命令"

    def _footer(self) -> str:
        from prompt_toolkit.utils import get_cwidth

        width = self.app.output.get_size().columns
        if self.selection_mode:
            return _line(
                "  拖选 · Ctrl+Shift+C 复制 · PgUp/PgDn 翻页 · F4 / Esc 返回"
                if width >= 70
                else "  拖选复制 · F4 / Esc 返回",
                width,
            )
        hint = "F2 Agent · F3 请求 · F4 选择 · F5 工具 · F8 取消 · Ctrl+Q 退出"
        if width < 100:
            hint = "F2 Agent · F4 选择 · Ctrl+Q 退出"
        if width < 60:
            hint = "F4 选择 · Ctrl+Q 退出"
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
        if self.dialog is not None or self.selection_mode or bot_id == self.selected_bot_id:
            return
        if not any(bot.bot_id == bot_id for bot in self.bot_summaries):
            return
        self._save_draft(None)
        self._set_selected_bot(bot_id)
        if self.app.is_running:
            self._spawn(self.refresh())

    def _set_selected_bot(self, bot_id: str | None) -> None:
        self.selected_bot_id = bot_id
        self.details = None
        if bot_id is not None:
            self.unread.discard(bot_id)
        draft = self.drafts.get(bot_id, "") if bot_id is not None else ""
        self.composer.buffer.set_document(Document(draft, len(draft)), bypass_readonly=True)
        self.transcript_control.reset()
        self.app.invalidate()

    def _discard_bot_state(self, bot_id: str, *, deleted: bool = False) -> None:
        self.unread.discard(bot_id)
        self._pending_index.pop(bot_id, None)
        for interaction_id, owner in list(self._answer_bot_ids.items()):
            if owner == bot_id:
                self.answer_drafts.pop(interaction_id, None)
                self._answer_bot_ids.pop(interaction_id, None)
        if self.selected_bot_id == bot_id:
            if deleted:
                self.selected_bot_id = None
            self.details = None
            self.composer.text = ""
            self.transcript_control.reset()
        self.drafts.pop(bot_id, None)

    def move_bot(self, delta: int) -> None:
        bots = self.filtered_bots
        if not bots:
            return
        index = next((i for i, b in enumerate(bots) if b.bot_id == self.selected_bot_id), -1)
        index = max(0, min(len(bots) - 1, index + delta))
        self.select_bot(bots[index].bot_id)

    async def refresh(self) -> None:
        async with self._refresh_lock:
            if self.selection_mode:
                return
            try:
                bots = await asyncio.to_thread(self.backend.list_bots)
                if self.selection_mode:
                    return
                old = {b.bot_id: b for b in self.bot_summaries}
                for bot in bots:
                    if bot.bot_id in old and old[bot.bot_id] != bot:
                        if bot.bot_id != self.selected_bot_id:
                            self.unread.add(bot.bot_id)
                self.bot_summaries = bots
                live_ids = {bot.bot_id for bot in bots}
                for removed_id in old.keys() - live_ids:
                    self._discard_bot_state(removed_id, deleted=True)
                if self.selected_bot_id not in live_ids and (
                    self.selected_bot_id is not None or bots
                ):
                    if self.selected_bot_id is not None:
                        self._discard_bot_state(self.selected_bot_id, deleted=True)
                    self._set_selected_bot(bots[0].bot_id if bots else None)
                bot_id = self.selected_bot_id
                if bot_id is not None:
                    detail = await asyncio.to_thread(self.backend.conversation, bot_id)
                    if self.selection_mode:
                        return
                    if self.selected_bot_id == bot_id:
                        self.details = detail
                online = await asyncio.to_thread(self.backend.worker_running)
                if self.selection_mode:
                    return
                self.worker_online = online
            except (HiroError, OSError, ValueError) as exc:
                if not self.selection_mode:
                    self.notice = "读取失败：" + _safe(str(exc))
            self.app.invalidate()

    def _spawn(self, coroutine: Coroutine[Any, Any, None]) -> None:
        self.app.create_background_task(coroutine)

    def send_current(self) -> None:
        text = self.composer.text.strip()
        if self.dialog is not None or self.selection_mode or not text:
            return
        if text.startswith("/"):
            if text.split(maxsplit=1)[0].casefold() in TUI_COMMANDS:
                self._command(text)
                return
            try:
                addressed_agent(
                    text, ((bot.bot_id, bot.name) for bot in self.bot_summaries), TUI_COMMANDS
                )
            except ConfigurationError as exc:
                self.notice = _safe(str(exc))
                self.app.invalidate()
                return
        if self._managing_agents:
            self.notice = "Agent 操作正在保存，请稍后发送消息"
            self.app.invalidate()
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
        elif command in {"/chat", "/exit", "/clear", "/delete-agent", "/init-agents"} and (
            self._submitting or self._replying or self._managing_agents
        ):
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
                    name = next(bot.name for bot in self.bot_summaries if bot.bot_id == bot_id)
                    self.notice = f"已切换 Agent：{_safe(name)}"
        else:
            self.composer.text = ""
            if command == "/tui":
                self.notice = "当前已在全屏工作台"
            elif command == "/select":
                self.toggle_selection_mode()
            elif command in {"/clear", "/delete-agent"}:
                self.open_agent_cleanup(delete=command == "/delete-agent")
            elif command == "/init-agents":
                self._managing_agents = True
                self._spawn(self._manage_agents("initialize"))
            elif command == "/help":
                close = Button("返回", handler=self.close_dialog)
                dialog = Dialog(
                    title="命令帮助",
                    body=Label(
                        "/Agent1 /Agent2 消息  指定多个通信对象（保持当前对话）\n"
                        "/agent              选择 Agent\n"
                        "/agent ID 或名称    切换到指定 Agent\n"
                        "/tui                全屏工作台\n"
                        "/chat               传统聊天\n"
                        "/clear              清空当前 Agent 的历史和上下文\n"
                        "/delete-agent       删除当前 Agent 及其历史\n"
                        "/init-agents        补齐三个默认 Agent（保留已有配置）\n"
                        "/select             选择文字（F4 / Esc 返回）\n"
                        "/help               显示帮助\n"
                        "/exit               退出界面\n\n"
                        "点击工具标题展开 / 收起，F5 切换所有工具结果。\n"
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

    def open_agent_cleanup(self, *, delete: bool = False) -> None:
        if self.dialog is not None or self.selection_mode or self._managing_agents:
            return
        bot = self.selected_bot
        if bot is None:
            self.notice = "请先选择 Agent；可用 /init-agents 创建默认 Agent"
            return
        bot_id = bot.bot_id
        action = "delete" if delete else "clear"
        title = "删除 Agent" if delete else "清空历史和上下文"
        explanation = (
            "将删除此 Agent 的身份、配置以及全部任务、会话和记忆。\n\n此操作无法撤销。"
            if delete
            else "将删除此 Agent 的全部任务、会话和记忆，清空历史和上下文。\n\n"
            "保留 Agent 身份和配置。此操作无法撤销。"
        )

        def accept() -> None:
            self.close_dialog()
            self._managing_agents = True
            self._spawn(self._manage_agents(action, bot_id))

        cancel = Button("返回", handler=self.close_dialog)
        confirm = Button("确认删除" if delete else "确认清空", handler=accept)
        dialog = Dialog(
            title=title,
            body=Label(f"{_safe(bot.name)} ({_safe(bot_id)})\n\n{explanation}"),
            buttons=[cancel, confirm],
            width=Dimension(min=28, preferred=76, max=90),
        )
        self._show_dialog(dialog, cancel)

    async def _manage_agents(self, action: str, bot_id: str | None = None) -> None:
        label = {"clear": "清空历史", "delete": "删除 Agent", "initialize": "初始化 Agent"}[action]
        prefer_default = action == "initialize" and self.selected_bot_id is None
        try:
            # Serialize with refresh so a previously loaded conversation cannot
            # reappear after its history has been removed.
            async with self._refresh_lock:
                if action == "initialize":
                    result = await asyncio.to_thread(self.backend.initialize_agents)
                    if prefer_default and self.selected_bot_id is None:
                        default_ids = result.get("created", []) + result.get("existing", [])
                        if "default" in default_ids:
                            self._set_selected_bot("default")
                    self.notice = f"默认 Agent 已就绪，新建 {len(result.get('created', []))} 个"
                else:
                    assert bot_id is not None
                    operation = (
                        self.backend.delete_bot
                        if action == "delete"
                        else self.backend.clear_history
                    )
                    await asyncio.to_thread(operation, bot_id)
                    self._discard_bot_state(bot_id, deleted=action == "delete")
                    self.notice = (
                        f"已删除 Agent：{_safe(bot_id)}"
                        if action == "delete"
                        else f"已清空历史和上下文：{_safe(bot_id)}（保留配置）"
                    )
            await self.refresh()
        except (HiroError, OSError, ValueError) as exc:
            self.notice = label + "失败：" + _safe(str(exc))
        finally:
            self._managing_agents = False
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
        if (
            self.answer is not None
            and self._answer_id is not None
            and self._answer_id in self._answer_bot_ids
        ):
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
        if self.dialog is not None or self._managing_agents or request is None or bot_id is None:
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
            self._answer_bot_ids[interaction_id] = bot_id
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
            self._answer_bot_ids.pop(interaction_id, None)
            self.notice = "答复已保存"
            if self.auto_start_worker:
                await self.ensure_worker()
        except (HiroError, OSError, ValueError) as exc:
            self.notice = "答复失败：" + _safe(str(exc))
        finally:
            self._replying.discard(interaction_id)
        await self.refresh()

    def open_cancel(self) -> None:
        if (
            self.dialog is not None
            or self._managing_agents
            or self.details is None
            or self.selected_bot_id is None
        ):
            return
        bot_id = self.selected_bot_id
        states = {
            TaskState.RUNNING,
            TaskState.QUEUED,
            TaskState.WAITING_APPROVAL,
            TaskState.WAITING_INPUT,
            TaskState.NEEDS_RECONCILIATION,
        }
        tasks = [task for task in reversed(self.details.tasks) if task.state in states]
        if not tasks:
            self.notice = "当前 Agent 没有可取消的任务"
            return
        choices = CheckboxList(
            [(task.id, f"{STATUS_LABELS[task.state]} · {_safe(task.text)[:65]}") for task in tasks]
        )
        validation = Label("")

        def select_all() -> None:
            choices.current_values = [task.id for task in tasks]
            validation.text = ""
            self.app.layout.focus(choices)
            self.app.invalidate()

        def select_none() -> None:
            choices.current_values = []
            self.app.layout.focus(choices)
            self.app.invalidate()

        def cancel() -> None:
            selected = tuple(task for task in tasks if task.id in choices.current_values)
            if not selected:
                validation.text = "请至少勾选一项任务。"
                self.app.invalidate()
                return
            self.close_dialog()
            if any(task.state == TaskState.NEEDS_RECONCILIATION for task in selected):
                self.open_reconciliation_cancel(bot_id, selected)
            else:
                self._spawn(self._cancel_tasks(bot_id, selected))

        dialog = Dialog(
            title="批量取消任务",
            body=HSplit(
                [
                    Label("空格勾选 · ↑↓ 移动 · Tab 切换按钮"),
                    Box(
                        choices,
                        padding=0,
                        height=Dimension(min=1, preferred=min(len(tasks), 10), max=10),
                    ),
                    validation,
                ]
            ),
            buttons=[
                Button("取消所选任务", handler=cancel, width=16),
                Button("全选", handler=select_all),
                Button("全不选", handler=select_none),
                Button("返回", self.close_dialog),
            ],
            width=Dimension(min=28, preferred=80, max=90),
        )
        self._show_dialog(dialog, choices)

    def open_reconciliation_cancel(self, bot_id: str, tasks: tuple[TaskRecord, ...]) -> None:
        note = TextArea(height=3, multiline=True)
        validation = Label("")
        uncertain = [task for task in tasks if task.state == TaskState.NEEDS_RECONCILIATION]

        def confirm() -> None:
            explanation = note.text.strip()
            if not explanation:
                validation.text = "请填写核实说明后再确认。"
                self.app.invalidate()
                return
            self.close_dialog()
            self._spawn(self._cancel_tasks(bot_id, tasks, explanation))

        dialog = Dialog(
            title="核实并取消任务",
            body=HSplit(
                [
                    Label(f"已选择 {len(tasks)} 项任务，其中 {len(uncertain)} 项待核实。"),
                    Box(
                        TextArea(
                            text="\n".join(
                                f"{_safe(task.text)[:80]}：{_safe(task.reason)[:120]}"
                                for task in uncertain
                            ),
                            read_only=True,
                            focusable=False,
                            scrollbar=True,
                        ),
                        padding=0,
                        height=Dimension(min=1, preferred=min(len(uncertain), 4), max=4),
                    ),
                    Label(
                        "请先核实文件或外部操作的实际结果。\n"
                        "取消只结束任务，不会撤销已发生的改动。\n\n"
                        "核实说明（必填，将记录到每项待核实任务）："
                    ),
                    note,
                    validation,
                ]
            ),
            buttons=[
                Button("返回", handler=self.close_dialog),
                Button("确认核实并取消", handler=confirm, width=20),
            ],
            width=Dimension(min=28, preferred=80, max=90),
        )
        self._show_dialog(dialog, note)

    async def _cancel_tasks(
        self, bot_id: str, tasks: tuple[TaskRecord, ...], note: str = ""
    ) -> None:
        cancelled = stopping = finished = 0
        errors: list[str] = []
        for task in tasks:
            try:
                if task.state == TaskState.NEEDS_RECONCILIATION:
                    saved = await asyncio.to_thread(
                        self.backend.resolve_cancel, bot_id, task.id, note
                    )
                else:
                    saved = await asyncio.to_thread(self.backend.cancel, bot_id, task.id)
                if saved.state == TaskState.NEEDS_RECONCILIATION:
                    raise ConfigurationError("任务已变为待核实，请重新选择并填写核实说明")
                if saved.state == TaskState.CANCELLED:
                    cancelled += 1
                elif saved.cancel_requested:
                    stopping += 1
                else:
                    finished += 1
            except (HiroError, OSError, ValueError) as exc:
                errors.append(f"{_safe(task.id)}：{_safe(str(exc))}")
        self.notice = f"已取消 {cancelled} 项 · 已请求停止 {stopping} 项"
        if finished:
            self.notice += f" · 已结束，跳过 {finished} 项"
        if errors:
            self.notice += f" · 失败 {len(errors)} 项（{errors[0]}）"
        await self.refresh()

    def open_import(self) -> None:
        if self.dialog is not None or self._managing_agents:
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
        no_dialog = Condition(lambda: self.dialog is None)
        normal = no_dialog & Condition(lambda: not self.selection_mode)
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

        @bindings.add("f4", filter=no_dialog)
        def select_text(event: KeyPressEvent) -> None:
            self.toggle_selection_mode()

        @bindings.add("f5", filter=normal)
        def toggle_tools(event: KeyPressEvent) -> None:
            self.transcript_control.toggle_tools()

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
            state = buffer.complete_state
            completion = state.current_completion if state else None
            target = completion
            if target is None and state is not None and len(state.completions) == 1:
                target = state.completions[0]
            if isinstance(target, AgentTargetCompletion):
                if completion is None:
                    buffer.apply_completion(target)
                buffer.complete_state = None
                return
            if (
                completion is None
                and state is not None
                and len(state.completions) == 1
                and isinstance(state.completions[0], AgentCompletion)
            ):
                buffer.apply_completion(state.completions[0])
                self.send_current()
            elif isinstance(completion, AgentCompletion):
                buffer.complete_state = None
                self.send_current()
            elif completion is not None:
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

        @bindings.add("tab", filter=~command_input & Condition(lambda: not self.selection_mode))
        def next_focus(event: KeyPressEvent) -> None:
            event.app.layout.focus_next()

        @bindings.add("s-tab", filter=~command_input & Condition(lambda: not self.selection_mode))
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

        @bindings.add("pageup", filter=no_dialog)
        def page_up(event: KeyPressEvent) -> None:
            self.transcript_control.scroll(-max(3, self.transcript_control.height - 2))

        @bindings.add("pagedown", filter=no_dialog)
        def page_down(event: KeyPressEvent) -> None:
            self.transcript_control.scroll(max(3, self.transcript_control.height - 2))

        @bindings.add("c-end", filter=no_dialog)
        def latest(event: KeyPressEvent) -> None:
            self.transcript_control.follow = True

        @bindings.add("escape")
        def escape(event: KeyPressEvent) -> None:
            if self.dialog is not None:
                self.close_dialog()
            elif self.selection_mode:
                self.toggle_selection_mode()
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
