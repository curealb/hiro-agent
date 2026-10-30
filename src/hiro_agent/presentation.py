"""Presentation for interactive sessions, with plain output for logs and pipes."""

from pathlib import Path

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Markdown, TableElement
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hiro_agent import __version__
from hiro_agent.config import Config
from hiro_agent.contracts import AssistantMessage, SessionEvent


class _WrappingTable(TableElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for renderable in super().__rich_console__(console, options):
            if isinstance(renderable, Table):
                for column in renderable.columns:
                    column.overflow = "fold"
            yield renderable


class _AssistantMarkdown(Markdown):
    elements = {**Markdown.elements, "table_open": _WrappingTable}


def show_session_header(
    console: Console, *, workspace: Path, session_id: str, config: Config
) -> None:
    if not console.is_terminal:
        console.print(f"Hiro · 工作区 {workspace.resolve()}", markup=False)
        console.print(f"Session: {session_id}", markup=False)
        return

    title = Text.assemble(
        (">_ ", "bold #63a8f5"),
        ("Hiro Agent", "bold #dce5ef"),
        (f" (v{__version__})", "#8390a0"),
    )
    details = Table.grid(padding=(0, 2), expand=True)
    details.add_column(style="#8390a0", no_wrap=True)
    details.add_column(style="#bdc6d2", overflow="fold", ratio=1)
    model = "Mock · 离线演示" if config.model.adapter == "mock" else config.model.model
    details.add_row("模型", Text(model))
    details.add_row("工作区", Text(str(workspace.resolve())))
    details.add_row("Session", Text(session_id))
    console.print(
        Panel(
            Group(title, Text(""), details),
            box=box.ROUNDED,
            style="#bdc6d2 on #171a1e",
            border_style="#384250",
            padding=(0, 2),
            width=min(console.width, 88),
        )
    )


def assistant_markdown(text: str) -> Markdown:
    """Render model Markdown without interpreting Rich's bracket markup."""
    return _AssistantMarkdown(text, justify="left")


def show_assistant_message(console: Console, text: str, *, prefix: str = "") -> None:
    if console.is_terminal:
        console.print(Text.assemble(("\n● ", "#63a8f5"), ("Hiro", "bold #dce5ef")))
        if prefix:
            console.print(Text(prefix.rstrip(), style="dim"))
        console.print(assistant_markdown(text))
    else:
        # Pipes and logs retain the original Markdown source.
        console.print(Text(prefix + text), highlight=False)


def show_session_history(console: Console, events: list[SessionEvent]) -> None:
    """Replay the human conversation without exposing or changing internal events."""
    console.print("\n历史对话", style="bold", markup=False)
    displayed = False
    for event in events:
        if event.type == "task.received":
            console.print(Text.assemble(("\n你：", "bold"), event.data["text"]))
            displayed = True
        elif event.type == "task.continued":
            console.print("\n你：/continue", style="dim", markup=False)
            displayed = True
        elif event.type == "assistant.message":
            message = AssistantMessage.model_validate(event.data["message"])
            if message.text:
                show_assistant_message(console, message.text)
                displayed = True
    if not displayed:
        console.print("这个会话还没有对话。", style="dim", markup=False)
