"""Restored conversation displays only durable human messages, in event order."""

import io
from typing import Any

import pytest
from rich.console import Console

from hiro_agent.contracts import SessionEvent
from hiro_agent.presentation import show_session_history


def events(*records: tuple[str, dict[str, Any]]) -> list[SessionEvent]:
    return [
        SessionEvent(session_id="history", seq=index, type=kind, data=data)
        for index, (kind, data) in enumerate(records, 1)
    ]


def history_console(*, terminal: bool = False) -> Console:
    return Console(
        file=io.StringIO(),
        force_terminal=terminal,
        color_system=None,
        width=120,
        record=True,
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_history_preserves_conversation_order_and_hides_internal_events(terminal: bool) -> None:
    transcript = events(
        ("prompt.system", {"text": "private system instructions"}),
        ("task.received", {"text": "first user task [red]literal brackets[/red]"}),
        (
            "assistant.message",
            {
                "message": {
                    "text": "first assistant answer",
                    "provider_metadata": {"internal": "private provider data"},
                }
            },
        ),
        (
            "assistant.message",
            {
                "message": {
                    "tool_calls": [
                        {"id": "call", "name": "read_file", "arguments": "private arguments"}
                    ]
                }
            },
        ),
        ("tool.result", {"value": "private tool output"}),
        ("task.continued", {"text": "private generated continuation instructions"}),
        ("assistant.message", {"message": {"text": "second assistant answer"}}),
        ("task.received", {"text": "second user task"}),
    )
    original = [event.model_dump() for event in transcript]
    console = history_console(terminal=terminal)

    show_session_history(console, transcript)

    output = console.export_text()
    ordered = [
        "历史对话",
        "first user task [red]literal brackets[/red]",
        "first assistant answer",
        "你：/continue",
        "second assistant answer",
        "second user task",
    ]
    positions = [output.index(text) for text in ordered]
    assert positions == sorted(positions)
    assert "private" not in output
    assert "read_file" not in output
    assert [event.model_dump() for event in transcript] == original


def test_pipe_history_preserves_full_assistant_markdown() -> None:
    markdown = "# Summary\n\n**Complete** answer\n\n```python\nprint('[bold]literal')\n```\n"
    markdown += "More context.\n" * 1000
    markdown += "final line beyond preview limits"
    console = history_console()

    show_session_history(console, events(("assistant.message", {"message": {"text": markdown}})))

    assert markdown in console.export_text()


def test_terminal_history_renders_assistant_markdown() -> None:
    console = history_console(terminal=True)
    show_session_history(
        console,
        events(("assistant.message", {"message": {"text": "**Formatted answer**"}})),
    )
    output = console.export_text()
    assert "Formatted answer" in output
    assert "**Formatted answer**" not in output
    assert "Hiro" in output


@pytest.mark.parametrize(
    "transcript",
    [[], events(("prompt.system", {"text": "hidden"}), ("assistant.message", {"message": {}}))],
)
def test_empty_history_explains_that_no_conversation_was_saved(
    transcript: list[SessionEvent],
) -> None:
    console = history_console()
    show_session_history(console, transcript)
    assert "这个会话还没有对话。" in console.export_text()
