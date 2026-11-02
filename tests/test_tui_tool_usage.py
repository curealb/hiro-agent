"""Tool disclosure and usage updates through the rendered terminal interface."""

from dataclasses import replace

import pytest
from test_bot_tui import ALPHA, ALT_DOWN, ALT_UP, BETA, Keyboard
from test_bot_tui import keyboard as keyboard  # noqa: F401
from test_tui_backend import backend as backend  # noqa: F401
from test_tui_backend import claim, journal

from hiro_agent.contracts import RunMetrics
from hiro_agent.kernel_contracts import TaskState
from hiro_agent.tui_backend import ChatEntry, TuiBackend

F5 = "\x1b[15~"


async def click_tool(keyboard: Keyboard, title: str, *, expanded: bool) -> None:
    row = next(index for index, line in enumerate(keyboard.lines) if title in line)
    column = keyboard.lines[row].index(title) + 1
    marker = "▾" if expanded else "▸"
    await keyboard.send(
        f"\x1b[<0;{column};{row + 1}M\x1b[<0;{column};{row + 1}m",
        lambda: any(marker in line and title in line for line in keyboard.lines),
    )


async def test_tool_mouse_and_f5_preserve_disclosure_across_refresh_and_bot_switch(
    keyboard: Keyboard,
) -> None:
    snapshot = keyboard.backend.conversations[ALPHA]
    task_id = snapshot.tasks[0].id
    keyboard.backend.conversations[ALPHA] = replace(
        snapshot,
        entries=(
            ChatEntry("tool", "read_file · 完成\nREAD_DETAIL", task_id, "alpha-read"),
            ChatEntry("tool", "run_shell · 完成\nSHELL_DETAIL", task_id, "alpha-shell"),
        ),
    )
    await keyboard.start()
    assert "READ_DETAIL" not in keyboard.screen
    assert "SHELL_DETAIL" not in keyboard.screen

    await click_tool(keyboard, "read_file", expanded=True)
    assert "READ_DETAIL" in keyboard.screen
    assert "SHELL_DETAIL" not in keyboard.screen
    await keyboard.ui.refresh()
    await keyboard.send(ALT_DOWN, lambda: keyboard.ui.selected_bot_id == BETA)
    await keyboard.send(ALT_UP, lambda: "read_file" in keyboard.screen)
    assert "READ_DETAIL" in keyboard.screen
    assert "SHELL_DETAIL" not in keyboard.screen

    await click_tool(keyboard, "read_file", expanded=False)
    assert "READ_DETAIL" not in keyboard.screen
    await keyboard.focus_composer()
    await keyboard.send("unsent draft" + F5, lambda: "SHELL_DETAIL" in keyboard.screen)
    assert "READ_DETAIL" in keyboard.screen
    await keyboard.send(F5, lambda: "SHELL_DETAIL" not in keyboard.screen)
    assert "READ_DETAIL" not in keyboard.screen
    assert keyboard.ui.composer.text == "unsent draft"
    assert not keyboard.backend.submissions


async def test_usage_refreshes_even_when_transcript_and_bot_summary_stay_the_same(
    keyboard: Keyboard,
) -> None:
    await keyboard.start()
    assert "Token 等待返回" in keyboard.screen
    snapshot = keyboard.backend.conversations[ALPHA]
    metrics = RunMetrics(
        model_requests=1,
        worker_steps=1,
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=75,
        cache_reported_input_tokens=100,
    )
    task = snapshot.tasks[0].model_copy(update={"cumulative_metrics": metrics})
    keyboard.backend.conversations[ALPHA] = replace(snapshot, tasks=(task,))
    await keyboard.rendered(
        lambda: keyboard.tasks.append(
            keyboard.ui.app.create_background_task(keyboard.ui.refresh())
        ),
        lambda: "Token 120" in keyboard.screen,
    )
    assert "输入 100 / 输出 20" in keyboard.screen
    assert "缓存命中 75 / 100 (75.0%)" in keyboard.screen

    # A second checkpoint and a resumed/final result replace the cumulative
    # snapshot; the UI must not add the previous snapshot's counters again.
    metrics = metrics.model_copy(
        update={
            "model_requests": 2,
            "worker_steps": 2,
            "input_tokens": 200,
            "output_tokens": 30,
            "cached_input_tokens": 100,
            "cache_reported_input_tokens": 200,
        }
    )
    task = task.model_copy(update={"state": TaskState.SUCCEEDED, "cumulative_metrics": metrics})
    keyboard.backend.conversations[ALPHA] = replace(snapshot, tasks=(task,))
    await keyboard.rendered(
        lambda: keyboard.tasks.append(
            keyboard.ui.app.create_background_task(keyboard.ui.refresh())
        ),
        lambda: "Token 230" in keyboard.screen,
    )
    assert "缓存命中 100 / 200 (50.0%)" in keyboard.screen
    assert "Token 120" not in keyboard.screen


@pytest.mark.parametrize("known_cache", [False, True])
async def test_tui_distinguishes_missing_cache_telemetry_from_zero_hits(
    keyboard: Keyboard,
    known_cache: bool,
) -> None:
    snapshot = keyboard.backend.conversations[ALPHA]
    metrics = RunMetrics(
        model_requests=1,
        worker_steps=1,
        input_tokens=10,
        output_tokens=2,
        cache_usage_missing=0 if known_cache else 1,
        cache_reported_input_tokens=10 if known_cache else 0,
    )
    task = snapshot.tasks[0].model_copy(
        update={
            "state": TaskState.SUCCEEDED,
            "cumulative_metrics": metrics,
        }
    )
    keyboard.backend.conversations[ALPHA] = replace(snapshot, tasks=(task,))
    await keyboard.start()
    assert "Token 12" in keyboard.screen
    expected = "缓存命中 0 / 10 (0.0%)" if known_cache else "缓存 未提供"
    assert expected in keyboard.screen


def test_tool_snapshot_keeps_full_output_and_stable_identity(backend: TuiBackend) -> None:
    claimed = claim(backend)
    call = {"id": "read", "name": "read_file", "arguments": '{"path":"long.txt"}'}
    output = "long output\n" * 300 + "FINAL_LINE"
    path = journal(
        backend.workspace,
        claimed.task.session_id,
        ("assistant.message", {"message": {"tool_calls": [call]}}),
        ("tool.result", {"call_id": "read", "ok": True, "value": output}),
    )
    entry = next(entry for entry in backend.conversation("writer").entries if entry.role == "tool")
    assert entry.text == "read_file · 完成\n" + output
    assert entry.entry_id
    assert backend.conversation("writer").entries[-1] == entry
    assert path.stat().st_size > 2000
