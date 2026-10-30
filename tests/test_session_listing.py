"""Useful, chronological session previews without modifying saved journals."""

import json
from pathlib import Path

from hiro_agent.session import JsonlSession, list_sessions


async def test_session_title_and_task_count_preserve_original_history(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "conversation")
    task = "  修复\n\t登录问题  " + "详细要求 " * 30
    await session.append("task.received", {"text": task})
    await session.append("task.continued", {"text": "continue"})
    await session.append("task.received", {"text": "add regression tests"})
    await session.close()
    original = session.path.read_bytes()

    [summary] = list_sessions(tmp_path)

    assert summary["title"] == " ".join(task.split())[:79] + "…"
    assert len(summary["title"]) == 80
    assert summary["task_count"] == 2
    assert summary["updated_at"] == session.events[-1].timestamp.isoformat()
    assert session.path.read_bytes() == original


async def test_sessions_sort_by_instant_with_stable_id_ties(tmp_path: Path) -> None:
    # Lexical dates and session filenames would both produce the wrong order.
    timestamps = {
        "a-older": "2026-10-31T10:00:00+08:00",
        "z-newer": "2026-10-31T03:00:00+00:00",
        "b-tie": "2026-10-31T11:00:00+08:00",
    }
    for session_id, timestamp in timestamps.items():
        session = await JsonlSession.open(tmp_path, session_id)
        await session.close()
        record = json.loads(session.path.read_text())
        record["timestamp"] = timestamp
        session.path.write_text(json.dumps(record) + "\n")

    summaries = list_sessions(tmp_path)

    assert [item["session_id"] for item in summaries] == ["b-tie", "z-newer", "a-older"]
    assert all(item["title"] == "" and item["task_count"] == 0 for item in summaries)


async def test_unavailable_sessions_sort_last_without_repair(tmp_path: Path) -> None:
    corrupt = await JsonlSession.open(tmp_path, "a-corrupt")
    await corrupt.close()
    damaged = corrupt.path.read_bytes() + b'{"version":'
    corrupt.path.write_bytes(damaged)
    ready = await JsonlSession.open(tmp_path, "z-ready")
    await ready.append("task.received", {"text": "review files"})
    await ready.close()
    busy = await JsonlSession.open(tmp_path, "b-busy")
    try:
        summaries = list_sessions(tmp_path)
        assert [(item["session_id"], item["status"]) for item in summaries] == [
            ("z-ready", "idle"),
            ("a-corrupt", "unreadable"),
            ("b-busy", "busy"),
        ]
        assert all("task_count" not in item for item in summaries[1:])
        assert corrupt.path.read_bytes() == damaged
    finally:
        await busy.close()


async def test_session_previews_exclude_child_sessions(tmp_path: Path) -> None:
    parent = await JsonlSession.open(tmp_path, "parent")
    await parent.append("task.received", {"text": "review project"})
    await parent.close()
    child = await JsonlSession.open(tmp_path, "child")
    await child.append(
        "subagent.created",
        {
            "parent_session_id": "parent",
            "parent_run_id": "parent-run",
            "child_session_id": "child",
            "child_run_id": "child-run",
            "call_id": "delegate-call",
            "role": "explorer",
        },
    )
    await child.append("task.received", {"text": "find files"})
    await child.close()

    [summary] = list_sessions(tmp_path)
    assert summary["session_id"] == "parent"
    assert summary["title"] == "review project"
