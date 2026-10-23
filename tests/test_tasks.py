"""Task inspection never starts recovery or changes existing journals."""

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from hiro_agent import cli
from hiro_agent.contracts import AssistantMessage, DelegationLink, RunMetrics, ToolCall, ToolResult
from hiro_agent.errors import PersistenceError
from hiro_agent.session import JsonlSession, MemorySession, read_session_events
from hiro_agent.tasks import list_tasks, summarize_tasks


async def declarations(session: MemorySession, run: str = "run-one") -> list[ToolCall]:
    await session.append("run.started", {}, run)
    calls = [
        ToolCall(
            id=f"{run}-{role}",
            name="delegate_task",
            arguments=json.dumps({"role": role, "task": f"Inspect {role} files"}),
        )
        for role in ("explorer", "reviewer")
    ]
    await session.append(
        "assistant.message", {"message": AssistantMessage(tool_calls=calls).model_dump()}, run
    )
    return calls


async def start(session: MemorySession, call: ToolCall, run: str = "run-one") -> DelegationLink:
    await session.append("tool.called", {"call": call.model_dump()}, run)
    await session.append("tool.started", {"call": call.model_dump()}, run)
    link = DelegationLink(
        call_id=call.id,
        parent_session_id=session.session_id,
        parent_run_id=run,
        child_session_id=f"child-{call.id}",
        child_run_id=f"child-run-{call.id}",
        role=json.loads(call.arguments)["role"],
        model_id="mock:demo",
    )
    await session.append("delegation.started", link.model_dump(), run)
    return link


async def finish(session: MemorySession, link: DelegationLink) -> None:
    metrics = RunMetrics(model_requests=2, tool_calls=1)
    result = ToolResult(
        call_id=link.call_id,
        ok=True,
        value={
            "delegation": link.model_dump(),
            "status": "completed",
            "metrics": metrics.model_dump(),
            "report": {"summary": f"Evidence for {link.role}"},
        },
    )
    await session.append(
        "delegation.finished",
        {"call_id": link.call_id, "result": result.model_dump(), "metrics": metrics.model_dump()},
        link.parent_run_id,
    )
    await session.append("tool.result", result.model_dump(), link.parent_run_id)


async def end(session: MemorySession, run: str = "run-one") -> None:
    await session.append("run.ended", {"status": "completed", "reason": "done", "metrics": {}}, run)


async def test_declaration_order_status_local_metrics_and_run_filter() -> None:
    session = MemorySession("parent")
    calls = await declarations(session)
    assert [t.status for t in summarize_tasks(session.events)] == ["queued", "queued"]
    links = [await start(session, call) for call in calls]
    assert [t.status for t in summarize_tasks(session.events)] == ["running", "running"]
    await finish(session, links[1])
    await finish(session, links[0])
    await end(session)
    await declarations(session, "run-two")
    summaries = summarize_tasks(session.events, "run-one")
    assert [t.task_id for t in summaries] == [call.id for call in calls]
    assert all(t.status == "completed" and t.model_id == "mock:demo" for t in summaries)
    assert [(t.model_requests, t.tool_calls) for t in summaries] == [(2, 1), (2, 1)]
    assert summaries[0].report_summary == "Evidence for explorer"
    assert summaries[1].child_session_id == links[1].child_session_id
    assert len(summarize_tasks(session.events)) == 4
    assert summarize_tasks(session.events, "missing") == []


async def test_read_only_history_and_cli_filter(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "parent")
    calls = await declarations(session)
    for call in calls:
        await finish(session, await start(session, call))
    await end(session)
    await declarations(session, "run-two")
    await session.close()
    original = {p.name: p.read_bytes() for p in session.path.parent.iterdir()}
    events = read_session_events(tmp_path, "parent")
    assert events[-1].type == "assistant.message"
    assert len(list_tasks(tmp_path, "parent")) == 4
    assert len(list_tasks(tmp_path, "parent", "run-one")) == 2
    result = CliRunner().invoke(
        cli.app,
        ["tasks", "--session", "parent", "--run", "run-one", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "run-one-explorer" in result.output
    assert "run-two" not in result.output
    assert "mock:demo" in result.output
    assert "Evidence for reviewer" in result.output
    assert original == {p.name: p.read_bytes() for p in session.path.parent.iterdir()}


async def test_busy_corrupt_missing_queries_do_not_modify_logs(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "parent")
    original = session.path.read_bytes()
    with pytest.raises(PersistenceError, match="不可读取.*占用"):
        list_tasks(tmp_path, "parent")
    busy = CliRunner().invoke(
        cli.app, ["tasks", "--session", "parent", "--workspace", str(tmp_path)]
    )
    assert busy.exit_code == 1
    assert "不可读取" in busy.output and "占用" in busy.output
    assert session.path.read_bytes() == original
    await session.close()
    session.path.write_bytes(original + b'{"version":')
    corrupt = session.path.read_bytes()
    with pytest.raises(PersistenceError, match="不可读取.*损坏"):
        list_tasks(tmp_path, "parent")
    assert session.path.read_bytes() == corrupt
    with pytest.raises(PersistenceError, match="不可读取"):
        list_tasks(tmp_path, "missing")
    assert not session.path.with_name("missing.lock").exists()
    assert not session.path.with_name("missing.jsonl").exists()


async def test_legacy_missing_model_metadata_and_unknown_outcomes(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "legacy")
    calls = await declarations(session)
    await start(session, calls[0])
    await end(session)
    await session.close()
    # Old journals have the same event version and no model metadata.
    lines = [json.loads(line) for line in session.path.read_text().splitlines()]
    for line in lines:
        if line["type"] == "delegation.started":
            del line["data"]["model_id"]
    session.path.write_text("\n".join(json.dumps(line) for line in lines))
    before = session.path.read_bytes()
    summaries = list_tasks(tmp_path, "legacy")
    assert [t.status for t in summaries] == ["unknown", "not_started"]
    assert summaries[0].model_id is None
    assert session.path.read_bytes() == before


async def test_failed_invalid_declarations_and_local_pause() -> None:
    session = MemorySession("parent")
    calls = await declarations(session)
    link = await start(session, calls[0])
    result = ToolResult(
        call_id=link.call_id,
        ok=False,
        error="SUBAGENT_INCOMPLETE",
        value={"status": "paused", "reason": "局部额度耗尽", "metrics": {"model_requests": 16}},
    )
    await session.append("tool.result", result.model_dump(), "run-one")
    await session.append(
        "tool.result",
        ToolResult(
            call_id=calls[1].id, ok=False, error="budget_exceeded", outcome="not_started"
        ).model_dump(),
        "run-one",
    )
    summaries = summarize_tasks(session.events)
    assert [(t.status, t.model_requests) for t in summaries] == [("paused", 16), ("not_started", 0)]


def test_progress_identifies_task_role_and_child(monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=200, color_system=None))
    data = {"call_id": "task-A", "role": "explorer", "child_session_id": "child-A"}
    cli.observe("delegation.queued", {**data, "child_session_id": None})
    cli.observe("delegation.started", data)
    cli.observe("delegation.finished", {**data, "status": "canceled", "ok": False})
    cli.observe(
        "tool.result",
        {**data, "task_id": "task-A", "call_id": "child-tool", "name": "read_file", "ok": True},
    )
    rendered = output.getvalue()
    assert rendered.count("任务 task-A") == 4
    assert "子会话 待创建 · 等待" in rendered
    assert "子会话 child-A · 运行" in rendered
    assert "子会话 child-A · 取消" in rendered
    assert "[任务 task-A · explorer · 子会话 child-A] read_file" in rendered
