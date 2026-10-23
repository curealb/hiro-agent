"""Read-only task summaries derived from the existing parent session journal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hiro_agent.contracts import AssistantMessage, DataModel, SessionEvent, ToolResult
from hiro_agent.errors import PersistenceError
from hiro_agent.session import read_session_events


class TaskSummary(DataModel):
    task_id: str
    session_id: str
    run_id: str | None = None
    task: str = ""
    role: str = "unknown"
    model_id: str | None = None
    status: str = "queued"
    child_session_id: str | None = None
    child_run_id: str | None = None
    model_requests: int = 0
    tool_calls: int = 0
    report_summary: str = ""
    reason: str = ""


def _brief(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result(summary: TaskSummary, result: ToolResult, metrics: Any = None) -> None:
    value = result.value if isinstance(result.value, dict) else {}
    link = value.get("delegation")
    if isinstance(link, dict):
        summary.child_session_id = link.get("child_session_id", summary.child_session_id)
        summary.child_run_id = link.get("child_run_id", summary.child_run_id)
        summary.model_id = link.get("model_id", summary.model_id)
    report = value.get("report")
    if isinstance(report, dict) and isinstance(report.get("summary"), str):
        summary.report_summary = report["summary"]
    metrics = metrics if isinstance(metrics, dict) else value.get("metrics", {})
    if isinstance(metrics, dict):
        summary.model_requests = metrics.get("model_requests", summary.model_requests)
        summary.tool_calls = metrics.get("tool_calls", summary.tool_calls)
    summary.reason = str(value.get("reason") or result.error or "")
    if result.outcome == "not_started":
        summary.status = "not_started"
    elif result.error == "cancelled":
        summary.status = "canceled"
    elif result.error == "budget_exceeded":
        summary.status = "paused"
    elif result.error == "TOOL_OUTCOME_UNKNOWN" or result.outcome == "unknown":
        summary.status = "unknown"
    elif isinstance(value.get("status"), str):
        summary.status = value["status"] if not result.error else "failed"
        if value["status"] != "completed":
            summary.status = value["status"]
    else:
        summary.status = "completed" if result.ok else "failed"


def summarize_tasks(events: list[SessionEvent], run_id: str | None = None) -> list[TaskSummary]:
    """Preserve declaration order even when child completions arrive out of order."""
    summaries: dict[str, TaskSummary] = {}
    started: set[str] = set()
    ended_runs = {event.run_id for event in events if event.type == "run.ended"}
    for event in events:
        if run_id is not None and event.run_id != run_id:
            continue
        if event.type == "assistant.message":
            message = AssistantMessage.model_validate(event.data["message"])
            for call in message.tool_calls:
                if call.name != "delegate_task":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except ValueError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                summaries[call.id] = TaskSummary(
                    task_id=call.id,
                    session_id=event.session_id,
                    run_id=event.run_id,
                    role=str(arguments.get("role", "unknown")),
                    task=_brief(str(arguments.get("task", ""))),
                )
            continue
        call_id = event.data.get("call_id")
        if event.type == "tool.started":
            call_id = event.data["call"]["id"]
            started.add(call_id)
        summary = summaries.get(call_id) if isinstance(call_id, str) else None
        if summary is None:
            continue
        if event.type == "delegation.started":
            summary.status = "running"
            summary.child_session_id = event.data["child_session_id"]
            summary.child_run_id = event.data["child_run_id"]
            summary.model_id = event.data.get("model_id")
        elif event.type == "delegation.finished":
            _result(summary, ToolResult.model_validate(event.data["result"]), event.data["metrics"])
        elif event.type == "tool.result":
            _result(summary, ToolResult.model_validate(event.data))
    for summary in summaries.values():
        if summary.run_id in ended_runs and summary.status in {"queued", "running"}:
            summary.status = "unknown" if summary.task_id in started else "not_started"
    return list(summaries.values())


def list_tasks(workspace: Path, session_id: str, run_id: str | None = None) -> list[TaskSummary]:
    events = read_session_events(workspace, session_id)
    if any(event.type == "subagent.created" for event in events):
        raise PersistenceError("请查询主会话；子会话没有委派任务。")
    return summarize_tasks(events, run_id)
