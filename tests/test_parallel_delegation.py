"""Concurrent delegation preserves authority, accounting, and durable call pairing."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, SubagentsConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import AssistantMessage, ModelResponse, RunStatus, ToolCall, ToolResult
from hiro_agent.core import EventBus
from hiro_agent.errors import ModelError, PersistenceError
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import JsonlSession, MemorySession, recover_session
from hiro_agent.subagents import SubagentManager, recover_delegations
from hiro_agent.tools import ToolExecutor, build_file_tools


def response(*calls: ToolCall, text: str = "") -> ModelResponse:
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=list(calls)),
        finish_reason="tool_calls" if calls else "stop",
    )


def tool(name: str, call_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def delegate(index: int) -> ToolCall:
    return tool("delegate_task", f"delegate-{index}", role="explorer", task=f"inspect-{index}")


def report(index: int) -> ModelResponse:
    return response(text=json.dumps({"summary": f"Evidence {index}", "findings": []}))


def results(session: MemorySession) -> dict[str, ToolResult]:
    return {
        event.data["call_id"]: ToolResult.model_validate(event.data)
        for event in session.events
        if event.type == "tool.result"
    }


ChildReply = Callable[[int, Any], Awaitable[ModelResponse]]


class RoutedModel:
    """Route by child task text, so responses do not depend on scheduling order."""

    def __init__(self, calls: list[ToolCall], child_reply: ChildReply) -> None:
        self.calls = calls
        self.child_reply = child_reply
        self.requests: list[Any] = []
        self.parent_requests = 0

    async def complete(self, request: Any) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if any(item["function"]["name"] == "delegate_task" for item in request.tools):
            self.parent_requests += 1
            if self.parent_requests == 1:
                return response(*self.calls)
            declared = [call.id for message in request.messages for call in message.tool_calls]
            paired = [
                message.tool_call_id for message in request.messages if message.role == "tool"
            ]
            assert sorted(declared) == sorted(paired)
            assert len(set(paired)) == len(paired)
            return response(text="Combined all available evidence")
        task = next(message.content for message in request.messages if message.role == "user")
        return await self.child_reply(int(task.removeprefix("inspect-")), request)

    async def close(self) -> None:
        pass


class Harness:
    def __init__(self) -> None:
        self.children: list[MemorySession] = []

    @classmethod
    async def open(
        cls,
        workspace: Path,
        calls: list[ToolCall],
        child_reply: ChildReply,
        *,
        concurrency: int = 2,
        budget: BudgetConfig | None = None,
        child_budget: int = 16,
        disk: bool = False,
        child_type: type[MemorySession] = MemorySession,
    ) -> Harness:
        instance = cls()
        instance.parent = (
            await JsonlSession.open(workspace, "parallel-parent")
            if disk
            else MemorySession("parallel-parent")
        )
        instance.model = RoutedModel(calls, child_reply)
        instance.finished = {call.id: asyncio.Event() for call in calls}
        instance.observed: list[tuple[str, dict[str, Any]]] = []
        events = EventBus()
        events.subscribe(lambda kind, data: instance.observed.append((kind, data)))
        registry = build_file_tools(workspace, ToolsConfig())

        async def sessions(session_id: str) -> MemorySession:
            assert any(
                event.type == "delegation.started" and event.data["child_session_id"] == session_id
                for event in instance.parent.events
            ), "Parent linkage must commit before a child session is opened"
            child = (
                await JsonlSession.open(workspace, session_id) if disk else child_type(session_id)
            )
            instance.children.append(child)
            return child

        async def approve(*_args: Any) -> bool:
            return True

        def observe(kind: str, data: dict[str, Any]) -> None:
            events.emit(kind, data)
            if kind == "tool.result":
                instance.finished[data["call_id"]].set()

        instance.manager = SubagentManager(
            instance.model,
            registry,
            sessions,
            SubagentsConfig(max_concurrency=concurrency, max_model_requests=child_budget),
            events,
        )
        instance.manager.register(registry)
        instance.agent = Agent(
            instance.parent,
            instance.model,
            ToolExecutor(
                registry,
                instance.parent,
                PermissionPolicy("ask-write", approve),
                observe,
                max_concurrency=concurrency,
            ),
            PassthroughContext(),
            IdentityPreprocessor(),
            budget or BudgetConfig(),
            events,
        )
        return instance

    async def close(self) -> None:
        await self.agent.close()
        await self.manager.close()
        await self.parent.close()


@pytest.mark.parametrize("concurrency", [1, 2, 3])
@pytest.mark.parametrize("disk", [False, True])
async def test_barriers_prove_overlap_and_bounded_admission(tmp_path, concurrency, disk):
    count = 5
    entered = [asyncio.Event() for _ in range(count)]
    release = [asyncio.Event() for _ in range(count)]
    active = 0
    peak = 0

    async def child(index, request):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        entered[index].set()
        try:
            await release[index].wait()
            return report(index)
        finally:
            active -= 1

    harness = await Harness.open(
        tmp_path,
        [delegate(index) for index in range(count)],
        child,
        concurrency=concurrency,
        disk=disk,
    )
    running = asyncio.create_task(harness.agent.run("Inspect five independent areas"))
    try:
        for start in range(0, count, concurrency):
            batch = range(start, min(start + concurrency, count))
            await asyncio.wait_for(asyncio.gather(*(entered[index].wait() for index in batch)), 5)
            assert active == len(batch)
            assert len(harness.children) == min(start + concurrency, count)
            started = [
                event.data["call"]["id"]
                for event in harness.parent.events
                if event.type == "tool.started"
            ]
            assert started == [
                f"delegate-{index}" for index in range(min(start + concurrency, count))
            ]
            assert harness.model.parent_requests == 1
            for index in batch:
                release[index].set()
        result = await asyncio.wait_for(running, 5)
        assert result.status == RunStatus.COMPLETED
        assert peak == concurrency
        assert active == 0
        assert result.metrics.model_requests == count + 2
        assert result.metrics.tool_calls == count
        assert len(results(harness.parent)) == count
        harness.parent.derive_messages()
        assert not harness.manager._active
        if disk:
            for old_child in harness.children:
                reopened = await JsonlSession.open(tmp_path, old_child.session_id, create=False)
                await reopened.close()
    finally:
        for event in release:
            event.set()
        harness.agent.cancel()
        await asyncio.wait_for(running, 5)
        await harness.close()


async def test_write_and_command_wait_for_prior_delegate_group(tmp_path):
    entered = [asyncio.Event() for _ in range(3)]
    release = [asyncio.Event() for _ in range(2)]

    async def child(index, request):
        entered[index].set()
        if index < 2:
            await release[index].wait()
        else:
            assert (tmp_path / "written").read_text() == "after reports"
            assert (tmp_path / "command").read_text() == "after write"
        return report(index)

    calls = [
        delegate(0),
        delegate(1),
        tool("write_file", "write", path="written", content="after reports"),
        tool(
            "run_command",
            "command",
            executable=sys.executable,
            args=[
                "-c",
                "from pathlib import Path; assert Path('written').exists(); "
                "Path('command').write_text('after write')",
            ],
        ),
        delegate(2),
    ]
    harness = await Harness.open(tmp_path, calls, child)
    running = asyncio.create_task(harness.agent.run("Inspect, write, run, inspect again"))
    try:
        await asyncio.wait_for(asyncio.gather(entered[0].wait(), entered[1].wait()), 2)
        assert not (tmp_path / "written").exists()
        assert not entered[2].is_set()
        release[1].set()
        # A terminal report for one sibling cannot release the sequential boundary.
        await asyncio.wait_for(harness.finished["delegate-1"].wait(), 2)
        assert not (tmp_path / "written").exists()
        assert not entered[2].is_set()
        release[0].set()
        result = await asyncio.wait_for(running, 5)
        assert result.status == RunStatus.COMPLETED
        assert entered[2].is_set()
        events = harness.parent.events
        write_started = next(
            e.seq for e in events if e.type == "tool.started" and e.data["call"]["id"] == "write"
        )
        assert all(
            e.seq < write_started
            for e in events
            if e.type == "delegation.finished" and e.data["call_id"] in {"delegate-0", "delegate-1"}
        )
    finally:
        for event in release:
            event.set()
        harness.agent.cancel()
        await asyncio.wait_for(running, 5)
        await harness.close()


@pytest.mark.parametrize("failure", ["model", "report", "local_budget"])
async def test_one_child_failure_does_not_abort_siblings(tmp_path, failure):
    barrier = asyncio.Barrier(2)

    async def child(index, request):
        await barrier.wait()
        if index == 1:
            return report(index)
        if failure == "model":
            raise ModelError("Injected isolated model failure")
        if failure == "report":
            return response(text="Invalid structured report")
        return response(tool("list_files", "inspect-files"))

    harness = await Harness.open(tmp_path, [delegate(0), delegate(1)], child, child_budget=1)
    try:
        result = await asyncio.wait_for(harness.agent.run("Inspect independently"), 5)
        assert result.status == RunStatus.COMPLETED
        settled = results(harness.parent)
        assert settled["delegate-1"].ok
        assert settled["delegate-0"].error == (
            "SUBAGENT_INVALID_REPORT" if failure == "report" else "SUBAGENT_INCOMPLETE"
        )
        assert len(harness.children) == 2
        assert result.metrics.model_requests == 4
        assert result.metrics.tool_errors >= 1
        assert not harness.manager._active
    finally:
        await harness.close()


async def test_root_budget_exhaustion_stops_queue_and_cleans_running_sibling(tmp_path):
    barrier = asyncio.Barrier(2)
    cleaned = asyncio.Event()

    async def child(index, request):
        await barrier.wait()
        if index == 0:
            return response(tool("list_files", "consume-local-step"))
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    harness = await Harness.open(
        tmp_path,
        [delegate(index) for index in range(4)],
        child,
        budget=BudgetConfig(max_model_requests=3),
    )
    try:
        result = await asyncio.wait_for(harness.agent.run("Inspect within root allowance"), 5)
        assert result.status == RunStatus.PAUSED
        assert result.metrics.model_requests == 3
        assert len(harness.children) == 2
        assert cleaned.is_set()
        assert harness.model.parent_requests == 1
        assert not harness.manager._active
        assert len(results(harness.parent)) == 4
        assert all(
            results(harness.parent)[f"delegate-{index}"].outcome == "not_started"
            for index in (2, 3)
        )
        harness.parent.derive_messages()
    finally:
        await harness.close()


async def test_exact_root_model_cap_stops_queued_admission_and_emits_terminal_progress(tmp_path):
    async def child(index, request):
        return report(index)

    harness = await Harness.open(
        tmp_path,
        [delegate(index) for index in range(4)],
        child,
        budget=BudgetConfig(max_model_requests=3),
    )
    try:
        result = await asyncio.wait_for(harness.agent.run("Inspect until root model cap"), 5)
        assert result.status == RunStatus.PAUSED
        assert result.metrics.model_requests == 3
        assert result.metrics.tool_calls == 2
        assert harness.model.parent_requests == 1
        assert len(harness.children) == 2
        started = [
            event.data["call"]["id"]
            for event in harness.parent.events
            if event.type == "tool.started"
        ]
        assert started == ["delegate-0", "delegate-1"]
        for index in (2, 3):
            task_id = f"delegate-{index}"
            assert results(harness.parent)[task_id].outcome == "not_started"
            terminal = [
                data
                for kind, data in harness.observed
                if kind == "delegation.finished" and data.get("call_id") == task_id
            ]
            assert len(terminal) == 1
            assert terminal[0]["status"] == "not_started"
            assert terminal[0]["role"] == "explorer"
            assert terminal[0]["child_session_id"] is None
        harness.parent.derive_messages()
        assert not harness.manager._active
    finally:
        await harness.close()


async def test_root_budget_reason_survives_user_cancel_during_child_settlement(tmp_path):
    barrier = asyncio.Barrier(2)
    budget_settling = asyncio.Event()
    sibling_cleaning = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def child(index, request):
        await barrier.wait()
        if index == 0:
            return response(tool("list_files", "consume-step-before-root-rejection"))
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cleaning.set()
            await release_cleanup.wait()

    class HoldBudgetSettlement(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if (
                event_type == "run.ended"
                and data["status"] == "paused"
                and data["reason"] == "模型请求预算耗尽"
            ):
                budget_settling.set()
                await release_cleanup.wait()
            return await super().append(event_type, data, run_id)

    harness = await Harness.open(
        tmp_path,
        [delegate(index) for index in range(4)],
        child,
        budget=BudgetConfig(max_model_requests=3),
        child_type=HoldBudgetSettlement,
    )
    running = asyncio.create_task(harness.agent.run("Preserve the initial interruption reason"))
    try:
        await asyncio.wait_for(budget_settling.wait(), 2)
        assert not running.done()
        harness.agent.cancel()
        await asyncio.wait_for(sibling_cleaning.wait(), 2)
        release_cleanup.set()
        result = await asyncio.wait_for(running, 5)
        assert result.status == RunStatus.PAUSED
        assert result.reason == "模型请求预算耗尽"
        assert result.metrics.model_requests == 3
        assert len(harness.children) == 2
        assert harness.model.parent_requests == 1
        assert not harness.manager._active
        assert len(results(harness.parent)) == 4
        for session in harness.children:
            session.derive_messages()
            with pytest.raises(PersistenceError, match="closed"):
                await session.append("probe", {})
        harness.parent.derive_messages()
    finally:
        release_cleanup.set()
        harness.agent.cancel()
        await asyncio.wait_for(running, 5)
        await harness.close()


async def test_child_recovered_event_preserves_tool_call_id_and_delegation_identity(tmp_path):
    async def child(index, request):
        return response(
            tool("list_files", "child-list-one"),
            tool("list_files", "child-list-two"),
        )

    class CancelAfterDeclaration(MemorySession):
        async def append(self, event_type, data, run_id=None):
            event = await super().append(event_type, data, run_id)
            if event_type == "assistant.message":
                harness.agent.cancel()
            return event

    harness = await Harness.open(tmp_path, [delegate(0)], child, child_type=CancelAfterDeclaration)
    try:
        result = await asyncio.wait_for(harness.agent.run("Inspect until declaration"), 5)
        assert result.status == RunStatus.CANCELED
        recovered = [
            data
            for kind, data in harness.observed
            if kind == "tool.recovered" and data.get("role") == "explorer"
        ]
        assert {data["call_id"] for data in recovered} == {"child-list-one", "child-list-two"}
        assert all(data["task_id"] == "delegate-0" for data in recovered)
        assert all(data["child_session_id"] == harness.children[0].session_id for data in recovered)
        assert len(results(harness.children[0])) == 2
        harness.parent.derive_messages()
        harness.children[0].derive_messages()
    finally:
        await harness.close()


@pytest.mark.parametrize("stop", ["cancel", "caller", "deadline"])
@pytest.mark.parametrize("persistence_failure", [False, True])
async def test_interruptions_wait_for_all_cleanup_and_prioritize_persistence(
    tmp_path,
    stop,
    persistence_failure,
):
    entered = [asyncio.Event(), asyncio.Event()]
    cleanup = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def child(index, request):
        entered[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup[index].set()
            await release.wait()

    class FailingCleanup(MemorySession):
        async def append(self, event_type, data, run_id=None):
            if persistence_failure and event_type == "model.interrupted":
                raise PersistenceError("Injected cleanup persistence failure")
            return await super().append(event_type, data, run_id)

    harness = await Harness.open(
        tmp_path,
        [delegate(index) for index in range(3)],
        child,
        budget=BudgetConfig(timeout_seconds=0.1 if stop == "deadline" else 600),
        child_type=FailingCleanup,
    )
    running = asyncio.create_task(harness.agent.run("Inspect until interrupted"))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 2)
        if stop == "cancel":
            harness.agent.cancel()
        elif stop == "caller":
            running.cancel()
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cleanup)), 2)
        assert not running.done()
        assert len(harness.children) == 2
        release.set()
        result = await asyncio.wait_for(running, 5)
        expected = RunStatus.PAUSED if stop == "deadline" else RunStatus.CANCELED
        assert result.status == (RunStatus.FAILED if persistence_failure else expected)
        assert result.persistence_failed == persistence_failure
        assert not harness.manager._active
        assert harness.model.parent_requests == 1
        for session in harness.children:
            with pytest.raises(PersistenceError, match="closed"):
                await session.append("probe", {})
        if not persistence_failure:
            assert len(results(harness.parent)) == 3
            harness.parent.derive_messages()
    finally:
        release.set()
        harness.agent.cancel()
        await asyncio.wait_for(running, 5)
        await harness.close()


async def test_multiple_child_crash_stages_recover_once_without_replay(tmp_path):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def child(index, request):
        if any(message.role == "tool" for message in request.messages):
            return report(index)
        if index < 2:
            entered[index].set()
            await release[index].wait()
        return response(tool("list_files", f"child-list-{index}"))

    harness = await Harness.open(
        tmp_path, [delegate(index) for index in range(3)], child, disk=True
    )
    running = asyncio.create_task(harness.agent.run("Inspect before simulated crash"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 5)
    release[1].set()
    await asyncio.wait_for(harness.finished["delegate-1"].wait(), 5)
    release[0].set()
    result = await asyncio.wait_for(running, 5)
    assert result.status == RunStatus.COMPLETED
    await harness.close()
    assert isinstance(harness.parent, JsonlSession)
    parent_events = harness.parent.events
    first_finished = next(
        index for index, event in enumerate(parent_events) if event.type == "delegation.finished"
    )
    assert parent_events[first_finished].data["call_id"] == "delegate-1"
    harness.parent.path.write_text(
        "".join(event.model_dump_json() + "\n" for event in parent_events[:first_finished])
    )
    child_by_call = {
        next(
            event.data["call_id"] for event in child.events if event.type == "subagent.created"
        ): child
        for child in harness.children
    }
    for call_id, boundary in {"delegate-0": "run.ended", "delegate-1": "model.requested"}.items():
        child_session = child_by_call[call_id]
        assert isinstance(child_session, JsonlSession)
        events = child_session.events
        last = next(index for index, event in enumerate(events) if event.type == boundary)
        child_session.path.write_text(
            "".join(event.model_dump_json() + "\n" for event in events[: last + 1])
        )
    never_started = child_by_call["delegate-2"]
    assert isinstance(never_started, JsonlSession)
    never_started.path.unlink()
    request_count = len(harness.model.requests)
    reopened = await JsonlSession.open(tmp_path, harness.parent.session_id, create=False)
    try:
        await recover_delegations(tmp_path, reopened, 16000)
        await recover_session(reopened)
        settled = results(reopened)
        assert settled["delegate-0"].ok
        assert settled["delegate-1"].error == "SUBAGENT_INCOMPLETE"
        assert settled["delegate-2"].outcome == "not_started"
        assert not never_started.path.exists()
        ended = next(event for event in reopened.events if event.type == "run.ended")
        assert ended.data["metrics"]["model_requests"] == 4
        assert ended.data["metrics"]["tool_calls"] == 3
        stable = reopened.path.read_bytes()
        await recover_delegations(tmp_path, reopened, 16000)
        await recover_session(reopened)
        assert reopened.path.read_bytes() == stable
        assert len(harness.model.requests) == request_count
        reopened.derive_messages()
    finally:
        await reopened.close()
