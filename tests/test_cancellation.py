"""Cancellation at durable journal boundaries must leave a usable conversation."""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from hiro_agent.agent import Agent
from hiro_agent.config import BudgetConfig, ToolsConfig
from hiro_agent.context import IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import AssistantMessage, ModelResponse, RunStatus, ToolCall
from hiro_agent.models import ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import JsonlSession
from hiro_agent.tools import ToolExecutor, build_file_tools


@pytest.mark.parametrize(
    "boundary", ["assistant.message", "tool.called", "tool.started", "tool.result"]
)
async def test_external_cancellation_during_durable_append_keeps_session_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    session = await JsonlSession.open(tmp_path)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    write = session._write_bytes
    blocked = False

    def delayed_write(raw: bytes) -> None:
        nonlocal blocked
        if json.loads(raw)["type"] == boundary and not blocked:
            blocked = True
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise TimeoutError("Test did not release the durable write")
        write(raw)

    monkeypatch.setattr(session, "_write_bytes", delayed_write)

    async def approve(tool: str, arguments: dict) -> bool:
        return True

    calls = [
        ToolCall(
            id=f"call-{index}",
            name="write_file",
            arguments=json.dumps({"path": f"output-{index}.txt", "content": "written"}),
        )
        for index in (1, 2)
    ]
    model = ScriptedModel(
        [
            ModelResponse(assistant=AssistantMessage(tool_calls=calls), finish_reason="tool_calls"),
            ModelResponse(
                assistant=AssistantMessage(text="continued safely"), finish_reason="stop"
            ),
        ]
    )
    tools = ToolExecutor(
        build_file_tools(tmp_path, ToolsConfig()), session, PermissionPolicy("ask-write", approve)
    )
    agent = Agent(
        session,
        model,
        tools,
        PassthroughContext(),
        IdentityPreprocessor(),
        BudgetConfig(),
    )
    running = asyncio.create_task(agent.run("Write both files"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        running.cancel()
        await asyncio.sleep(0)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done(), "Cancellation must wait for the pending durable acknowledgment"
        release.set()
        result = await asyncio.wait_for(running, 3)
        assert result.status == RunStatus.CANCELED
        assert session.events[-1].type == "run.ended"
        assert session.events[-1].data["status"] == "canceled"

        results = [e.data for e in session.events if e.type == "tool.result"]
        assert [item["call_id"] for item in results] == ["call-1", "call-2"]
        assert results[1]["outcome"] == "not_started"
        expected_first = {
            "assistant.message": "not_started",
            "tool.called": "not_started",
            "tool.started": "unknown",
            "tool.result": "completed",
        }
        assert results[0]["outcome"] == expected_first[boundary]
        called = [e.data["call"]["id"] for e in session.events if e.type == "tool.called"]
        assert len(called) == len(set(called))
        assert (tmp_path / "output-1.txt").exists() == (boundary == "tool.result")
        assert not (tmp_path / "output-2.txt").exists()
        assert [m.tool_call_id for m in session.derive_messages() if m.role == "tool"] == [
            "call-1",
            "call-2",
        ]

        continued = await agent.continue_run()
        assert continued.status == RunStatus.COMPLETED
        assert model.requests[-1].messages[-1].continuation is True
        assert len([e for e in session.events if e.type == "tool.result"]) == 2
    finally:
        release.set()
        agent.cancel()
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 3)
        await agent.close()
        await session.close()
        await model.close()
