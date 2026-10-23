from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from hiro_agent.budget import Budget
from hiro_agent.config import BudgetConfig, ToolsConfig
from hiro_agent.contracts import Cancellation, DataModel, Message, SessionEvent, ToolCall
from hiro_agent.errors import BudgetExceeded, PersistenceError, RunCancelled
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.tools import ToolExecutor, ToolRegistry, build_file_tools


class MemorySession:
    session_id = "tools-test"

    def __init__(self, fail_on: str | None = None) -> None:
        self.events: list[SessionEvent] = []
        self.fail_on = fail_on

    async def append(
        self, event_type: str, data: dict[str, Any], run_id: str | None = None
    ) -> SessionEvent:
        if event_type == self.fail_on:
            raise PersistenceError("disk failure")
        event = SessionEvent(
            session_id=self.session_id,
            seq=len(self.events) + 1,
            type=event_type,
            data=data,
            run_id=run_id,
        )
        self.events.append(event)
        return event

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def derive_messages(self) -> list[Message]:
        return []


async def allow(_tool: str, _arguments: dict[str, Any]) -> bool:
    return True


def setup_tools(
    root: Path,
    *,
    approved: bool = True,
    config: ToolsConfig | None = None,
    excluded_env: tuple[str, ...] = (),
) -> tuple[ToolExecutor, MemorySession, Budget]:
    session = MemorySession()
    registry = build_file_tools(root, config or ToolsConfig(), excluded_env=excluded_env)
    executor = ToolExecutor(
        registry, session, PermissionPolicy(callback=allow if approved else None)
    )
    return executor, session, Budget(BudgetConfig())


async def invoke(
    executor: ToolExecutor,
    budget: Budget,
    name: str,
    args: dict[str, Any],
    cancellation: Cancellation | None = None,
    call_id: str = "call-1",
) -> Any:
    return await executor.execute(
        ToolCall(id=call_id, name=name, arguments=json.dumps(args)),
        "run-1",
        budget,
        cancellation or Cancellation(),
    )


async def test_permission_is_per_call_and_defaults_to_denial() -> None:
    assert await PermissionPolicy().approve("read_file", {}, False)
    assert not await PermissionPolicy().approve("write_file", {}, True)
    assert not await PermissionPolicy("read-only", allow).approve("run_command", {}, True)
    observed = []

    async def approve(tool: str, arguments: dict[str, Any]) -> bool:
        observed.append((tool, arguments))
        return len(observed) == 1

    policy = PermissionPolicy(callback=approve)
    assert await policy.approve("write_file", {"path": "a"}, True)
    assert not await policy.approve("write_file", {"path": "b"}, True)
    assert len(observed) == 2


async def test_schema_unknown_and_invalid_attempts_count(tmp_path: Path) -> None:
    executor, session, _ = setup_tools(tmp_path)
    budget = Budget(BudgetConfig(max_tool_calls=2))
    schema = executor.schemas()
    assert len(schema) == 6
    assert all(item["function"]["parameters"]["additionalProperties"] is False for item in schema)
    invalid = await invoke(executor, budget, "write_file", {"path": "a", "content": "x", "bad": 1})
    unknown = await invoke(executor, budget, "not_a_tool", {})
    assert invalid.error == "invalid_arguments"
    assert unknown.error == "unknown_tool"
    assert budget.metrics.tool_calls == 2
    with pytest.raises(BudgetExceeded):
        await invoke(executor, budget, "list_files", {})
    assert [event.type for event in session.events] == ["tool.called", "tool.result"] * 3
    assert not (tmp_path / "a").exists()


async def test_denial_never_mutates_and_result_is_exact(tmp_path: Path) -> None:
    executor, session, budget = setup_tools(tmp_path, approved=False)
    result = await invoke(executor, budget, "write_file", {"path": "a", "content": "x"})
    assert not result.ok and result.error == "permission_denied"
    assert result.outcome == "not_started"
    assert budget.metrics.permission_denials == 1
    assert budget.metrics.tool_calls == 1
    assert not (tmp_path / "a").exists()
    assert session.events[-1].data == result.model_dump(mode="json")


async def test_durable_started_before_effect_and_result_before_return(tmp_path: Path) -> None:
    session = MemorySession()
    observed = []
    registry = ToolRegistry()

    async def handler(_args: DataModel) -> dict[str, int]:
        observed.extend(event.type for event in session.events)
        return {"answer": 42}

    registry.register("probe", "probe", DataModel, handler)
    executor = ToolExecutor(registry, session, PermissionPolicy())
    call = ToolCall(id="probe", name="probe", arguments="{}")
    result = await executor.execute(call, "run", Budget(BudgetConfig()), Cancellation())
    assert observed == ["tool.called", "tool.started"]
    assert session.events[0].data == {"call": call.model_dump(mode="json"), "accepted": True}
    assert session.events[-1].data == result.model_dump(mode="json")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("probe", "duplicate", DataModel, handler)


@pytest.mark.parametrize("fail_on", ["tool.called", "tool.started", "tool.result"])
async def test_persistence_failure_propagates_without_retry(tmp_path: Path, fail_on: str) -> None:
    session = MemorySession(fail_on)
    registry = build_file_tools(tmp_path, ToolsConfig())
    executor = ToolExecutor(registry, session, PermissionPolicy(callback=allow))
    with pytest.raises(PersistenceError):
        await invoke(executor, Budget(BudgetConfig()), "write_file", {"path": "a", "content": "x"})
    assert (tmp_path / "a").exists() == (fail_on == "tool.result")
    assert sum(event.type == "tool.started" for event in session.events) <= 1


@pytest.mark.parametrize("path", ["../outside.txt", ".hiro-agent/session.json", ".git/config"])
async def test_protected_or_escaping_writes(tmp_path: Path, path: str) -> None:
    executor, _, budget = setup_tools(tmp_path)
    result = await invoke(
        executor, budget, "write_file", {"path": path, "content": "x", "create_parents": True}
    )
    assert not result.ok
    assert not (tmp_path / ".hiro-agent").exists()
    assert not (tmp_path / ".git").exists()


async def test_symlink_escape_and_runtime_alias_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / ".hiro-agent").mkdir()
    (workspace / ".hiro-agent" / "journal").write_text("private")
    (workspace / "alias").symlink_to(workspace / ".hiro-agent", target_is_directory=True)
    executor, _, budget = setup_tools(workspace)
    for name, args in [
        ("read_file", {"path": "escape/secret"}),
        ("write_file", {"path": "escape/new", "content": "x"}),
        ("read_file", {"path": "alias/journal"}),
        ("run_command", {"executable": sys.executable, "cwd": "escape", "args": ["-c", "pass"]}),
    ]:
        assert not (await invoke(executor, budget, name, args)).ok
    assert not (outside / "new").exists()
    assert (outside / "secret").read_text() == "secret"


async def test_precise_edit_rejects_ambiguous_match(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("hello hello")
    executor, _, budget = setup_tools(tmp_path)
    rejected = await invoke(
        executor, budget, "edit_file", {"path": path.name, "old_text": "hello", "new_text": "bye"}
    )
    assert not rejected.ok
    assert path.read_text() == "hello hello"
    success = await invoke(
        executor,
        budget,
        "edit_file",
        {
            "path": path.name,
            "old_text": "hello",
            "new_text": "bye",
            "expected_matches": 2,
        },
    )
    assert success.ok
    assert path.read_text() == "bye bye"


async def test_output_file_sizes_and_traversal_are_bounded(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("needle" * 100)
    (tmp_path / "small.txt").write_text("needle " * 50)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden").write_text("needle")
    (tmp_path / ".hiro-agent").mkdir()
    (tmp_path / ".hiro-agent" / "private").write_text("needle")
    executor, _, budget = setup_tools(
        tmp_path, config=ToolsConfig(max_output_chars=256, max_file_bytes=512)
    )
    assert not (await invoke(executor, budget, "read_file", {"path": "large.txt"})).ok
    read = await invoke(executor, budget, "read_file", {"path": "small.txt"})
    assert len(read.value["content"]) == 256 and read.value["truncated"]
    found = await invoke(executor, budget, "search_text", {"query": "needle"})
    assert found.value["matches"][0]["path"] == "small.txt"
    assert len(found.value["matches"][0]["text"]) <= 256
    listed = await invoke(executor, budget, "list_files", {"recursive": True})
    assert all(not path.startswith(".") for path in listed.value["paths"])
    refused = await invoke(
        executor, budget, "write_file", {"path": "too-big", "content": "界" * 200}
    )
    assert not refused.ok and not (tmp_path / "too-big").exists()
    capped, _, cap_budget = setup_tools(tmp_path, config=ToolsConfig(max_search_files=1))
    result = await invoke(capped, cap_budget, "list_files", {"recursive": True})
    assert result.value["truncated"]
    assert len(result.value["paths"]) <= 1


async def test_command_env_whitelist_and_explicit_configured_key_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIRO_API_KEY", "credential-do-not-forward")
    monkeypatch.setenv("UNRELATED_SECRET", "also-do-not-forward")
    monkeypatch.setenv("LANG", "configured-model-key")
    executor, _, budget = setup_tools(
        tmp_path, excluded_env=("LANG",), config=ToolsConfig(command_timeout_seconds=3)
    )
    result = await invoke(
        executor,
        budget,
        "run_command",
        {
            "executable": sys.executable,
            "args": ["-c", "import os,json; print(json.dumps(dict(os.environ)))"],
        },
    )
    assert result.ok
    environment = json.loads(result.value["stdout"])
    assert "HIRO_API_KEY" not in environment
    assert "UNRELATED_SECRET" not in environment
    assert "LANG" not in environment


async def test_command_drains_both_streams_and_limits_combined_output(tmp_path: Path) -> None:
    executor, _, budget = setup_tools(
        tmp_path, config=ToolsConfig(max_output_chars=256, command_timeout_seconds=3)
    )
    result = await invoke(
        executor,
        budget,
        "run_command",
        {
            "executable": sys.executable,
            "args": ["-c", "import os; os.write(1,b'a'*1000000); os.write(2,b'b'*1000000)"],
        },
    )
    assert result.ok
    assert len(result.value["stdout"]) + len(result.value["stderr"]) <= 256
    assert result.value["truncated"]
    failed = await invoke(
        executor,
        budget,
        "run_command",
        {
            "executable": sys.executable,
            "args": ["-c", "print('failure detail'); raise SystemExit(7)"],
        },
    )
    assert failed.error == "command_failed", failed
    assert failed.value["exit_code"] == 7
    assert "failure detail" in failed.value["stdout"]


def process_alive(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    if path.exists():
        return path.read_text().split(") ", 1)[1].split()[0] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def wait_for_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not path.exists():  # noqa: ASYNC110 - observe a child process's readiness file
            await asyncio.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="process-group verification uses /proc")
@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
async def test_command_cleanup_kills_descendants_after_leader_exits(
    tmp_path: Path, interrupt: str
) -> None:
    child_pid = tmp_path / "child.pid"
    executor, session, budget = setup_tools(
        tmp_path, config=ToolsConfig(command_timeout_seconds=0.3 if interrupt == "timeout" else 5)
    )
    cancellation = Cancellation()
    # The leader exits immediately; its child keeps inherited output pipes open.
    code = (
        "import subprocess,sys,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path('child.pid').write_text(str(p.pid))"
    )
    task = asyncio.create_task(
        invoke(
            executor,
            budget,
            "run_command",
            {
                "executable": sys.executable,
                "args": ["-c", code],
            },
            cancellation,
        )
    )
    await wait_for_file(child_pid)
    pid = int(child_pid.read_text())
    try:
        if interrupt == "cancel":
            cancellation.cancel()
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 5)
            result = session.events[-1].data
            assert result["outcome"] == "unknown"
        else:
            result = await asyncio.wait_for(task, 5)
            assert not result.ok and "超时" in result.message
        assert not process_alive(pid)
        assert len([e for e in session.events if e.type == "tool.result"]) == 1
    finally:
        if process_alive(pid):
            os.kill(pid, 9)


async def test_cancel_approval_records_not_started(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def approve(_tool: str, _arguments: dict[str, Any]) -> bool:
        entered.set()
        await asyncio.Event().wait()
        return True

    executor, session, budget = setup_tools(tmp_path)
    executor.permission = PermissionPolicy(callback=approve)
    cancellation = Cancellation()
    task = asyncio.create_task(
        invoke(executor, budget, "write_file", {"path": "a", "content": "x"}, cancellation)
    )
    await entered.wait()
    cancellation.cancel()
    with pytest.raises(RunCancelled):
        await task
    assert not (tmp_path / "a").exists()
    assert [event.type for event in session.events] == ["tool.called", "tool.result"]
    assert session.events[-1].data["outcome"] == "not_started"


async def test_command_timeout_keeps_draining_until_reaped(tmp_path: Path) -> None:
    executor, session, budget = setup_tools(
        tmp_path, config=ToolsConfig(max_output_chars=256, command_timeout_seconds=0.1)
    )
    async with asyncio.timeout(5):
        result = await invoke(
            executor,
            budget,
            "run_command",
            {
                "executable": sys.executable,
                "args": ["-c", "import os\nwhile True: os.write(1,b'x'*1000000)"],
            },
        )
    assert not result.ok
    assert "超时" in result.message
    assert session.events[-1].type == "tool.result"


async def test_observer_failure_does_not_change_execution_order(tmp_path: Path) -> None:
    executor, session, budget = setup_tools(tmp_path)

    def broken_observer(_kind: str, _data: dict[str, Any]) -> None:
        raise RuntimeError("display unavailable")

    executor.observer = broken_observer
    result = await invoke(executor, budget, "write_file", {"path": "a", "content": "x"})
    assert result.ok and (tmp_path / "a").read_text() == "x"
    assert [event.type for event in session.events] == [
        "tool.called",
        "tool.started",
        "tool.result",
    ]
