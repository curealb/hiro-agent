"""Independent R1 acceptance: process boundaries, revocation and fail-closed storage."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hiro_agent.bots import BotStore
from hiro_agent.budget import Budget
from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig, SubagentsConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from hiro_agent.errors import PersistenceError
from hiro_agent.kernel_contracts import InteractionReply, OperationIntent
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.models import ScriptedModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import MemorySession
from hiro_agent.subagents import SubagentManager
from hiro_agent.tools import ToolExecutor, build_file_tools


def configuration(*tools: str, identity: str = "r1-bot") -> Config:
    return Config(
        model=ModelConfig(adapter="mock", model="r1-persisted-model"),
        bot=BotConfig(id=identity, instructions="R1 persistent instructions", allowed_tools=tools),
        budget=BudgetConfig(max_model_requests=7, max_tool_calls=11, timeout_seconds=45),
    )


def python_process(code: str, *args: str, stdin: str = "") -> Any:
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return json.loads(result.stdout)


def test_session_and_complete_bot_config_survive_separate_processes(tmp_path: Path) -> None:
    code = """
import asyncio, json, sys
from pathlib import Path
from hiro_agent.bootstrap import create_harness
from hiro_agent.config import Config
async def main():
    config = Config.model_validate_json(sys.stdin.read())
    harness = await create_harness(Path(sys.argv[1]), config, sys.argv[2] or None)
    try:
        record = harness.runtime.services['bot']
        print(json.dumps({
            'session_id': harness.session.session_id,
            'config': record.config.model_dump(mode='json'),
            'authority': record.authority.model_dump(mode='json'),
            'prompt': harness.session.derive_messages()[0].content,
            'bindings': sum(e.type == 'session.bot' for e in harness.session.events),
            'schemas': harness.runtime.services['tools'].schemas(),
        }))
    finally:
        await harness.close()
asyncio.run(main())
"""
    config = configuration("list_files", "read_file")
    first = python_process(code, str(tmp_path), "", stdin=config.model_dump_json())
    reopened = python_process(code, str(tmp_path), first["session_id"], stdin="{}")
    assert first == reopened
    assert reopened["config"] == config.model_dump(mode="json")
    assert reopened["bindings"] == 1
    assert reopened["authority"]["permission_version"] == 1
    assert "R1 persistent instructions" in reopened["prompt"]


def test_concurrent_process_updates_have_one_permission_version_winner(tmp_path: Path) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(configuration("list_files"), expected_version=0)
    store.save(configuration("read_file", identity="separate-bot"))
    other = store.get("separate-bot")
    code = """
import json, sys
from pathlib import Path
from hiro_agent.bots import BotStore
from hiro_agent.config import Config
from hiro_agent.errors import ConfigurationError
from hiro_agent.kernel_db import KernelDatabase
config = Config.model_validate_json(sys.stdin.readline())
store = BotStore(KernelDatabase.for_workspace(Path(sys.argv[1])))
try:
    record = store.save(config, expected_version=1)
except ConfigurationError:
    print(json.dumps({'state': 'conflict'}))
else:
    print(json.dumps({'state': 'saved', 'version': record.authority.permission_version}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process, tool in zip(processes, ("read_file", "search_text"), strict=True):
            assert process.stdin is not None
            process.stdin.write(configuration(tool).model_dump_json() + "\n")
            process.stdin.flush()
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            outputs.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert sorted(item["state"] for item in outputs) == ["conflict", "saved"]
    assert store.authority("r1-bot").permission_version == 2
    assert store.get("separate-bot") == other


async def declared_call(executor: ToolExecutor, call: ToolCall) -> Any:
    await executor.session.append("run.started", {}, "r1-run")
    await executor.session.append(
        "assistant.message",
        {"message": AssistantMessage(tool_calls=[call]).model_dump(mode="json")},
        "r1-run",
    )
    return await executor.execute(call, "r1-run", Budget(BudgetConfig()), Cancellation())


@pytest.mark.parametrize("name", ["write_file", "run_command", "read_file"])
async def test_direct_calls_cannot_bypass_empty_tool_policy(tmp_path: Path, name: str) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(configuration())
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")

    async def approval(_tool: str, _args: dict[str, Any]) -> bool:
        pytest.fail("Disabled calls must never reach approval")

    executor = ToolExecutor(
        build_file_tools(tmp_path, Config().tools),
        MemorySession(),
        PermissionPolicy(callback=approval),
        authority=store.provider("r1-bot"),
    )
    # Never ask for schemas: the executor must defend an independently supplied call.
    result = await declared_call(executor, ToolCall(id="bypass", name=name, arguments="{}"))
    assert (result.error, result.outcome, result.value) == (
        "permission_denied",
        "not_started",
        None,
    )
    assert sentinel.read_text() == "unchanged"
    assert not any(event.type == "tool.started" for event in executor.session.events)


@pytest.mark.parametrize("change", ["revoke-and-restore", "rename"])
async def test_approval_distinguishes_restored_permissions_from_cosmetic_update(
    tmp_path: Path, change: str
) -> None:
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    config = configuration("write_file")
    original = store.save(config)

    async def approval(_tool: str, _args: dict[str, Any]) -> bool:
        if change == "revoke-and-restore":
            store.save(configuration(), expected_version=1)
            store.save(config, expected_version=2)
        else:
            changed = config.model_dump()
            changed["bot"]["name"] = "Renamed while awaiting approval"
            store.save(Config.model_validate(changed), expected_version=1)
        return True

    executor = ToolExecutor(
        build_file_tools(tmp_path, config.tools),
        MemorySession(),
        PermissionPolicy(callback=approval),
        authority=store.provider("r1-bot"),
    )
    result = await declared_call(
        executor,
        ToolCall(id="write", name="write_file", arguments='{"path":"output","content":"ok"}'),
    )
    assert store.authority("r1-bot").permission_digest == original.authority.permission_digest
    if change == "revoke-and-restore":
        assert result.error == "permission_changed"
        assert result.outcome == "not_started"
        assert not (tmp_path / "output").exists()
        assert store.authority("r1-bot").permission_version == 3
    else:
        assert result.ok
        assert (tmp_path / "output").read_text() == "ok"
        assert store.authority("r1-bot").permission_version == 1


async def test_child_checks_live_bot_policy_after_model_request(tmp_path: Path) -> None:
    (tmp_path / "secret").write_text("PRIVATE_R1_SENTINEL")
    store = BotStore(KernelDatabase.for_workspace(tmp_path))
    store.save(configuration("delegate_task", "read_file"))

    class RevokingModel(ScriptedModel):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            if not self.requests:
                assert {schema["function"]["name"] for schema in request.tools} == {"read_file"}
                store.save(configuration("delegate_task"), expected_version=1)
            return await super().complete(request)

    model = RevokingModel(
        [
            ModelResponse(
                assistant=AssistantMessage(
                    tool_calls=[
                        ToolCall(id="read", name="read_file", arguments='{"path":"secret"}')
                    ]
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                assistant=AssistantMessage(text='{"summary":"Read denied"}'), finish_reason="stop"
            ),
        ]
    )
    children: list[MemorySession] = []

    async def child_factory(session_id: str) -> MemorySession:
        child = MemorySession(session_id)
        children.append(child)
        return child

    registry = build_file_tools(tmp_path, Config().tools)
    manager = SubagentManager(model, registry, child_factory, SubagentsConfig(enabled=True))
    manager.register(registry)
    executor = ToolExecutor(
        registry, MemorySession(), PermissionPolicy(), authority=store.provider("r1-bot")
    )
    try:
        result = await declared_call(
            executor,
            ToolCall(
                id="delegate",
                name="delegate_task",
                arguments='{"role":"explorer","task":"Read secret"}',
            ),
        )
        assert result.ok
        assert len(children) == 1
        child_results = [event for event in children[0].events if event.type == "tool.result"]
        assert len(child_results) == 1
        assert child_results[0].data["error"] == "permission_denied"
        assert child_results[0].data["outcome"] == "not_started"
        assert model.requests[1].tools == []
        assert "PRIVATE_R1_SENTINEL" not in json.dumps(model.requests[1].model_dump())
    finally:
        await manager.close()


def test_database_open_failure_uses_persistence_error_boundary(tmp_path: Path) -> None:
    database = KernelDatabase.for_workspace(tmp_path)
    store = BotStore(database)
    store.save(configuration("read_file"))
    database.path.unlink()
    database.path.mkdir()
    with pytest.raises(PersistenceError):
        store.authority("r1-bot")


async def test_unreadable_authority_stops_executor_instead_of_ordinary_tool_error(
    tmp_path: Path,
) -> None:
    database = KernelDatabase.for_workspace(tmp_path)
    store = BotStore(database)
    store.save(configuration("write_file"))
    executor = ToolExecutor(
        build_file_tools(tmp_path, Config().tools),
        MemorySession(),
        PermissionPolicy(),
        authority=store.provider("r1-bot"),
    )
    database.path.unlink()
    database.path.mkdir()
    with pytest.raises(PersistenceError):
        await declared_call(
            executor,
            ToolCall(id="write", name="write_file", arguments='{"path":"output","content":"x"}'),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("arguments", ["[]", '{"value":NaN}', '{"value":Infinity}'])
def test_approval_intent_rejects_noncanonical_json_values(arguments: str) -> None:
    with pytest.raises(ValidationError):
        OperationIntent(
            task_id="task",
            bot_id="r1-bot",
            session_id="session",
            run_id="run",
            call_id="call",
            tool_name="write_file",
            arguments_json=arguments,
            permission_version=1,
            permission_digest="digest",
        )


@pytest.mark.parametrize("approved", [1, "true", "yes"])
def test_approval_reply_does_not_coerce_truthy_input(approved: Any) -> None:
    with pytest.raises(ValidationError):
        InteractionReply(approved=approved)
