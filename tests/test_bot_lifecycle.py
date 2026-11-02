"""Bot cleanup preserves peers and can run while a compatible worker is idle."""

import asyncio
import fcntl
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hiro_agent import bot_lifecycle
from hiro_agent.bot_lifecycle import clear_bot_history
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.contracts import AssistantMessage, DelegationLink, ModelResponse, ToolCall
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import TaskSubmission
from hiro_agent.kernel_db import maintenance_lock
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.routines import RoutineCreate
from hiro_agent.session import JsonlSession


def runtime(workspace: Path) -> KernelRuntime:
    kernel = KernelRuntime(
        workspace,
        model_factory=lambda config: ScriptedModel(
            [ModelResponse(assistant=AssistantMessage(text="fresh output"), finish_reason="stop")]
        ),
    )
    for bot_id in ("owner", "peer"):
        kernel.bots.save(
            Config(
                model=ModelConfig(adapter="mock"),
                bot=BotConfig(id=bot_id, allowed_tools=("read_file", "send_handoff")),
            )
        )
    return kernel


async def journal(kernel: KernelRuntime, session_id: str, bot_id: str = "owner") -> Path:
    session = await JsonlSession.open(kernel.workspace, session_id)
    try:
        await session.append("session.bot", kernel.bots.authority(bot_id).model_dump(mode="json"))
        await session.append("task.received", {"text": "private old input"})
        return session.path
    finally:
        await session.close()


def terminal(kernel: KernelRuntime, bot_id: str = "owner"):
    task = kernel.tasks.submit(TaskSubmission(bot_id=bot_id, text="old secret"))
    return kernel.tasks.cancel(task.id)


async def test_clear_all_context_then_new_task_with_idle_online_worker(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    task = terminal(kernel)
    own_path = await journal(kernel, task.session_id)
    peer = terminal(kernel, "peer")
    peer_path = await journal(kernel, peer.session_id, "peer")
    peer_bytes = peer_path.read_bytes()
    kernel.memory.create("owner", "old note")
    kernel.memory.remember_task("owner", task.id)
    kernel.memory.remember_task("peer", peer.id)
    worker = kernel.worker()
    try:
        assert not await worker.tick()
        result = clear_bot_history(tmp_path, "owner")
        assert result["counts"]["tasks"] == 1
        assert result["counts"]["sessions"] == 1
        assert result["counts"]["memories"] == 2
        assert not own_path.exists()
        assert peer_path.read_bytes() == peer_bytes
        assert kernel.memory.sync_terminal("owner") == []
        assert kernel.tasks.list(bot_id="owner") == []
        assert len(kernel.memory.list("peer")) == 1
        assert kernel.bots.get("owner").bot_id == "owner"
        assert clear_bot_history(tmp_path, "owner")["counts"]["sessions"] == 0
        fresh = kernel.tasks.submit(TaskSubmission(bot_id="owner", text="new request"))
        assert await worker.tick()
        assert kernel.tasks.get(fresh.id).output == "fresh output"
        assert all("old secret" not in record.content for record in kernel.memory.list("owner"))
    finally:
        await worker.close()
    assert len(KernelRuntime(tmp_path).tasks.list(bot_id="owner")) == 1


async def test_traditional_bound_parent_and_child_removed_but_project_kept(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    parent = await JsonlSession.open(tmp_path, "traditional")
    try:
        await parent.append("session.bot", kernel.bots.authority("owner").model_dump(mode="json"))
        await parent.append("run.started", {}, "run")
        await parent.append("task.received", {"text": "old delegated task"}, "run")
        call = ToolCall(id="delegate", name="delegate_task", arguments="{}")
        await parent.append(
            "assistant.message",
            {"message": AssistantMessage(tool_calls=[call]).model_dump()},
            "run",
        )
        await parent.append("tool.called", {"call": call.model_dump()}, "run")
        await parent.append("tool.started", {"call": call.model_dump()}, "run")
        link = DelegationLink(
            parent_session_id="traditional",
            parent_run_id="run",
            child_session_id="child",
            child_run_id="child-run",
            call_id="delegate",
            role="explorer",
        )
        await parent.append("delegation.started", link.model_dump(), "run")
        child = await JsonlSession.open(tmp_path, "child")
        await child.append("subagent.created", link.model_dump())
        await child.close()
    finally:
        await parent.close()
    peer_path = await journal(kernel, "unrelated", "peer")
    project = tmp_path / "source.py"
    project.write_text("keep")
    skill = tmp_path / ".hiro-agent" / "skills" / "shared.md"
    skill.write_text("# keep shared skill")
    result = clear_bot_history(tmp_path, "owner")
    assert result["counts"]["sessions"] == 2
    assert peer_path.exists() and project.read_text() == "keep" and skill.exists()
    assert (parent.path.parent / "traditional.lock").exists()
    assert (parent.path.parent / "child.lock").exists()


async def test_busy_later_journal_prevents_any_partial_deletion(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    first = terminal(kernel)
    second = terminal(kernel)
    paths = [await journal(kernel, task.session_id) for task in (first, second)]
    before = [path.read_bytes() for path in paths]
    kernel.memory.create("owner", "preserve on failure")
    busy = await JsonlSession.open(tmp_path, second.session_id)
    try:
        with pytest.raises(PersistenceError, match="正在使用"):
            clear_bot_history(tmp_path, "owner", delete_bot=True)
        assert [path.read_bytes() for path in paths] == before
        assert len(kernel.tasks.list(bot_id="owner")) == 2
        assert len(kernel.memory.list("owner")) == 1
    finally:
        await busy.close()


def test_active_task_and_unfinished_peer_links_refuse_without_mutation(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    source = kernel.tasks.submit(TaskSubmission(bot_id="owner", text="source"))
    peer = kernel.tasks.submit(TaskSubmission(bot_id="peer", text="peer", parent_task_id=source.id))
    with pytest.raises(ConfigurationError, match="其他 Bot"):
        clear_bot_history(tmp_path, "owner")
    kernel.tasks.cancel(peer.id)
    claim = kernel.tasks.claim("test")
    assert claim is not None
    with pytest.raises(ConfigurationError, match="正在执行"):
        clear_bot_history(tmp_path, "owner")
    assert len(kernel.tasks.list()) == 2


async def test_database_failure_restores_every_staged_journal(tmp_path: Path, monkeypatch) -> None:
    kernel = runtime(tmp_path)
    task = terminal(kernel)
    path = await journal(kernel, task.session_id)
    before = path.read_bytes()
    kernel.memory.create("owner", "keep")
    original = bot_lifecycle._delete_records

    def fail(connection, bot_id, delete_bot):
        original(connection, bot_id, delete_bot)
        raise RuntimeError("injected SQL-side failure")

    monkeypatch.setattr(bot_lifecycle, "_delete_records", fail)
    with pytest.raises(RuntimeError, match="injected"):
        clear_bot_history(tmp_path, "owner", delete_bot=True)
    assert path.read_bytes() == before
    assert kernel.tasks.get(task.id).id == task.id
    assert len(kernel.memory.list("owner")) == 1
    assert not (tmp_path / ".hiro-agent" / "history-cleanup").exists()


def test_delete_owned_definitions_and_handoff_fks_preserves_peer_chain(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    source = kernel.tasks.submit(TaskSubmission(bot_id="owner", text="source"))
    message = kernel.handoffs.send("owner", source.id, "peer", "help", idempotency_key="handoff")
    kernel.handoffs.sync()
    target = kernel.handoffs.get("owner", message.id).target_task_id
    assert target is not None
    kernel.tasks.cancel(source.id)
    kernel.tasks.cancel(target)
    kernel.handoffs.sync()
    # Result delivery is also a Task in the source Bot's chain.
    for task in kernel.tasks.list(bot_id="owner"):
        kernel.tasks.cancel(task.id)
    routine = kernel.routines.create(
        RoutineCreate(
            bot_id="owner", text="later", scheduled_at=datetime.now(UTC) + timedelta(days=1)
        )
    )
    skill = tmp_path / ".hiro-agent" / "skills" / "shared.md"
    skill.write_text("# Shared\nUseful instruction")
    kernel.skills.enable("owner", "shared")
    chain_before = kernel.tasks.chain(target)
    result = clear_bot_history(tmp_path, "owner", delete_bot=True)
    assert result["deleted"]
    assert kernel.bots.list()[0].bot_id == "peer"
    assert kernel.tasks.get(target).bot_id == "peer"
    assert kernel.tasks.chain(target) == chain_before
    assert skill.exists()
    with kernel.database.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute("SELECT 1 FROM routines WHERE id=?", (routine.id,)).fetchone()
            is None
        )
        assert (
            connection.execute("SELECT 1 FROM skill_settings WHERE bot_id='owner'").fetchone()
            is None
        )
        assert connection.execute("SELECT 1 FROM handoff_messages").fetchone() is None
    assert clear_bot_history(tmp_path, "owner", delete_bot=True)["counts"]["tasks"] == 0


@pytest.mark.parametrize("committed", [False, True])
async def test_interrupted_staging_recovered_before_retry(tmp_path: Path, committed: bool) -> None:
    kernel = runtime(tmp_path)
    task = terminal(kernel)
    path = await journal(kernel, task.session_id)
    clear_bot_history(tmp_path, "peer")  # Install recovery receipt schema.
    staging = tmp_path / ".hiro-agent" / "history-cleanup"
    staging.mkdir()
    (staging / "manifest.json").write_text(
        json.dumps({"id": "interrupted", "journals": [path.name]})
    )
    path.rename(staging / path.name)
    if committed:
        with maintenance_lock(kernel.database.path.parent, exclusive=True):
            with kernel.database.transaction() as connection:
                bot_lifecycle._delete_records(connection, "owner", False)
                connection.execute("INSERT INTO history_cleanup_commits VALUES ('interrupted')")
    with pytest.raises(PersistenceError, match="重试"):
        kernel.tasks.submit(TaskSubmission(bot_id="owner", text="blocked until repaired"))
    result = clear_bot_history(tmp_path, "owner")
    assert result["counts"]["tasks"] == (0 if committed else 1)
    assert not staging.exists() and not path.exists()


def test_legacy_online_worker_requires_restart(tmp_path: Path) -> None:
    runtime(tmp_path)
    descriptor = os.open(tmp_path / ".hiro-agent" / "worker.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(ConfigurationError, match="重启 worker"):
            clear_bot_history(tmp_path, "owner")
    finally:
        os.close(descriptor)


async def test_serving_idle_worker_keeps_running_after_clear(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    worker = kernel.worker(poll_interval=0.01)
    stop = asyncio.Event()
    serving = asyncio.create_task(worker.serve(stop))
    try:
        await asyncio.sleep(0.03)
        clear_bot_history(tmp_path, "owner")
        task = kernel.tasks.submit(TaskSubmission(bot_id="owner", text="new start"))
        for _ in range(100):
            if kernel.tasks.get(task.id).output:
                break
            await asyncio.sleep(0.01)
        assert kernel.tasks.get(task.id).output == "fresh output"
        assert not serving.done()
    finally:
        stop.set()
        await serving


@pytest.mark.parametrize(
    "manifest",
    ["{", "[]", "{}", '{"id": "x", "journals": null}', '{"id":"x","journals":["../bad.jsonl"]}'],
)
def test_invalid_recovery_manifest_blocks_writes_without_raw_errors(
    tmp_path: Path, manifest: str
) -> None:
    kernel = runtime(tmp_path)
    clear_bot_history(tmp_path, "owner")
    staging = tmp_path / ".hiro-agent" / "history-cleanup"
    staging.mkdir()
    (staging / "manifest.json").write_text(manifest)
    with pytest.raises(PersistenceError, match="清单无效"):
        clear_bot_history(tmp_path, "owner")
    assert kernel.bots.get("owner").bot_id == "owner"


def test_clear_discards_unacknowledged_routine_slot_without_recreating_task(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    now = datetime.now(UTC)
    routine = kernel.routines.create(
        RoutineCreate(bot_id="owner", text="old scheduled request", scheduled_at=now)
    )
    trigger = kernel.routines._reserve(now)
    assert trigger is not None
    task = kernel.tasks.submit(trigger.submission)
    # Simulate crash between Task.submit and routine acknowledgment.
    clear_bot_history(tmp_path, "owner")
    assert kernel.routines.tick(now=now) == []
    assert kernel.tasks.list(bot_id="owner") == []
    remaining = kernel.routines.get(routine.id)
    assert remaining.definition == routine.definition
    assert remaining.last_task_id is None and remaining.next_scheduled_at is None
    with pytest.raises(ConfigurationError):
        kernel.tasks.get(task.id)


@pytest.mark.parametrize("consumer", ["routine", "handoff"])
def test_direct_dispatch_cannot_race_clear_between_snapshot_and_submit(
    tmp_path: Path, monkeypatch, consumer: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    kernel = runtime(tmp_path)
    if consumer == "routine":
        kernel.routines.create(
            RoutineCreate(bot_id="owner", text="reserved", scheduled_at=datetime.now(UTC))
        )
        action = kernel.routines.tick
    else:
        source = kernel.tasks.submit(TaskSubmission(bot_id="owner", text="source"))
        kernel.handoffs.send("owner", source.id, "peer", "handoff", idempotency_key="race")
        action = kernel.handoffs.sync
    original = kernel.tasks.submit

    def during_submit(request):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(clear_bot_history, tmp_path, "owner")
            with pytest.raises(PersistenceError, match="正在执行"):
                future.result(timeout=5)
        return original(request)

    monkeypatch.setattr(kernel.tasks, "submit", during_submit)
    action()
    assert kernel.tasks.list()


async def test_stale_compatibility_files_do_not_authorize_legacy_worker(tmp_path: Path) -> None:
    kernel = runtime(tmp_path)
    worker = kernel.worker()
    await worker.close()
    protocol = tmp_path / ".hiro-agent" / "maintenance-worker.lock"
    assert protocol.exists()
    descriptor = os.open(tmp_path / ".hiro-agent" / "worker.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, b"hiro-maintenance-v1\n")
        with pytest.raises(ConfigurationError, match="重启 worker"):
            clear_bot_history(tmp_path, "owner")
    finally:
        os.close(descriptor)
