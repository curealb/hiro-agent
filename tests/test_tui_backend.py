"""Live display snapshots, durable Bot isolation, and independent worker lifecycle."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.contracts import RunResult, RunStatus, SessionEvent
from hiro_agent.errors import ConfigurationError, ModelError, PersistenceError
from hiro_agent.kernel_contracts import (
    InteractionRecord,
    InteractionReply,
    InteractionRequest,
    InteractionState,
    OperationIntent,
    RunClaim,
    RunEvidence,
    TaskState,
)
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.models import ScriptedModel
from hiro_agent.session import JsonlSession, read_session_events, read_session_snapshot
from hiro_agent.tui_backend import TuiBackend


@pytest.fixture
def backend(tmp_path: Path) -> TuiBackend:
    kernel = KernelRuntime(tmp_path)
    for bot_id in ("writer", "reader"):
        kernel.bots.save(
            Config(
                bot=BotConfig(id=bot_id, name=bot_id.title()),
                model=ModelConfig(adapter="mock", model="offline"),
            )
        )
    return TuiBackend(tmp_path, kernel=kernel)


def journal(workspace: Path, session_id: str, *events: tuple[str, dict]) -> Path:
    directory = workspace / ".hiro-agent" / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.jsonl"
    rows = [("session.created", {"workspace": str(workspace)}), *events]
    path.write_text(
        "".join(
            SessionEvent(session_id=session_id, seq=index, type=kind, data=data).model_dump_json()
            + "\n"
            for index, (kind, data) in enumerate(rows, 1)
        )
    )
    return path


def claim(backend: TuiBackend, text: str = "work", bot_id: str = "writer") -> RunClaim:
    task = backend.submit(bot_id, text)
    result = backend.kernel.tasks.claim("test-worker")
    assert result is not None and result.task.id == task.id
    return result


def finish(backend: TuiBackend, claimed: RunClaim, output: str = "done") -> None:
    backend.kernel.tasks.finish(
        claimed.run.id,
        RunResult(
            run_id=claimed.run.id,
            session_id=claimed.run.session_id,
            status=RunStatus.COMPLETED,
            reason="finished",
            output=output,
        ),
    )


def waiting(backend: TuiBackend) -> tuple[RunClaim, InteractionRecord]:
    claimed = claim(backend)
    record = backend.kernel.interactions.create(
        InteractionRequest(
            task_id=claimed.task.id,
            bot_id="writer",
            kind="question",
            prompt="Which branch?",
            idempotency_key="question",
        )
    )
    backend.kernel.tasks.finish(
        claimed.run.id,
        RunResult(
            run_id=claimed.run.id,
            session_id=claimed.run.session_id,
            status=RunStatus.PAUSED,
            reason="await reply",
        ),
        interaction_id=record.id,
        waiting_for="question",
    )
    return claimed, record


async def test_live_snapshot_ignores_partial_tail_without_locking_or_repair(tmp_path: Path) -> None:
    session = await JsonlSession.open(tmp_path, "live")
    try:
        await session.append("task.received", {"text": "hello"})
        with pytest.raises(PersistenceError, match="占用"):
            read_session_events(tmp_path, "live")
        with session.path.open("ab") as stream:
            stream.write(b'{"version":1,"text":"partial')
        original = session.path.read_bytes()
        metadata = session.path.stat()
        events = read_session_snapshot(tmp_path, "live")
        assert [event.type for event in events] == ["session.created", "task.received"]
        assert session.path.read_bytes() == original
        assert session.path.stat().st_mtime_ns == metadata.st_mtime_ns
    finally:
        await session.close()


def test_snapshot_excludes_even_complete_json_until_newline(tmp_path: Path) -> None:
    path = journal(tmp_path, "live", ("task.received", {"text": "visible after newline"}))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_session_snapshot(tmp_path, "live")) == 1
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert len(read_session_snapshot(tmp_path, "live")) == 2


def test_snapshot_read_is_bounded_by_initial_file_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = journal(tmp_path, "growing")
    extra = SessionEvent(
        session_id="growing", seq=2, type="task.received", data={"text": "next snapshot"}
    ).model_dump_json()
    real_read = os.read
    appended = False

    def append_during_read(fd: int, size: int) -> bytes:
        nonlocal appended
        if not appended:
            with path.open("a") as stream:
                stream.write(extra + "\n")
            appended = True
        return real_read(fd, size)

    monkeypatch.setattr("hiro_agent.session.os.read", append_during_read)
    assert len(read_session_snapshot(tmp_path, "growing")) == 1
    assert len(read_session_snapshot(tmp_path, "growing")) == 2


@pytest.mark.parametrize(
    "row",
    [
        b"not json\n",
        b"[]\n",
        b'{"version":1,"seq":true}\n',
        SessionEvent(session_id="bad", seq=3, type="task.received", data={"text": "x"})
        .model_dump_json()
        .encode()
        + b"\n",
        SessionEvent(session_id="other", seq=2, type="task.received", data={"text": "x"})
        .model_dump_json()
        .encode()
        + b"\n",
        SessionEvent(
            session_id="bad", seq=2, type="tool.result", data={"call_id": "missing", "ok": True}
        )
        .model_dump_json()
        .encode()
        + b"\n",
    ],
)
def test_snapshot_rejects_complete_corruption_without_mutating(tmp_path: Path, row: bytes) -> None:
    path = journal(tmp_path, "bad")
    original = path.read_bytes() + row
    path.write_bytes(original)
    with pytest.raises(PersistenceError):
        read_session_snapshot(tmp_path, "bad")
    assert path.read_bytes() == original


@pytest.mark.parametrize("target", ["journal", "sessions", "parent", "fifo", "id"])
def test_snapshot_rejects_unsafe_paths(tmp_path: Path, target: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    path = journal(tmp_path, "safe")
    if target == "journal":
        path.rename(outside / "saved")
        path.symlink_to(outside / "saved")
    elif target == "sessions":
        path.parent.rename(outside / "sessions")
        path.parent.symlink_to(outside / "sessions", target_is_directory=True)
    elif target == "parent":
        path.parent.parent.rename(outside / "parent")
        path.parent.parent.symlink_to(outside / "parent", target_is_directory=True)
    elif target == "fifo":
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(PersistenceError):
        read_session_snapshot(tmp_path, "../safe" if target == "id" else "safe")


def test_missing_snapshot_does_not_create_paths(tmp_path: Path) -> None:
    with pytest.raises(PersistenceError):
        read_session_snapshot(tmp_path, "missing")
    assert list(tmp_path.iterdir()) == []


def test_bot_history_isolation_queued_text_and_no_duplicate_output(backend: TuiBackend) -> None:
    first = claim(backend, "first message")
    journal(
        backend.workspace,
        first.task.session_id,
        ("prompt.system", {"text": "private runtime prompt"}),
        ("task.received", {"text": "first message"}),
        ("assistant.message", {"message": {"text": "first answer"}}),
    )
    finish(backend, first, "first answer")
    second = backend.submit("writer", "queued second message")
    backend.submit("reader", "other Bot secret")

    conversation = TuiBackend(backend.workspace).conversation("writer")
    assert [entry.text for entry in conversation.entries] == [
        "first message",
        "first answer",
        "queued second message",
    ]
    assert [task.id for task in conversation.tasks] == [first.task.id, second.id]
    assert conversation.error is None
    assert all(entry.task_id in {first.task.id, second.id} for entry in conversation.entries)


def test_history_is_limited_to_latest_thirty_tasks(backend: TuiBackend) -> None:
    for index in range(35):
        backend.submit("writer", f"task {index}")
    conversation = backend.conversation("writer")
    assert len(conversation.tasks) == 30
    assert conversation.entries[0].text == "task 5"
    assert conversation.entries[-1].text == "task 34"


async def test_model_failure_without_output_displays_durable_reason(backend: TuiBackend) -> None:
    model = ScriptedModel([ModelError("模型服务暂时不可用")])
    kernel = KernelRuntime(backend.workspace, model_factory=lambda _: model)
    display = TuiBackend(backend.workspace, kernel=kernel)
    task = display.submit("writer", "request a report")
    worker = kernel.worker(poll_interval=0.01)
    try:
        assert await worker.tick()
    finally:
        await worker.close()
    saved = kernel.tasks.get(task.id)
    assert saved.state == TaskState.FAILED and saved.output == ""
    conversation = display.conversation("writer")
    assert conversation.error is None
    assert [(entry.role, entry.text) for entry in conversation.entries] == [
        ("user", "request a report"),
        ("system", "任务失败：模型服务暂时不可用"),
    ]
    assert kernel.tasks.get(task.id) == saved


def test_uncertain_task_explains_required_verification_without_replay(backend: TuiBackend) -> None:
    claimed = claim(backend, "write exactly once")
    journal(backend.workspace, claimed.task.session_id)
    saved = backend.kernel.tasks.reconcile(
        RunEvidence(
            run_id=claimed.run.id,
            session_id=claimed.task.session_id,
            started=True,
            unknown_side_effect=True,
        )
    )
    conversation = backend.conversation("writer")
    [status] = [entry for entry in conversation.entries if entry.role == "system"]
    assert status.task_id == claimed.task.id
    assert "待核实" in status.text and "人工明确核实" in status.text
    assert "不会自动重新执行" in status.text and saved.reason in status.text
    assert backend.kernel.tasks.get(claimed.task.id) == saved
    assert backend.kernel.tasks.claim("observer") is None


def test_cancelled_task_has_one_visible_status_and_reason(backend: TuiBackend) -> None:
    task = backend.submit("writer", "cancel before starting")
    saved = backend.cancel("writer", task.id)
    conversation = backend.conversation("writer")
    assert len(conversation.entries) == 2
    status = conversation.entries[-1]
    assert status.role == "system" and status.task_id == task.id
    assert "已取消" in status.text and saved.reason in status.text


def test_failure_status_does_not_repeat_identical_assistant_output(backend: TuiBackend) -> None:
    claimed = claim(backend)
    journal(backend.workspace, claimed.task.session_id)
    backend.kernel.tasks.finish(
        claimed.run.id,
        RunResult(
            run_id=claimed.run.id,
            session_id=claimed.task.session_id,
            status=RunStatus.FAILED,
            reason="upstream unavailable",
            output="upstream unavailable",
        ),
    )
    conversation = backend.conversation("writer")
    assert [entry.text for entry in conversation.entries] == [
        "work",
        "upstream unavailable",
        "任务失败",
    ]


def test_missing_and_corrupt_history_preserve_task_and_fallback(backend: TuiBackend) -> None:
    first = claim(backend, "missing history")
    finish(backend, first, "saved answer")
    second = claim(backend, "corrupt history")
    path = journal(backend.workspace, second.task.session_id)
    path.write_bytes(path.read_bytes() + b"bad row\n")
    finish(backend, second)
    original = path.read_bytes()
    conversation = backend.conversation("writer")
    assert conversation.error is not None
    assert first.task.id[:8] in conversation.error and second.task.id[:8] in conversation.error
    assert [entry.text for entry in conversation.entries] == [
        "missing history",
        "saved answer",
        "corrupt history",
        "done",
    ]
    assert path.read_bytes() == original


def test_unchanged_journals_are_cached_but_appends_and_replacements_refresh(
    backend: TuiBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    claimed = claim(backend)
    path = journal(backend.workspace, claimed.task.session_id)
    real_read = read_session_snapshot
    reads = 0

    def counted(workspace: Path, session_id: str) -> list[SessionEvent]:
        nonlocal reads
        reads += 1
        return real_read(workspace, session_id)

    monkeypatch.setattr("hiro_agent.tui_backend.read_session_snapshot", counted)
    backend.conversation("writer")
    backend.conversation("writer")
    assert reads == 1
    with path.open("a") as stream:
        stream.write(
            SessionEvent(
                session_id=claimed.task.session_id,
                seq=2,
                type="assistant.message",
                data={"message": {"text": "live update"}},
            ).model_dump_json()
            + "\n"
        )
    assert backend.conversation("writer").entries[-1].text == "live update"
    assert reads == 2
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    backend.conversation("writer")
    assert reads == 3


def test_durable_answer_is_visible_before_resume_and_not_duplicated(backend: TuiBackend) -> None:
    claimed, record = waiting(backend)
    backend.reply("writer", record.id, InteractionReply(text="feature branch"))
    journal(
        backend.workspace,
        claimed.task.session_id,
        ("interaction.answer", {"interaction_id": record.id, "text": "feature branch"}),
        ("task.continued", {"text": "用户对问题的答复 synthetic context"}),
        ("assistant.message", {"message": {"text": "hidden"}, "prepared_operation": True}),
        ("assistant.message", {"message": {"text": "continuing"}}),
    )
    conversation = backend.conversation("writer")
    assert [entry.text for entry in conversation.entries] == [
        "work",
        "Which branch?",
        "feature branch",
        "continuing",
    ]
    assert conversation.interactions == ()


def test_tool_activity_is_readable_without_provider_metadata(backend: TuiBackend) -> None:
    claimed = claim(backend)
    call = {"id": "read", "name": "read_file", "arguments": '{"path":"report.txt"}'}
    journal(
        backend.workspace,
        claimed.task.session_id,
        ("assistant.message", {"message": {"tool_calls": [call], "provider_metadata": {"x": 1}}}),
        ("tool.called", {"call": call}),
        ("tool.started", {"call": call}),
        ("tool.result", {"call_id": "read", "ok": True, "value": "report contents"}),
    )
    entries = backend.conversation("writer").entries
    assert len(entries) == 3
    assert entries[1].role == entries[2].role == "tool"
    assert "read_file" in entries[1].text and "report contents" in entries[2].text
    assert "provider_metadata" not in str(entries)


def test_summary_pending_excludes_stale_interactions_and_prioritizes_waiting(
    backend: TuiBackend,
) -> None:
    claimed, current = waiting(backend)
    backend.submit("writer", "later queued work")
    backend.kernel.interactions.create(
        current.request.model_copy(update={"idempotency_key": "stale"})
    )
    summaries = {summary.bot_id: summary for summary in backend.list_bots()}
    assert summaries["writer"].status == "waiting_input"
    assert summaries["writer"].pending_count == 1
    assert summaries["writer"].task_count == 2
    assert summaries["writer"].preview == "later queued work"
    assert summaries["reader"].status == "idle"
    assert [record.id for record in backend.conversation("writer").interactions] == [current.id]
    backend.cancel("writer", claimed.task.id)
    assert {summary.bot_id: summary for summary in backend.list_bots()}["writer"].pending_count == 0


def test_submit_idempotency_and_owner_guarded_cancel(backend: TuiBackend) -> None:
    task = backend.submit("writer", "persist", idempotency_key="once")
    assert backend.submit("writer", "persist", idempotency_key="once") == task
    with pytest.raises(ConfigurationError):
        backend.submit("writer", " \n ")
    with pytest.raises(ConfigurationError):
        backend.cancel("reader", task.id)
    assert backend.cancel("writer", task.id).state == TaskState.CANCELLED


def test_reply_rejects_wrong_bot_stale_pointer_and_cancelled_task(backend: TuiBackend) -> None:
    claimed, current = waiting(backend)
    stale = backend.kernel.interactions.create(
        current.request.model_copy(update={"idempotency_key": "stale"})
    )
    response = InteractionReply(text="branch")
    with pytest.raises(ConfigurationError):
        backend.reply("reader", current.id, response)
    with pytest.raises(ConfigurationError):
        backend.reply("writer", stale.id, response)
    backend.cancel("writer", claimed.task.id)
    with pytest.raises(ConfigurationError):
        backend.reply("writer", current.id, response)
    assert backend.kernel.interactions.get(current.id).reply is None


def test_reply_rejects_interaction_pointing_to_other_bots_task(backend: TuiBackend) -> None:
    task = backend.submit("reader", "other Bot")
    record = backend.kernel.interactions.create(
        InteractionRequest(
            bot_id="writer",
            task_id=task.id,
            kind="question",
            prompt="bad owner",
            idempotency_key="x",
        )
    )
    with pytest.raises(ConfigurationError):
        backend.reply("writer", record.id, InteractionReply(text="no"))
    assert backend.kernel.interactions.get(record.id).state == InteractionState.PENDING


def test_approval_reply_is_bound_to_current_saved_operation(backend: TuiBackend) -> None:
    claimed = claim(backend)
    authority = backend.kernel.bots.authority("writer")
    operation = OperationIntent(
        task_id=claimed.task.id,
        bot_id="writer",
        session_id=claimed.task.session_id,
        run_id=claimed.run.id,
        call_id="write",
        tool_name="write_file",
        arguments_json='{"path":"report.txt","content":"saved"}',
        permission_version=authority.permission_version,
        permission_digest=authority.permission_digest,
    )
    record = backend.kernel.interactions.create(
        InteractionRequest(
            bot_id="writer",
            task_id=claimed.task.id,
            kind="approval",
            prompt="Approve write_file",
            operation=operation,
            idempotency_key="approval",
        )
    )
    backend.kernel.tasks.finish(
        claimed.run.id,
        RunResult(
            run_id=claimed.run.id,
            session_id=claimed.task.session_id,
            status=RunStatus.PAUSED,
            reason="approve",
        ),
        pending_operation=operation,
        interaction_id=record.id,
        waiting_for="approval",
    )
    summary = next(item for item in backend.list_bots() if item.bot_id == "writer")
    assert summary.status == "waiting_approval" and summary.pending_count == 1
    answer = backend.reply("writer", record.id, InteractionReply(approved=True))
    assert backend.reply("writer", record.id, InteractionReply(approved=True)) == answer
    assert backend.kernel.tasks.get(claimed.task.id).state == TaskState.WAITING_APPROVAL
    assert not (backend.workspace / "report.txt").exists()
    backend.kernel.tasks.wake(claimed.task.id, record.id)
    with pytest.raises(ConfigurationError):
        backend.reply("writer", record.id, InteractionReply(approved=False))


def test_lock_probe_does_not_create_unlink_or_recover(backend: TuiBackend) -> None:
    path = backend.workspace / ".hiro-agent" / "worker.lock"
    assert not backend.worker_running()
    assert not path.exists()
    with path.open("w") as stream:
        inode = path.stat().st_ino
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert backend.worker_running()
        assert backend.start_worker() is None
        assert path.stat().st_ino == inode
    assert not backend.worker_running()
    assert path.stat().st_ino == inode


def test_real_worker_is_detached_and_survives_backend_recreation(backend: TuiBackend) -> None:
    (backend.workspace / "hiro-demo.txt").write_text("read-only worker fixture\n")
    task = backend.submit("reader", "independent execution")
    process = backend.start_worker()
    assert process is not None
    try:
        assert os.getsid(process.pid) == process.pid
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not backend.worker_running():
            assert process.poll() is None
            time.sleep(0.05)
        assert backend.worker_running()
        reopened = TuiBackend(backend.workspace)
        assert reopened.worker_running()
        assert reopened.start_worker() is None
        assert process.poll() is None
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and reopened.kernel.tasks.get(task.id).state != TaskState.SUCCEEDED
        ):
            assert process.poll() is None
            time.sleep(0.05)
        completed = reopened.kernel.tasks.get(task.id)
        assert completed.state == TaskState.SUCCEEDED
        assert completed.output
        assert reopened.conversation("reader").error is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 0
    assert not backend.worker_running()
    assert (backend.workspace / ".hiro-agent" / "worker.lock").exists()
    assert (backend.workspace / ".hiro-agent" / "worker.log").exists()
