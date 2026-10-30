"""Resume real journals across processes without replaying work or losing the current chat."""

import asyncio
import fcntl
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from hiro_agent import cli
from hiro_agent.bootstrap import create_harness
from hiro_agent.config import Config
from hiro_agent.context import SYSTEM_PROMPT
from hiro_agent.contracts import AssistantMessage, ModelRequest, ModelResponse
from hiro_agent.errors import HiroError
from hiro_agent.models import MockModel
from hiro_agent.session import JsonlSession, delete_session, read_session_events


def run_cli(*arguments: str, input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hiro_agent.cli", *arguments],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"},
    )


async def create_session(workspace: Path, session_id: str, task: str = "") -> Path:
    session = await JsonlSession.open(workspace, session_id)
    try:
        await session.append("prompt.system", {"text": SYSTEM_PROMPT})
        if task:
            await session.append("task.received", {"text": task})
            await session.append(
                "assistant.message",
                {"message": AssistantMessage(text=f"Reply to {task}").model_dump(mode="json")},
            )
        return session.path
    finally:
        await session.close()


def saved_logs(workspace: Path) -> dict[str, bytes]:
    return {
        path.stem: path.read_bytes()
        for path in (workspace / ".hiro-agent" / "sessions").glob("*.jsonl")
    }


@pytest.mark.parametrize("numbered", [False, True])
def test_history_listing_keeps_copyable_ids_in_normal_terminal(
    monkeypatch: pytest.MonkeyPatch, numbered: bool
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=80))
    session_id = "0123456789abcdef0123456789abcdef"
    timestamp = "2026-10-31T10:24:02.012345+00:00"
    title = "查看 [bold]配置文件[/bold] 和模型上下文"

    cli.show_sessions(
        [
            {
                "session_id": session_id,
                "status": "completed",
                "updated_at": timestamp,
                "title": title,
            }
        ],
        numbered=numbered,
    )

    assert session_id in output.getvalue()
    assert timestamp in output.getvalue()
    assert title in output.getvalue()


@pytest.mark.parametrize("entrypoint", ["resume", "root", "chat"])
def test_reopen_saved_conversation_replays_history_without_running_again(
    tmp_path: Path, entrypoint: str
) -> None:
    first = run_cli(
        "--mock", "--workspace", str(tmp_path), input_text="original request\nn\n/exit\n"
    )
    assert first.returncode == 0, first.stdout + first.stderr
    original = saved_logs(tmp_path)
    [session_id] = original
    arguments = {
        "resume": ["resume", session_id],
        "root": ["--resume", session_id],
        "chat": ["chat", "--resume", session_id],
    }[entrypoint]

    # Global model/workspace options must also be inherited by the new command.
    result = run_cli("--mock", "--workspace", str(tmp_path), *arguments, input_text="/exit\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Session: {session_id}" in result.stdout
    assert "历史对话" in result.stdout
    assert "original request" in result.stdout
    assert "演示停止" in result.stdout
    assert "审批：" not in result.stdout
    assert saved_logs(tmp_path) == original
    assert not (tmp_path / "hiro-demo.txt").exists()


@pytest.mark.parametrize("selection", ["1", "z-newer"])
def test_selector_opens_newest_session_by_number_or_id(tmp_path: Path, selection: str) -> None:
    asyncio.run(create_session(tmp_path, "a-older", "older task"))
    asyncio.run(create_session(tmp_path, "z-newer", "newer task"))
    original = saved_logs(tmp_path)

    result = run_cli(
        "resume", "--mock", "-w", str(tmp_path), input_text=f"invalid\n{selection}\n/exit\n"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("z-newer") < result.stdout.index("a-older")
    assert "请输入列表中的序号" in result.stdout
    assert "Session: z-newer" in result.stdout
    assert "Reply to newer task" in result.stdout
    assert saved_logs(tmp_path) == original


@pytest.mark.parametrize("answer", ["\n", "q\n", ""])
def test_canceling_selector_does_not_create_or_change_sessions(tmp_path: Path, answer: str) -> None:
    asyncio.run(create_session(tmp_path, "saved", "saved task"))
    original = saved_logs(tmp_path)

    result = run_cli("resume", "--mock", "-w", str(tmp_path), input_text=answer)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Session:" not in result.stdout
    assert saved_logs(tmp_path) == original


@pytest.mark.parametrize("arguments", [[], ["--last"]])
def test_resume_empty_workspace_does_not_create_a_session(
    tmp_path: Path, arguments: list[str]
) -> None:
    result = run_cli("resume", *arguments, "--mock", "-w", str(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "没有可恢复的历史会话" in result.stdout
    assert not list(tmp_path.iterdir())


def test_last_skips_empty_busy_and_unreadable_sessions(tmp_path: Path) -> None:
    asyncio.run(create_session(tmp_path, "ready", "ready task"))
    busy = asyncio.run(create_session(tmp_path, "busy", "busy task"))
    asyncio.run(create_session(tmp_path, "empty"))
    corrupt = asyncio.run(create_session(tmp_path, "corrupt", "corrupt task"))
    corrupt.write_bytes(corrupt.read_bytes() + b"not json\n")
    original = saved_logs(tmp_path)
    with busy.with_suffix(".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_cli("resume", "--last", "--mock", "-w", str(tmp_path), input_text="/exit\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Session: ready" in result.stdout
    assert "Reply to ready task" in result.stdout
    assert saved_logs(tmp_path) == original


@pytest.mark.parametrize("command", ["/resume target", "/resume\n1", "/resume --last"])
def test_slash_resume_routes_followup_to_selected_session_and_releases_locks(
    tmp_path: Path, command: str
) -> None:
    asyncio.run(create_session(tmp_path, "target", "target task"))
    source = asyncio.run(create_session(tmp_path, "source", "source task"))
    original_source = source.read_bytes()

    result = run_cli(
        "--mock",
        "-w",
        str(tmp_path),
        "--resume",
        "source",
        input_text=f"{command}\nfollowup task\nn\n/exit\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Session: target" in result.stdout
    assert "Reply to target task" in result.stdout
    assert source.read_bytes() == original_source
    events = read_session_events(tmp_path, "target")
    assert [event.data["text"] for event in events if event.type == "task.received"] == [
        "target task",
        "followup task",
    ]
    assert len(saved_logs(tmp_path)) == 2
    assert read_session_events(tmp_path, "source")


@pytest.mark.parametrize("failure", ["missing", "busy", "corrupt"])
def test_failed_slash_resume_preserves_current_session(tmp_path: Path, failure: str) -> None:
    asyncio.run(create_session(tmp_path, "source", "source task"))
    target = None
    lock = None
    try:
        if failure != "missing":
            target = asyncio.run(create_session(tmp_path, "target", "target task"))
            if failure == "corrupt":
                target.write_bytes(target.read_bytes() + b"not json\n")
            else:
                lock = target.with_suffix(".lock").open("r")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_target = target.read_bytes() if target else None
        result = run_cli(
            "--mock",
            "-w",
            str(tmp_path),
            "--resume",
            "source",
            input_text="/resume target\nstill in source\nn\n/exit\n",
        )
    finally:
        if lock is not None:
            lock.close()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Session: target" not in result.stdout
    events = read_session_events(tmp_path, "source")
    assert [event.data["text"] for event in events if event.type == "task.received"] == [
        "source task",
        "still in source",
    ]
    assert len(saved_logs(tmp_path)) == (1 if failure == "missing" else 2)
    if target:
        assert target.read_bytes() == original_target


async def test_resumed_model_request_contains_prior_user_and_assistant_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await create_session(tmp_path, "saved", "remember this task")
    answers = iter(["followup question", "/exit"])
    requests: list[ModelRequest] = []

    async def read(self: cli.Terminal, prompt: str, **kwargs: object) -> str:
        return next(answers)

    async def complete(self: MockModel, request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return ModelResponse(assistant=AssistantMessage(text="new answer"), finish_reason="stop")

    config = Config()
    config.model.adapter = "mock"
    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(MockModel, "complete", complete)

    await cli.chat_loop(tmp_path, config, "saved")

    assert len(requests) == 1
    assert [(message.role, message.content) for message in requests[0].messages][-3:] == [
        ("user", "remember this task"),
        ("assistant", "Reply to remember this task"),
        ("user", "followup question"),
    ]
    [log] = saved_logs(tmp_path).values()
    events = [json.loads(line) for line in log.splitlines()]
    assert sum(event["type"] == "model.requested" for event in events) == 1


def test_text_selector_deletes_confirmed_session_then_resumes_remaining(tmp_path: Path) -> None:
    keep = asyncio.run(create_session(tmp_path, "keep", "keep this task"))
    remove = asyncio.run(create_session(tmp_path, "remove", "remove this task"))
    original = keep.read_bytes()

    result = run_cli("resume", "--mock", "-w", str(tmp_path), input_text="d 1\ny\n1\n/exit\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "已删除会话 remove" in result.stdout
    assert "Session: keep" in result.stdout
    assert not remove.exists()
    assert keep.read_bytes() == original


@pytest.mark.parametrize("answer", ["", "n", "no", "wrong"])
def test_text_selector_requires_explicit_delete_confirmation(tmp_path: Path, answer: str) -> None:
    asyncio.run(create_session(tmp_path, "saved", "preserve this task"))
    original = saved_logs(tmp_path)

    result = run_cli("resume", "--mock", "-w", str(tmp_path), input_text=f"d saved\n{answer}\nq\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "确认删除" in result.stdout
    assert "已删除会话" not in result.stdout
    assert saved_logs(tmp_path) == original


def test_text_selector_can_delete_last_session_without_creating_another(tmp_path: Path) -> None:
    asyncio.run(create_session(tmp_path, "saved", "delete this task"))

    result = run_cli("resume", "--mock", "-w", str(tmp_path), input_text="d saved\nyes\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "没有其他历史会话" in result.stdout
    assert not saved_logs(tmp_path)


async def test_resume_does_not_recreate_concurrently_deleted_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = await create_session(tmp_path, "saved", "deleted task")
    open_session = JsonlSession.open

    async def delete_before_open(cls, workspace, session_id=None, *, create=True):
        delete_session(workspace, session_id)
        return await open_session(workspace, session_id, create=create)

    monkeypatch.setattr(JsonlSession, "open", classmethod(delete_before_open))
    config = Config()
    config.model.adapter = "mock"

    with pytest.raises(HiroError, match="会话不存在"):
        await create_harness(tmp_path, config, "saved")
    assert not path.exists()


@pytest.mark.parametrize("tail", [b"", b'{"version":'])
async def test_resume_initializes_existing_empty_journal_after_repair(
    tmp_path: Path, tail: bytes
) -> None:
    path = await create_session(tmp_path, "empty")
    path.write_bytes(tail)
    config = Config()
    config.model.adapter = "mock"

    harness = await create_harness(tmp_path, config, "empty")
    await harness.close()

    assert [event.type for event in read_session_events(tmp_path, "empty")] == [
        "session.created",
        "prompt.system",
    ]


async def test_keyboard_picker_deletes_history_then_restores_other_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = await create_session(tmp_path, "source", "current task")
    target = await create_session(tmp_path, "target", "resume this task")
    remove = await create_session(tmp_path, "remove", "delete this task")
    originals = {path: path.read_bytes() for path in (source, target)}
    output = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    config = Config()
    config.model.adapter = "mock"
    picker_rendered = asyncio.Event()
    chat_rendered = asyncio.Event()
    picker_class = cli.SessionPicker

    def picker(*args, **kwargs):
        instance = picker_class(*args, **kwargs)
        instance.app.after_render += lambda _: picker_rendered.set()
        return instance

    monkeypatch.setattr(cli, "SessionPicker", picker)
    with create_pipe_input() as pipe:
        terminal = cli.Terminal()
        prompt: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
        terminal._prompt = prompt
        prompt.app.after_render += lambda _: chat_rendered.set()
        monkeypatch.setattr(cli, "Terminal", lambda: terminal)
        task = asyncio.create_task(cli.chat_loop(tmp_path, config, "source"))
        try:
            await asyncio.wait_for(chat_rendered.wait(), 2)
            pipe.send_text("/resume\r")
            await asyncio.wait_for(picker_rendered.wait(), 2)
            chat_rendered.clear()
            # Newest is selected first; delete it, then restore the next row.
            pipe.send_text("\x1b[3~y\r")
            await asyncio.wait_for(chat_rendered.wait(), 2)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            terminal.close()

    assert "Session: target" in output.getvalue()
    assert "Reply to resume this task" in output.getvalue()
    assert not remove.exists()
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert set(saved_logs(tmp_path)) == {"source", "target"}
