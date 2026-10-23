"""CLI smoke tests and real prompt-toolkit keyboard/signal regressions."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from hiro_agent.cli import Terminal
from hiro_agent.contracts import AgentService
from hiro_agent.errors import RunCancelled


class CancelTarget:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    def cancel(self) -> None:
        self.cancelled.set()


@pytest.mark.parametrize("answer,approved", [("y\n", True), ("n\n", False)])
async def test_approval_preserves_run_sigint_handler(
    monkeypatch: pytest.MonkeyPatch, answer: str, approved: bool
) -> None:
    loop = asyncio.get_running_loop()
    target = CancelTarget()
    previous_handler = signal.getsignal(signal.SIGINT)
    loop.add_signal_handler(signal.SIGINT, target.cancel)
    installed_handler = loop._signal_handlers[signal.SIGINT]
    try:
        with create_pipe_input() as pipe:
            terminal = Terminal()
            terminal.agent = cast(AgentService, target)
            terminal._prompt = PromptSession(input=pipe, output=DummyOutput())
            rendered = asyncio.Event()
            terminal._prompt.app.before_render += lambda _: rendered.set()
            monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
            task = asyncio.create_task(terminal.approve("write_file", {"path": "demo.txt"}))
            await asyncio.wait_for(rendered.wait(), 2)
            pipe.send_text(answer)
            assert await asyncio.wait_for(task, 2) is approved
        # A prompt must leave the event loop's callback installed, not just the
        # low-level Python signal function restored by prompt-toolkit.
        assert loop._signal_handlers.get(signal.SIGINT) is installed_handler
        assert not target.cancelled.is_set()
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(target.cancelled.wait(), 2)
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_handler)


async def test_ctrl_c_during_approval_cancels_run(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()
    target = CancelTarget()
    previous_handler = signal.getsignal(signal.SIGINT)
    loop.add_signal_handler(signal.SIGINT, target.cancel)
    installed_handler = loop._signal_handlers[signal.SIGINT]
    try:
        with create_pipe_input() as pipe:
            terminal = Terminal()
            terminal.agent = cast(AgentService, target)
            terminal._prompt = PromptSession(input=pipe, output=DummyOutput())
            rendered = asyncio.Event()
            terminal._prompt.app.before_render += lambda _: rendered.set()
            monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
            task = asyncio.create_task(terminal.approve("write_file", {"path": "demo.txt"}))
            await asyncio.wait_for(rendered.wait(), 2)
            pipe.send_text("\x03")
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 2)
        assert target.cancelled.is_set()
        assert loop._signal_handlers.get(signal.SIGINT) is installed_handler
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_handler)


def run_cli(*arguments: str, input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hiro_agent.cli", *arguments],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )


def test_cli_help_and_empty_sessions(tmp_path: Path) -> None:
    help_result = run_cli("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--workspace" in help_result.stdout
    assert "chat" in help_result.stdout
    listed = run_cli("sessions", "--workspace", str(tmp_path))
    assert listed.returncode == 0, listed.stderr
    assert "Session" in listed.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("subcommand", [[], ["chat"]])
def test_mock_repl_default_and_chat_paths(tmp_path: Path, subcommand: list[str]) -> None:
    result = run_cli(
        *subcommand,
        "--mock",
        "--workspace",
        str(tmp_path),
        input_text="demo\ny\ny\n/continue\n/exit\n",
    )
    assert result.returncode == 0, result.stderr
    assert "演示完成" in result.stdout
    assert "[completed]" in result.stdout
    assert "没有可继续的未完成任务" in result.stdout
    assert (tmp_path / "hiro-demo.txt").read_text() == "Hello from Hiro!\n"
    logs = list((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
    assert len(logs) == 1
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert sum(event["type"] == "task.received" for event in events) == 1
    assert [event["data"]["status"] for event in events if event["type"] == "run.ended"] == [
        "completed"
    ]
    listed = run_cli("sessions", "--workspace", str(tmp_path))
    assert listed.returncode == 0
    assert "completed" in listed.stdout


def test_mock_denial_returns_to_repl_without_writing(tmp_path: Path) -> None:
    result = run_cli("--mock", "--workspace", str(tmp_path), input_text="demo\nn\n/help\n/exit\n")
    assert result.returncode == 0, result.stderr
    assert "演示停止" in result.stdout
    assert "演示完成" not in result.stdout
    assert result.stdout.count("/help") >= 2
    assert not (tmp_path / "hiro-demo.txt").exists()


def test_mock_eof_at_approval_denies_and_exits(tmp_path: Path) -> None:
    result = run_cli("--mock", "--workspace", str(tmp_path), input_text="demo\n")
    assert result.returncode == 0, result.stderr
    assert "演示停止" in result.stdout
    assert not (tmp_path / "hiro-demo.txt").exists()


def test_resume_paused_demo_continues_without_duplicate_write(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"adapter": "mock"},
                "budget": {"max_model_requests": 2},
            }
        )
    )
    arguments = ["--workspace", str(tmp_path), "--config", str(config_path)]
    first = run_cli(*arguments, input_text="original demo\ny\n/exit\n")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "[paused]" in first.stdout
    assert "演示完成" not in first.stdout
    assert (tmp_path / "hiro-demo.txt").read_text() == "Hello from Hiro!\n"
    logs = list((tmp_path / ".hiro-agent" / "sessions").glob("*.jsonl"))
    assert len(logs) == 1
    resumed = run_cli(*arguments, "--resume", logs[0].stem, input_text="/continue\ny\n/exit\n")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "演示完成" in resumed.stdout
    assert "[completed]" in resumed.stdout
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [event["data"]["status"] for event in events if event["type"] == "run.ended"] == [
        "paused",
        "completed",
    ]
    assert [event["data"]["text"] for event in events if event["type"] == "task.received"] == [
        "original demo"
    ]
    assert sum(event["type"] == "task.continued" for event in events) == 1
    assert [
        event["data"]["call"]["name"] for event in events if event["type"] == "tool.started"
    ] == ["read_file", "write_file", "run_command"]
    assert len({event["run_id"] for event in events if event["type"] == "run.started"}) == 2


def test_missing_resume_rejected_without_creating_journal(tmp_path: Path) -> None:
    result = run_cli(
        "--mock", "--workspace", str(tmp_path), "--resume", "missing-session", input_text="/exit\n"
    )
    assert result.returncode == 1
    assert "会话不存在" in result.stdout
    assert not list(tmp_path.iterdir())


def test_global_options_are_inherited_by_subcommands(tmp_path: Path) -> None:
    result = run_cli(
        "--mock", "--workspace", str(tmp_path), "chat", input_text="demo\ny\ny\n/exit\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "演示完成" in result.stdout
    listed = run_cli("--workspace", str(tmp_path), "sessions")
    assert listed.returncode == 0
    assert "completed" in listed.stdout
