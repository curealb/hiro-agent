"""CLI configuration and a real PTY flow for persistent, separate Bot chats."""

from __future__ import annotations

import fcntl
import json
import os
import pty
import select
import sqlite3
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.text import Text
from typer.testing import CliRunner

from hiro_agent import cli
from hiro_agent.bot_tui import BotTUI
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.kernel_contracts import TaskState
from hiro_agent.kernel_runtime import KernelRuntime


@pytest.fixture
def interfaces(monkeypatch: pytest.MonkeyPatch) -> list[BotTUI]:
    opened: list[BotTUI] = []

    async def capture(interface: BotTUI) -> None:
        opened.append(interface)

    monkeypatch.setattr(BotTUI, "run", capture)
    # Only the command's terminal guard is replaced. Click keeps its own capture
    # streams, while actual Bot configuration and refresh still run normally.
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            platform="linux",
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    return opened


def save_config(directory: Path, bot_id: str) -> Path:
    path = directory / f"{bot_id}.json"
    path.write_text(
        Config(
            bot=BotConfig(id=bot_id, name=f"Name {bot_id}"),
            model=ModelConfig(adapter="mock", model=f"model-{bot_id}"),
        ).model_dump_json()
    )
    return path


@pytest.mark.parametrize("color", [False, True])
def test_tui_help_is_available_without_terminal_or_database(tmp_path: Path, color: bool) -> None:
    runner = CliRunner()
    root_help = runner.invoke(cli.app, ["--help"], terminal_width=160, color=color)
    assert root_help.exit_code == 0 and "tui" in Text.from_ansi(root_help.output).plain
    help_result = runner.invoke(
        cli.app, ["--workspace", str(tmp_path), "tui", "--help"], terminal_width=160, color=color
    )
    assert help_result.exit_code == 0, help_result.output
    # Rich can insert ANSI styles between the dashes and option name in CI.
    help_text = Text.from_ansi(help_result.output).plain
    for option in ("--config", "--agent", "--demo", "--no-start-worker"):
        assert option in help_text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, False), (False, True), (True, False)])
def test_noninteractive_tui_rejects_before_database_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdin_tty: bool, stdout_tty: bool
) -> None:
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            platform="linux",
            stdin=SimpleNamespace(isatty=lambda: stdin_tty),
            stdout=SimpleNamespace(isatty=lambda: stdout_tty),
        ),
    )
    result = CliRunner().invoke(cli.app, ["tui", "--workspace", str(tmp_path), "--demo"])
    assert result.exit_code == 2
    assert "交互式终端" in result.output
    assert list(tmp_path.iterdir()) == []


def test_global_workspace_and_config_are_inherited(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    workspace = tmp_path / "inherited-workspace"
    workspace.mkdir()
    config = save_config(tmp_path, "inherited")
    result = CliRunner().invoke(
        cli.app,
        ["--workspace", str(workspace), "--config", str(config), "tui", "--no-start-worker"],
    )
    assert result.exit_code == 0, result.output
    [interface] = interfaces
    assert interface.backend.workspace == workspace
    assert interface.selected_bot_id == "inherited"
    assert interface.auto_start_worker is False
    stored = KernelRuntime(workspace).bots.get("inherited")
    assert stored.config.bot is not None and stored.config.bot.name == "Name inherited"
    assert stored.config.model.model == "model-inherited"
    assert not (tmp_path / ".hiro-agent").exists()
    assert not (workspace / ".hiro-agent" / "worker.lock").exists()


def test_command_workspace_and_configs_override_inherited_values(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    global_workspace = tmp_path / "global"
    command_workspace = tmp_path / "command"
    global_workspace.mkdir()
    command_workspace.mkdir()
    global_config = save_config(tmp_path, "global-bot")
    command_config = save_config(tmp_path, "command-bot")
    result = CliRunner().invoke(
        cli.app,
        [
            "--workspace",
            str(global_workspace),
            "--config",
            str(global_config),
            "tui",
            "--workspace",
            str(command_workspace),
            "--config",
            str(command_config),
            "--no-start-worker",
        ],
    )
    assert result.exit_code == 0, result.output
    [interface] = interfaces
    assert interface.backend.workspace == command_workspace
    assert [bot.bot_id for bot in interface.backend.kernel.bots.list()] == ["command-bot"]
    assert not (global_workspace / ".hiro-agent").exists()


@pytest.mark.parametrize("disable_worker", [False, True])
def test_repeated_configs_persist_and_agent_and_worker_options_are_forwarded(
    tmp_path: Path, interfaces: list[BotTUI], disable_worker: bool
) -> None:
    configs = [save_config(tmp_path, bot_id) for bot_id in ("alpha", "beta")]
    arguments = ["tui", "--workspace", str(tmp_path)]
    for config in configs:
        arguments.extend(["--config", str(config)])
    arguments.extend(["--agent", "beta"])
    if disable_worker:
        arguments.append("--no-start-worker")
    result = CliRunner().invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    [interface] = interfaces
    assert interface.selected_bot_id == "beta"
    assert interface.auto_start_worker is (not disable_worker)
    records = KernelRuntime(tmp_path).bots.list()
    assert [record.bot_id for record in records] == ["alpha", "beta"]
    assert [record.config.model.model for record in records] == ["model-alpha", "model-beta"]
    assert not (tmp_path / ".hiro-agent" / "worker.lock").exists()


def test_unknown_initial_agent_fails_before_opening_interface(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    result = CliRunner().invoke(
        cli.app,
        ["tui", "--workspace", str(tmp_path), "--demo", "--agent", "missing", "--no-start-worker"],
    )
    assert result.exit_code == 1
    assert "Bot 不存在" in result.output
    assert interfaces == []
    assert not (tmp_path / ".hiro-agent" / "worker.lock").exists()


def test_demo_preserves_existing_bot_permissions_and_configuration(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    kernel = KernelRuntime(tmp_path)
    original = kernel.bots.save(
        Config(
            bot=BotConfig(
                id="demo-a", name="Existing restricted Bot", allowed_tools=("read_file",)
            ),
            model=ModelConfig(adapter="compatible", model="preserved-model"),
            permissions="read-only",
        )
    )
    result = CliRunner().invoke(
        cli.app, ["tui", "--workspace", str(tmp_path), "--demo", "--no-start-worker"]
    )
    assert result.exit_code == 0, result.output
    assert len(interfaces) == 1
    assert kernel.bots.get("demo-a") == original
    assert kernel.bots.get("demo-b").config.model.adapter == "mock"
    assert [record.bot_id for record in kernel.bots.list()] == ["demo-a", "demo-b"]
    assert not (tmp_path / ".hiro-agent" / "worker.lock").exists()


def test_inherited_mock_option_creates_offline_demo_bots(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    result = CliRunner().invoke(
        cli.app, ["--workspace", str(tmp_path), "--mock", "tui", "--no-start-worker"]
    )
    assert result.exit_code == 0, result.output
    [interface] = interfaces
    assert [record.bot_id for record in interface.backend.kernel.bots.list()] == [
        "demo-a",
        "demo-b",
    ]


def _saved_tasks(workspace: Path) -> list[dict]:
    path = workspace / ".hiro-agent" / "kernel.sqlite3"
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT record_json FROM durable_tasks ORDER BY created_at, id"
                )
            ]
    except sqlite3.OperationalError:
        # Startup migrations can still be in progress on the first screen.
        return []


@pytest.mark.skipif(sys.platform != "linux", reason="TUI currently supports Linux terminals")
def test_real_pty_cli_switches_bots_and_persists_owned_tasks(tmp_path: Path) -> None:
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
    attributes = termios.tcgetattr(slave)
    attributes[0] &= ~(termios.IXON | termios.IXOFF)
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    process: subprocess.Popen[bytes] | None = None
    transcript = bytearray()
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hiro_agent.cli",
                "tui",
                "--workspace",
                str(tmp_path),
                "--demo",
                "--no-start-worker",
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            start_new_session=True,
            env={
                **os.environ,
                "TERM": "xterm-256color",
                "PROMPT_TOOLKIT_NO_CPR": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        os.close(slave)
        slave = -1

        def wait_until(predicate: Callable[[], bool], description: str) -> None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if predicate():
                    return
                assert process is not None and process.poll() is None, transcript.decode(
                    errors="replace"
                )
                if select.select([master], [], [], 0.05)[0]:
                    transcript.extend(os.read(master, 65536))
            pytest.fail(
                f"Timed out waiting for {description}: {transcript.decode(errors='replace')}"
            )

        wait_until(lambda: "给 Demo A 发消息".encode() in transcript, "initial composer")
        os.write(master, b"message for Agent A\r")
        wait_until(lambda: len(_saved_tasks(tmp_path)) == 1, "first durable task")
        # Alt+Down is an Escape prefix followed by the ordinary Down sequence.
        os.write(master, b"\x1b\x1b[Bmessage for Agent B\r")
        wait_until(lambda: len(_saved_tasks(tmp_path)) == 2, "second durable task")
        os.write(master, b"\x11")
        process.wait(timeout=5)
        assert process.returncode == 0, transcript.decode(errors="replace")
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        os.close(master)
        if slave != -1:
            os.close(slave)

    tasks = KernelRuntime(tmp_path).tasks.list()
    assert [(task.bot_id, task.text) for task in tasks] == [
        ("demo-a", "message for Agent A"),
        ("demo-b", "message for Agent B"),
    ]
    assert all(task.state == TaskState.QUEUED for task in tasks)
    assert len({task.session_id for task in tasks}) == 2
    assert not (tmp_path / ".hiro-agent" / "worker.lock").exists()
    assert not (tmp_path / ".hiro-agent" / "sessions").exists()
