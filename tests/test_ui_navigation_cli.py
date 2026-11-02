"""Default routing and interface navigation with real Bot and session stores."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hiro_agent import cli
from hiro_agent.bot_tui import BotTUI
from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.session import JsonlSession
from hiro_agent.tui_backend import TuiBackend


def terminal_streams(monkeypatch: pytest.MonkeyPatch, stdin: bool, stdout: bool) -> None:
    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            platform="linux",
            stdin=SimpleNamespace(isatty=lambda: stdin),
            stdout=SimpleNamespace(isatty=lambda: stdout),
        ),
    )


@pytest.fixture
def interfaces(monkeypatch: pytest.MonkeyPatch) -> list[BotTUI]:
    opened: list[BotTUI] = []
    monkeypatch.delenv("HIRO_MODEL", raising=False)
    monkeypatch.delenv("HIRO_BASE_URL", raising=False)

    async def capture(interface: BotTUI) -> None:
        opened.append(interface)

    monkeypatch.setattr(BotTUI, "run", capture)
    terminal_streams(monkeypatch, True, True)
    return opened


def journals(workspace: Path) -> dict[str, list[dict]]:
    return {
        path.stem: [json.loads(line) for line in path.read_text().splitlines()]
        for path in (workspace / ".hiro-agent" / "sessions").glob("*.jsonl")
    }


def assert_unlocked(workspace: Path) -> None:
    for path in (workspace / ".hiro-agent" / "sessions").glob("*.lock"):
        with path.open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def test_root_tty_bootstraps_default_with_model_environment_and_subagents(
    tmp_path: Path, interfaces: list[BotTUI], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIRO_MODEL", "configured-model")
    monkeypatch.setenv("HIRO_BASE_URL", "https://provider.example/v1")
    result = CliRunner().invoke(
        cli.app, ["--workspace", str(tmp_path), "--subagents", "--no-start-worker"]
    )
    assert result.exit_code == 0, result.output
    [interface] = interfaces
    assert interface.selected_bot_id == "default"
    assert not interface.auto_start_worker
    [record] = KernelRuntime(tmp_path).bots.list()
    assert record.bot_id == "default"
    assert record.config.model.model == "configured-model"
    assert record.config.model.base_url == "https://provider.example/v1"
    assert record.config.subagents.enabled
    assert record.config.permissions == "ask-write"
    assert record.config.bot is not None
    assert {"write_file", "edit_file", "run_command", "delegate_task"}.issubset(
        record.config.bot.allowed_tools
    )
    assert not journals(tmp_path)


@pytest.mark.parametrize("entrypoint", [[], ["tui"]])
def test_single_legacy_config_bootstraps_default_without_losing_settings(
    tmp_path: Path, interfaces: list[BotTUI], entrypoint: list[str]
) -> None:
    config = Config(permissions="read-only", model=ModelConfig(adapter="mock", model="legacy"))
    config.subagents.enabled = True
    config.budget.max_model_requests = 7
    path = tmp_path / "legacy.json"
    path.write_text(config.model_dump_json())
    result = CliRunner().invoke(
        cli.app,
        [*entrypoint, "--workspace", str(tmp_path), "--config", str(path), "--no-start-worker"],
    )
    assert result.exit_code == 0, result.output
    [record] = KernelRuntime(tmp_path).bots.list()
    assert record.bot_id == "default"
    assert record.config.permissions == "read-only"
    assert record.config.subagents.enabled
    assert record.config.budget == config.budget
    assert record.config.model == config.model
    assert len(interfaces) == 1


def test_root_existing_bots_keep_saved_configuration_and_authority(
    tmp_path: Path, interfaces: list[BotTUI], monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = KernelRuntime(tmp_path)
    original = kernel.bots.save(
        Config(
            bot=BotConfig(id="restricted", name="Restricted", allowed_tools=("read_file",)),
            model=ModelConfig(adapter="mock", model="saved-model"),
            permissions="read-only",
        )
    )
    monkeypatch.setenv("HIRO_MODEL", "must-not-overwrite")
    monkeypatch.setenv("HIRO_BASE_URL", "https://must-not-overwrite.example/v1")
    result = CliRunner().invoke(
        cli.app, ["--workspace", str(tmp_path), "--subagents", "--no-start-worker"]
    )
    assert result.exit_code == 0, result.output
    assert kernel.bots.list() == [original]
    assert interfaces[0].selected_bot_id == "restricted"


def test_root_mock_keeps_two_demo_bots(tmp_path: Path, interfaces: list[BotTUI]) -> None:
    result = CliRunner().invoke(
        cli.app, ["--workspace", str(tmp_path), "--mock", "--no-start-worker"]
    )
    assert result.exit_code == 0, result.output
    records = KernelRuntime(tmp_path).bots.list()
    assert [record.bot_id for record in records] == ["demo-a", "demo-b"]
    assert all(record.config.model.adapter == "mock" for record in records)
    assert len(interfaces) == 1


def test_multiple_legacy_configs_require_explicit_bot_ids(
    tmp_path: Path, interfaces: list[BotTUI]
) -> None:
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path in paths:
        path.write_text(Config().model_dump_json())
    result = CliRunner().invoke(
        cli.app,
        [
            "tui",
            "--workspace",
            str(tmp_path),
            "--config",
            str(paths[0]),
            "--config",
            str(paths[1]),
            "--no-start-worker",
        ],
    )
    assert result.exit_code != 0
    assert "bot.id" in result.output
    assert not interfaces


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, False), (False, True), (True, False)])
def test_root_noninteractive_streams_keep_traditional_chat(
    tmp_path: Path,
    interfaces: list[BotTUI],
    monkeypatch: pytest.MonkeyPatch,
    stdin_tty: bool,
    stdout_tty: bool,
) -> None:
    terminal_streams(monkeypatch, stdin_tty, stdout_tty)

    async def read(*args, **kwargs) -> str:
        return "/exit"

    monkeypatch.setattr(cli.Terminal, "read", read)
    result = CliRunner().invoke(cli.app, ["--workspace", str(tmp_path), "--mock"])
    assert result.exit_code == 0, result.output
    assert not interfaces
    assert len(journals(tmp_path)) == 1
    assert not (tmp_path / ".hiro-agent" / "kernel.sqlite3").exists()


@pytest.mark.parametrize("entrypoint", ["chat", "root-resume", "resume"])
def test_explicit_chat_and_resume_stay_traditional_in_tty(
    tmp_path: Path,
    interfaces: list[BotTUI],
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    async def seed() -> None:
        session = await JsonlSession.open(tmp_path, "saved")
        await session.append("prompt.system", {"text": "saved system prompt"})
        await session.close()

    asyncio.run(seed())

    async def read(*args, **kwargs) -> str:
        return "/exit"

    monkeypatch.setattr(cli.Terminal, "read", read)
    arguments = {
        "chat": ["chat"],
        "root-resume": ["--resume", "saved"],
        "resume": ["resume", "saved"],
    }[entrypoint]
    result = CliRunner().invoke(cli.app, ["--workspace", str(tmp_path), "--mock", *arguments])
    assert result.exit_code == 0, result.output
    assert not interfaces
    assert len(journals(tmp_path)) == (2 if entrypoint == "chat" else 1)
    assert not (tmp_path / ".hiro-agent" / "kernel.sqlite3").exists()


def test_roundtrip_reuses_bot_sessions_and_tui_without_replaying_worker_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_streams(monkeypatch, True, True)
    backend = TuiBackend(tmp_path)
    for bot_id, name in (
        ("alpha", "Alpha Agent"),
        ("beta", "Beta Agent"),
        ("collision", "beta"),
        ("duplicate-a", "Shared"),
        ("duplicate-b", "Shared"),
    ):
        backend.kernel.bots.save(
            Config(bot=BotConfig(id=bot_id, name=name), model=ModelConfig(adapter="mock"))
        )
    worker_task = backend.submit("alpha", "keep this durable task queued")
    original_records = backend.kernel.bots.list()
    opened: list[tuple[str, str]] = []
    ui_instances: list[BotTUI] = []
    create_harness = cli.create_harness

    async def capture_harness(workspace, config, *args, **kwargs):
        # Every navigation transition must release the previous journal first.
        assert_unlocked(tmp_path)
        harness = await create_harness(workspace, config, *args, **kwargs)
        assert config.bot is not None
        opened.append((config.bot.id, harness.session.session_id))
        return harness

    monkeypatch.setattr(cli, "create_harness", capture_harness)
    commands = iter(
        [
            ("alpha", "/chat"),
            ("alpha", "/agent"),
            ("alpha", "/agent missing"),
            ("alpha", "/agent Shared"),
            ("alpha", "/agent beta"),
            ("beta", "/agent Alpha Agent"),
            ("alpha", "/tui"),
            ("beta", "/tui"),
        ]
    )

    async def read(*args, **kwargs) -> str:
        bot_id, command = next(commands)
        assert opened[-1][0] == bot_id
        return command

    async def run(interface: BotTUI) -> None:
        assert_unlocked(tmp_path)
        ui_instances.append(interface)
        assert interface is ui_instances[0]
        if len(ui_instances) == 1:
            interface.select_bot("alpha")
            interface.drafts["beta"] = "unsent beta draft"
            interface.exit_action = "chat"
        elif len(ui_instances) == 2:
            assert interface.selected_bot_id == "alpha"
            assert interface.drafts["beta"] == "unsent beta draft"
            interface.select_bot("beta")
            assert interface.composer.text == "unsent beta draft"
            interface.exit_action = "chat"
        else:
            assert len(ui_instances) == 3
            assert interface.selected_bot_id == "beta"
            assert interface.drafts["beta"] == "unsent beta draft"
            interface.exit_action = None

    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(BotTUI, "run", run)
    result = CliRunner().invoke(cli.app, ["--workspace", str(tmp_path), "--no-start-worker"])
    assert result.exit_code == 0, result.output
    assert "找不到 Agent" in result.output
    assert "名称不唯一" in result.output
    assert "Alpha Agent" in result.output and "Beta Agent" in result.output
    assert [bot_id for bot_id, _ in opened] == ["alpha", "beta", "alpha", "beta"]
    assert opened[0] == opened[2]
    assert opened[1] == opened[3]
    assert len(ui_instances) == 3
    saved = journals(tmp_path)
    assert len(saved) == 2
    assert worker_task.session_id not in saved
    assert all(
        event["type"] not in {"task.received", "run.started"}
        for events in saved.values()
        for event in events
    )
    assert backend.kernel.tasks.list() == [worker_task]
    assert backend.kernel.bots.list() == original_records
    assert_unlocked(tmp_path)


def test_legacy_readonly_chat_roundtrip_keeps_identity_when_default_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_streams(monkeypatch, True, True)
    monkeypatch.delenv("HIRO_MODEL", raising=False)
    monkeypatch.delenv("HIRO_BASE_URL", raising=False)
    kernel = KernelRuntime(tmp_path)
    original = kernel.bots.save(
        Config(
            bot=BotConfig(
                id="default",
                name="Existing writer",
                instructions="Private instructions belonging only to the existing writer.",
                allowed_tools=("read_file", "write_file", "run_command"),
            ),
            model=ModelConfig(adapter="mock", model="existing-writer"),
            permissions="ask-write",
        )
    )
    legacy = Config(
        permissions="read-only", model=ModelConfig(adapter="mock", model="legacy-reader")
    )
    legacy.subagents.enabled = True
    legacy.budget.max_model_requests = 7
    legacy.tools.max_search_files = 12
    path = tmp_path / "legacy-reader.json"
    path.write_text(legacy.model_dump_json())
    opened: list[tuple[Config, str]] = []
    selected: list[str] = []
    create_harness = cli.create_harness

    async def capture_harness(workspace, config, *args, **kwargs):
        assert_unlocked(tmp_path)
        harness = await create_harness(workspace, config, *args, **kwargs)
        opened.append((config.model_copy(deep=True), harness.session.session_id))
        return harness

    commands = iter(["/tui", "/exit"])

    async def read(*args, **kwargs) -> str:
        return next(commands)

    async def run(interface: BotTUI) -> None:
        assert_unlocked(tmp_path)
        bot_id = interface.selected_bot_id
        assert bot_id is not None and bot_id != "default"
        selected.append(bot_id)
        stored = interface.backend.kernel.bots.get(bot_id)
        assert stored.config.model_dump(exclude={"bot"}) == legacy.model_dump(exclude={"bot"})
        assert stored.config.bot is not None and stored.config.bot.instructions == ""
        interface.exit_action = "chat"

    monkeypatch.setattr(cli, "create_harness", capture_harness)
    monkeypatch.setattr(cli.Terminal, "read", read)
    monkeypatch.setattr(BotTUI, "run", run)
    result = CliRunner().invoke(
        cli.app,
        [
            "--no-start-worker",
            "chat",
            "--workspace",
            str(tmp_path),
            "--config",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(opened) == 2
    assert opened[0][1] == opened[1][1]
    assert opened[0][0].bot is None
    assert opened[1][0].bot is not None and opened[1][0].bot.id == selected[0]
    assert opened[1][0].model_dump(exclude={"bot"}) == legacy.model_dump(exclude={"bot"})
    assert kernel.bots.get("default") == original
    assert len(kernel.bots.list()) == 2
    [events] = journals(tmp_path).values()
    assert not any(event["type"] in {"task.received", "run.started"} for event in events)
    assert all("existing writer" not in json.dumps(event) for event in events)
    assert_unlocked(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="TUI supports Linux terminals")
def test_real_pty_root_chat_tui_roundtrip_exits_without_model_task(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("The real root entrypoint probe requires uv")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
    process: subprocess.Popen[bytes] | None = None
    transcript = bytearray()
    try:
        process = subprocess.Popen(
            [
                uv,
                "run",
                "--offline",
                "--no-sync",
                "hiro-agent",
                "--workspace",
                str(tmp_path),
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
                "HIRO_MODEL": "pty-model",
                "HIRO_API_KEY": "unused-no-model-task",
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

        wait_until(lambda: "Enter 发送".encode() in transcript, "default TUI composer")
        os.write(master, b"/chat\r")
        wait_until(
            lambda: "输入任务，或输入 / 选择命令".encode() in transcript,
            "traditional chat composer",
        )
        transcript.clear()
        os.write(master, b"/tui\r")
        wait_until(lambda: "Enter 发送".encode() in transcript, "restored TUI composer")
        os.write(master, b"/exit\r")
        assert process.wait(timeout=5) == 0, transcript.decode(errors="replace")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        os.close(master)
        if slave != -1:
            os.close(slave)

    kernel = KernelRuntime(tmp_path)
    assert [record.bot_id for record in kernel.bots.list()] == ["default"]
    assert kernel.tasks.list() == []
    [events] = journals(tmp_path).values()
    assert not any(event["type"] in {"task.received", "run.started"} for event in events)
    assert not (tmp_path / ".hiro-agent" / "worker.lock").exists()
    assert_unlocked(tmp_path)
