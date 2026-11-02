"""Real CLI subprocesses share SQLite/JSONL while using a deterministic offline model."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.kernel_runtime import KernelRuntime
from hiro_agent.session import read_session_events

OFFLINE_WORKER = """
import json
from uuid import uuid4
from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall
from hiro_agent.kernel_runtime import KernelRuntime
class OfflineModel:
    async def complete(self, request):
        results = [json.loads(m.content) for m in request.messages if m.role == "tool"]
        answers = [m.content for m in request.messages
                   if m.role == "user" and "用户对问题" in (m.content or "")]
        if answers or any(r["ok"] for r in results):
            return ModelResponse(
                assistant=AssistantMessage(text=answers[-1] if answers else "written"),
                finish_reason="stop")
        original = next(m.content for m in request.messages if m.role == "user")
        name = "ask_user" if original == "question" else "write_file"
        args = ({"question": "Which color?"} if name == "ask_user" else
                {"path": "approved.txt", "content": "exact saved contents"})
        call = ToolCall(id=uuid4().hex, name=name, arguments=json.dumps(args))
        return ModelResponse(assistant=AssistantMessage(tool_calls=[call]),
                             finish_reason="tool_calls")
    async def close(self):
        pass
class OfflineKernel(KernelRuntime):
    def __init__(self, workspace):
        super().__init__(workspace, model_factory=lambda config: OfflineModel())
kernel_cli.KernelRuntime = OfflineKernel
app()
"""


def cli(workspace: Path, *args: str) -> dict | list:
    result = subprocess.run(
        [sys.executable, "-m", "hiro_agent.cli", "kernel", *args, "-w", str(workspace)],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def wait_state(kernel: KernelRuntime, task_id: str, state: str, worker: subprocess.Popen) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        task = kernel.tasks.get(task_id)
        if task.state == state:
            return
        assert worker.poll() is None, f"worker exited: {worker.returncode}; Task={task}"
        time.sleep(0.05)
    pytest.fail(f"Task did not reach {state}: {kernel.tasks.get(task_id)}")


def stop(worker: subprocess.Popen) -> None:
    if worker.poll() is None:
        worker.terminate()
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=10)
        pytest.fail("worker did not stop cooperatively")
    assert worker.returncode == 0


@pytest.mark.parametrize("kind", ["approval", "question"])
def test_cli_exit_independent_worker_and_reply_across_restart(tmp_path: Path, kind: str) -> None:
    kernel = KernelRuntime(tmp_path)
    kernel.bots.save(
        Config(
            bot=BotConfig(id="bot", allowed_tools=("write_file", "ask_user")),
            model=ModelConfig(adapter="mock"),
            permissions="ask-write",
        )
    )
    args = [
        sys.executable,
        "-c",
        OFFLINE_WORKER,
        "kernel",
        "worker",
        "-w",
        str(tmp_path),
        "--poll-interval",
        "0.02",
    ]
    with (tmp_path / "worker.log").open("w") as log:
        first = subprocess.Popen(
            args, stdout=log, stderr=log, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        try:
            submitted = cli(tmp_path, "submit", "bot", kind, "--idempotency-key", kind)
            assert isinstance(submitted, dict)
            task_id = submitted["id"]
            # submit CLI has exited, but this separate worker remains alive.
            assert first.poll() is None
            wait_state(
                kernel,
                task_id,
                "waiting_approval" if kind == "approval" else "waiting_input",
                first,
            )
        finally:
            stop(first)
        task = kernel.tasks.get(task_id)
        assert task.interaction_id is not None
        interaction_id = task.interaction_id
        assert not (tmp_path / "approved.txt").exists()
        reply_args = ("--approve",) if kind == "approval" else ("--text", "blue")
        replied = cli(tmp_path, "reply", interaction_id, *reply_args)
        assert cli(tmp_path, "reply", interaction_id, *reply_args) == replied
        # Reply persists while no worker exists; task wake is recovered next startup.
        assert kernel.tasks.get(task_id).state.startswith("waiting_")
        second = subprocess.Popen(
            args, stdout=log, stderr=log, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        try:
            wait_state(kernel, task_id, "succeeded", second)
            cli(tmp_path, "reply", interaction_id, *reply_args)
            time.sleep(0.1)
            assert len(kernel.tasks.runs(task_id)) == 2
            assert second.poll() is None
        finally:
            stop(second)
    final = kernel.tasks.get(task_id)
    events = read_session_events(tmp_path, final.session_id)
    assert final.cumulative_metrics.model_requests == 2
    if kind == "approval":
        assert (tmp_path / "approved.txt").read_text() == "exact saved contents"
        starts = [e for e in events if e.type == "tool.started"]
        assert len(starts) == 1
        assert final.cumulative_metrics.tool_calls == 2
    else:
        assert "blue" in final.output
        assert not any(e.type == "tool.started" for e in events)
    assert kernel.interactions.get(interaction_id).state == "consumed"


def test_cli_bot_config_submission_query_cancel(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        Config(bot=BotConfig(id="saved"), model=ModelConfig(adapter="mock")).model_dump_json()
    )
    assert cli(tmp_path, "bot-save", str(config))["bot_id"] == "saved"
    submission = cli(tmp_path, "submit", "saved", "task", "--idempotency-key", "one")
    assert cli(tmp_path, "submit", "saved", "task", "--idempotency-key", "one") == submission
    assert cli(tmp_path, "task", submission["id"])["task"]["state"] == "queued"
    assert cli(tmp_path, "cancel", submission["id"])["state"] == "cancelled"
