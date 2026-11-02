"""Real process CLI visibility of persistent sending, result delivery and chain limits."""

import json
import os
import subprocess
import sys
from pathlib import Path

from hiro_agent.config import BotConfig, Config, ModelConfig
from hiro_agent.kernel_runtime import KernelRuntime

OFFLINE_WORKER = """
from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse
from hiro_agent.kernel_runtime import KernelRuntime
class Model:
    async def complete(self, request):
        return ModelResponse(assistant=AssistantMessage(text="review done"), finish_reason="stop")
    async def close(self):
        pass
class OfflineKernel(KernelRuntime):
    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, model_factory=lambda config: Model(), **kwargs)
kernel_cli.KernelRuntime = OfflineKernel
app()
"""


def cli(workspace: Path, *args: str, worker: bool = False):
    prefix = (
        [sys.executable, "-c", OFFLINE_WORKER]
        if worker
        else [sys.executable, "-m", "hiro_agent.cli"]
    )
    result = subprocess.run(
        [*prefix, "kernel", *args, "-w", str(workspace)],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def test_cli_handoff_to_review_and_return_survives_each_process_exit(tmp_path: Path) -> None:
    kernel = KernelRuntime(tmp_path)
    for bot_id in ("writer", "reviewer"):
        kernel.bots.save(
            Config(
                bot=BotConfig(id=bot_id, allowed_tools=("send_handoff", "read_handoff")),
                model=ModelConfig(adapter="mock"),
            )
        )
    task = cli(
        tmp_path,
        "submit",
        "writer",
        "inspect",
        "--chain-max-hops",
        "2",
        "--chain-model-requests",
        "5",
        "--chain-tool-calls",
        "8",
        "--chain-timeout-seconds",
        "30",
    )
    cli(tmp_path, "worker", "--once", worker=True)
    message = cli(tmp_path, "handoff-send", "writer", task["id"], "reviewer", "review", "once")
    assert cli(tmp_path, "handoffs", "reviewer") == [message]
    assert cli(tmp_path, "handoffs", "writer", "--outgoing") == [message]
    assert cli(tmp_path, "handoff-result", "writer", message["id"]) == {
        "result": None,
        "delivery_task": None,
    }
    cli(tmp_path, "handoff-sync")
    cli(tmp_path, "handoff-sync")
    cli(tmp_path, "worker", "--once", worker=True)
    returned = cli(tmp_path, "handoff-result", "writer", message["id"])
    assert returned["result"]["source_task_id"] == task["id"]
    assert returned["result"]["state"] == "succeeded"
    assert returned["delivery_task"]["bot_id"] == "writer"
    assert returned["delivery_task"]["state"] == "queued"
    cli(tmp_path, "worker", "--once", worker=True)
    final = cli(tmp_path, "handoff-result", "writer", message["id"])
    assert final["delivery_task"]["state"] == "succeeded"
    assert final["delivery_task"]["chain_id"] == task["chain_id"]
    chain = cli(tmp_path, "chain", task["id"])
    assert chain["max_hops"] == 2 and chain["hops_used"] == 1
    assert chain["budget"]["max_model_requests"] == 5
    assert chain["cumulative_metrics"]["model_requests"] == 3
    assert len(cli(tmp_path, "tasks")) == 3
