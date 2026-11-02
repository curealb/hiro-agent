"""Process-boundary coverage for C4's memory, skill, routine and worker commands."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hiro_agent.config import BotConfig, Config, ModelConfig

OFFLINE_WORKER = """
from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse
from hiro_agent.kernel_runtime import KernelRuntime
class OfflineModel:
    async def complete(self, request):
        return ModelResponse(
            assistant=AssistantMessage(text="offline success"), finish_reason="stop")
    async def close(self):
        pass
class OfflineKernel(KernelRuntime):
    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, model_factory=lambda config: OfflineModel(), **kwargs)
kernel_cli.KernelRuntime = OfflineKernel
app()
"""


def cli(workspace: Path, *args: str, offline_worker: bool = False) -> Any:
    invocation = (
        [sys.executable, "-c", OFFLINE_WORKER]
        if offline_worker
        else [sys.executable, "-m", "hiro_agent.cli"]
    )
    result = subprocess.run(
        [*invocation, "kernel", *args, "-w", str(workspace)],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def test_cli_memory_crud_explicit_skill_draft_and_routine_worker_flow(tmp_path: Path) -> None:
    config = tmp_path / "bot.json"
    config.write_text(
        Config(bot=BotConfig(id="writer"), model=ModelConfig(adapter="mock")).model_dump_json()
    )
    assert cli(tmp_path, "bot-save", str(config))["bot_id"] == "writer"

    note = cli(tmp_path, "memory-add", "writer", "literal %_ CLI memory")
    assert cli(tmp_path, "memory", "writer", note["id"]) == note
    assert cli(tmp_path, "memories", "writer", "--query", "%_") == [note]
    updated = cli(
        tmp_path,
        "memory-update",
        "writer",
        note["id"],
        "updated content",
        "--expected-revision",
        str(note["revision"]),
    )
    assert updated["revision"] == note["revision"] + 1
    assert cli(tmp_path, "memory", "writer", note["id"])["content"] == "updated content"
    cli(tmp_path, "memory-delete", "writer", note["id"])
    assert cli(tmp_path, "memories", "writer") == []

    submitted = cli(tmp_path, "submit", "writer", "produce a reusable result")
    cli(
        tmp_path,
        "worker",
        "--once",
        "--context-max-chars",
        "12000",
        "--context-max-messages",
        "12",
        offline_worker=True,
    )
    assert cli(tmp_path, "task", submitted["id"])["task"]["state"] == "succeeded"
    assert cli(tmp_path, "skills", "writer") == []
    summaries = cli(tmp_path, "memories", "writer")
    assert len(summaries) == 1 and summaries[0]["task_id"] == submitted["id"]

    draft = cli(tmp_path, "skill-draft", "writer", submitted["id"])
    assert cli(tmp_path, "skill-draft", "writer", submitted["id"]) == draft
    skill_id = draft["document"]["id"]
    assert Path(draft["document"]["path"]).is_file()
    assert cli(tmp_path, "skills", "writer", "--enabled-only") == []
    assert cli(tmp_path, "skill-enable", "writer", skill_id)["enabled"] is True
    assert cli(tmp_path, "skill", "writer", skill_id)["enabled"] is True
    skill = cli(tmp_path, "skill", "writer", skill_id, "--include-content")
    assert skill["id"] == skill_id and "offline success" in skill["content"]
    assert len(cli(tmp_path, "skills", "writer", "--enabled-only")) == 1
    assert cli(tmp_path, "skill-disable", "writer", skill_id)["enabled"] is False
    assert cli(tmp_path, "skills", "writer", "--enabled-only") == []

    scheduled = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    routine = cli(
        tmp_path, "routine-add", "writer", "scheduled from CLI", "--at", scheduled, "--disabled"
    )
    inspected = cli(tmp_path, "routine", routine["id"])
    assert inspected["routine"]["id"] == routine["id"] and inspected["triggers"] == []
    assert cli(tmp_path, "routines", "--bot-id", "writer") == [routine]
    assert cli(tmp_path, "routine-tick", "--max-triggers", "1") == []
    assert cli(tmp_path, "routine-enable", routine["id"], "--bot-id", "writer")["enabled"] is True
    assert cli(tmp_path, "routine-disable", routine["id"], "--bot-id", "writer")["enabled"] is False
    cli(tmp_path, "routine-enable", routine["id"])
    triggers = cli(tmp_path, "routine-tick", "--max-triggers", "1")
    assert len(triggers) == 1 and triggers[0]["routine_id"] == routine["id"]
    scheduled_task_id = triggers[0]["task_id"]
    assert cli(tmp_path, "routine-tick") == []
    cli(tmp_path, "worker", "--once", offline_worker=True)
    task = cli(tmp_path, "task", scheduled_task_id)["task"]
    assert task["state"] == "succeeded" and task["source"] == "routine"
    assert len(cli(tmp_path, "tasks", "--bot-id", "writer")) == 2
