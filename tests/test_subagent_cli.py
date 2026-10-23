"""Exercise the public offline delegation entry point and session resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def cli(*args: str, input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hiro_agent.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )


def journals(workspace: Path) -> list[tuple[Path, list[dict]]]:
    return [
        (path, [json.loads(line) for line in path.read_text().splitlines()])
        for path in (workspace / ".hiro-agent" / "sessions").glob("*.jsonl")
    ]


@pytest.mark.parametrize("subcommand", [[], ["chat"]])
def test_mock_subagents_public_flag_and_child_resume_rejected(tmp_path, subcommand):
    result = cli(
        "--mock",
        "--subagents",
        "--workspace",
        str(tmp_path),
        *subcommand,
        input_text="demo\ny\ny\n/tasks\n/exit\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[completed]" in result.stdout
    assert "委派 explorer" in result.stdout
    assert "委派 reviewer" in result.stdout
    assert "只读探索与审查报告" in result.stdout
    assert (tmp_path / "hiro-demo.txt").read_text() == "Hello from Hiro!\n"
    logs = journals(tmp_path)
    assert len(logs) == 3
    parent = next((p, es) for p, es in logs if not p.stem.startswith("sub-"))
    children = [(p, es) for p, es in logs if p.stem.startswith("sub-")]
    end = next(e for e in parent[1] if e["type"] == "run.ended")
    assert end["data"]["metrics"]["model_requests"] == 9
    assert end["data"]["metrics"]["tool_calls"] == 7
    assert "局部调用:" in result.stdout
    task_query = cli("tasks", "--session", parent[0].stem, "--workspace", str(tmp_path))
    assert task_query.returncode == 0, task_query.stdout + task_query.stderr
    assert "explorer" in task_query.stdout and "reviewer" in task_query.stdout
    assert "局部调用:" in task_query.stdout
    listed = cli("sessions", "--workspace", str(tmp_path))
    assert listed.returncode == 0
    assert "sub-" not in listed.stdout
    child_resume = cli(
        "--mock",
        "--workspace",
        str(tmp_path),
        "--resume",
        children[0][0].stem,
        input_text="/continue\n/exit\n",
    )
    assert child_resume.returncode == 1
    assert "子会话不能直接续跑" in child_resume.stdout
    original = parent[0].read_bytes()
    resume = cli(
        "--mock",
        "--workspace",
        str(tmp_path),
        "--resume",
        parent[0].stem,
        input_text="/exit\n",
    )
    assert resume.returncode == 0, resume.stdout + resume.stderr
    assert parent[0].read_bytes() == original


def test_mock_subagents_resume_uses_new_root_budget_without_repeating_writes(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {"adapter": "mock"},
                "subagents": {"enabled": True},
                "budget": {"max_model_requests": 3},
            }
        )
    )
    args = ["--config", str(config), "--workspace", str(tmp_path)]
    first = cli(*args, input_text="original task\ny\ny\n/exit\n")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "[paused]" in first.stdout
    parent_path = next(p for p, _ in journals(tmp_path) if not p.stem.startswith("sub-"))
    # Continue with a new root budget large enough for the parallel group and summary.
    config.write_text(
        json.dumps(
            {
                "model": {"adapter": "mock"},
                "subagents": {"enabled": True},
                "budget": {"max_model_requests": 6},
            }
        )
    )
    second = cli(*args, "--resume", parent_path.stem, input_text="/continue\n/tasks\n/exit\n")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "[completed]" in second.stdout
    logs = journals(tmp_path)
    assert len(logs) == 3
    events = next(es for p, es in logs if p == parent_path)
    assert len([e for e in events if e["type"] == "task.received"]) == 1
    names = [e["data"]["call"]["name"] for e in events if e["type"] == "tool.started"]
    assert names == ["read_file", "write_file", "run_command", "delegate_task", "delegate_task"]
    ends = [e for e in events if e["type"] == "run.ended"]
    assert [e["data"]["metrics"]["model_requests"] for e in ends] == [3, 6]
