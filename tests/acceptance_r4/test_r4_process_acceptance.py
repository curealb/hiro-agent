"""Independent R4: real subprocess CLI, worker, Agent, SQLite and raw JSONL."""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, BudgetConfig, Config, ModelConfig
from hiro_agent.session import read_session_events

DRIVER = Path(__file__).with_name("process_driver.py")


def invoke(workspace, *arguments, cut="", mode="pipeline", expected=0):
    command = [sys.executable, str(DRIVER), "kernel", *arguments, "-w", str(workspace)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=25,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "R4_WORKSPACE": str(workspace),
            "R4_CUT": cut,
            "R4_MODE": mode,
        },
    )
    with (workspace / "r4-process-audit.jsonl").open("a") as audit:
        audit.write(
            json.dumps(
                {
                    "command": command,
                    "cut": cut,
                    "mode": mode,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            + "\n"
        )
    assert result.returncode == expected, result.stdout + result.stderr
    return json.loads(result.stdout) if expected == 0 else result.stderr


def setup(workspace, *, budget=None):
    for bot_id, tools in (
        ("a", ("read_file", "write_file", "send_handoff", "read_handoff")),
        ("b", ("read_file", "read_handoff", "send_handoff")),
        ("outsider", ("read_handoff",)),
    ):
        config = workspace / f"{bot_id}.json"
        config.write_text(
            Config(
                bot=BotConfig(id=bot_id, allowed_tools=tools),
                model=ModelConfig(adapter="mock"),
                budget=budget or BudgetConfig(max_model_requests=100, max_tool_calls=100),
            ).model_dump_json()
        )
        invoke(workspace, "bot-save", str(config))
    (workspace / "evidence.txt").write_text("R4 VERIFIED inspection input\n")
    (workspace / "report.txt").write_text("R4 VERIFIED\n")


def worker(workspace, **kwargs):
    return invoke(workspace, "worker", "--once", **kwargs)


def task(workspace, task_id):
    return invoke(workspace, "task", task_id)["task"]


def events(workspace, record, kind):
    return [e for e in read_session_events(workspace, record["session_id"]) if e.type == kind]


def assert_accounted(workspace, source):
    members = [t for t in invoke(workspace, "tasks") if t["chain_id"] == source["chain_id"]]
    chain = invoke(workspace, "chain", source["id"])
    for name, total in chain["cumulative_metrics"].items():
        assert total == sum(t["cumulative_metrics"][name] for t in members)
    return chain, members


def test_full_routine_approval_process_restart_review_and_consumed_return(tmp_path):
    setup(tmp_path)
    (tmp_path / "report.txt").unlink()
    invoke(tmp_path, "memory-add", "a", "R4-A-PRIVATE")
    invoke(tmp_path, "memory-add", "b", "R4-B-PRIVATE")
    routine = invoke(
        tmp_path,
        "routine-add",
        "a",
        "R4 inspect",
        "--at",
        (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    worker(tmp_path)
    routine = invoke(tmp_path, "routine", routine["id"])
    source = task(tmp_path, routine["routine"]["last_task_id"])
    assert source["source"] == "routine" and source["state"] == "waiting_approval"
    assert not (tmp_path / "report.txt").exists()
    original_metrics = source["cumulative_metrics"]
    raw = read_session_events(tmp_path, source["session_id"])
    report = next(e for e in raw if "R4 inspection report:" in json.dumps(e.data))
    waiting = next(e for e in raw if e.type == "interaction.waiting")
    assert report.seq < waiting.seq
    time.sleep(0.1)
    worker(tmp_path)
    assert task(tmp_path, source["id"])["cumulative_metrics"] == original_metrics
    interaction_id = source["interaction_id"]
    approval = invoke(tmp_path, "reply", interaction_id, "--approve")
    assert invoke(tmp_path, "reply", interaction_id, "--approve") == approval
    worker(tmp_path)
    assert (tmp_path / "report.txt").read_text() == "R4 VERIFIED\n"
    source = task(tmp_path, source["id"])
    assert source["state"] == "succeeded"
    [message] = invoke(tmp_path, "handoffs", "a", "--outgoing")
    assert message["source_task_id"] == source["id"]
    assert (
        invoke(tmp_path, "handoff-send", "a", source["id"], "b", "R4 review", "review") == message
    )
    worker(tmp_path)
    result = invoke(tmp_path, "handoff-result", "a", message["id"])
    returned = result["result"]
    assert returned["output"] == "R4 reviewer VERIFIED report"
    assert returned["source_task_id"] == source["id"]
    assert returned["target_task_id"] == message["target_task_id"]
    delivery = result["delivery_task"]
    assert delivery["source"] == "handoff_result" and delivery["bot_id"] == "a"
    assert delivery["parent_task_id"] == returned["target_task_id"]
    assert delivery["metadata"] == {
        "message_id": message["id"],
        "source_task_id": source["id"],
        "target_task_id": returned["target_task_id"],
    }
    worker(tmp_path)
    delivery = task(tmp_path, delivery["id"])
    assert delivery["state"] == "succeeded"
    assert "R4 received and processed:" in delivery["output"]
    assert returned["output"] in delivery["output"]
    assert source["id"] in delivery["output"] and message["target_task_id"] in delivery["output"]
    review = task(tmp_path, returned["target_task_id"])
    for record, private, forbidden in (
        (source, "R4-A-PRIVATE", "R4-B-PRIVATE"),
        (review, "R4-B-PRIVATE", "R4-A-PRIVATE"),
    ):
        requests = str([e.data for e in events(tmp_path, record, "model.requested")])
        assert private in requests and forbidden not in requests
    for _ in range(2):
        invoke(tmp_path, "handoff-sync")
        worker(tmp_path)
    invoke(tmp_path, "reply", interaction_id, "--approve")
    worker(tmp_path)
    chain, members = assert_accounted(tmp_path, source)
    assert len(members) == 3 and chain["hops_used"] == 1
    assert delivery["cumulative_metrics"]["model_requests"] == 2
    assert delivery["cumulative_metrics"]["tool_calls"] == 1
    assert len(invoke(tmp_path, "task", source["id"])["runs"]) == 2
    writes = [
        e
        for t in members
        for e in events(tmp_path, t, "tool.started")
        if e.data["call"]["name"] == "write_file"
    ]
    assert len(writes) == 1
    attempts = [
        json.loads(line) for line in (tmp_path / "r4-model-audit.jsonl").read_text().splitlines()
    ]
    assert len({row["pid"] for row in attempts}) == 4
    assert len(invoke(tmp_path, "handoffs", "a", "--outgoing")) == 1
    assert invoke(tmp_path, "handoffs", "b", "--outgoing") == []


@pytest.mark.parametrize(
    "cut,code",
    [
        ("handoff-before", 81),
        ("handoff-after", 82),
        ("terminal", 83),
        ("handoff_result-before", 81),
        ("handoff_result-after", 82),
    ],
)
def test_process_crash_in_handoff_and_result_delivery_gaps(tmp_path, cut, code):
    setup(tmp_path)
    source = invoke(tmp_path, "submit", "a", "R4 finish")
    worker(tmp_path)
    message = invoke(tmp_path, "handoff-send", "a", source["id"], "b", "R4 review", "cut")
    if cut.startswith("handoff-"):
        invoke(tmp_path, "handoff-sync", cut=cut, expected=code)
    else:
        invoke(tmp_path, "handoff-sync")
        worker(tmp_path, cut=cut, expected=code)
    for _ in range(3):
        invoke(tmp_path, "handoff-sync")
        worker(tmp_path)
    result = invoke(tmp_path, "handoff-result", "a", message["id"])
    assert result["result"]["output"] == "R4 reviewer VERIFIED report"
    assert result["delivery_task"]["state"] == "succeeded"
    chain, members = assert_accounted(tmp_path, source)
    assert len(members) == 3 and chain["hops_used"] == 1
    assert [t["source"] for t in members].count("handoff") == 1
    assert [t["source"] for t in members].count("handoff_result") == 1
    assert all(len(invoke(tmp_path, "task", t["id"])["runs"]) == 1 for t in members)


def test_accepted_message_survives_unknown_sender_without_replay(tmp_path):
    setup(tmp_path)
    source = invoke(tmp_path, "submit", "a", "R4 send")
    worker(tmp_path, cut="accepted", expected=84)
    for _ in range(3):
        worker(tmp_path)
    assert task(tmp_path, source["id"])["state"] == "needs_reconciliation"
    [message] = invoke(tmp_path, "handoffs", "a", "--outgoing")
    result = invoke(tmp_path, "handoff-result", "a", message["id"])
    assert result["delivery_task"]["state"] == "succeeded"
    assert len(invoke(tmp_path, "task", source["id"])["runs"]) == 1
    assert len(events(tmp_path, source, "tool.started")) == 1
    assert len(invoke(tmp_path, "tasks")) == 3


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_terminal_result_and_recipient_permissions_are_observable(tmp_path, outcome):
    setup(tmp_path)
    source = invoke(tmp_path, "submit", "a", "R4 finish")
    worker(tmp_path)
    message = invoke(tmp_path, "handoff-send", "a", source["id"], "b", "R4 forbidden", "bad")
    invoke(tmp_path, "handoff-sync")
    [message] = invoke(tmp_path, "handoffs", "a", "--outgoing")
    if outcome == "cancelled":
        invoke(tmp_path, "cancel", message["target_task_id"])
    worker(tmp_path)
    worker(tmp_path)
    result = invoke(tmp_path, "handoff-result", "a", message["id"])
    assert result["result"]["state"] == outcome and result["result"]["reason"]
    assert result["delivery_task"]["state"] == "succeeded"
    assert outcome in result["delivery_task"]["output"]
    assert not (tmp_path / "unauthorized.txt").exists()
    assert invoke(tmp_path, "interactions") == []
    invoke(tmp_path, "handoff-result", "outsider", message["id"], expected=1)
    invoke(tmp_path, "handoff-result", "b", message["id"], expected=1)
    error = invoke(
        tmp_path, "handoff-send", "a", source["id"], "missing", "bad", "unknown", expected=1
    )
    assert "Bot" in error
    assert invoke(tmp_path, "chain", source["id"])["hops_used"] == 1


@pytest.mark.parametrize(
    "limit,arguments",
    [
        ("models", ("--chain-model-requests", "4")),
        ("tools", ("--chain-tool-calls", "2")),
        ("active", ("--chain-timeout-seconds", "0.5")),
    ],
)
def test_delivery_obeys_exhausted_shared_budget_but_preserves_inbox(tmp_path, limit, arguments):
    setup(tmp_path)
    source = invoke(
        tmp_path, "submit", "a", "R4 send slow" if limit == "active" else "R4 send", *arguments
    )
    for _ in range(4):
        worker(tmp_path)
    [message] = invoke(tmp_path, "handoffs", "a", "--outgoing")
    result = invoke(tmp_path, "handoff-result", "a", message["id"])
    assert result["result"] is not None
    assert result["delivery_task"]["state"] == "failed"
    assert "budget" in result["delivery_task"]["reason"]
    assert invoke(tmp_path, "task", result["delivery_task"]["id"])["runs"] == []
    chain, members = assert_accounted(tmp_path, source)
    assert len(members) == 3 and chain["hops_used"] == 1
    if limit == "models":
        assert chain["cumulative_metrics"]["model_requests"] == 4
    elif limit == "tools":
        assert chain["cumulative_metrics"]["tool_calls"] == 2
    else:
        assert 500 <= chain["cumulative_metrics"]["elapsed_ms"] < 1000
        assert result["result"]["state"] == "failed"
    before = chain["cumulative_metrics"]
    worker(tmp_path)
    assert invoke(tmp_path, "chain", source["id"])["cumulative_metrics"] == before


def test_explicit_cross_bot_chat_stops_at_shared_hop_limit_after_every_restart(tmp_path):
    setup(tmp_path)
    source = invoke(tmp_path, "submit", "a", "R4 explicit chat", "--chain-max-hops", "2")
    for _ in range(8):
        worker(tmp_path, mode="chat")
    chain, members = assert_accounted(tmp_path, source)
    assert chain["hops_used"] == 2 and len(members) == 5
    assert all(t["state"] in {"succeeded", "failed"} for t in members)
    assert {t["bot_id"] for t in members} == {"a", "b"}
    assert any("hop limit" in t["reason"] for t in members)
    before = chain["cumulative_metrics"]
    worker(tmp_path, mode="chat")
    assert invoke(tmp_path, "chain", source["id"])["cumulative_metrics"] == before
