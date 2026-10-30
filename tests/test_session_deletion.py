"""Deleting history validates and locks every linked journal before changing files."""

import json
import os
from pathlib import Path

import pytest

from hiro_agent.contracts import AssistantMessage, DelegationLink, ToolCall
from hiro_agent.errors import PersistenceError
from hiro_agent.session import JsonlSession, delete_session, list_sessions


async def create_family(workspace: Path, children: tuple[str, ...] = ("child",)) -> Path:
    parent = await JsonlSession.open(workspace, "parent")
    try:
        await parent.append("run.started", {}, "parent-run")
        await parent.append("task.received", {"text": "Review this project"}, "parent-run")
        calls = [
            ToolCall(id=f"call-{child}", name="delegate_task", arguments="{}") for child in children
        ]
        await parent.append(
            "assistant.message",
            {"message": AssistantMessage(tool_calls=calls).model_dump()},
            "parent-run",
        )
        for child_id, call in zip(children, calls, strict=True):
            await parent.append("tool.called", {"call": call.model_dump()}, "parent-run")
            await parent.append("tool.started", {"call": call.model_dump()}, "parent-run")
            link = DelegationLink(
                parent_session_id="parent",
                parent_run_id="parent-run",
                child_session_id=child_id,
                child_run_id=f"run-{child_id}",
                call_id=call.id,
                role="explorer",
            )
            await parent.append("delegation.started", link.model_dump(), "parent-run")
            child = await JsonlSession.open(workspace, child_id)
            try:
                await child.append("subagent.created", link.model_dump())
            finally:
                await child.close()
        return parent.path.parent
    finally:
        await parent.close()


def journal_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.glob("*.jsonl")}


async def test_deletes_parent_and_children_but_keeps_other_history_and_stable_locks(
    tmp_path: Path,
) -> None:
    directory = await create_family(tmp_path, ("child-a", "child-b"))
    unrelated = await JsonlSession.open(tmp_path, "unrelated")
    await unrelated.append("task.received", {"text": "Keep this conversation"})
    await unrelated.close()
    original_unrelated = unrelated.path.read_bytes()
    lock_inodes = {path.name: path.stat().st_ino for path in directory.glob("*.lock")}
    project_file = tmp_path / "source.py"
    project_file.write_text("preserved = True\n")

    delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == {"unrelated.jsonl": original_unrelated}
    assert {path.name: path.stat().st_ino for path in directory.glob("*.lock")} == lock_inodes
    assert all(path.stat().st_size == 0 for path in directory.glob("*.lock"))
    assert project_file.read_text() == "preserved = True\n"
    assert [item["session_id"] for item in list_sessions(tmp_path)] == ["unrelated"]
    with pytest.raises(FileNotFoundError):
        await JsonlSession.open(tmp_path, "parent", create=False)


@pytest.mark.parametrize("busy_id", ["parent", "child-b"])
async def test_busy_parent_or_child_leaves_every_journal_untouched(
    tmp_path: Path, busy_id: str
) -> None:
    directory = await create_family(tmp_path, ("child-a", "child-b"))
    busy = await JsonlSession.open(tmp_path, busy_id)
    before = journal_bytes(directory)
    try:
        with pytest.raises(PersistenceError, match="正在使用"):
            delete_session(tmp_path, "parent")
        assert journal_bytes(directory) == before
    finally:
        await busy.close()

    delete_session(tmp_path, "parent")
    assert journal_bytes(directory) == {}


async def test_child_cannot_be_deleted_independently(tmp_path: Path) -> None:
    directory = await create_family(tmp_path)
    before = journal_bytes(directory)

    with pytest.raises(PersistenceError, match="子会话不能单独删除"):
        delete_session(tmp_path, "child")

    assert journal_bytes(directory) == before
    delete_session(tmp_path, "parent")
    assert journal_bytes(directory) == {}


@pytest.mark.parametrize("damage", ["partial", "invalid", "lineage", "missing-link"])
async def test_invalid_child_prevents_all_deletions_without_repair(
    tmp_path: Path, damage: str
) -> None:
    directory = await create_family(tmp_path, ("child-a", "child-b"))
    child_path = directory / "child-b.jsonl"
    if damage in {"partial", "invalid"}:
        child_path.write_bytes(
            child_path.read_bytes() + (b'{"seq":' if damage == "partial" else b"invalid\n")
        )
    else:
        events = [json.loads(line) for line in child_path.read_text().splitlines()]
        if damage == "lineage":
            events[1]["data"]["parent_session_id"] = "different-parent"
        else:
            events[1]["type"] = "task.received"
            events[1]["data"] = {"text": "An independent conversation"}
        child_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    before = journal_bytes(directory)

    with pytest.raises(PersistenceError, match="会话删除失败"):
        delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == before
    # All acquired locks must be released even after validation fails.
    parent = await JsonlSession.open(tmp_path, "parent")
    await parent.close()


@pytest.mark.parametrize("lock_remains", [False, True])
async def test_missing_child_from_interrupted_creation_is_tolerated(
    tmp_path: Path, lock_remains: bool
) -> None:
    directory = await create_family(tmp_path)
    (directory / "child.jsonl").unlink()
    if not lock_remains:
        (directory / "child.lock").unlink()

    delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == {}


async def test_child_creation_stub_is_removed_with_parent(tmp_path: Path) -> None:
    directory = await create_family(tmp_path)
    child_path = directory / "child.jsonl"
    child_path.write_bytes(child_path.read_bytes().splitlines(keepends=True)[0])

    delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == {}


@pytest.mark.parametrize("session_id", ["../outside", "", "/absolute", "missing"])
def test_invalid_or_missing_session_never_creates_files(tmp_path: Path, session_id: str) -> None:
    with pytest.raises(PersistenceError, match="会话删除失败"):
        delete_session(tmp_path, session_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["parent.jsonl", "parent.lock", "child.jsonl", "child.lock"])
async def test_symlink_file_is_refused_without_touching_target(tmp_path: Path, name: str) -> None:
    directory = await create_family(tmp_path)
    target = tmp_path / "keep-target"
    original = (directory / name).read_bytes()
    target.write_bytes(original)
    (directory / name).unlink()
    (directory / name).symlink_to(target)
    before = journal_bytes(directory)

    with pytest.raises(PersistenceError, match="会话删除失败"):
        delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == before
    assert target.read_bytes() == original
    assert (directory / name).is_symlink()


@pytest.mark.parametrize("component", [".hiro-agent", "sessions"])
async def test_symlink_directory_is_refused(tmp_path: Path, component: str) -> None:
    directory = await create_family(tmp_path)
    source = tmp_path / ".hiro-agent" if component == ".hiro-agent" else directory
    moved = tmp_path / "keep-directory"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=True)
    before = journal_bytes(directory)

    with pytest.raises(PersistenceError, match="会话删除失败"):
        delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == before
    assert source.is_symlink()


async def test_nonregular_journal_is_rejected_without_blocking(tmp_path: Path) -> None:
    directory = await create_family(tmp_path)
    child_path = directory / "child.jsonl"
    child_path.unlink()
    os.mkfifo(child_path)
    parent = (directory / "parent.jsonl").read_bytes()

    with pytest.raises(PersistenceError, match="不是普通文件"):
        delete_session(tmp_path, "parent")

    assert (directory / "parent.jsonl").read_bytes() == parent


async def test_corrupt_parent_is_not_repaired_or_deleted(tmp_path: Path) -> None:
    directory = await create_family(tmp_path)
    parent = directory / "parent.jsonl"
    parent.write_bytes(parent.read_bytes() + b'{"seq":')
    before = journal_bytes(directory)

    with pytest.raises(PersistenceError, match="日志损坏"):
        delete_session(tmp_path, "parent")

    assert journal_bytes(directory) == before
