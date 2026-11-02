"""Independent failure evidence for the integrated scheduler shutdown contract."""

import asyncio
from pathlib import Path

import pytest

from hiro_agent.config import BotConfig, Config
from hiro_agent.contracts import ModelRequest, ModelResponse
from hiro_agent.errors import PersistenceError
from hiro_agent.kernel_contracts import TaskState, TaskSubmission
from hiro_agent.kernel_runtime import KernelRuntime


def exception_messages(error: BaseException) -> set[str]:
    messages, seen = set(), set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        messages.add(str(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        pending.extend(
            nested for nested in (current.__cause__, current.__context__) if nested is not None
        )
    return messages


async def test_scheduler_failure_survives_a_second_failure_during_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class Model:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            raise AssertionError("unreachable")

        async def close(self) -> None:
            pass

    kernel = KernelRuntime(tmp_path, model_factory=lambda config: Model())
    kernel.bots.save(Config(bot=BotConfig(id="alpha")))
    task = kernel.tasks.submit(TaskSubmission(bot_id="alpha", text="wait during two failures"))
    worker = kernel.worker(poll_interval=0.01)
    stop = asyncio.Event()
    serving = asyncio.create_task(worker.serve(stop))
    original_close = kernel.runner.close

    def scheduler_failure() -> None:
        raise PersistenceError("R3 scheduler primary failure")

    async def shutdown_failure() -> None:
        await original_close()
        raise RuntimeError("R3 runner secondary close failure")

    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        monkeypatch.setattr(kernel.routines, "tick", scheduler_failure)
        monkeypatch.setattr(kernel.runner, "close", shutdown_failure)
        with pytest.raises(BaseException) as raised:
            await asyncio.wait_for(serving, timeout=5)
    finally:
        stop.set()
        await asyncio.gather(worker.close(), serving, return_exceptions=True)
    assert cleaned.is_set() and stop.is_set()
    assert kernel.tasks.get(task.id).state == TaskState.CANCELLED
    monkeypatch.setattr(kernel.runner, "close", original_close)
    replacement = kernel.worker()
    await replacement.close()
    messages = exception_messages(raised.value)
    assert "R3 runner secondary close failure" in messages
    assert "R3 scheduler primary failure" in messages, (
        f"Integrated shutdown lost the durable scheduler error; visible exception tree: {messages}"
    )
