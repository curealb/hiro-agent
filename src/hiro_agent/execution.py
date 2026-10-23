"""Cancellation propagates to the operation and waits for its cleanup."""

import asyncio
import inspect
from collections.abc import Awaitable

from hiro_agent.contracts import Cancellation
from hiro_agent.errors import BudgetExceeded, PersistenceError, RunCancelled


async def cancellable[T](operation: Awaitable[T], cancellation: Cancellation, timeout: float) -> T:
    if cancellation.cancelled or timeout <= 0:
        if inspect.iscoroutine(operation):
            operation.close()
        cancellation.check()
        raise BudgetExceeded("运行时间预算耗尽")
    task = asyncio.ensure_future(operation)
    canceled = asyncio.create_task(cancellation.event.wait())
    try:
        done, _ = await asyncio.wait(
            {task, canceled}, timeout=max(0, timeout), return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        # Cleanup may signal a shared cancellation token itself. Preserve the
        # original interruption cause so a deadline does not become a user cancel.
        reason = (
            cancellation.reason if cancellation.cancelled else BudgetExceeded("运行时间预算耗尽")
        )
        cancellation.cancel(reason)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cancellation.check()
        raise RunCancelled("用户取消")  # event is always set above
    finally:
        canceled.cancel()
        if not task.done():
            task.cancel()
        outcomes = await asyncio.gather(task, canceled, return_exceptions=True)
        # A protected handler can discover a journal failure while completing
        # cleanup. Never acknowledge an ordinary cancellation in that case: the
        # caller must stop and poison its runtime until journals are reopened.
        if isinstance(outcomes[0], PersistenceError):
            raise outcomes[0]
