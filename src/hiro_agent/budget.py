"""Per-run counters with optional shared ancestor limits."""

from __future__ import annotations

import time

from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import Cancellation, RunMetrics
from hiro_agent.errors import BudgetExceeded


class Budget:
    def __init__(self, config: BudgetConfig, parent: Budget | None = None) -> None:
        self.config = config
        self.parent = parent
        self.started = time.monotonic()
        self.metrics = RunMetrics()
        self._exhausted_reason: str | None = None
        self._cancellation: Cancellation | None = None

    def bind_cancellation(self, cancellation: Cancellation) -> None:
        root = self
        while root.parent is not None:
            root = root.parent
        root._cancellation = cancellation

    @property
    def remaining(self) -> float:
        local = max(0.0, self.config.timeout_seconds - (time.monotonic() - self.started))
        return min(local, self.parent.remaining) if self.parent is not None else local

    def check(self) -> None:
        current: Budget | None = self
        while current is not None:
            if current._exhausted_reason is not None:
                raise BudgetExceeded(current._exhausted_reason)
            current = current.parent
        if self.remaining <= 0:
            raise BudgetExceeded("运行时间预算耗尽")

    def record(self, **counts: int) -> None:
        """Attribute counts locally and once to each ancestor, never elapsed time."""
        for name, count in counts.items():
            if name not in RunMetrics.model_fields or name == "elapsed_ms":
                raise ValueError(f"Invalid budget counter: {name}")
            if type(count) is not int or count < 0:
                raise ValueError("Budget counter increments must be nonnegative integers")
        for name, count in counts.items():
            setattr(self.metrics, name, getattr(self.metrics, name) + count)
        if self.parent is not None:
            self.parent.record(**counts)

    def _check_capacity(self, counter: str, limit: str, message: str) -> None:
        # No awaits: all limits are checked before charging any budget. Local
        # rejection must not consume a root allowance, nor vice versa.
        current: Budget | None = self
        while current is not None:
            current.check()
            if getattr(current.metrics, counter) >= getattr(current.config, limit):
                current._exhausted_reason = message
                if current.parent is None and current._cancellation is not None:
                    # Signal at the failed claim, before journal/child cleanup.
                    # A later Ctrl+C must not replace an earlier root exhaustion.
                    current._cancellation.cancel(BudgetExceeded(message))
                raise BudgetExceeded(message)
            current = current.parent

    def check_model_capacity(self) -> None:
        """Do not admit queued delegations that cannot make even one model request."""
        self._check_capacity("model_requests", "max_model_requests", "模型请求预算耗尽")

    def _claim(self, counter: str, limit: str, message: str) -> None:
        self._check_capacity(counter, limit, message)
        self.record(**{counter: 1})

    def claim_model(self) -> None:
        self._claim("model_requests", "max_model_requests", "模型请求预算耗尽")

    def claim_tool(self) -> None:
        self._claim("tool_calls", "max_tool_calls", "工具调用预算耗尽")

    def finish(self) -> RunMetrics:
        self.metrics.elapsed_ms = int((time.monotonic() - self.started) * 1000)
        return self.metrics.model_copy(deep=True)
