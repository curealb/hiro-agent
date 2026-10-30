"""Service contracts shared by plugins; no provider imports belong here."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hiro_agent.errors import PersistenceError, RunCancelled

if TYPE_CHECKING:
    from hiro_agent.budget import Budget


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(DataModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str


class AssistantMessage(DataModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class Message(DataModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    continuation: bool = False


class Usage(DataModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_cached_input(self) -> Self:
        if self.cached_input_tokens is not None and self.cached_input_tokens > self.input_tokens:
            raise ValueError("Cached input tokens cannot exceed input tokens")
        return self


def usage_counts(usage: Usage | None) -> dict[str, int]:
    """Counters for one completed response; cached tokens are part of input."""
    counts = {"worker_steps": 1}
    if usage is None:
        counts.update(usage_missing=1, cache_usage_missing=1)
    else:
        counts.update(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        if usage.cached_input_tokens is None:
            counts["cache_usage_missing"] = 1
        else:
            counts.update(
                cached_input_tokens=usage.cached_input_tokens,
                cache_reported_input_tokens=usage.input_tokens,
            )
    return counts


class ModelRequest(DataModel):
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)


class ModelResponse(DataModel):
    assistant: AssistantMessage
    finish_reason: str
    usage: Usage | None = None


class SessionEvent(DataModel):
    version: Literal[1] = 1
    session_id: str
    run_id: str | None = None
    seq: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ToolResult(DataModel):
    call_id: str
    ok: bool
    value: Any = None
    error: str | None = None
    message: str = ""
    outcome: Literal["completed", "not_started", "unknown"] = "completed"


class RunStatus(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    REQUESTING_MODEL = "requesting_model"
    EXECUTING_TOOLS = "executing_tools"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELED = "canceled"
    FAILED = "failed"


class RunMetrics(DataModel):
    model_requests: int = 0
    worker_steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    permission_denials: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_missing: int = 0
    cached_input_tokens: int = 0
    cache_reported_input_tokens: int = 0
    cache_usage_missing: int = 0
    elapsed_ms: int = 0

    @model_validator(mode="before")
    @classmethod
    def preserve_unknown_legacy_cache(cls, value: Any) -> Any:
        # Older completed runs and delegation summaries have no cache telemetry.
        if isinstance(value, dict) and not {
            "cached_input_tokens",
            "cache_reported_input_tokens",
            "cache_usage_missing",
        }.intersection(value):
            return {**value, "cache_usage_missing": value.get("worker_steps", 0)}
        return value


class RunResult(DataModel):
    session_id: str
    run_id: str
    status: RunStatus
    reason: str
    output: str = ""
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    persistence_failed: bool = False


class DelegationLink(DataModel):
    call_id: str = Field(min_length=1)
    parent_session_id: str = Field(min_length=1)
    parent_run_id: str = Field(min_length=1)
    child_session_id: str = Field(min_length=1)
    child_run_id: str = Field(min_length=1)
    role: Literal["explorer", "reviewer"]
    model_id: str | None = None


class Finding(DataModel):
    path: str = Field(min_length=1, max_length=4096)
    line: int | None = Field(default=None, ge=1)
    description: str = Field(min_length=1, max_length=2000)


class SubagentReport(DataModel):
    summary: str = Field(min_length=1, max_length=4000)
    findings: list[Finding] = Field(default_factory=list, max_length=20)
    suggestions: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class Cancellation:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: BaseException | None = None

    def cancel(self, reason: BaseException | None = None) -> None:
        # Cleanup and sibling cancellation must not replace the initial cause.
        # A journal failure always takes precedence over an ordinary interruption.
        if not self.event.is_set() or (
            isinstance(reason, PersistenceError) and not isinstance(self.reason, PersistenceError)
        ):
            self.reason = reason or RunCancelled("用户取消")
        self.event.set()

    def check(self) -> None:
        if self.cancelled:
            raise self.reason or RunCancelled("用户取消")

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()


@dataclass(frozen=True)
class RunScope:
    run_id: str
    budget: Budget
    cancellation: Cancellation
    root_session_id: str
    root_run_id: str
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    role: str = "main"
    call_id: str | None = None


@dataclass(frozen=True)
class ToolExecutionContext:
    session: SessionService
    run_id: str
    call_id: str
    budget: Budget
    cancellation: Cancellation


class ModelService(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    async def close(self) -> None: ...


@runtime_checkable
class StreamingModelService(Protocol):
    """Optional capability; callbacks carry transient public text, not tool calls."""

    async def complete_stream(
        self, request: ModelRequest, on_delta: Callable[[str], None]
    ) -> ModelResponse: ...


class SessionService(Protocol):
    session_id: str

    @property
    def events(self) -> list[SessionEvent]: ...
    async def append(
        self, event_type: str, data: dict[str, Any], run_id: str | None = None
    ) -> SessionEvent: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...
    def derive_messages(self) -> list[Message]: ...


class PermissionService(Protocol):
    async def approve(self, tool: str, arguments: dict[str, Any], mutating: bool) -> bool: ...


class ToolsService(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...
    async def execute(
        self, call: ToolCall, run_id: str, budget: Budget, cancellation: Cancellation
    ) -> ToolResult: ...
    async def execute_many(
        self, calls: list[ToolCall], run_id: str, budget: Budget, cancellation: Cancellation
    ) -> list[ToolResult]: ...


class ContextService(Protocol):
    def prepare(self, session: SessionService) -> list[Message]: ...


class TaskPreprocessor(Protocol):
    async def prepare(self, task: str) -> str: ...


class AgentService(Protocol):
    @property
    def status(self) -> RunStatus: ...
    async def run(self, task: str, *, scope: RunScope | None = None) -> RunResult: ...
    async def continue_run(self) -> RunResult: ...
    def cancel(self) -> None: ...
    async def close(self) -> None: ...


class SubagentService(Protocol):
    async def delegate(
        self, role: Literal["explorer", "reviewer"], task: str, context: ToolExecutionContext
    ) -> ToolResult: ...
    async def close(self) -> None: ...


ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]
Observer = Callable[[str, dict[str, Any]], None]
