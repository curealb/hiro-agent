"""Durable kernel DTOs and module boundaries; independent of the single-run loop."""

from __future__ import annotations

import builtins
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, Self
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import DataModel, RunMetrics, RunResult


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """Normalize validated JSON arguments, including schema defaults, before binding."""
    return json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_RECONCILIATION = "needs_reconciliation"


TERMINAL_TASK_STATES = frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED})
WAITING_TASK_STATES = frozenset({TaskState.WAITING_APPROVAL, TaskState.WAITING_INPUT})


class OperationIntent(DataModel):
    """The settled, not-executed invocation, never a still-open tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    bot_id: str
    session_id: str
    run_id: str
    call_id: str
    tool_name: str
    arguments_json: str
    permission_version: int = Field(ge=1)
    permission_digest: str

    @field_validator("arguments_json")
    @classmethod
    def normalize_arguments(cls, value: str) -> str:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be an object")
        return canonical_arguments(parsed)

    @property
    def binding(self) -> str:
        """Original call IDs are audit data; authority binds the exact task and operation."""
        payload = {
            "task_id": self.task_id,
            "bot_id": self.bot_id,
            "tool_name": self.tool_name,
            "arguments_json": self.arguments_json,
            "permission_version": self.permission_version,
            "permission_digest": self.permission_digest,
        }
        return hashlib.sha256(canonical_arguments(payload).encode()).hexdigest()


class TaskSubmission(DataModel):
    bot_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    idempotency_key: str | None = Field(default=None, min_length=1)
    source: str = "manual"
    budget: BudgetConfig | None = None
    chain_id: str | None = None
    chain_budget: BudgetConfig | None = None
    chain_max_hops: int | None = Field(default=None, ge=1, le=1000)
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChainRecord(DataModel):
    id: str
    budget: BudgetConfig
    max_hops: int = Field(default=8, ge=1)
    hops_used: int = Field(default=0, ge=0)
    cumulative_metrics: RunMetrics = Field(default_factory=RunMetrics)


class TaskRecord(DataModel):
    id: str
    bot_id: str
    session_id: str
    text: str
    state: TaskState = TaskState.QUEUED
    revision: int = Field(default=1, ge=1)
    source: str = "manual"
    idempotency_key: str | None = None
    chain_id: str | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cumulative_metrics: RunMetrics = Field(default_factory=RunMetrics)
    active_run_id: str | None = None
    cancel_requested: bool = False
    pending_operation: OperationIntent | None = None
    interaction_id: str | None = None
    output: str = ""
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DurableRunState(StrEnum):
    CLAIMED = "claimed"
    STARTED = "started"
    FINISHED = "finished"
    UNCERTAIN = "uncertain"


class DurableRunRecord(DataModel):
    id: str
    task_id: str
    session_id: str
    worker_id: str
    state: DurableRunState = DurableRunState.CLAIMED
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    result: RunResult | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunClaim(DataModel):
    task: TaskRecord
    run: DurableRunRecord


class RunEvidence(DataModel):
    """Facts read from a validated, recovered JSONL session by the coordinator."""

    run_id: str
    session_id: str
    started: bool
    result: RunResult | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    unknown_side_effect: bool = False
    pending_operation: OperationIntent | None = None
    interaction_id: str | None = None
    waiting_for: Literal["approval", "question"] | None = None


class TaskRunOutcome(DataModel):
    result: RunResult
    pending_operation: OperationIntent | None = None
    interaction_id: str | None = None
    waiting_for: Literal["approval", "question"] | None = None
    unknown_side_effect: bool = False


class TaskStoreService(Protocol):
    def submit(self, request: TaskSubmission) -> TaskRecord: ...
    def get(self, task_id: str, *, bot_id: str | None = None) -> TaskRecord: ...
    def list(
        self, *, bot_id: str | None = None, states: tuple[TaskState, ...] = ()
    ) -> list[TaskRecord]: ...
    def cancel(self, task_id: str, *, bot_id: str | None = None) -> TaskRecord: ...
    def claim(self, worker_id: str) -> RunClaim | None: ...
    def mark_started(self, run_id: str) -> DurableRunRecord: ...
    def checkpoint(self, run_id: str, metrics: RunMetrics) -> TaskRecord: ...
    def finish(
        self,
        run_id: str,
        result: RunResult,
        *,
        pending_operation: OperationIntent | None = None,
        interaction_id: str | None = None,
        waiting_for: Literal["approval", "question"] | None = None,
    ) -> TaskRecord: ...
    def wake(self, task_id: str, interaction_id: str) -> TaskRecord: ...
    def reconcile(self, evidence: RunEvidence) -> TaskRecord: ...
    def runs(self, task_id: str) -> builtins.list[DurableRunRecord]: ...
    def resolve(
        self, task_id: str, *, action: Literal["retry", "fail", "cancel"], note: str
    ) -> TaskRecord: ...


class TaskRunner(Protocol):
    """B3 owns the real Agent adapter; B1 tests its worker with controlled runners."""

    async def run(
        self, claim: RunClaim, checkpoint: Callable[[RunMetrics], None]
    ) -> TaskRunOutcome: ...
    async def recover(self, claim: RunClaim) -> RunEvidence: ...
    def cancel(self, task_id: str) -> None: ...
    async def close(self) -> None: ...


class InteractionState(StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


class InteractionRequest(DataModel):
    task_id: str
    bot_id: str
    kind: Literal["approval", "question"]
    prompt: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    operation: OperationIntent | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        if self.kind == "approval" and self.operation is None:
            raise ValueError("Approval requires the exact operation")
        if self.operation is not None and (
            self.operation.task_id != self.task_id or self.operation.bot_id != self.bot_id
        ):
            raise ValueError("Interaction and operation must have the same owner")
        return self


class InteractionReply(DataModel):
    approved: bool | None = Field(default=None, strict=True)
    text: str | None = None


class InteractionRecord(DataModel):
    id: str
    request: InteractionRequest
    state: InteractionState = InteractionState.PENDING
    reply: InteractionReply | None = None
    consumed_by_run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    answered_at: datetime | None = None
    consumed_at: datetime | None = None


class InteractionService(Protocol):
    def create(self, request: InteractionRequest) -> InteractionRecord: ...
    def get(self, interaction_id: str, *, bot_id: str | None = None) -> InteractionRecord: ...
    def list(
        self,
        *,
        bot_id: str | None = None,
        task_id: str | None = None,
        state: InteractionState | None = None,
    ) -> list[InteractionRecord]: ...
    def reply(self, interaction_id: str, reply: InteractionReply) -> InteractionRecord: ...
    def ready(self) -> builtins.list[InteractionRecord]: ...
    def consume_authorization(
        self, interaction_id: str, operation: OperationIntent, run_id: str
    ) -> bool: ...
    def consume_answer(self, interaction_id: str, task_id: str, run_id: str) -> str | None: ...
    def cancel_for_task(self, task_id: str) -> None: ...
