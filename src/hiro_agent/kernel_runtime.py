"""Durable Task integration with the single-Run Agent and its execution journal."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from hiro_agent.bootstrap import Harness, create_harness
from hiro_agent.bots import BotStore
from hiro_agent.budget import Budget
from hiro_agent.config import BudgetConfig, Config
from hiro_agent.contracts import (
    AssistantMessage,
    Cancellation,
    DataModel,
    ModelService,
    RunMetrics,
    RunResult,
    RunScope,
    RunStatus,
    SessionService,
    ToolCall,
)
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.errors import (
    ConfigurationError,
    HiroError,
    PersistenceError,
    RunSuspended,
    ToolError,
)
from hiro_agent.handoffs import HandoffStore
from hiro_agent.interactions import InteractionStore
from hiro_agent.kernel_context import KernelContext
from hiro_agent.kernel_contracts import (
    TERMINAL_TASK_STATES,
    InteractionRecord,
    InteractionRequest,
    OperationIntent,
    RunClaim,
    RunEvidence,
    TaskRecord,
    TaskRunOutcome,
    TaskState,
    canonical_arguments,
)
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.memory import MemoryStore
from hiro_agent.routines import RoutineStore
from hiro_agent.session import JsonlSession, recover_session
from hiro_agent.skills import SkillStore
from hiro_agent.subagents import recover_delegations
from hiro_agent.tools import ToolExecutor, ToolRegistry
from hiro_agent.worker import TaskWorker


class QuestionArguments(DataModel):
    question: str = Field(min_length=1, max_length=16000)


class SkillArguments(DataModel):
    skill_id: str = Field(min_length=1, max_length=128)


class SendHandoffArguments(DataModel):
    target_bot_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=32000)
    idempotency_key: str = Field(min_length=1, max_length=200)


class ReadHandoffArguments(DataModel):
    message_id: str = Field(min_length=1, max_length=128)


async def _question_handler(arguments: QuestionArguments) -> None:
    # Questions must suspend before tool.started, through the durable gate.
    raise HiroError("Question execution requires a persistent Task")


class _Interactions:
    def __init__(
        self, claim: RunClaim, bots: BotStore, tasks: TaskStore, store: InteractionStore
    ) -> None:
        self.claim = claim
        self.bots = bots
        self.tasks = tasks
        self.store = store
        self.session: SessionService | None = None
        self.waiting: InteractionRecord | None = None
        self.resume_call: str | None = None

    async def gate(
        self, call: ToolCall, arguments: dict[str, Any], mutating: bool, run_id: str
    ) -> bool:
        task = self.tasks.get(self.claim.task.id, bot_id=self.claim.task.bot_id)
        if task.active_run_id != run_id or task.session_id != self.claim.run.session_id:
            raise PersistenceError("Tool gate does not belong to the active Task Run")
        authority = self.bots.authority(task.bot_id)
        if call.id == self.resume_call:
            saved = self.claim.task.pending_operation
            if saved is None or self.claim.task.interaction_id is None:
                raise PersistenceError("Saved authorization is missing")
            if (
                call.name != saved.tool_name
                or canonical_arguments(arguments) != saved.arguments_json
            ):
                return False
            return self.store.consume_authorization(self.claim.task.interaction_id, saved, run_id)
        if call.name != "ask_user" and not mutating:
            return True
        if mutating and not authority.allows(call.name, mutating=True):
            return False
        operation = None
        kind: Literal["approval", "question"] = "question"
        prompt = str(arguments.get("question", ""))
        if mutating:
            kind = "approval"
            operation = OperationIntent(
                task_id=task.id,
                bot_id=task.bot_id,
                session_id=task.session_id,
                run_id=run_id,
                call_id=call.id,
                tool_name=call.name,
                arguments_json=canonical_arguments(arguments),
                permission_version=authority.permission_version,
                permission_digest=authority.permission_digest,
            )
            prompt = f"Approve {call.name}: {operation.arguments_json}"
        request = InteractionRequest(
            task_id=task.id,
            bot_id=task.bot_id,
            kind=kind,
            prompt=prompt,
            idempotency_key=f"{run_id}:{call.id}",
            operation=operation,
        )
        record = self.store.create(request)
        assert self.session is not None
        await self.session.append(
            "interaction.waiting",
            {"interaction_id": record.id, "request": request.model_dump(mode="json")},
            run_id,
        )
        self.waiting = record
        raise RunSuspended(f"等待{'审批' if mutating else '用户答复'}: {record.id}")


class KernelRunner:
    """Real runner. The optional model factory is an offline test/provider injection seam."""

    def __init__(
        self,
        workspace: Path,
        bots: BotStore,
        tasks: TaskStore,
        interactions: InteractionStore,
        memory: MemoryStore,
        skills: SkillStore,
        handoffs: HandoffStore,
        *,
        model_factory: Callable[[Config], ModelService] | None = None,
        context_max_chars: int = 64_000,
        context_max_messages: int = 64,
    ) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (context_max_chars, context_max_messages)
        ):
            raise ConfigurationError("Context limits must be positive integers")
        self.workspace = workspace.resolve(strict=True)
        self.bots = bots
        self.tasks = tasks
        self.interactions = interactions
        self.memory = memory
        self.skills = skills
        self.handoffs = handoffs
        self.model_factory = model_factory
        self.context_max_chars = context_max_chars
        self.context_max_messages = context_max_messages
        self._active: dict[str, Harness] = {}
        self._cancellations: dict[str, Cancellation] = {}

    @staticmethod
    def _configure_tools(registry: ToolRegistry) -> None:
        registry.register(
            "ask_user",
            "Ask the user a question and wait for a durable reply.",
            QuestionArguments,
            _question_handler,
        )

    async def run(
        self, claim: RunClaim, checkpoint: Callable[[RunMetrics], None]
    ) -> TaskRunOutcome:
        task = claim.task
        config = self.bots.get(task.bot_id).config
        gate = _Interactions(claim, self.bots, self.tasks, self.interactions)
        cancellation = Cancellation()
        self._cancellations[task.id] = cancellation
        harness: Harness | None = None
        heartbeat: asyncio.Task[None] | None = None
        try:

            async def read_skill(arguments: SkillArguments) -> str:
                try:
                    return self.skills.content(task.bot_id, arguments.skill_id)
                except ConfigurationError as exc:
                    # A missing/disabled Skill is an expected read rejection,
                    # not evidence of an unknown side effect after tool.started.
                    raise ToolError(str(exc)) from exc

            async def send_handoff(arguments: SendHandoffArguments) -> str:
                try:
                    return self.handoffs.send(
                        task.bot_id,
                        task.id,
                        arguments.target_bot_id,
                        arguments.text,
                        idempotency_key=arguments.idempotency_key,
                    ).model_dump_json()
                except ConfigurationError as exc:
                    raise ToolError(str(exc)) from exc

            async def read_handoff(arguments: ReadHandoffArguments) -> str:
                try:
                    result = self.handoffs.result(task.bot_id, arguments.message_id)
                    return result.model_dump_json() if result is not None else '{"pending":true}'
                except ConfigurationError as exc:
                    raise ToolError(str(exc)) from exc

            def configure_tools(registry: ToolRegistry) -> None:
                self._configure_tools(registry)
                registry.register(
                    "read_skill",
                    "Read the full text of a Skill enabled for this Bot by its skill_id.",
                    SkillArguments,
                    read_skill,
                )
                registry.register(
                    "send_handoff",
                    "Send a durable Task to another Bot; results return as a linked Task. "
                    "Reuse the same idempotency_key when retrying the same request.",
                    SendHandoffArguments,
                    send_handoff,
                )
                registry.register(
                    "read_handoff",
                    "Read a returned handoff result owned by this Bot.",
                    ReadHandoffArguments,
                    read_handoff,
                )

            harness = await create_harness(
                self.workspace,
                config,
                task.session_id,
                create_session=not any(
                    run.id != claim.run.id and run.started_at is not None
                    for run in self.tasks.runs(task.id)
                ),
                persist_bot=False,
                model=self.model_factory(config) if self.model_factory else None,
                gate=gate.gate,
                configure_tools=configure_tools,
                context=KernelContext(
                    task.bot_id,
                    task.text,
                    self.memory,
                    self.skills,
                    max_chars=self.context_max_chars,
                    max_messages=self.context_max_messages,
                ),
            )
            self._active[task.id] = harness
            gate.session = harness.session
            used = task.cumulative_metrics
            chain = self.tasks.chain(task.id)
            chain_used = chain.cumulative_metrics
            remaining = BudgetConfig(
                max_model_requests=min(
                    task.budget.max_model_requests - used.model_requests,
                    chain.budget.max_model_requests - chain_used.model_requests,
                ),
                max_tool_calls=min(
                    task.budget.max_tool_calls - used.tool_calls,
                    chain.budget.max_tool_calls - chain_used.tool_calls,
                ),
                timeout_seconds=min(
                    task.budget.timeout_seconds - used.elapsed_ms / 1000,
                    chain.budget.timeout_seconds - chain_used.elapsed_ms / 1000,
                ),
            )
            budget = Budget(remaining, checkpoint=checkpoint)

            async def prepare() -> None:
                if task.interaction_id is None:
                    return
                record = self.interactions.get(task.interaction_id, bot_id=task.bot_id)
                if record.request.task_id != task.id:
                    raise PersistenceError("Interaction belongs to another Task")
                if record.request.kind == "question":
                    if record.reply is None or record.reply.text is None:
                        raise HiroError("用户答复不可用；本次未继续执行")
                    # The answer is durable in both stores before its one-time
                    # token is consumed. A crash in the following gap stops the
                    # Run, but leaves the full input available for inspection.
                    await harness.session.append(
                        "interaction.answer",
                        {"interaction_id": record.id, "text": record.reply.text},
                        claim.run.id,
                    )
                    answer = self.interactions.consume_answer(record.id, task.id, claim.run.id)
                    if answer is None:
                        raise HiroError("用户答复已消费或不可用；本次未继续执行")
                    await harness.session.append(
                        "interaction.resumed",
                        {"interaction_id": record.id, "kind": "question"},
                        claim.run.id,
                    )
                    await harness.session.append(
                        "task.continued",
                        {"text": f"用户对问题「{record.request.prompt}」的答复：\n{answer}"},
                        claim.run.id,
                    )
                    return
                if record.reply is None or record.reply.approved is not True:
                    raise HiroError("审批已拒绝；保存的操作未执行")
                operation = task.pending_operation
                if operation is None or record.request.operation != operation:
                    raise PersistenceError("Saved operation and interaction do not match")
                call = ToolCall(
                    id=uuid4().hex, name=operation.tool_name, arguments=operation.arguments_json
                )
                gate.resume_call = call.id
                await harness.session.append(
                    "interaction.resumed",
                    {"interaction_id": record.id, "operation_id": operation.id, "call_id": call.id},
                    claim.run.id,
                )
                await harness.session.append(
                    "assistant.message",
                    {
                        "message": AssistantMessage(tool_calls=[call]).model_dump(mode="json"),
                        "prepared_operation": True,
                    },
                    claim.run.id,
                )
                executor: ToolExecutor = harness.runtime.services["tools"]
                result = await executor.execute(call, claim.run.id, budget, cancellation)
                if not result.ok:
                    raise HiroError(f"保存的操作未成功执行: {result.error}: {result.message}")

            async def pulse() -> None:
                while True:
                    await asyncio.sleep(0.1)
                    try:
                        checkpoint(budget.finish())
                    except PersistenceError as exc:
                        cancellation.cancel(exc)
                        raise

            heartbeat = asyncio.create_task(pulse())
            result = await harness.agent.run(
                task.text
                if task.interaction_id is None
                else "继续持久任务，先处理已保存的用户答复。",
                scope=RunScope(
                    run_id=claim.run.id,
                    budget=budget,
                    cancellation=cancellation,
                    root_session_id=task.session_id,
                    root_run_id=claim.run.id,
                    persistent_task_id=task.id,
                    prepare=prepare,
                ),
            )
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if result.persistence_failed:
                raise PersistenceError("Run journal or Task checkpoint could not be saved")
            checkpoint(result.metrics)
            outcome = TaskRunOutcome(
                result=result, unknown_side_effect=self._unknown(harness.session, claim.run.id)
            )
            if result.status == RunStatus.PAUSED and gate.waiting is not None:
                outcome.pending_operation = gate.waiting.request.operation
                outcome.interaction_id = gate.waiting.id
                outcome.waiting_for = gate.waiting.request.kind
            return outcome
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                outcomes = await asyncio.gather(heartbeat, return_exceptions=True)
                error = next(
                    (value for value in outcomes if isinstance(value, PersistenceError)), None
                )
            else:
                error = None
            try:
                if harness is not None:
                    await harness.close()
            finally:
                self._active.pop(task.id, None)
                self._cancellations.pop(task.id, None)
            if error is not None:
                raise error

    @staticmethod
    def _unknown(session: SessionService, run_id: str) -> bool:
        return any(
            event.run_id == run_id
            and event.type == "tool.result"
            and event.data.get("outcome") == "unknown"
            for event in session.events
        )

    async def recover(self, claim: RunClaim) -> RunEvidence:
        evidence = RunEvidence(run_id=claim.run.id, session_id=claim.task.session_id, started=False)
        try:
            session = await JsonlSession.open(self.workspace, claim.task.session_id, create=False)
        except FileNotFoundError:
            return evidence
        try:
            bindings = [e for e in session.events if e.type == "session.bot"]
            if bindings and (
                len(bindings) != 1 or bindings[0].data.get("bot_id") != claim.task.bot_id
            ):
                raise PersistenceError("Task journal Bot binding does not match")
            starts = [
                e for e in session.events if e.type == "run.started" and e.run_id == claim.run.id
            ]
            if not starts:
                return evidence
            if not bindings or starts[0].data.get("persistent_task_id") != claim.task.id:
                raise PersistenceError("Task journal Run linkage does not match")
            config = self.bots.get(claim.task.bot_id).config
            await recover_delegations(self.workspace, session, config.subagents.max_output_chars)
            await recover_session(session)
            ended = next(
                e for e in session.events if e.type == "run.ended" and e.run_id == claim.run.id
            )
            result = RunResult(
                session_id=session.session_id,
                run_id=claim.run.id,
                status=RunStatus(ended.data["status"]),
                reason=ended.data["reason"],
                output=ended.data.get("output", ""),
                metrics=RunMetrics.model_validate(ended.data["metrics"]),
            )
            evidence.started = True
            evidence.result = result
            evidence.metrics = result.metrics
            evidence.unknown_side_effect = self._unknown(session, claim.run.id)
            waiting = [
                e
                for e in session.events
                if e.run_id == claim.run.id and e.type == "interaction.waiting"
            ]
            if waiting and result.status == RunStatus.PAUSED:
                if len(waiting) != 1:
                    raise PersistenceError("Run has multiple pending interactions")
                request = InteractionRequest.model_validate(waiting[0].data["request"])
                record = self.interactions.get(
                    waiting[0].data["interaction_id"], bot_id=claim.task.bot_id
                )
                if record.request != request or request.task_id != claim.task.id:
                    raise PersistenceError("Journal interaction does not match durable reply")
                evidence.pending_operation = request.operation
                evidence.interaction_id = record.id
                evidence.waiting_for = request.kind
            return evidence
        except (KeyError, ValueError, ValidationError) as exc:
            raise PersistenceError("Invalid durable Task journal metadata") from exc
        finally:
            await session.close()

    def cancel(self, task_id: str) -> None:
        cancellation = self._cancellations.get(task_id)
        if cancellation is not None:
            cancellation.cancel()

    async def close(self) -> None:
        for cancellation in self._cancellations.values():
            cancellation.cancel()
        for harness in list(self._active.values()):
            await harness.agent.close()


class KernelWorker(TaskWorker):
    """Reconcile durable consumers and schedule routines within one worker lifecycle."""

    def __init__(self, kernel: KernelRuntime, *, poll_interval: float = 1.0) -> None:
        self.kernel = kernel
        self._scheduler: asyncio.Task[None] | None = None
        super().__init__(kernel.database, kernel.tasks, kernel.runner, poll_interval=poll_interval)

    def _summaries(self) -> None:
        self.kernel.handoffs.sync()
        for bot in self.kernel.bots.list():
            self.kernel.memory.sync_terminal(bot.bot_id)

    def _interactions(self) -> None:
        for interaction in self.kernel.interactions.ready():
            task = self.kernel.tasks.get(
                interaction.request.task_id, bot_id=interaction.request.bot_id
            )
            if task.state in TERMINAL_TASK_STATES or task.state == TaskState.NEEDS_RECONCILIATION:
                self.kernel.interactions.cancel_for_task(task.id)
            else:
                self.kernel.tasks.wake(task.id, interaction.id)
        for task in self.kernel.tasks.list():
            if task.state in TERMINAL_TASK_STATES or task.state == TaskState.NEEDS_RECONCILIATION:
                self.kernel.interactions.cancel_for_task(task.id)

    async def recover(self) -> list[TaskRecord]:
        try:
            recovered = await super().recover()
            self._interactions()
            self._summaries()
            return recovered
        except BaseException:
            await self.close()
            raise

    async def tick(self) -> bool:
        try:
            self._ensure_open()
            self.kernel.routines.tick()
            self._interactions()
            self._summaries()
            result = await super().tick()
            self._interactions()
            self._summaries()
            return result
        except BaseException:
            await self.close()
            raise

    async def serve(self, stop: asyncio.Event) -> None:
        self._ensure_open()
        if self._scheduler is not None or self._stop is not None:
            raise HiroError("Worker is already serving")

        async def schedule() -> None:
            try:
                while not stop.is_set():
                    self.kernel.routines.tick()
                    self._summaries()
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
                    except TimeoutError:
                        pass
            except BaseException:
                # A durable scheduling failure stops the executor too; do not
                # silently lose timers while leaving a seemingly healthy worker.
                stop.set()
                raise

        self._scheduler = asyncio.create_task(schedule())
        try:
            await super().serve(stop)
        finally:
            await self.close()

    async def _shutdown(self) -> None:
        # TaskWorker.close shields this entire shutdown from repeated caller
        # cancellation, keeping flock until both scheduler and runner are gone.
        if self._stop is not None:
            self._stop.set()
        errors: list[BaseException] = []
        if self._scheduler is not None:
            self._scheduler.cancel()
            outcomes = await asyncio.gather(self._scheduler, return_exceptions=True)
            if isinstance(outcomes[0], BaseException) and not isinstance(
                outcomes[0], asyncio.CancelledError
            ):
                errors.append(outcomes[0])
        try:
            await super()._shutdown()
        except BaseException as error:
            errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Kernel worker shutdown failed", errors)


class KernelRuntime:
    def __init__(
        self,
        workspace: Path,
        *,
        model_factory: Callable[[Config], ModelService] | None = None,
        context_max_chars: int = 64_000,
        context_max_messages: int = 64,
    ) -> None:
        self.workspace = workspace.resolve(strict=True)
        self.database = KernelDatabase.for_workspace(self.workspace)
        self.bots = BotStore(self.database)
        self.tasks = TaskStore(self.database, self.bots)
        self.interactions = InteractionStore(self.database, self.bots)
        self.memory = MemoryStore(self.database, self.bots, self.tasks)
        self.skills = SkillStore(
            self.database, self.bots, self.tasks, self.workspace / ".hiro-agent" / "skills"
        )
        self.routines = RoutineStore(self.database, self.bots, self.tasks)
        self.handoffs = HandoffStore(self.database, self.bots, self.tasks)
        self.runner = KernelRunner(
            self.workspace,
            self.bots,
            self.tasks,
            self.interactions,
            self.memory,
            self.skills,
            self.handoffs,
            model_factory=model_factory,
            context_max_chars=context_max_chars,
            context_max_messages=context_max_messages,
        )

    def worker(self, *, poll_interval: float = 1.0) -> KernelWorker:
        return KernelWorker(self, poll_interval=poll_interval)
