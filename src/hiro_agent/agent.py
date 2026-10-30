"""The provider-independent agent state machine."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from hiro_agent.budget import Budget
from hiro_agent.config import BudgetConfig
from hiro_agent.contracts import (
    Cancellation,
    ContextService,
    ModelRequest,
    ModelService,
    RunMetrics,
    RunResult,
    RunScope,
    RunStatus,
    SessionService,
    StreamingModelService,
    TaskPreprocessor,
    ToolsService,
    usage_counts,
)
from hiro_agent.core import EventBus
from hiro_agent.errors import (
    BudgetExceeded,
    ContextLimitError,
    HiroError,
    ModelError,
    PersistenceError,
    RunCancelled,
)
from hiro_agent.execution import cancellable
from hiro_agent.session import delegation_terminal_event, settle_tool_calls


class Agent:
    def __init__(
        self,
        session: SessionService,
        model: ModelService,
        tools: ToolsService,
        context: ContextService,
        preprocessor: TaskPreprocessor,
        budget: BudgetConfig,
        events: EventBus | None = None,
    ) -> None:
        self.session = session
        self.model = model
        self.tools = tools
        self.context = context
        self.preprocessor = preprocessor
        self.budget_config = budget
        self.events = events or EventBus()
        self._status = RunStatus.IDLE
        self._lock = asyncio.Lock()
        self._cancellation = Cancellation()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._broken = False
        self._active_task: asyncio.Task[RunResult] | None = None
        self._scope: RunScope | None = None

    @property
    def status(self) -> RunStatus:
        return self._status

    def cancel(self) -> None:
        self._cancellation.cancel()

    async def close(self) -> None:
        self._closed = True
        if self._active_task is not None and not self._active_task.done():
            self.cancel()
        await self._idle.wait()

    async def run(self, task: str, *, scope: RunScope | None = None) -> RunResult:
        if not task.strip():
            raise HiroError("任务不能为空")
        return await self._invoke(task, scope)

    async def continue_run(self) -> RunResult:
        events = self.session.events
        tasks = [event for event in events if event.type == "task.received"]
        ended = [event for event in events if event.type == "run.ended"]
        if not tasks or not ended or ended[-1].data.get("status") == "completed":
            raise HiroError("没有可继续的未完成任务")
        started = [event for event in events if event.type == "run.started"]
        if started and not started[-1].data.get("continued", False):
            if tasks[-1].run_id != started[-1].run_id:
                raise HiroError("最近的 Run 尚未记录任务，请重新输入任务")
        return await self._invoke(None)

    async def _invoke(self, task_text: str | None, scope: RunScope | None = None) -> RunResult:
        if self._active_task is not None:
            raise HiroError("同一会话已有正在执行的 Run")
        if scope is None:
            run_id = uuid.uuid4().hex
            scope = RunScope(
                run_id=run_id,
                budget=Budget(self.budget_config),
                cancellation=Cancellation(),
                root_session_id=self.session.session_id,
                root_run_id=run_id,
            )
        self._scope = scope
        self._cancellation = scope.cancellation
        scope.budget.bind_cancellation(scope.cancellation)
        self._idle.clear()
        task = asyncio.create_task(self._run(task_text, scope))
        self._active_task = task
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    # Translate caller cancellation to the same cooperative path
                    # as Ctrl+C. Journal acknowledgments must not be interrupted:
                    # they may commit after an awaiting coroutine is canceled.
                    self.cancel()
        finally:
            self._active_task = None
            self._idle.set()

    def _lineage(self) -> dict[str, Any]:
        data: dict[str, Any] = {"session_id": self.session.session_id}
        if self._scope is not None:
            data.update(
                run_id=self._scope.run_id,
                root_session_id=self._scope.root_session_id,
                root_run_id=self._scope.root_run_id,
                parent_session_id=self._scope.parent_session_id,
                parent_run_id=self._scope.parent_run_id,
                role=self._scope.role,
                task_id=self._scope.call_id,
                child_session_id=self.session.session_id if self._scope.call_id else None,
            )
        return data

    def _emit(self, name: str, data: dict[str, Any]) -> None:
        self.events.emit(name, {**data, **self._lineage()})

    def _state(self, status: RunStatus) -> None:
        self._status = status
        self._emit("run.state", {"status": status.value})

    def _check(self, budget: Budget) -> None:
        self._cancellation.check()
        budget.check()

    async def _run(self, task: str | None, scope: RunScope) -> RunResult:
        if self._closed or self._broken:
            raise HiroError("运行时已关闭或持久化失败，请重新打开会话")
        if self._lock.locked():
            raise HiroError("同一会话已有正在执行的 Run")
        async with self._lock:
            self._idle.clear()
            budget = scope.budget
            run_id = scope.run_id
            output = ""
            status = RunStatus.FAILED
            reason = "运行失败"
            persistence_failed = False
            self._state(RunStatus.PREPARING)
            try:
                await self.session.append(
                    "run.started", {"continued": task is None, **self._lineage()}, run_id
                )
                self._emit("run.started", {})
                if task is not None:
                    await self.session.append("task.received", {"text": task}, run_id)
                    prepared = await cancellable(
                        self.preprocessor.prepare(task), self._cancellation, budget.remaining
                    )
                    if prepared != task:
                        # v1 deliberately has an identity preprocessor: do not inject
                        # unjournaled model-visible text through an extension.
                        raise HiroError("当前版本仅支持保留原始任务的预处理器")
                    await self.session.append("task.prepared", {"text": prepared}, run_id)
                else:
                    await self.session.append(
                        "task.continued",
                        {
                            "text": "请根据已有结果继续完成上一个未完成的任务；"
                            "先核实未知结果，不要盲目重做操作。"
                        },
                        run_id,
                    )
                while True:
                    self._check(budget)
                    request = ModelRequest(
                        messages=self.context.prepare(self.session), tools=self.tools.schemas()
                    )
                    budget.claim_model()
                    attempt_id = uuid.uuid4().hex
                    output = ""
                    self._state(RunStatus.REQUESTING_MODEL)
                    await self.session.append(
                        "model.requested",
                        {
                            "attempt_id": attempt_id,
                            "purpose": "worker",
                            "request": request.model_dump(mode="json"),
                        },
                        run_id,
                    )
                    self._emit("model.requested", {"attempt_id": attempt_id})

                    def on_delta(text: str, *, current_attempt: str = attempt_id) -> None:
                        self._check(budget)
                        if text:
                            self._emit(
                                "assistant.delta", {"attempt_id": current_attempt, "text": text}
                            )

                    try:
                        completion = (
                            self.model.complete_stream(request, on_delta)
                            if isinstance(self.model, StreamingModelService)
                            else self.model.complete(request)
                        )
                        response = await cancellable(
                            completion, self._cancellation, budget.remaining
                        )
                    except (RunCancelled, BudgetExceeded, asyncio.CancelledError):
                        await self.session.append(
                            "model.interrupted", {"attempt_id": attempt_id}, run_id
                        )
                        self._emit("model.interrupted", {"attempt_id": attempt_id})
                        raise
                    except Exception as exc:
                        if isinstance(exc, PersistenceError):
                            raise
                        await self.session.append(
                            "model.failed",
                            {"attempt_id": attempt_id, "error": type(exc).__name__},
                            run_id,
                        )
                        self._emit("model.failed", {"attempt_id": attempt_id})
                        raise
                    await self.session.append(
                        "assistant.message",
                        {
                            "attempt_id": attempt_id,
                            "message": response.assistant.model_dump(mode="json"),
                            "finish_reason": response.finish_reason,
                            "usage": response.usage.model_dump() if response.usage else None,
                        },
                        run_id,
                    )
                    budget.record(**usage_counts(response.usage))
                    self._emit(
                        "model.finished",
                        {
                            "attempt_id": attempt_id,
                            "finish_reason": response.finish_reason,
                            "usage": response.usage.model_dump() if response.usage else None,
                        },
                    )
                    output = response.assistant.text or ""
                    if response.finish_reason == "length":
                        status = RunStatus.PAUSED
                        reason = "模型输出达到长度限制，未执行截断响应中的工具"
                        break
                    if response.finish_reason not in {"stop", "tool_calls"}:
                        raise ModelError(f"模型未正常完成：{response.finish_reason}")
                    calls = response.assistant.tool_calls
                    if not calls:
                        if response.finish_reason == "tool_calls":
                            raise ModelError("模型报告工具调用但未提供工具")
                        status, reason = RunStatus.COMPLETED, "模型完成回答"
                        break
                    if output:
                        self._emit("assistant.message", {"text": output})
                    self._state(RunStatus.EXECUTING_TOOLS)
                    await self.tools.execute_many(calls, run_id, budget, self._cancellation)
            except (RunCancelled, asyncio.CancelledError):
                status, reason = RunStatus.CANCELED, "用户取消；已停止启动新工具"
            except (BudgetExceeded, ContextLimitError) as exc:
                status, reason = RunStatus.PAUSED, str(exc)
            except PersistenceError:
                persistence_failed = self._broken = True
                status, reason = RunStatus.FAILED, "会话持久化失败；已停止执行，请重新打开会话"
            except HiroError as exc:
                status, reason = RunStatus.FAILED, str(exc)
            except Exception:
                status, reason = RunStatus.FAILED, "运行时出现未预期错误，已停止执行"
            final_metrics: RunMetrics | None = None
            try:
                if not persistence_failed:
                    # All calls, including those never handed to the executor, must
                    # have a terminal result before another user/model turn.
                    recovered = await settle_tool_calls(self.session)
                    for tool_result in recovered:
                        self._emit("tool.recovered", tool_result.model_dump(mode="json"))
                        progress = delegation_terminal_event(
                            self.session, tool_result, recovered=True
                        )
                        if progress is not None:
                            self.events.emit("delegation.finished", progress)
                    await self.session.append(
                        "run.checkpoint", {"status": status.value, "reason": reason}, run_id
                    )
                    final_metrics = budget.finish()
                    await self.session.append(
                        "run.ended",
                        {
                            "status": status.value,
                            "reason": reason,
                            "metrics": final_metrics.model_dump(),
                            "output": output,
                        },
                        run_id,
                    )
            except PersistenceError:
                persistence_failed = self._broken = True
                status, reason = RunStatus.FAILED, "会话结算写入失败；请重新打开会话恢复"
            finally:
                self._state(status)
                self._idle.set()
            result = RunResult(
                session_id=self.session.session_id,
                run_id=run_id,
                status=status,
                reason=reason,
                output=output,
                metrics=final_metrics if final_metrics is not None else budget.finish(),
                persistence_failed=persistence_failed,
            )
            self._emit("run.ended", result.model_dump(mode="json"))
            return result
