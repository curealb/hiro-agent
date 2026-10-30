"""Transient run animation with confirmed, provider-reported token usage."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.segment import Segments
from rich.text import Text

from hiro_agent.contracts import RunMetrics, Usage, usage_counts
from hiro_agent.presentation import assistant_markdown

_FINISHED = {"completed", "paused", "canceled", "failed"}
_STAGES = {
    "preparing": "准备中",
    "requesting_model": "思考中",
    "executing_tools": "执行工具",
    "completed": "已完成",
    "paused": "已暂停",
    "canceled": "已取消",
    "failed": "运行失败",
}
_FRAMES = ("✶", "✸", "✹", "✺", "✹", "✸")
_PREVIEW_CHARACTERS = 4000
_PREVIEW_LINES = 12


def usage_summary(metrics: RunMetrics, *, pending: bool = False) -> str:
    """Cache tokens are a subset of input, and only reported inputs form the rate."""
    if metrics.worker_steps == 0:
        unavailable = "等待返回" if pending else "未提供"
        return f"Token {unavailable} · 缓存 {unavailable}"

    if metrics.worker_steps > metrics.usage_missing:
        total = metrics.input_tokens + metrics.output_tokens
        tokens = f"Token {total:,} · 输入 {metrics.input_tokens:,} / 输出 {metrics.output_tokens:,}"
        if metrics.usage_missing or (not pending and metrics.model_requests > metrics.worker_steps):
            tokens += "（部分）"
    else:
        tokens = "Token 未提供"

    if metrics.worker_steps > metrics.cache_usage_missing:
        cached = metrics.cached_input_tokens
        denominator = metrics.cache_reported_input_tokens
        rate = f"{cached / denominator:.1%}" if denominator else "—"
        cache = f"缓存命中 {cached:,} / {denominator:,} ({rate})"
        if metrics.cache_usage_missing or (
            not pending and metrics.model_requests > metrics.worker_steps
        ):
            cache += "（部分）"
    else:
        cache = "缓存 未提供"
    return f"{tokens} · {cache}"


class RunDisplay:
    """Refresh on the event loop so approvals can take exclusive terminal control."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.metrics = RunMetrics()
        self._running = False
        self._suspended = 0
        self._started = 0.0
        self._ended = False
        self._root: tuple[str, str] | None = None
        self._children: set[tuple[str, str]] = set()
        self._known_runs: set[tuple[str, str]] = set()
        self._responses: set[tuple[str, str, str]] = set()
        self._pending_models: set[tuple[str, str]] = set()
        self._stage = "preparing"
        self._tool = ""
        self._attempt_id: str | None = None
        self._seen_attempts: set[str] = set()
        self._partial_chunks: list[str] = []
        self._partial_characters = 0
        self._preview_tail = ""
        self._attempt_interrupted = False
        self._attempt_open = False
        self._live: Live | None = None
        self._refresh_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.stop()
        self.metrics = RunMetrics()
        self._root = None
        self._children.clear()
        self._known_runs.clear()
        self._responses.clear()
        self._pending_models.clear()
        self._stage = "preparing"
        self._tool = ""
        self._attempt_id = None
        self._seen_attempts.clear()
        self._reset_partial()
        self._ended = False
        self._started = time.monotonic()
        self._running = True
        self._show()

    def stop(self) -> None:
        self._running = False
        self._hide()

    @property
    def interrupted_output(self) -> str:
        """Return incomplete root text for the CLI, including after stop()."""
        return "".join(self._partial_chunks) if self._attempt_interrupted else ""

    def _reset_partial(self) -> None:
        self._partial_chunks.clear()
        self._partial_characters = 0
        self._preview_tail = ""
        self._attempt_interrupted = False
        self._attempt_open = False

    def _show(self) -> None:
        if (
            not self._running
            or self._suspended
            or self._live is not None
            or not self.console.is_terminal
            or self.console.is_dumb_terminal
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        live = Live(
            console=self.console,
            get_renderable=self.render,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live = live
        try:
            live.start(refresh=True)
            self._refresh_task = loop.create_task(self._refresh(live))
        except Exception:
            self._hide()

    def _hide(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass

    async def _refresh(self, live: Live) -> None:
        try:
            while self._running and self._live is live:
                await asyncio.sleep(0.125)
                if self._live is live:
                    live.refresh()
        except Exception:
            # Rendering must not interrupt an agent run or leak a background task.
            self._hide()

    @contextmanager
    def suspend(self) -> Iterator[None]:
        self._suspended += 1
        self._hide()
        try:
            yield
        finally:
            self._suspended -= 1
            self._show()

    def _identity(self, data: dict[str, Any]) -> tuple[str, str] | None:
        session_id, run_id = data.get("session_id"), data.get("run_id")
        if isinstance(session_id, str) and isinstance(run_id, str):
            return session_id, run_id
        return None

    def observe(self, name: str, data: dict[str, Any]) -> None:
        if not self._running or self._ended:
            return
        identity = self._identity(data)
        if identity is None:
            return
        if self._root is None:
            if (
                name not in {"run.state", "run.started"}
                or data.get("role", "main") != "main"
                or data.get("parent_run_id") is not None
            ):
                return
            self._root = identity
            self._known_runs.add(identity)
        if identity not in self._known_runs:
            root = (data.get("root_session_id"), data.get("root_run_id"))
            if root != self._root:
                return
            self._known_runs.add(identity)

        root_event = identity == self._root
        if name == "run.state":
            status = str(data.get("status", ""))
            if root_event:
                self._stage = status
                self._tool = ""
            elif status in _FINISHED:
                self._children.discard(identity)
            else:
                self._children.add(identity)
            if status == "requesting_model":
                self._pending_models.add(identity)
                self.metrics.model_requests += 1
            elif status in _FINISHED:
                self._pending_models.discard(identity)
        elif name == "model.requested" and root_event:
            attempt_id = data.get("attempt_id")
            if (
                not isinstance(attempt_id, str)
                or not attempt_id
                or attempt_id in self._seen_attempts
            ):
                return
            self._seen_attempts.add(attempt_id)
            self._attempt_id = attempt_id
            self._reset_partial()
            self._attempt_open = True
        elif name == "assistant.delta" and root_event:
            text = data.get("text")
            if (
                self._attempt_open
                and data.get("attempt_id") == self._attempt_id
                and isinstance(text, str)
                and text
            ):
                self._partial_chunks.append(text)
                self._partial_characters += len(text)
                self._preview_tail = (self._preview_tail + text)[-_PREVIEW_CHARACTERS:]
        elif name in {"model.interrupted", "model.failed"} and root_event:
            if self._attempt_open and data.get("attempt_id") == self._attempt_id:
                self._attempt_open = False
                self._attempt_interrupted = True
        elif name == "model.finished":
            attempt_id = data.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                return
            key = (*identity, attempt_id)
            if key in self._responses:
                return
            raw_usage = data.get("usage")
            usage = Usage.model_validate(raw_usage) if raw_usage is not None else None
            for field, count in usage_counts(usage).items():
                setattr(self.metrics, field, getattr(self.metrics, field) + count)
            self._responses.add(key)
            self._pending_models.discard(identity)
            if root_event and attempt_id == self._attempt_id and self._attempt_open:
                # The existing CLI renders the durable result exactly once.
                self._reset_partial()
                if self._live is not None:
                    try:
                        # Rich prints the last rendered Live frame underneath
                        # console messages; clear it before the durable reply.
                        self._live.refresh()
                    except Exception:
                        self._hide()
        elif name == "tool.called" and data.get("accepted", True):
            self.metrics.tool_calls += 1
        elif name == "tool.started" and root_event:
            call = data.get("call", {})
            self._tool = str(call.get("name", "")) if isinstance(call, dict) else ""
        elif name == "tool.result" and root_event:
            self._tool = ""
        elif name == "run.ended":
            self._children.discard(identity)
            self._pending_models.discard(identity)
            if root_event:
                # The root budget already includes every child; replace, never add.
                self.metrics = RunMetrics.model_validate(data["metrics"])
                self._stage = str(data.get("status", "completed"))
                self._children.clear()
                if self._attempt_open and self._partial_chunks:
                    self._attempt_open = False
                    self._attempt_interrupted = self._stage != "completed"
                    if not self._attempt_interrupted:
                        self._reset_partial()
                self._ended = True

    def render(self) -> RenderableType:
        elapsed = (
            self.metrics.elapsed_ms / 1000
            if self._ended
            else max(0.0, time.monotonic() - self._started)
        )
        frame = _FRAMES[int(elapsed * 8) % len(_FRAMES)]
        if self._attempt_open and self._partial_characters:
            stage = f"生成中 · 已接收 {self._partial_characters:,} 字符"
        elif self._children:
            stage = f"子任务 {len(self._children)} 个运行中"
        else:
            stage = _STAGES.get(self._stage, "处理中")
            if self._tool:
                stage += " · " + self._tool
        title = Text.assemble(
            (f"{frame} ", "bold #b69aff"),
            (stage, "#b9cced"),
            (f"  ·  {elapsed:.1f}s", "#8b95a5"),
        )
        usage = Text("  " + usage_summary(self.metrics, pending=not self._ended), style="#aebdce")
        status = [title, usage]
        if self.console.width >= 60:
            title.append("  ·  Ctrl+C 取消", style="#8b95a5")
        else:
            status.append(Text("  Ctrl+C 取消", style="#8b95a5"))
        if not self._preview_tail:
            return Group(*status)

        width = max(1, self.console.width)
        status_height = sum(len(line.wrap(self.console, width)) for line in status)
        preview_height = min(_PREVIEW_LINES, max(0, self.console.height - status_height - 2))
        if not preview_height:
            return Group(*status)
        # Parse only the bounded preview. Once its beginning is trimmed, retain
        # literal text: a missing opening fence could turn code into Markdown.
        # The durable reply always renders the complete Markdown source.
        trimmed = self._partial_characters > len(self._preview_tail)
        body = (
            Text(self._preview_tail, style="#dce5ef")
            if trimmed
            else assistant_markdown(self._preview_tail)
        )
        lines = self.console.render_lines(body, pad=False, new_lines=True)
        truncated = trimmed or len(lines) > preview_height
        label = "● Hiro · 生成预览" + ("（仅显示末尾）" if truncated else "")
        return Group(
            Text(label, style="#63a8f5", no_wrap=True, overflow="ellipsis"),
            Segments(segment for line in lines[-preview_height:] for segment in line),
            Text(""),
            *status,
        )
