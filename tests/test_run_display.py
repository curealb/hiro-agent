"""Confirmed usage, shared child events, and transient rendering lifecycle."""

import asyncio
import io
from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from hiro_agent import run_display
from hiro_agent.contracts import RunMetrics
from hiro_agent.run_display import RunDisplay, usage_summary


def event(*, child: str | None = None, **data):
    return {
        "session_id": f"session-{child}" if child else "root-session",
        "run_id": child or "root-run",
        "root_session_id": "root-session",
        "root_run_id": "root-run",
        "parent_run_id": "root-run" if child else None,
        "role": "explorer" if child else "main",
        **data,
    }


def make_display(*, terminal=False, width=100):
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=terminal,
        color_system="truecolor" if terminal else None,
        width=width,
        _environ={"TERM": "xterm"},
    )
    display = RunDisplay(console)
    display.start()
    display.observe("run.state", event(status="preparing"))
    return display, output


def render_text(display):
    output = io.StringIO()
    Console(file=output, width=display.console.width).print(display.render())
    return output.getvalue()


def finish_response(display, attempt_id, *, child=None, usage=None):
    display.observe("model.finished", event(child=child, attempt_id=attempt_id, usage=usage))


def request_response(display, attempt_id, *, child=None):
    display.observe("run.state", event(child=child, status="requesting_model"))
    display.observe("model.requested", event(child=child, attempt_id=attempt_id))


def send_delta(display, attempt_id, text, *, child=None):
    display.observe("assistant.delta", event(child=child, attempt_id=attempt_id, text=text))


def test_missing_usage_cache_and_zero_hits_are_distinct():
    display, output = make_display()
    assert "Token 等待返回" in render_text(display)
    assert "缓存 等待返回" in render_text(display)
    finish_response(display, "missing")
    assert "Token 未提供" in render_text(display)
    assert "缓存 未提供" in render_text(display)

    finish_response(display, "known-tokens", usage={"input_tokens": 10, "output_tokens": 5})
    assert "Token 15" in render_text(display)
    assert "输出 5（部分）" in render_text(display)
    assert "缓存 未提供" in render_text(display)

    finish_response(
        display,
        "zero-cache",
        usage={"input_tokens": 100, "output_tokens": 8, "cached_input_tokens": 0},
    )
    assert "缓存命中 0 / 100 (0.0%)（部分）" in usage_summary(display.metrics)
    assert output.getvalue() == ""
    display.stop()


def test_response_cache_is_subset_and_percentage_uses_only_reported_inputs():
    display, _ = make_display()
    finish_response(
        display,
        "known-cache",
        usage={"input_tokens": 100, "output_tokens": 8, "cached_input_tokens": 75},
    )
    finish_response(display, "unknown-cache", usage={"input_tokens": 900, "output_tokens": 2})
    summary = usage_summary(display.metrics)
    assert "Token 1,010" in summary
    assert "缓存命中 75 / 100 (75.0%)（部分）" in summary
    display.stop()


def test_child_events_are_aggregated_once_and_parent_end_replaces_totals():
    display, _ = make_display()
    display.observe("run.state", event(status="requesting_model"))
    finish_response(
        display,
        "root-response",
        usage={"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 5},
    )
    for child in ("first", "second"):
        display.observe("run.state", event(child=child, status="requesting_model"))
    assert "子任务 2 个运行中" in render_text(display)
    data = event(
        child="first",
        attempt_id="child-response",
        usage={"input_tokens": 30, "output_tokens": 4, "cached_input_tokens": 10},
    )
    display.observe("model.finished", data)
    display.observe("model.finished", data)
    assert display.metrics.worker_steps == 2
    assert display.metrics.input_tokens == 40
    assert display.metrics.output_tokens == 6
    assert display.metrics.cached_input_tokens == 15
    display.observe(
        "run.ended", event(child="first", status="completed", metrics={"input_tokens": 30})
    )
    assert display.metrics.input_tokens == 40
    assert "子任务 1 个运行中" in render_text(display)

    final = RunMetrics(
        model_requests=3,
        worker_steps=3,
        input_tokens=60,
        output_tokens=9,
        cached_input_tokens=25,
        cache_reported_input_tokens=60,
        cache_usage_missing=0,
        elapsed_ms=2300,
    )
    display.observe("run.ended", event(status="completed", metrics=final.model_dump()))
    assert display.metrics == final
    assert "已完成" in render_text(display)
    assert "2.3s" in render_text(display)
    display.observe("model.finished", data)
    assert display.metrics == final
    display.stop()


def test_stale_child_events_do_not_change_root_stage_or_usage():
    display, _ = make_display()
    display.observe("run.state", event(status="requesting_model"))
    display.observe(
        "model.finished",
        event(
            child="stale",
            root_run_id="previous-run",
            attempt_id="a",
            usage={"input_tokens": 99, "output_tokens": 2},
        ),
    )
    assert display.metrics.worker_steps == 0
    assert "思考中" in render_text(display)
    display.observe("run.state", event(status="executing_tools"))
    display.observe("tool.started", event(call={"name": "read_file"}))
    assert "执行工具 · read_file" in render_text(display)
    display.observe("tool.result", event(call_id="read"))
    assert "read_file" not in render_text(display)
    display.stop()


def test_zero_input_cache_rate_and_interrupted_summary():
    zero = RunMetrics(worker_steps=1, cache_usage_missing=0)
    assert "缓存命中 0 / 0 (—)" in usage_summary(zero)
    interrupted = RunMetrics(
        model_requests=2,
        worker_steps=1,
        input_tokens=100,
        output_tokens=10,
        cached_input_tokens=50,
        cache_reported_input_tokens=100,
        cache_usage_missing=0,
    )
    summary = usage_summary(interrupted)
    assert "输出 10（部分）" in summary
    assert "(50.0%)（部分）" in summary
    assert "未提供" in usage_summary(RunMetrics(model_requests=1))


def test_animation_and_elapsed_change_without_inventing_tokens(monkeypatch):
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(run_display, "time", SimpleNamespace(monotonic=lambda: clock.value))
    display, _ = make_display()
    first = render_text(display)
    clock.value += 0.25
    second = render_text(display)
    assert first[0] != second[0]
    assert "0.0s" in first and "0.2s" in second
    assert display.metrics.worker_steps == 0
    assert "Token 等待返回" in first and "Token 等待返回" in second
    display.stop()


def test_narrow_terminal_wraps_usage_and_preserves_cancellation_hint():
    display, _ = make_display(width=30)
    finish_response(
        display, "a", usage={"input_tokens": 1000, "output_tokens": 20, "cached_input_tokens": 750}
    )
    text = render_text(display)
    assert all(get_cwidth(line) <= 30 for line in text.splitlines())
    assert "Ctrl+C 取消" in text
    assert "750" in text and "75.0%" in text
    display.stop()


@pytest.mark.parametrize("terminal,term", [(False, "xterm"), (True, "dumb")])
async def test_noninteractive_output_never_starts_animation(terminal, term):
    output = io.StringIO()
    console = Console(file=output, force_terminal=terminal, _environ={"TERM": term})
    display = RunDisplay(console)
    display.start()
    assert display._refresh_task is None
    assert not console._live_stack
    with display.suspend():
        pass
    display.stop()
    assert output.getvalue() == ""


async def test_live_refreshes_and_nested_approval_suspends_then_resumes(monkeypatch):
    display, output = make_display(terminal=True)
    task = display._refresh_task
    assert task is not None
    assert display.console._live_stack
    rendered = asyncio.Event()
    original = display.render

    def record_render():
        rendered.set()
        return original()

    # The live callable is captured at construction; observe its next refresh.
    live = display._live
    assert live is not None
    monkeypatch.setattr(live, "_get_renderable", record_render)
    await asyncio.wait_for(rendered.wait(), 2)
    assert output.getvalue()
    with pytest.raises(RuntimeError, match="approval failed"):
        with display.suspend():
            assert not display.console._live_stack
            with display.suspend():
                assert not display.console._live_stack
            assert not display.console._live_stack
            raise RuntimeError("approval failed")
    assert display.console._live_stack
    assert display._refresh_task is not task
    display.stop()
    await asyncio.sleep(0)
    assert not display.console._live_stack
    assert task.done()
    assert display._refresh_task is None


async def test_stop_while_suspended_does_not_restart_live_and_new_run_resets():
    display, _ = make_display(terminal=True)
    finish_response(display, "a", usage={"input_tokens": 100, "output_tokens": 10})
    with display.suspend():
        display.stop()
    assert not display.console._live_stack
    assert display._refresh_task is None
    display.start()
    assert display.metrics.worker_steps == display.metrics.input_tokens == 0
    assert "等待返回" in render_text(display)
    display.stop()
    await asyncio.sleep(0)


def test_root_stream_renders_progressively_and_clears_before_durable_output():
    display, output = make_display()
    request_response(display, "a")
    send_delta(display, "a", "开始分析 [bold]源码[/bold]。\n")
    first = render_text(display)
    assert "开始分析 [bold]源码[/bold]。" in first
    assert first.index("开始分析") < first.index("生成中")
    assert "Token 等待返回" in first
    assert display.metrics.input_tokens == display.metrics.output_tokens == 0
    assert not display.interrupted_output

    send_delta(display, "a", "这是第二段内容。")
    second = render_text(display)
    assert "开始分析" in second and "这是第二段内容。" in second
    assert "字符" in second
    assert display.metrics.worker_steps == 0
    finish_response(
        display, "a", usage={"input_tokens": 100, "output_tokens": 12, "cached_input_tokens": 20}
    )
    final = render_text(display)
    assert "开始分析" not in final and "生成预览" not in final
    assert "Token 112" in final
    assert not display.interrupted_output
    # Deltas are only a transient TTY preview, never extra pipe/log output.
    assert output.getvalue() == ""
    display.stop()


def test_child_streams_and_unrelated_attempts_never_enter_root_preview():
    display, _ = make_display()
    request_response(display, "root-a")
    send_delta(display, "root-a", "父任务内容")
    request_response(display, "child-a", child="first")
    send_delta(display, "child-a", "子任务敏感中间内容", child="first")
    send_delta(display, "different", "不属于当前请求")
    send_delta(display, "root-a", "旧会话文本", child="unrelated")
    text = render_text(display)
    assert "父任务内容" in text
    assert "子任务敏感中间内容" not in text
    assert "不属于当前请求" not in text and "旧会话文本" not in text
    display.observe("model.interrupted", event(child="first", attempt_id="child-a"))
    assert not display.interrupted_output
    assert "父任务内容" in render_text(display)
    finish_response(
        display, "child-a", child="first", usage={"input_tokens": 5, "output_tokens": 2}
    )
    assert "父任务内容" in render_text(display)
    assert display.metrics.input_tokens == 5
    display.stop()


@pytest.mark.parametrize(
    "failure,status", [("model.failed", "failed"), ("model.interrupted", "canceled")]
)
def test_interrupted_root_output_survives_stop_without_late_delta_contamination(failure, status):
    display, _ = make_display()
    request_response(display, "a")
    for chunk in ("第一段\n", "中文 [literal]", " tail"):
        send_delta(display, "a", chunk)
    display.observe(failure, event(attempt_id="a"))
    send_delta(display, "a", "晚到的片段")
    assert display.interrupted_output == "第一段\n中文 [literal] tail"
    display.observe(
        "run.ended", event(status=status, metrics=RunMetrics(model_requests=1).model_dump())
    )
    display.stop()
    assert display.interrupted_output == "第一段\n中文 [literal] tail"
    assert display.metrics.worker_steps == 0
    display.start()
    assert not display.interrupted_output
    assert "第一段" not in render_text(display)
    display.stop()


def test_next_attempt_resets_preview_and_ignores_old_requests_and_chunks():
    display, _ = make_display()
    request_response(display, "first")
    send_delta(display, "first", "旧尝试内容")
    display.observe("model.failed", event(attempt_id="first"))
    assert display.interrupted_output == "旧尝试内容"

    request_response(display, "second")
    assert not display.interrupted_output
    assert "旧尝试内容" not in render_text(display)
    send_delta(display, "second", "新尝试内容")
    display.observe("model.requested", event(attempt_id="first"))
    send_delta(display, "first", "旧尝试晚到内容")
    finish_response(display, "first", usage=None)
    assert "新尝试内容" in render_text(display)
    assert "旧尝试" not in render_text(display)
    finish_response(display, "second", usage=None)
    send_delta(display, "second", "完成之后的片段")
    assert "完成之后" not in render_text(display)
    assert "新尝试内容" not in render_text(display)
    display.stop()


def test_failed_run_preserves_partial_even_without_request_failure_event():
    display, _ = make_display()
    request_response(display, "a")
    send_delta(display, "a", "持久化错误前已收到的文本")
    display.observe(
        "run.ended",
        event(status="failed", metrics=RunMetrics(model_requests=1).model_dump()),
    )
    display.stop()
    assert display.interrupted_output == "持久化错误前已收到的文本"


@pytest.mark.parametrize("width,height", [(100, 25), (30, 16)])
def test_long_stream_preview_is_bounded_but_keeps_complete_interrupted_text(width, height):
    display, _ = make_display(width=width)
    display.console.height = height
    request_response(display, "large")
    chunks = [f"中文内容第 {index} 行：" + "内容" * 25 + "\n" for index in range(200)]
    for chunk in chunks:
        send_delta(display, "large", chunk)
    rendered = render_text(display)
    assert "第 0 行" not in rendered
    assert "仅显示末尾" in rendered
    assert "生成中" in rendered and "Ctrl+C 取消" in rendered
    assert len(rendered.splitlines()) <= height
    assert all(get_cwidth(line) <= width for line in rendered.splitlines())
    assert "Token 等待返回" in rendered
    assert len(rendered) < 5000
    display.observe("model.interrupted", event(attempt_id="large"))
    display.stop()
    assert display.interrupted_output == "".join(chunks)


async def test_live_refresh_shows_delta_before_response_finishes(monkeypatch):
    display, output = make_display(terminal=True)
    request_response(display, "stream")
    rendered = asyncio.Event()
    live = display._live
    assert live is not None
    original = display.render

    def record_render():
        rendered.set()
        return original()

    monkeypatch.setattr(live, "_get_renderable", record_render)
    try:
        send_delta(display, "stream", "正在逐步生成的回答")
        await asyncio.wait_for(rendered.wait(), 2)
        assert "正在逐步生成的回答" in output.getvalue()
        assert display.metrics.worker_steps == 0
        assert not display.interrupted_output
        output.seek(0)
        output.truncate()
        finish_response(display, "stream", usage={"input_tokens": 10, "output_tokens": 4})
        display.console.print("正在逐步生成的回答")
        assert output.getvalue().count("正在逐步生成的回答") == 1
    finally:
        display.stop()
        await asyncio.sleep(0)


def test_stream_preview_renders_markdown_across_delta_boundaries():
    display, output = make_display(terminal=True)
    request_response(display, "markdown")
    chunks = [
        "### 工具\n\n**只",
        "读**：使用 `read_",
        "file`。\n\n| 工具 | 用途 |\n| --- | --- |\n",
        "| `read_file` | 读取文件 |\n",
    ]
    try:
        for chunk in chunks:
            send_delta(display, "markdown", chunk)
            # Incomplete delimiters must be safe to render between chunks.
            render_text(display)
        rendered = render_text(display)
        assert "工具" in rendered and "只读" in rendered
        assert "read_file" in rendered and "读取文件" in rendered
        assert "###" not in rendered and "**" not in rendered
        assert "`" not in rendered and "| --- |" not in rendered
        assert "生成中" in rendered
        assert output.getvalue() == ""
        finish_response(display, "markdown")
        assert "read_file" not in render_text(display)
        assert not display.interrupted_output
    finally:
        display.stop()


def test_stream_preview_keeps_code_literal_before_and_after_closing_fence():
    display, _ = make_display(terminal=True)
    request_response(display, "code")
    try:
        send_delta(display, "code", "```python\nprint('**literal** [bold]')\n")
        unfinished = render_text(display)
        assert "print('**literal** [bold]')" in unfinished
        assert "```" not in unfinished
        send_delta(display, "code", "```\n\n**完成**")
        finished = render_text(display)
        assert "print('**literal** [bold]')" in finished
        assert "完成" in finished and "**完成**" not in finished
        assert "```" not in finished
    finally:
        display.stop()


@pytest.mark.parametrize("width,height", [(100, 25), (30, 16)])
def test_markdown_table_preview_fits_terminal_and_keeps_latest_rows(width, height):
    display, _ = make_display(terminal=True, width=width)
    display.console.height = height
    request_response(display, "table")
    source = "| 工具 | 用途 |\n| --- | --- |\n" + "".join(
        f"| item_{index} | 读取中文内容 |\n" for index in range(20)
    )
    try:
        send_delta(display, "table", source)
        rendered = render_text(display)
        assert "item_19" in rendered and "item_0" not in rendered
        assert "仅显示末尾" in rendered
        assert "生成中" in rendered and "Ctrl+C 取消" in rendered
        assert len(rendered.splitlines()) <= height
        assert all(get_cwidth(line) <= width for line in rendered.splitlines())
        display.observe("model.interrupted", event(attempt_id="table"))
        assert display.interrupted_output == source
    finally:
        display.stop()


def test_preview_with_discarded_code_fence_keeps_code_literal_and_recovers_full_source():
    display, _ = make_display(terminal=True)
    request_response(display, "long-code")
    source = "```text\n" + "code line\n" * 500 + "### literal\n**literal** [bold]code[/bold]"
    try:
        send_delta(display, "long-code", source)
        rendered = render_text(display)
        assert "仅显示末尾" in rendered
        assert "### literal" in rendered
        assert "**literal** [bold]code[/bold]" in rendered
        display.observe("model.interrupted", event(attempt_id="long-code"))
        display.stop()
        assert display.interrupted_output == source
    finally:
        display.stop()
