"""Assistant replies render Markdown in terminals and retain source in pipes."""

import io

import pytest
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from hiro_agent.presentation import show_assistant_message


def render_message(text, *, terminal=True, width=100, prefix=""):
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=terminal,
        color_system="truecolor" if terminal else None,
        width=width,
        record=True,
        _environ={"TERM": "xterm"},
    )
    show_assistant_message(console, text, prefix=prefix)
    return console, output, console.export_text(clear=False)


def test_terminal_renders_markdown_and_preserves_literal_code_and_rich_brackets():
    source = (
        "### 工具层\n\n"
        "使用 **只读工具** 和 `read_file`。\n\n"
        "- 读取文件\n- 搜索文本\n\n"
        "[bold]这是模型原文[/bold]\n\n"
        "```python\nprint('**literal** [bold]')\n```"
    )
    console, output, rendered = render_message(source)

    assert "● Hiro" in rendered
    assert "工具层" in rendered and "### 工具层" not in rendered
    assert "只读工具" in rendered and "**只读工具**" not in rendered
    assert "read_file" in rendered and "`read_file`" not in rendered
    assert "读取文件" in rendered and "搜索文本" in rendered
    assert "- 读取文件" not in rendered
    assert "[bold]这是模型原文[/bold]" in rendered
    assert "print('**literal** [bold]')" in rendered
    assert "```" not in rendered
    assert "font-weight: bold" in console.export_html(inline_styles=True)
    assert "\x1b[" in output.getvalue()


@pytest.mark.parametrize("width", [32, 100])
def test_terminal_formats_chinese_tables_within_available_width(width):
    _, _, rendered = render_message(
        "| 工具 | 用途 |\n"
        "| --- | --- |\n"
        "| `read_file` | 读取中文文件内容 |\n"
        "| `write_file` | 创建文件 |\n"
        "| `multi_tool_use.parallel` | |\n"
        "| | 并行调用互不依赖的工具，保持全部中文描述 |",
        width=width,
    )

    assert "工具" in rendered and "用途" in rendered
    assert "read_file" in rendered and "write_file" in rendered
    assert "创建文件" in rendered
    # Empty adjacent cells let wrapped values be joined independently of the
    # table's column widths without depending on its exact border characters.
    compact = "".join(rendered.split())
    assert "multi_tool_use.parallel" in compact
    assert "并行调用互不依赖的工具，保持全部中文描述" in compact
    assert "…" not in rendered
    assert "| --- |" not in rendered and "`read_file`" not in rendered
    assert all(get_cwidth(line) <= width for line in rendered.splitlines())


def test_child_prefix_does_not_prevent_first_markdown_heading_from_rendering():
    prefix = "[任务 task-1 · explorer · 子会话 child-1] "
    _, _, rendered = render_message("### 检查结果\n\n**通过**", prefix=prefix)

    assert prefix.strip() in rendered
    assert "检查结果" in rendered and "### 检查结果" not in rendered
    assert "通过" in rendered and "**通过**" not in rendered


def test_redirected_message_keeps_markdown_source_and_prefix_without_ansi():
    source = (
        "### 工具\n\n**只读** `read_file` [bold]原文[/bold]\n\n"
        "| 工具 | 用途 |\n| --- | --- |\n| read_file | 读取 |\n\n"
        "```python\nprint('hello')\n```"
    )
    prefix = "[任务 task-1 · explorer · 子会话 child-1] "
    _, output, _ = render_message(source, terminal=False, width=200, prefix=prefix)

    assert output.getvalue() == prefix + source + "\n"
    assert "\x1b" not in output.getvalue()
