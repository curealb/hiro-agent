"""Font changes target one Terminal TTY and never execute user-supplied code."""

import subprocess
from types import SimpleNamespace

import pytest

from hiro_agent.errors import ConfigurationError
from hiro_agent.terminal_font import TerminalFont


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr("hiro_agent.terminal_font.sys.platform", "darwin")
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    for key in ("SSH_CONNECTION", "SSH_TTY", "TMUX", "STY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hiro_agent.terminal_font.sys.stdin", SimpleNamespace(fileno=lambda: 0))
    monkeypatch.setattr("hiro_agent.terminal_font.os.ttyname", lambda _: "/dev/ttys123")
    return TerminalFont()


def test_font_changes_and_reset_target_only_the_current_tty(terminal, monkeypatch) -> None:
    calls = []
    replies = iter(["14 16", "16 17", "17 14"])

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=next(replies))

    monkeypatch.setattr("hiro_agent.terminal_font.subprocess.run", run)
    assert terminal.change("16") == 16
    assert terminal.change("+") == 17
    assert terminal.change("reset") == 14
    assert terminal.original_size == 14
    assert [args[-3:] for args, _ in calls] == [
        ["/dev/ttys123", "16", "0"],
        ["/dev/ttys123", "+", "14"],
        ["/dev/ttys123", "reset", "14"],
    ]
    assert all(kwargs.get("shell", False) is False for _, kwargs in calls)


@pytest.mark.parametrize("argument", ["7", "49", "1.5", "-2", "hello", "16; exit", "９９"])
def test_invalid_font_never_invokes_terminal(terminal, monkeypatch, argument: str) -> None:
    monkeypatch.setattr("hiro_agent.terminal_font.subprocess.run", lambda *a, **k: pytest.fail())
    with pytest.raises(ConfigurationError, match="用法"):
        terminal.change(argument)


@pytest.mark.parametrize("variable", ["TERM_PROGRAM", "TMUX", "SSH_CONNECTION"])
def test_unsupported_terminal_has_a_manual_fallback(terminal, monkeypatch, variable: str) -> None:
    monkeypatch.setenv(variable, "unsupported")
    monkeypatch.setattr("hiro_agent.terminal_font.subprocess.run", lambda *a, **k: pytest.fail())
    with pytest.raises(ConfigurationError, match="/font"):
        terminal.change("16")


def test_terminal_failure_does_not_record_a_false_font_size(terminal, monkeypatch) -> None:
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("osascript", 10)

    monkeypatch.setattr("hiro_agent.terminal_font.subprocess.run", run)
    with pytest.raises(ConfigurationError, match="字号调整失败"):
        terminal.change("16")
    assert terminal.original_size is None
