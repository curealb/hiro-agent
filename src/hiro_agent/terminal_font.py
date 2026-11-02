"""Change the current macOS Terminal tab's font, matched by its TTY."""

import os
import subprocess
import sys

from hiro_agent.errors import ConfigurationError

FONT_HELP = (
    "用法：/font 16 · /font + · /font - · /font reset\n\n"
    "字号范围为 8–48；reset 恢复本次首次调整前的字号。\n"
    "自动调整支持本机 macOS 自带 Terminal。\n"
    "仅修改当前标签页；首次使用可能需允许自动化控制。\n"
    "其他终端请使用自己的缩放快捷键：\n"
    "macOS 通常为 ⌘+ / ⌘-；\n"
    "Linux / Windows 通常为 Ctrl+Shift+加号 / 减号。"
)

_SCRIPT = """on run argv
    set targetTTY to item 1 of argv
    set action to item 2 of argv
    set originalSize to item 3 of argv as integer
    tell application "/System/Applications/Utilities/Terminal.app"
        repeat with w in windows
            repeat with t in tabs of w
                if tty of t is targetTTY then
                    set oldSize to font size of t
                    if action is "+" then
                        set newSize to oldSize + 1
                    else if action is "-" then
                        set newSize to oldSize - 1
                    else if action is "reset" then
                        if originalSize is 0 then
                            set newSize to oldSize
                        else
                            set newSize to originalSize
                        end if
                    else
                        set newSize to action as integer
                    end if
                    if action is not "reset" then
                        if newSize < 8 then set newSize to 8
                        if newSize > 48 then set newSize to 48
                    end if
                    set font size of t to newSize
                    return (oldSize as text) & " " & (newSize as text)
                end if
            end repeat
        end repeat
    end tell
    error "No matching Terminal tab"
end run
"""


class TerminalFont:
    def __init__(self) -> None:
        self.original_size: int | None = None

    def change(self, argument: str) -> int:
        if argument not in {"+", "-", "reset"} and not (
            len(argument) <= 2
            and argument.isascii()
            and argument.isdecimal()
            and 8 <= int(argument) <= 48
        ):
            raise ConfigurationError("用法：/font 8–48、/font +、/font -、/font reset")
        if (
            sys.platform != "darwin"
            or os.environ.get("TERM_PROGRAM") != "Apple_Terminal"
            or any(os.environ.get(key) for key in ("SSH_CONNECTION", "SSH_TTY", "TMUX", "STY"))
        ):
            raise ConfigurationError("当前终端不支持自动调整字号；输入 /font 查看缩放方法。")
        try:
            tty = os.ttyname(sys.stdin.fileno())
            result = subprocess.run(
                ["/usr/bin/osascript", "-e", _SCRIPT, tty, argument, str(self.original_size or 0)],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            old, new = map(int, result.stdout.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            raise ConfigurationError(
                "字号调整失败；请检查 macOS 的自动化权限，或输入 /font 查看手动缩放方法。"
            ) from None
        if self.original_size is None:
            self.original_size = old
        return new
