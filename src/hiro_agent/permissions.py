"""Explicit approval for tools that can change local state."""

from typing import Literal

from hiro_agent.contracts import ApprovalCallback


class PermissionPolicy:
    def __init__(
        self,
        mode: Literal["ask-write", "read-only"] = "ask-write",
        callback: ApprovalCallback | None = None,
    ) -> None:
        if mode not in {"ask-write", "read-only"}:
            raise ValueError("Unknown permission mode")
        self.mode = mode
        self.callback = callback

    async def approve(self, tool: str, arguments: dict[str, object], mutating: bool) -> bool:
        if not mutating:
            return True
        if self.mode == "read-only" or self.callback is None:
            return False
        # Only an affirmative response grants this particular invocation.
        return await self.callback(tool, arguments) is True
