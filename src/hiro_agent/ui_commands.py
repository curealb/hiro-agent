"""Navigation shared by the full-screen workbench and traditional chat."""

from collections.abc import Callable, Iterable, Mapping

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from hiro_agent.errors import ConfigurationError

NAVIGATION_COMMANDS = {
    "/agent": "切换 Agent：/agent ID 或名称",
    "/tui": "切换到全屏工作台",
    "/chat": "切换到传统聊天",
}


def resolve_agent(query: str, agents: Iterable[tuple[str, str]]) -> str:
    """Prefer an exact ID; names must identify precisely one persisted Bot."""
    choices = list(agents)
    for bot_id, _ in choices:
        if query == bot_id:
            return bot_id
    matches = {
        bot_id
        for bot_id, name in choices
        if query.casefold() in {bot_id.casefold(), name.casefold()}
    }
    if len(matches) == 1:
        return matches.pop()
    if matches:
        raise ConfigurationError("Agent 名称不唯一，请使用完整 ID：" + "、".join(sorted(matches)))
    raise ConfigurationError(f"找不到 Agent：{query}；输入 /agent 查看可用 Agent")


class AgentCommandCompleter(Completer):
    def __init__(
        self,
        commands: Mapping[str, str],
        agents: Callable[[], Iterable[tuple[str, str]]],
    ) -> None:
        self.commands = commands
        self.agents = agents

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        if document.cursor_position != len(document.text) or "\n" in document.text:
            return
        prefix = document.text.lstrip()
        if prefix.startswith("/agent "):
            query = prefix[len("/agent ") :]
            for bot_id, name in self.agents():
                if any(value.casefold().startswith(query.casefold()) for value in (bot_id, name)):
                    yield Completion(bot_id, start_position=-len(query), display_meta=name)
        elif prefix.startswith("/") and not any(c.isspace() for c in prefix):
            for command, description in self.commands.items():
                if command.startswith(prefix):
                    yield Completion(command, start_position=-len(prefix), display_meta=description)
