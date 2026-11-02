"""Navigation shared by the full-screen workbench and traditional chat."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from hiro_agent.errors import ConfigurationError

NAVIGATION_COMMANDS = {
    "/agent": "选择 Agent，或输入 /agent 名称切换",
    "/tui": "切换到全屏工作台",
    "/chat": "切换到传统聊天",
}


class AgentCompletion(Completion):
    """An explicit /agent navigation selection."""


class AgentTargetCompletion(Completion):
    """Select a communication target; leave the composer open for the message."""


@dataclass(frozen=True)
class AgentRecipient:
    bot_id: str
    name: str


@dataclass(frozen=True)
class AgentAddress:
    targets: tuple[AgentRecipient, ...]
    text: str


def _target_prefix(
    text: str,
    choices: list[tuple[str, str]],
) -> tuple[list[AgentRecipient], str]:
    """Consume adjacent slash targets, leaving the shared message untouched."""
    targets: dict[str, AgentRecipient] = {}
    body = text.strip()
    while body.startswith("/"):
        body = body[1:].lstrip()
        matches = [
            label
            for bot_id, name in choices
            for label in (bot_id, name)
            if body.casefold() == label.casefold()
            or (
                body[: len(label)].casefold() == label.casefold()
                and body[len(label) : len(label) + 1].isspace()
            )
        ]
        if not matches:
            raise ConfigurationError(
                "未知命令或通信对象；用 /Agent1 /Agent2 消息内容，切换对话用 /agent 名称"
            )
        label = max(matches, key=len)
        bot_id = resolve_agent(label, choices)
        targets[bot_id] = AgentRecipient(bot_id, next(n for key, n in choices if key == bot_id))
        body = body[len(label) :].lstrip()
    return list(targets.values()), body


def addressed_agent(
    text: str,
    agents: Iterable[tuple[str, str]],
    commands: Iterable[str],
) -> AgentAddress | None:
    """Parse /ID /name shared message; navigation remains explicit /agent."""
    text = text.strip()
    if not text.startswith("/"):
        return None
    first = text.split(maxsplit=1)[0].casefold()
    if first in commands:
        return None
    targets, payload = _target_prefix(text, list(agents))
    # Explicitly delimit a message beginning with a slash, e.g. /review -- /tmp/file.
    if payload.startswith("-- "):
        payload = payload[3:].lstrip()
    if not payload:
        raise ConfigurationError("请在通信 Agent 后填写消息内容，例如 /通信测试应答方 重新测试")
    return AgentAddress(tuple(targets), payload)


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
                    yield AgentCompletion(
                        bot_id, start_position=-len(query), display=name, display_meta=bot_id
                    )
        elif prefix.startswith("/"):
            choices = list(self.agents())
            offset = prefix.rfind("/")
            if offset > 0 and not prefix[offset - 1].isspace():
                return
            selected: list[AgentRecipient] = []
            if offset > 0:
                try:
                    selected, remainder = _target_prefix(prefix[:offset], choices)
                except ConfigurationError:
                    return
                if remainder:
                    return
            query = prefix[offset + 1 :].lstrip().casefold()
            commands = (
                [
                    Completion(command, start_position=-len(prefix), display_meta=description)
                    for command, description in self.commands.items()
                    if command.startswith(prefix)
                ]
                if offset == 0 and not any(c.isspace() for c in prefix)
                else []
            )
            if query:
                yield from commands
            for bot_id, name in choices:
                if any(target.bot_id == bot_id for target in selected):
                    continue
                if any(query in value.casefold() for value in (bot_id, name)):
                    yield AgentTargetCompletion(
                        f"/{bot_id} ",
                        start_position=offset - len(prefix),
                        display=name,
                        display_meta=f"通信对象 · {bot_id}",
                    )
            if not query:
                yield from commands
