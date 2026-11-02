"""Fresh Bot references and bounded projections for persistent Task Runs."""

from hiro_agent.contracts import Message, SessionService
from hiro_agent.errors import ConfigurationError
from hiro_agent.memory import MemoryStore, prepare_context
from hiro_agent.skills import SkillStore


class KernelContext:
    def __init__(
        self,
        bot_id: str,
        current_task: str,
        memory: MemoryStore,
        skills: SkillStore,
        *,
        max_chars: int = 64_000,
        max_messages: int = 64,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (max_chars, max_messages)):
            raise ConfigurationError("Context limits must be positive integers")
        self.bot_id = bot_id
        self.current_task = current_task
        self.memory = memory
        self.skills = skills
        self.max_chars = max_chars
        self.max_messages = max_messages

    def prepare(self, session: SessionService) -> list[Message]:
        messages = session.derive_messages()
        descriptions = self.skills.describe(self.bot_id)
        if descriptions:
            # This projection never becomes a Session message. The executor's
            # live authority remains the only source of tool permission.
            messages = [Message(role="system", content=descriptions), *messages]
        memories = self.memory.list(self.bot_id)
        prepared = prepare_context(
            messages,
            bot_id=self.bot_id,
            current_task=self.current_task,
            memories=memories,
            max_chars=self.max_chars,
            max_messages=self.max_messages,
        )
        if prepared.omitted_memories < len(memories):
            # The pure projection inserts references before the first user turn.
            # Keep the actual Task as the first user turn for existing provider
            # adapters; move only the synthetic reference message, never a unit
            # containing tool declarations/results or original Session history.
            index = next(
                i for i, message in enumerate(prepared.messages) if message.role != "system"
            )
            reference = prepared.messages.pop(index)
            task_index = max(
                i
                for i, message in enumerate(prepared.messages)
                if message.role == "user"
                and not message.continuation
                and message.content == self.current_task
            )
            prepared.messages.insert(task_index + 1, reference)
        return prepared.messages
