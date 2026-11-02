"""Direct Agent request/reply tools over the durable handoff transport."""

from pydantic import Field

from hiro_agent.bots import BotStore
from hiro_agent.contracts import DataModel
from hiro_agent.errors import ConfigurationError, ToolError
from hiro_agent.handoffs import HandoffStore
from hiro_agent.tools import ToolRegistry

COMMUNICATION_TOOLS = ("list_agents", "send_message", "read_message")


class SendMessageArguments(DataModel):
    target_agent_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=200)
    payload: str = Field(min_length=1, max_length=32000)
    idempotency_key: str = Field(min_length=1, max_length=200)


class ReadMessageArguments(DataModel):
    message_id: str = Field(min_length=1, max_length=128)


def enable_communication(store: BotStore, bot_id: str) -> None:
    """Explicitly grant messaging to an existing identity, preserving its settings."""
    record = store.get(bot_id)
    config = record.config.model_copy(deep=True)
    assert config.bot is not None
    config.bot.allowed_tools = (*config.bot.allowed_tools, *COMMUNICATION_TOOLS)
    store.save(config, expected_version=record.authority.permission_version)


def register_communication_tools(
    registry: ToolRegistry, store: HandoffStore, bot_id: str, task_id: str
) -> None:
    async def send_message(arguments: SendMessageArguments) -> dict[str, object]:
        try:
            message = store.send_message(
                bot_id,
                task_id,
                arguments.target_agent_id,
                arguments.payload,
                correlation_id=arguments.correlation_id,
                idempotency_key=arguments.idempotency_key,
            )
        except ConfigurationError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "message_id": message.id,
            "source_agent_id": message.source_bot_id,
            "target_agent_id": message.target_bot_id,
            "correlation_id": message.correlation_id,
            "status": "accepted",
            "reply_pending": True,
        }

    async def read_message(arguments: ReadMessageArguments) -> dict[str, object]:
        try:
            message = store.get(bot_id, arguments.message_id)
            result = store.result(bot_id, arguments.message_id)
        except ConfigurationError:
            return {
                "message_id": arguments.message_id,
                "status": "unavailable",
                "error": "message_not_accessible",
                "message": "该消息不存在或不属于你的发件箱。请读取你自己发送的消息 ID；"
                "委派的最终报告和下游证据通过原消息的 result.delegated_results 回传。"
                "不能直接读取其他 Agent 的消息，也不要重试同一无权访问的 ID。",
            }
        return {
            "message_id": message.id,
            "source_agent_id": message.source_bot_id,
            "target_agent_id": message.target_bot_id,
            "correlation_id": message.correlation_id,
            "pending": result is None,
            "result": result.model_dump(mode="json") if result is not None else None,
        }

    registry.register(
        "send_message",
        "通过真实持久消息接口直接联系另一个 Agent。使用 list_agents 查询目标 ID；"
        "提供 correlation_id 和消息 payload，重试相同请求须复用 idempotency_key。"
        "成功只表示已接受发送，不代表已收到回复。发送后结束当前阶段，"
        "worker 将运行接收 Agent 并自动唤醒你处理回复；不要在当前任务循环等待。"
        "接收方若继续委派，系统会等待下游核验完成后，再回传最终报告和相关证据。",
        SendMessageArguments,
        send_message,
    )
    registry.register(
        "read_message",
        "用 message_id 读取自己发出的消息及真实接收 Agent 的持久回复。"
        "pending=true 表示尚未返回；返回后核对回复方、状态和 output 正文。"
        "再委派的证据在 result.delegated_results，无需读取其中的其他 Agent 消息 ID。"
        "status=unavailable 表示该 ID 不可读取，请改读自己的原消息或报告无法核验。"
        "顶层 correlation_id 来自原请求，须另行核对回复正文中的标识及 payload。",
        ReadMessageArguments,
        read_message,
    )
