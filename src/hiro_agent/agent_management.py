"""Permanent Agent creation through the same Bot authority and approval boundary."""

from hiro_agent.bots import BotAuthority, BotRecord, BotStore, CreateAgentArguments
from hiro_agent.config import BotConfig, Config
from hiro_agent.contracts import DataModel, ToolExecutionContext
from hiro_agent.errors import ConfigurationError, PersistenceError, ToolError
from hiro_agent.tools import ToolRegistry

MANAGER_INSTRUCTIONS = """你是 Agent 管理员，负责按用户要求创建 TUI 左侧长期保留的 Agent。
先使用 list_agents 查看已有 Agent，选择未使用的稳定英文 ID，并为新 Agent 编写明确的职责指令。
使用 create_agent 实际创建；只有工具成功后才能声称创建完成，并告诉用户名称和 ID。
新 Agent 自动继承你的模型、预算和工具配置，不需要用户提供 API Key。
根据任务分配最少必要工具；纯阅读和审查默认使用 list_files、read_file、search_text，
permissions 设为 read-only；需要修改代码时可授予 write_file、edit_file、run_command，
permissions 设为 ask-write。只能授予你已有的工具，写入仍遵守审批流程。
需要 Agent 间通信时，授予 list_agents、send_message、read_message；
send_message 是真实异步消息接口，使用 correlation_id 关联请求和回复。
发送后结束当前阶段，等待 worker 自动运行接收方并回传；不能把已发送当成已收到回复。
核验委派结果时读取自己发出的原消息；下游收发证据在 result.delegated_results，
不要直接读取其他 Agent 发件箱的消息 ID；无法读取时如实报告并改查自己的原委派。
用户要求新 Agent 也能创建 Agent 时，才授予 create_agent。
不能覆盖已有 ID；遇到重名应说明情况并选择新 ID。创建不会自动向新 Agent 提交任务。
不要把永久 Agent 与 delegate_task 创建的临时只读子任务混淆。
"""


class ListAgentsArguments(DataModel):
    pass


def agent_summary(record: BotRecord) -> dict[str, object]:
    """Expose identity and capabilities without model settings or credentials."""
    assert record.config.bot is not None
    return {
        "bot_id": record.bot_id,
        "name": record.config.bot.name,
        "allowed_tools": list(record.authority.allowed_tools),
        "permissions": record.authority.permission_mode,
    }


def manager_config(
    source: Config, *, bot_id: str = "agent-manager", name: str = "Agent 管理员"
) -> Config:
    if source.bot is None:
        raise ConfigurationError("请先保存用于提供模型和权限的源 Bot")
    if source.permissions == "read-only":
        raise ConfigurationError("Agent 管理员需要 ask-write 权限，请选择允许写入的源 Bot")
    config = source.model_copy(deep=True)
    config.bot = BotConfig(
        id=bot_id,
        name=name,
        instructions=MANAGER_INSTRUCTIONS,
        allowed_tools=(
            *source.bot.allowed_tools,
            "list_agents",
            "create_agent",
            "send_message",
            "read_message",
        ),
    )
    return Config.model_validate_json(config.model_dump_json())


def register_agent_tools(registry: ToolRegistry, store: BotStore, bot_id: str) -> None:
    async def list_agents(arguments: ListAgentsArguments) -> list[dict[str, object]]:
        return [agent_summary(record) for record in store.list()]

    async def create_agent(
        arguments: CreateAgentArguments, context: ToolExecutionContext
    ) -> dict[str, object]:
        # Use the authority that authorized this exact invocation, not a new
        # snapshot taken after the executor's last awaited journal append.
        started = next(
            (
                event
                for event in reversed(context.session.events)
                if event.type == "tool.started"
                and event.run_id == context.run_id
                and event.data["call"]["id"] == context.call_id
                and event.data["call"]["name"] == "create_agent"
            ),
            None,
        )
        if started is None or "authority" not in started.data:
            raise PersistenceError("创建 Agent 缺少工具执行授权记录")
        authority = BotAuthority.model_validate(started.data["authority"])
        try:
            record = store.create_from_creator(bot_id, arguments, expected_authority=authority)
        except ConfigurationError as exc:
            raise ToolError(str(exc)) from exc
        return agent_summary(record)

    registry.register(
        "list_agents",
        "列出工作区中长期保留的 Agent 的 ID、名称和工具权限。创建前可用它检查 ID 是否已存在。",
        ListAgentsArguments,
        list_agents,
    )
    registry.register_contextual(
        "create_agent",
        "创建一个在 TUI 左侧长期保留的 Agent，需要审批。"
        "指定 id、name、instructions、allowed_tools，可选 permissions（read-only 或 ask-write）。"
        "模型及运行配置继承当前 Agent；工具只能是当前 Agent 已有权限的子集。"
        "ID 必须未被使用；不会覆盖已有 Agent，也不会自动提交任务。",
        CreateAgentArguments,
        create_agent,
        mutating=True,
    )
