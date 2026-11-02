"""Create the standard persistent Agents without replacing saved configuration."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from hiro_agent.agent_communication import COMMUNICATION_TOOLS
from hiro_agent.agent_management import manager_config
from hiro_agent.bots import BotStore, permission_digest
from hiro_agent.config import BotConfig, Config, resolve_model_environment
from hiro_agent.errors import ConfigurationError

DEFAULT_AGENT_IDS = ("default", "agent-manager", "code-writer")
STANDARD_TOOLS = (
    "list_files",
    "read_file",
    "search_text",
    "write_file",
    "edit_file",
    "run_command",
    *COMMUNICATION_TOOLS,
)
WRITER_INSTRUCTIONS = """你是代码编写员，负责根据用户需求检查、实现和验证代码变更。
先阅读相关代码和项目说明，确认现有行为与需求，再实施范围明确的修改。
遵守项目约定；写入文件、编辑代码和运行命令必须遵守工具权限与审批流程。
修改后运行适当的测试和检查，验证结果；遇到失败应查明原因并修复。
完成后说明修改内容、验证结果和测试情况；未运行的测试或仍存在的问题必须如实报告。
"""


def _standard_config(source: Config, bot_id: str, name: str, instructions: str = "") -> Config:
    config = source.model_copy(deep=True)
    tools = STANDARD_TOOLS + (("delegate_task",) if source.subagents.enabled else ())
    config.bot = BotConfig(id=bot_id, name=name, instructions=instructions, allowed_tools=tools)
    return Config.model_validate_json(config.model_dump_json())


def initialize_agents(
    store: BotStore, source: Config | None = None, *, workspace: Path | None = None
) -> dict[str, list[str]]:
    """Atomically create missing presets using one consistent saved-config snapshot."""
    with store.database.transaction() as connection:
        rows = {
            row["bot_id"]: row for row in connection.execute("SELECT * FROM bots ORDER BY bot_id")
        }
        missing = [bot_id for bot_id in DEFAULT_AGENT_IDS if bot_id not in rows]
        result = {
            "created": missing,
            "existing": [bot_id for bot_id in DEFAULT_AGENT_IDS if bot_id in rows],
        }
        if not missing:
            return result

        if source is None and rows:
            source_id = next(
                (bot_id for bot_id in ("default", "agent-manager") if bot_id in rows),
                next(iter(rows)),
            )
            source = store._record_from_row(source_id, rows[source_id]).config
        if workspace is not None:
            source = resolve_model_environment(source or Config(), workspace=workspace)
        if source is None:
            raise ConfigurationError(
                "初始化 Agent 需要模型配置：请使用 --config 指定配置文件，"
                "或设置 HIRO_MODEL 等模型环境变量。"
            )
        try:
            source = Config.model_validate_json(source.model_dump_json())
        except ValidationError:
            raise ConfigurationError("初始化 Agent 的源配置字段不合法，请检查 --config。") from None
        if source.model.adapter == "compatible" and not source.model.model.strip():
            raise ConfigurationError(
                "初始化 Agent 需要模型名称：请在 .env 中设置 HIRO_MODEL，"
                "或在 --config 中填写 model.model（也可导出 HIRO_MODEL）。"
            )

        default = _standard_config(source, "default", "Hiro")
        configs: dict[str, Config] = {"default": default}
        if "agent-manager" in missing:
            configs["agent-manager"] = manager_config(default)
        if "code-writer" in missing:
            configs["code-writer"] = _standard_config(
                source, "code-writer", "代码编写员", WRITER_INSTRUCTIONS
            )

        # Materialize every validated row before the first insert. The write lock
        # also prevents concurrent initializers from racing to insert these IDs.
        now = datetime.now(UTC).isoformat()
        values = [
            (
                bot_id,
                configs[bot_id].model_dump_json(),
                1,
                permission_digest(configs[bot_id]),
                now,
                now,
            )
            for bot_id in missing
        ]
        connection.executemany(
            "INSERT INTO bots (bot_id, config_json, permission_version, permission_digest, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            values,
        )
    return result
