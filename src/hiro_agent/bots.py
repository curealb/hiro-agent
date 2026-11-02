"""Persistent Bot configurations and versioned execution authority."""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from hiro_agent.config import BotConfig, Config
from hiro_agent.contracts import DataModel
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_db import KernelDatabase


class BotAuthority(DataModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bot_id: str
    allowed_tools: tuple[str, ...]
    permission_mode: Literal["ask-write", "read-only"]
    permission_version: int = Field(ge=1)
    permission_digest: str

    def allows(self, tool: str, *, mutating: bool = False) -> bool:
        return tool in self.allowed_tools and not (mutating and self.permission_mode == "read-only")


AuthorityProvider = Callable[[], BotAuthority]


class CreateAgentArguments(BotConfig):
    """Identity and reduced authority for a new persistent Bot."""

    instructions: str = Field(default="", max_length=16000)
    permissions: Literal["ask-write", "read-only"] | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Bot name cannot be blank")
        return value


class BotRecord(DataModel):
    bot_id: str
    config: Config
    authority: BotAuthority
    created_at: datetime
    updated_at: datetime


def permission_digest(config: Config) -> str:
    if config.bot is None:
        raise ConfigurationError("持久 Bot 必须配置稳定 ID")
    policy = {
        "bot_id": config.bot.id,
        "allowed_tools": sorted(set(config.bot.allowed_tools)),
        "permission_mode": config.permissions,
    }
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class BotStore:
    def __init__(self, database: KernelDatabase) -> None:
        self.database = database
        database.migrate(
            "bots",
            1,
            (
                "CREATE TABLE bots (bot_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, "
                "permission_version INTEGER NOT NULL CHECK(permission_version >= 1), "
                "permission_digest TEXT NOT NULL, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)",
            ),
        )

    def save(self, config: Config, *, expected_version: int | None = None) -> BotRecord:
        """Persist settings; every permission change increments even after reverting."""
        # Revalidate mutable Config instances before storing authority.
        config = Config.model_validate_json(config.model_dump_json())
        if config.bot is None:
            raise ConfigurationError("持久 Bot 必须配置稳定 ID")
        digest = permission_digest(config)
        now = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM bots WHERE bot_id=?", (config.bot.id,)
            ).fetchone()
            current = previous["permission_version"] if previous is not None else 0
            if expected_version is not None and expected_version != current:
                raise ConfigurationError("Bot 权限版本已变化，请重新读取配置")
            version = current + (
                1 if previous is None or previous["permission_digest"] != digest else 0
            )
            created = previous["created_at"] if previous is not None else now
            connection.execute(
                "INSERT INTO bots VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(bot_id) DO UPDATE SET config_json=excluded.config_json, "
                "permission_version=excluded.permission_version, "
                "permission_digest=excluded.permission_digest, updated_at=excluded.updated_at",
                (config.bot.id, config.model_dump_json(), version, digest, created, now),
            )
        return self.get(config.bot.id)

    def create_from_creator(
        self,
        creator_id: str,
        arguments: CreateAgentArguments,
        *,
        expected_authority: BotAuthority,
    ) -> BotRecord:
        """Create once, inheriting current settings under the approved authority."""
        arguments = CreateAgentArguments.model_validate_json(arguments.model_dump_json())
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM bots WHERE bot_id=?", (creator_id,)).fetchone()
            creator = self._record_from_row(creator_id, row)
            if creator.authority != expected_authority:
                raise ConfigurationError("创建者 Bot 权限已变化，请重新审批")
            if not creator.authority.allows("create_agent", mutating=True):
                raise ConfigurationError("创建者 Bot 没有 create_agent 写入权限")
            forbidden = set(arguments.allowed_tools) - set(creator.authority.allowed_tools)
            if forbidden:
                raise ConfigurationError(
                    f"新 Bot 的工具不能超出创建者权限: {', '.join(sorted(forbidden))}"
                )
            if creator.config.permissions == "read-only" and arguments.permissions == "ask-write":
                raise ConfigurationError("新 Bot 不能将 read-only 权限提升为 ask-write")
            delegates = "delegate_task" in arguments.allowed_tools
            if delegates and not creator.config.subagents.enabled:
                raise ConfigurationError("创建者尚未启用子代理，不能授予 delegate_task")
            if (
                connection.execute("SELECT 1 FROM bots WHERE bot_id=?", (arguments.id,)).fetchone()
                is not None
            ):
                raise ConfigurationError(f"Bot ID 已存在: {arguments.id}；请选择新的 ID")

            config = creator.config.model_copy(deep=True)
            config.bot = BotConfig.model_validate(arguments.model_dump(exclude={"permissions"}))
            config.permissions = arguments.permissions or creator.config.permissions
            config.subagents.enabled = delegates
            config = Config.model_validate_json(config.model_dump_json())
            digest = permission_digest(config)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO bots VALUES (?, ?, ?, ?, ?, ?)",
                (arguments.id, config.model_dump_json(), 1, digest, now, now),
            )
            row = connection.execute(
                "SELECT * FROM bots WHERE bot_id=?", (arguments.id,)
            ).fetchone()
            record = self._record_from_row(arguments.id, row)
        return record

    def get(self, bot_id: str) -> BotRecord:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
        return self._record_from_row(bot_id, row)

    @staticmethod
    def _record_from_row(bot_id: str, row: sqlite3.Row | None) -> BotRecord:
        if row is None:
            raise ConfigurationError(f"Bot 不存在: {bot_id}")
        config = Config.model_validate_json(row["config_json"])
        if (
            config.bot is None
            or config.bot.id != bot_id
            or permission_digest(config) != row["permission_digest"]
        ):
            raise PersistenceError("Bot 配置与权限摘要不一致")
        return BotRecord(
            bot_id=bot_id,
            config=config,
            authority=BotAuthority(
                bot_id=bot_id,
                allowed_tools=config.bot.allowed_tools,
                permission_mode=config.permissions,
                permission_version=row["permission_version"],
                permission_digest=row["permission_digest"],
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list(self) -> list[BotRecord]:
        with self.database.connect() as connection:
            ids = [row[0] for row in connection.execute("SELECT bot_id FROM bots ORDER BY bot_id")]
        return [self.get(bot_id) for bot_id in ids]

    def authority(self, bot_id: str) -> BotAuthority:
        return self.get(bot_id).authority

    def provider(self, bot_id: str) -> AuthorityProvider:
        return lambda: self.authority(bot_id)
