"""Persistent Bot configurations and versioned execution authority."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field

from hiro_agent.config import Config
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

    def get(self, bot_id: str) -> BotRecord:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
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
