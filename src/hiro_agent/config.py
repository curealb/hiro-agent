"""JSON configuration and local environment defaults for application entry points."""

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator

from hiro_agent.contracts import DataModel
from hiro_agent.errors import ConfigurationError


class ModelConfig(DataModel):
    adapter: Literal["compatible", "mock"] = "compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "HIRO_API_KEY"
    request_timeout_seconds: float = Field(default=120, gt=0)
    stream: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_body")
    @classmethod
    def validate_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"messages", "tools", "stream", "stream_options", "model", "n", "api_key"}
        if forbidden.intersection(value):
            raise ValueError("extra_body cannot override protocol or credential fields")
        return value


class BudgetConfig(DataModel):
    max_model_requests: int = Field(default=64, ge=1)
    max_tool_calls: int = Field(default=128, ge=1)
    timeout_seconds: float = Field(default=600, gt=0)


class ToolsConfig(DataModel):
    max_output_chars: int = Field(default=32768, ge=256)
    max_file_bytes: int = Field(default=1_048_576, ge=256)
    max_search_files: int = Field(default=2000, ge=1)
    command_timeout_seconds: float = Field(default=60, gt=0)


class SubagentRoleConfig(DataModel):
    model: ModelConfig | None = None
    instructions: str = ""


class SubagentRolesConfig(DataModel):
    explorer: SubagentRoleConfig = Field(default_factory=SubagentRoleConfig)
    reviewer: SubagentRoleConfig = Field(default_factory=SubagentRoleConfig)


class SubagentsConfig(DataModel):
    enabled: bool = False
    max_concurrency: int = Field(default=2, ge=1, le=8)
    max_model_requests: int = Field(default=16, ge=1)
    max_tool_calls: int = Field(default=32, ge=1)
    max_output_chars: int = Field(default=16000, ge=256, le=131072)
    roles: SubagentRolesConfig = Field(default_factory=SubagentRolesConfig)


class BotConfig(DataModel):
    """An explicit identity; reading the same config never invents a new Bot."""

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    name: str = Field(default="Bot", min_length=1, max_length=200)
    instructions: str = ""
    allowed_tools: tuple[str, ...] = ("list_files", "read_file", "search_text")

    @field_validator("allowed_tools")
    @classmethod
    def normalize_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not name or not name.isascii() or not name.replace("_", "").isalnum() for name in value
        ):
            raise ValueError("Tool names must be nonempty ASCII identifiers")
        return tuple(sorted(set(value)))


class Config(DataModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    permissions: Literal["ask-write", "read-only"] = "ask-write"
    bot: BotConfig | None = None


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config()
    try:
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        # Pydantic's original errors may include the supplied secrets.
        raise ConfigurationError("配置文件无法读取或字段不合法，请检查示例配置。") from exc


def load_environment(workspace: Path | None = None) -> None:
    """Load only explicit local files, preserving exported environment variables."""
    directories = [workspace.resolve()] if workspace is not None else []
    if Path.cwd() not in directories:
        directories.append(Path.cwd())
    for directory in directories:
        path = directory / ".env"
        try:
            if path.is_file():
                load_dotenv(path, override=False, encoding="utf-8-sig")
        except (OSError, UnicodeError):
            raise ConfigurationError(".env 无法读取，请检查文件权限和 UTF-8 编码。") from None


def load_runtime_config(path: Path | None = None, *, workspace: Path | None = None) -> Config:
    """Resolve shell > workspace .env > launch-directory .env > JSON/defaults."""
    load_environment(workspace)
    config = load_config(path)
    if "HIRO_BASE_URL" in os.environ:
        config.model.base_url = os.environ["HIRO_BASE_URL"]
    if "HIRO_MODEL" in os.environ:
        config.model.model = os.environ["HIRO_MODEL"]
    return config
