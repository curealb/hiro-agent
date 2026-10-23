"""Explicit configuration; credentials are read only by the model adapter."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator

from hiro_agent.contracts import DataModel
from hiro_agent.errors import ConfigurationError


class ModelConfig(DataModel):
    adapter: Literal["compatible", "mock"] = "compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "HIRO_API_KEY"
    request_timeout_seconds: float = Field(default=120, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_body")
    @classmethod
    def validate_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"messages", "tools", "stream", "model", "n", "api_key"}
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


class Config(DataModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    permissions: Literal["ask-write", "read-only"] = "ask-write"


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return Config()
    try:
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        # Pydantic's original errors may include the supplied secrets.
        raise ConfigurationError("配置文件无法读取或字段不合法，请检查示例配置。") from exc
