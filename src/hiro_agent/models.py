"""Model plugins keep provider details outside the agent loop."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from hiro_agent.config import ModelConfig
from hiro_agent.contracts import (
    AssistantMessage,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)
from hiro_agent.errors import ConfigurationError, ContextLimitError, ModelError
from hiro_agent.streaming import collect_stream

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_RESERVED_BODY_FIELDS = {"messages", "tools", "stream", "stream_options", "model", "n", "api_key"}


def _endpoint(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and "?" not in base_url
            and "#" not in base_url
            and not any(char.isspace() for char in base_url)
        )
        # Accessing port checks both the numeric form and valid range.
        _ = parsed.port
        if not valid:
            raise ValueError
        return str(httpx.URL(base_url.rstrip("/") + "/chat/completions"))
    except (ValueError, httpx.InvalidURL):
        raise ConfigurationError(
            "base_url 必须是无凭据、查询参数或片段的 HTTP(S) API 地址。"
        ) from None


def _wire_call(call: ToolCall) -> dict[str, Any]:
    if not call.id.strip() or not call.name.strip():
        raise ModelError("工具调用的 ID 和名称不能为空。")
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }


def _wire_message(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        if message.role != "assistant":
            raise ModelError("只有 assistant 消息可以包含工具调用。")
        ids = [call.id for call in message.tool_calls]
        if len(ids) != len(set(ids)):
            raise ModelError("同一消息中的工具调用 ID 不能重复。")
        wire["tool_calls"] = [_wire_call(call) for call in message.tool_calls]
    if message.role == "tool":
        if not message.tool_call_id or not message.tool_call_id.strip():
            raise ModelError("工具结果必须关联工具调用 ID。")
        wire["tool_call_id"] = message.tool_call_id
    elif message.tool_call_id is not None:
        raise ModelError("只有工具结果可以关联工具调用 ID。")
    return wire


def _parse_response(body: Any) -> ModelResponse:
    """Validate the envelope while leaving argument JSON to the tool executor."""
    try:
        if not isinstance(body, dict):
            raise ValueError
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError
        if "index" in choice and (type(choice["index"]) is not int or choice["index"] != 0):
            raise ValueError
        finish_reason = choice["finish_reason"]
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise ValueError
        message = choice["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ValueError
        calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict) or raw_call.get("type") != "function":
                raise ValueError
            call_id = raw_call["id"]
            function = raw_call["function"]
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or call_id in seen_ids
                or not isinstance(function, dict)
            ):
                raise ValueError
            name, arguments = function["name"], function["arguments"]
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
                raise ValueError
            seen_ids.add(call_id)
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        usage = None
        raw_usage = body.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise ValueError
            input_tokens, output_tokens = raw_usage["prompt_tokens"], raw_usage["completion_tokens"]
            if (
                type(input_tokens) is not int
                or type(output_tokens) is not int
                or input_tokens < 0
                or output_tokens < 0
            ):
                raise ValueError
            details = raw_usage.get("prompt_tokens_details")
            if details is not None and not isinstance(details, dict):
                raise ValueError
            cached_input_tokens = details.get("cached_tokens") if details is not None else None
            if cached_input_tokens is None:
                cached_input_tokens = raw_usage.get("prompt_cache_hit_tokens")
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
        metadata = {
            key: value
            for key, value in message.items()
            if key not in {"role", "content", "tool_calls"}
        }
        return ModelResponse(
            assistant=AssistantMessage(text=content, tool_calls=calls, provider_metadata=metadata),
            finish_reason=finish_reason,
            usage=usage,
        )
    except (KeyError, TypeError, ValueError):
        raise ModelError("模型返回了不合法的 Chat Completions 响应。") from None


class CompatibleModel:
    """Chat Completions over any explicitly configured API prefix."""

    def __init__(self, config: ModelConfig, client: httpx.AsyncClient | None = None) -> None:
        self._url = _endpoint(config.base_url)
        if not config.model.strip():
            raise ConfigurationError("请在配置中填写模型名称。")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config.api_key_env):
            raise ConfigurationError("api_key_env 必须是合法的环境变量名称。")
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise ConfigurationError("请通过 api_key_env 指定的环境变量设置 API 密钥。")
        if not api_key.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in api_key):
            raise ConfigurationError("API 密钥的格式不合法。")
        if _RESERVED_BODY_FIELDS.intersection(config.extra_body):
            raise ConfigurationError("extra_body 不能覆盖协议字段或凭据。")
        self._model = config.model.strip()
        self._extra_body = dict(config.extra_body)
        self._api_key = api_key
        self._timeout = config.request_timeout_seconds
        self._stream = config.stream
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(follow_redirects=False)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await self._complete(request, None)

    async def complete_stream(
        self, request: ModelRequest, on_delta: Callable[[str], None]
    ) -> ModelResponse:
        return await self._complete(request, on_delta)

    async def _complete(
        self, request: ModelRequest, on_delta: Callable[[str], None] | None
    ) -> ModelResponse:
        body: dict[str, Any] = {
            **self._extra_body,
            "model": self._model,
            "messages": [_wire_message(message) for message in request.messages],
            "stream": self._stream,
            "n": 1,
        }
        if request.tools:
            body["tools"] = request.tools
        if self._stream:
            body["stream_options"] = {"include_usage": True}
        try:
            async with self._client.stream(
                "POST",
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
                follow_redirects=False,
            ) as response:
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                if (
                    200 <= response.status_code < 300
                    and self._stream
                    and content_type.lower() == "text/event-stream"
                ):
                    envelope = await collect_stream(
                        response.aiter_bytes(), _MAX_RESPONSE_BYTES, on_delta
                    )
                    return _parse_response(envelope)
                payload = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(payload) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise ModelError("模型响应超过允许的大小。")
                    payload.extend(chunk)
                try:
                    decoded = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    decoded = None
                if not 200 <= response.status_code < 300:
                    error = decoded.get("error") if isinstance(decoded, dict) else None
                    if isinstance(error, dict) and (
                        error.get("code") == "context_length_exceeded"
                        or error.get("type") == "context_length_exceeded"
                    ):
                        raise ContextLimitError("模型上下文长度已超出限制。")
                    raise ModelError(f"模型请求失败（HTTP {response.status_code}）。")
        except httpx.TimeoutException:
            raise ModelError("模型请求超时。") from None
        except httpx.HTTPError:
            raise ModelError("模型请求发生网络错误。") from None
        return _parse_response(decoded)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class ScriptedModel:
    """A deterministic, inspectable adapter for loop and integration tests."""

    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise ModelError("脚本模型响应已耗尽。")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response.model_copy(deep=True)

    async def close(self) -> None:
        pass


class MockModel:
    """An offline demonstration whose next step is derived from durable history."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @staticmethod
    def _call(name: str, arguments: dict[str, Any]) -> ModelResponse:
        return MockModel._calls([(name, arguments)])

    @staticmethod
    def _calls(calls: list[tuple[str, dict[str, Any]]]) -> ModelResponse:
        return ModelResponse(
            assistant=AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid4().hex}",
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                    for name, arguments in calls
                ]
            ),
            finish_reason="tool_calls",
        )

    @staticmethod
    def _answer(text: str) -> ModelResponse:
        return ModelResponse(assistant=AssistantMessage(text=text), finish_reason="stop")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        start = next(
            (
                i + 1
                for i in range(len(request.messages) - 1, -1, -1)
                if request.messages[i].role == "user" and not request.messages[i].continuation
            ),
            0,
        )
        messages = request.messages[start:]
        calls = [call for message in messages for call in message.tool_calls]
        names = {tool["function"]["name"] for tool in request.tools}
        is_child = names == {"list_files", "read_file", "search_text"}
        if not calls:
            return self._call("read_file", {"path": "hiro-demo.txt"})
        last_call = calls[-1]
        result_message = next(
            (
                message
                for message in reversed(messages)
                if message.role == "tool" and message.tool_call_id == last_call.id
            ),
            None,
        )
        if result_message is None:
            return self._answer("演示暂停：上一步工具尚无结果。")
        try:
            result = json.loads(result_message.content or "")
        except (ValueError, TypeError):
            return self._answer("演示停止：无法确认上一步工具的结果。")
        if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
            return self._answer("演示停止：无法确认上一步工具的结果。")
        error = result.get("error")
        if is_child:
            report = {
                "summary": "已读取演示文件；未执行命令或修改文件。"
                if result["ok"]
                else "当前无法读取演示文件，请由主 Agent 检查或创建。",
                "findings": [
                    {
                        "path": "hiro-demo.txt",
                        "line": 1,
                        "description": "文件读取成功。" if result["ok"] else "文件读取失败。",
                    }
                ],
                "suggestions": ["由主 Agent 执行内容验证。"],
            }
            return self._answer(json.dumps(report, ensure_ascii=False))
        if last_call.name == "delegate_task":
            # Older Mock journals began with an explorer before file creation.
            # Resume their durable stage instead of claiming verification early.
            if not any(call.name == "run_command" for call in calls):
                if result["ok"] and json.loads(last_call.arguments).get("role") == "explorer":
                    return self._call("read_file", {"path": "hiro-demo.txt"})
                return self._answer("演示停止：子任务未完成，请检查委派结果。")
            delegate_calls = [call for call in calls if call.name == "delegate_task"]
            reports = []
            for call in delegate_calls:
                result_text = next(
                    (
                        message.content
                        for message in reversed(messages)
                        if message.role == "tool" and message.tool_call_id == call.id
                    ),
                    None,
                )
                try:
                    delegated = json.loads(result_text or "")
                    if not delegated["ok"]:
                        raise ValueError
                    role = json.loads(call.arguments)["role"]
                    summary = delegated["value"]["report"]["summary"]
                    reports.append(f"{role}：{summary}")
                except (ValueError, TypeError, KeyError):
                    return self._answer("演示停止：子任务未完成，请检查委派结果。")
            if not reports:
                return self._answer("演示停止：子任务未完成，请检查委派结果。")
            return self._answer(
                "演示完成：已创建 hiro-demo.txt，通过 Python 校验，并取得只读探索与审查报告。\n"
                + "\n".join(reports)
            )
        if (
            not result["ok"]
            and isinstance(error, str)
            and error.lower() in {"permission_denied", "permission_rejected"}
        ):
            return self._answer("演示停止：工具权限未获批准，尚未完成文件创建和校验。")
        if last_call.name == "read_file":
            return self._call(
                "write_file", {"path": "hiro-demo.txt", "content": "Hello from Hiro!\n"}
            )
        if not result["ok"]:
            return self._answer("演示停止：上一步工具执行失败，尚未完成校验。")
        if last_call.name == "write_file":
            return self._call(
                "run_command",
                {
                    "executable": "python3",
                    "args": [
                        "-c",
                        "from pathlib import Path; "
                        "assert Path('hiro-demo.txt').read_text(encoding='utf-8') "
                        "== 'Hello from Hiro!\\n'; print('verified')",
                    ],
                },
            )
        if last_call.name == "run_command":
            value = result.get("value")
            if isinstance(value, dict) and value.get("exit_code", value.get("returncode", 0)) != 0:
                return self._answer("演示停止：Python 校验失败。")
            if "delegate_task" in names:
                return self._calls(
                    [
                        (
                            "delegate_task",
                            {
                                "role": "explorer",
                                "task": "读取 hiro-demo.txt，报告内容和文件证据。",
                            },
                        ),
                        (
                            "delegate_task",
                            {
                                "role": "reviewer",
                                "task": "只读审查 hiro-demo.txt 的内容，返回文件证据。",
                            },
                        ),
                    ]
                )
            return self._answer("演示完成：已创建 hiro-demo.txt，并通过 Python 校验内容。")
        return self._answer("演示停止：遇到了未知的工具步骤。")

    async def close(self) -> None:
        pass
