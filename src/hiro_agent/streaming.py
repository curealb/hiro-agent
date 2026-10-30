"""Bounded SSE decoding and Chat Completions delta assembly."""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from hiro_agent.errors import ContextLimitError, ModelError

_INVALID_STREAM = "模型返回了不合法的 Chat Completions 流式响应。"


async def _sse_data(chunks: AsyncIterator[bytes], limit: int) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    data: list[str] = []
    size = 0
    first = True

    def read_line(line: str) -> str | None:
        if not line:
            if data:
                event = "\n".join(data)
                data.clear()
                return event
            return None
        key, separator, value = line.partition(":")
        if key == "data":
            data.append(value.removeprefix(" ") if separator else "")
        return None

    try:
        async for chunk in chunks:
            size += len(chunk)
            if size > limit:
                raise ModelError("模型响应超过允许的大小。")
            decoded = decoder.decode(chunk)
            if first and decoded:
                decoded = decoded.removeprefix("\ufeff")
                first = False
            pending += decoded
            while match := re.search(r"[\r\n]", pending):
                index = match.start()
                # A CRLF may be split between network chunks.
                if pending[index] == "\r" and index + 1 == len(pending):
                    break
                delimiter = 2 if pending[index : index + 2] == "\r\n" else 1
                line, pending = pending[:index], pending[index + delimiter :]
                event = read_line(line)
                if event is not None:
                    yield event
        pending += decoder.decode(b"", final=True)
        if pending:
            event = read_line(pending.rstrip("\r"))
            if event is not None:
                yield event
        event = read_line("")
        if event is not None:
            yield event
    except UnicodeDecodeError:
        raise ModelError(_INVALID_STREAM) from None


@dataclass
class _ToolDelta:
    ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    type: str | None = None

    def extend(self, delta: dict[str, Any]) -> None:
        call_type = delta.get("type")
        if call_type is not None:
            if call_type != "function":
                raise ValueError
            self.type = call_type
        call_id = delta.get("id")
        if call_id is not None:
            if not isinstance(call_id, str):
                raise ValueError
            self.ids.append(call_id)
        function = delta.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ValueError
            for key, target in (("name", self.names), ("arguments", self.arguments)):
                value = function.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError
                    target.append(value)

    def message(self) -> dict[str, Any]:
        return {
            "id": "".join(self.ids),
            "type": self.type,
            "function": {"name": "".join(self.names), "arguments": "".join(self.arguments)},
        }


async def collect_stream(
    chunks: AsyncIterator[bytes], limit: int, on_delta: Callable[[str], None] | None
) -> dict[str, Any]:
    """Emit public text immediately; return a complete envelope only after DONE."""
    text: list[str] = []
    tools: dict[int, _ToolDelta] = {}
    metadata: dict[str, Any] = {}
    usage: Any = None
    finish_reason: str | None = None
    done = False
    async for event in _sse_data(chunks, limit):
        try:
            if event == "[DONE]":
                done = True
                break
            payload = json.loads(event)
            if not isinstance(payload, dict):
                raise ValueError
            if "error" in payload:
                error = payload["error"]
                if isinstance(error, dict) and (
                    error.get("code") == "context_length_exceeded"
                    or error.get("type") == "context_length_exceeded"
                ):
                    raise ContextLimitError("模型上下文长度已超出限制。")
                raise ModelError("模型流式请求失败。")
            choices = payload["choices"]
            if not isinstance(choices, list) or len(choices) > 1:
                raise ValueError
            if payload.get("usage") is not None:
                if not isinstance(payload["usage"], dict):
                    raise ValueError
                usage = payload["usage"]
            if not choices:
                if payload.get("usage") is None:
                    raise ValueError
                continue
            choice = choices[0]
            if (
                not isinstance(choice, dict)
                or type(choice.get("index")) is not int
                or choice["index"] != 0
                or finish_reason is not None
            ):
                raise ValueError
            delta = choice["delta"]
            if not isinstance(delta, dict):
                raise ValueError
            if delta.get("role") not in (None, "assistant"):
                raise ValueError
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError
            call_deltas = delta.get("tool_calls")
            if call_deltas is not None:
                if not isinstance(call_deltas, list):
                    raise ValueError
                for call in call_deltas:
                    if (
                        not isinstance(call, dict)
                        or type(call.get("index")) is not int
                        or call["index"] < 0
                    ):
                        raise ValueError
                    tools.setdefault(call["index"], _ToolDelta()).extend(call)
            for key, value in delta.items():
                if key in {"role", "content", "tool_calls"} or value is None:
                    continue
                if isinstance(value, str) and isinstance(metadata.get(key), str):
                    metadata[key] += value
                elif isinstance(value, list) and isinstance(metadata.get(key), list):
                    metadata[key].extend(value)
                else:
                    metadata[key] = value
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError
                finish_reason = reason
            if content is not None:
                text.append(content)
        except (KeyError, TypeError, ValueError):
            raise ModelError(_INVALID_STREAM) from None
        if content and on_delta is not None:
            on_delta(content)
    if not done or finish_reason is None:
        raise ModelError("模型流式响应未完整结束。")
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    **metadata,
                    "role": "assistant",
                    "content": "".join(text) if text else None,
                    "tool_calls": [tools[index].message() for index in sorted(tools)],
                },
            }
        ],
        "usage": usage,
    }
