"""R2-only process fixture: real CLI/worker/storage with an offline model and crash cuts."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall, Usage
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.kernel_runtime import KernelRunner, KernelRuntime

CUT = os.environ.get("R2_CRASH_CUT", "")
WORKSPACE = Path(os.environ["R2_WORKSPACE"])


class OfflineModel:
    async def complete(self, request):
        original = next(message.content for message in request.messages if message.role == "user")
        with (WORKSPACE / "model-attempts").open("a") as audit:
            audit.write("request\n")
        if original == "slow":
            (WORKSPACE / "model-entered").touch()
            await asyncio.Event().wait()
        answers = [
            message.content
            for message in request.messages
            if message.role == "user" and "用户对问题" in (message.content or "")
        ]
        results = [
            json.loads(message.content) for message in request.messages if message.role == "tool"
        ]
        if original == "finish" or answers or any(result["ok"] for result in results):
            return ModelResponse(
                assistant=AssistantMessage(text=answers[-1] if answers else "accepted result"),
                finish_reason="stop",
                usage=Usage(input_tokens=13, output_tokens=5),
            )
        if original == "question":
            name, arguments = "ask_user", {"question": "Exact report title?"}
        else:
            name, arguments = "write_file", {"path": "approved.txt", "content": "精确内容\n"}
        calls = [ToolCall(id=uuid4().hex, name=name, arguments=json.dumps(arguments))]
        if original == "approval":
            calls.append(ToolCall(id=uuid4().hex, name="read_file", arguments='{"path":"later"}'))
        return ModelResponse(
            assistant=AssistantMessage(tool_calls=calls),
            finish_reason="tool_calls",
            usage=Usage(input_tokens=13, output_tokens=5),
        )

    async def close(self):
        pass


class OfflineKernel(KernelRuntime):
    def __init__(self, workspace):
        super().__init__(workspace, model_factory=lambda config: OfflineModel())


if CUT == "claim":
    original_claim = TaskStore.claim

    def crash_after_claim(self, worker_id):
        claim = original_claim(self, worker_id)
        if claim is not None:
            os._exit(71)
        return claim

    TaskStore.claim = crash_after_claim
elif CUT == "ended":

    def crash_before_finish(self, *args, **kwargs):
        os._exit(72)

    TaskStore.finish = crash_before_finish
elif CUT == "effect":
    original_configure = KernelRunner._configure_tools

    def configure_crashing_write(registry):
        original_configure(registry)
        original_handler = registry.definitions["write_file"].handler

        async def crash_after_effect(arguments):
            await original_handler(arguments)
            with (WORKSPACE / "effect-count").open("a") as audit:
                audit.write("effect\n")
            os._exit(73)

        registry.definitions["write_file"] = replace(
            registry.definitions["write_file"], handler=crash_after_effect
        )

    KernelRunner._configure_tools = staticmethod(configure_crashing_write)

kernel_cli.KernelRuntime = OfflineKernel
app()
