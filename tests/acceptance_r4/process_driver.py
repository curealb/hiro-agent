"""Independent R4 process fixture: only model substitution and explicit crash cuts."""

import asyncio
import json
import os
import re
from pathlib import Path
from uuid import uuid4

from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.handoffs import HandoffStore
from hiro_agent.kernel_runtime import KernelRuntime

WORKSPACE = Path(os.environ["R4_WORKSPACE"])
CUT = os.environ.get("R4_CUT", "")
MODE = os.environ.get("R4_MODE", "pipeline")


def response(name=None, arguments=None, text=None):
    calls = (
        []
        if name is None
        else [ToolCall(id=uuid4().hex, name=name, arguments=json.dumps(arguments))]
    )
    return ModelResponse(
        assistant=AssistantMessage(text=text, tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop",
    )


class Model:
    def __init__(self, bot_id):
        self.bot_id = bot_id

    async def complete(self, request):
        task = next(m.content for m in request.messages if m.role == "user")
        results = [json.loads(m.content) for m in request.messages if m.role == "tool"]
        successful = [r for r in results if r["ok"]]
        with (WORKSPACE / "r4-model-audit.jsonl").open("a") as audit:
            audit.write(json.dumps({"pid": os.getpid(), "bot": self.bot_id, "task": task}) + "\n")
        if MODE == "chat":
            if not results:
                return response(
                    "send_handoff",
                    {
                        "target_bot_id": "b" if self.bot_id == "a" else "a",
                        "text": "R4 explicit chat",
                        "idempotency_key": "chat",
                    },
                )
            return response(text="Explicit chat finished")
        if task.startswith("A Bot handoff result has returned"):
            if not results:
                match = re.search(r"message_id ([a-f0-9]+)", task)
                assert match
                return response("read_handoff", {"message_id": match[1]})
            assert successful
            return response(text="R4 received and processed: " + successful[-1]["value"])
        if task == "R4 inspect":
            if not successful:
                return response("read_file", {"path": "evidence.txt"})
            if len(successful) == 1:
                return response(
                    "write_file",
                    {"path": "report.txt", "content": "R4 VERIFIED\n"},
                    text="R4 inspection report: evidence is VERIFIED.",
                )
            if len(successful) == 2:
                return response(
                    "send_handoff",
                    {
                        "target_bot_id": "b",
                        "text": "R4 review",
                        "idempotency_key": "review",
                    },
                )
            return response(text="R4 report committed and review requested")
        if task == "R4 review":
            if not results:
                return response("read_file", {"path": "report.txt"})
            assert "R4 VERIFIED" in successful[-1]["value"]["content"]
            return response(text="R4 reviewer VERIFIED report")
        if task == "R4 slow":
            await asyncio.sleep(2)
            return response(text="should exceed the chain active time limit")
        if task == "R4 forbidden":
            return response("write_file", {"path": "unauthorized.txt", "content": "bad"})
        if task in {"R4 send", "R4 send slow"} and not results:
            return response(
                "send_handoff",
                {
                    "target_bot_id": "b",
                    "text": "R4 slow" if task.endswith("slow") else "R4 review",
                    "idempotency_key": "review",
                },
            )
        return response(text="R4 completed")

    async def close(self):
        pass


class Kernel(KernelRuntime):
    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, model_factory=lambda config: Model(config.bot.id), **kwargs)


original_submit = TaskStore.submit


def submit(self, request):
    if CUT == request.source + "-before":
        os._exit(81)
    result = original_submit(self, request)
    if CUT == request.source + "-after":
        os._exit(82)
    return result


original_capture = HandoffStore._capture_result


def capture(self, message):
    if CUT == "terminal" and self.tasks.get(message.target_task_id).state == "succeeded":
        os._exit(83)
    return original_capture(self, message)


original_send = HandoffStore.send


def send(self, *args, **kwargs):
    result = original_send(self, *args, **kwargs)
    if CUT == "accepted":
        os._exit(84)
    return result


TaskStore.submit = submit
HandoffStore._capture_result = capture
HandoffStore.send = send
kernel_cli.KernelRuntime = Kernel
app()
