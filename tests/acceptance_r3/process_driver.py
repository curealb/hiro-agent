"""R3 process fixture: real CLI/Agent/stores, deterministic model, abrupt crash cuts."""

import json
import os
from uuid import uuid4

from hiro_agent import kernel_cli
from hiro_agent.cli import app
from hiro_agent.contracts import AssistantMessage, ModelResponse, ToolCall
from hiro_agent.durable_tasks import TaskStore
from hiro_agent.kernel_runtime import KernelRuntime


class AcceptanceModel:
    async def complete(self, request):
        task = next(message.content for message in request.messages if message.role == "user")
        results = [message for message in request.messages if message.role == "tool"]
        if task in {"load guide", "load then attack"} and not results:
            name, arguments = "read_skill", {"skill_id": "guide"}
        elif task == "load then attack" and len(results) == 1:
            name, arguments = "write_file", {"path": "unauthorized.txt", "content": "bad"}
        else:
            return ModelResponse(
                assistant=AssistantMessage(text=f"R3 finished: {task}"), finish_reason="stop"
            )
        return ModelResponse(
            assistant=AssistantMessage(
                tool_calls=[ToolCall(id=uuid4().hex, name=name, arguments=json.dumps(arguments))]
            ),
            finish_reason="tool_calls",
        )

    async def close(self):
        pass


class AcceptanceKernel(KernelRuntime):
    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, model_factory=lambda config: AcceptanceModel(), **kwargs)


cut = os.environ.get("R3_ROUTINE_CUT")
if cut:
    original_submit = TaskStore.submit

    def interrupted_submit(self, request):
        if request.source == "routine" and cut == "accepted":
            os._exit(81)
        result = original_submit(self, request)
        if request.source == "routine" and cut == "submitted":
            os._exit(82)
        return result

    TaskStore.submit = interrupted_submit

kernel_cli.KernelRuntime = AcceptanceKernel
app()
