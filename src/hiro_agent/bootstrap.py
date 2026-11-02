"""The application composition root. The loop never imports concrete providers."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hiro_agent.agent import Agent
from hiro_agent.bots import AuthorityProvider, BotStore
from hiro_agent.config import Config
from hiro_agent.context import SYSTEM_PROMPT, IdentityPreprocessor, PassthroughContext
from hiro_agent.contracts import (
    AgentService,
    ApprovalCallback,
    ContextService,
    ModelService,
    Observer,
    SessionService,
    ToolResult,
)
from hiro_agent.core import PluginContext, Runtime, ServicePlugin
from hiro_agent.errors import HiroError
from hiro_agent.kernel_db import KernelDatabase
from hiro_agent.models import CompatibleModel, MockModel
from hiro_agent.permissions import PermissionPolicy
from hiro_agent.session import JsonlSession, recover_session
from hiro_agent.subagents import SubagentManager, recover_delegations
from hiro_agent.tools import ToolExecutor, ToolGate, ToolRegistry, build_file_tools


@dataclass
class Harness:
    runtime: Runtime
    agent: AgentService
    session: SessionService
    recovered: list[ToolResult]

    async def close(self) -> None:
        await self.runtime.close()


async def create_harness(
    workspace: Path,
    config: Config,
    session_id: str | None = None,
    approval: ApprovalCallback | None = None,
    observer: Observer | None = None,
    *,
    create_session: bool = False,
    persist_bot: bool = True,
    model: ModelService | None = None,
    gate: ToolGate | None = None,
    configure_tools: Callable[[ToolRegistry], None] | None = None,
    context: ContextService | None = None,
) -> Harness:
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("工作区必须是目录")
    if session_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", session_id):
            raise HiroError("Session ID 不合法")
        path = workspace / ".hiro-agent" / "sessions" / f"{session_id}.jsonl"
        if not path.is_file() and not create_session:
            raise HiroError("指定的会话不存在，请使用 sessions 查看当前工作区的会话")
    runtime = Runtime()
    recovered: list[ToolResult] = []
    authority: AuthorityProvider | None = None
    if observer:
        runtime.events.subscribe(observer)

    async def session_setup(ctx: PluginContext) -> None:
        nonlocal config, authority
        try:
            # A selected journal can be deleted after the existence check.
            # Resuming must never recreate it after the deletion lock releases.
            session = await JsonlSession.open(
                workspace, session_id, create=session_id is None or create_session
            )
        except FileNotFoundError:
            raise HiroError("指定的会话不存在，请使用 sessions 查看当前工作区的会话") from None
        ctx.on_dispose(session.close)
        ctx.provide("session", session)
        if not session.events:
            await session.append("session.created", {"workspace": str(workspace)})
        if any(event.type == "subagent.created" for event in session.events):
            raise HiroError("子会话不能直接续跑，请打开对应的父会话")
        bindings = [event for event in session.events if event.type == "session.bot"]
        if len(bindings) > 1:
            raise HiroError("会话存在重复 Bot 绑定")
        bound_id = bindings[0].data.get("bot_id") if bindings else None
        if bindings and not isinstance(bound_id, str):
            raise HiroError("会话 Bot 绑定无效")
        if config.bot is not None and bound_id is not None and bound_id != config.bot.id:
            raise HiroError("会话属于其他 Bot，不能切换身份续跑")
        if config.bot is not None or bound_id is not None:
            database = KernelDatabase.for_workspace(workspace)
            store = BotStore(database)
            if config.bot is None:
                assert isinstance(bound_id, str)
                record = store.get(bound_id)
                config = record.config
            elif persist_bot:
                record = store.save(config)
            else:
                record = store.get(config.bot.id)
                config = record.config
            authority = store.provider(record.bot_id)
            ctx.provide("kernel_database", database)
            ctx.provide("bot_store", store)
            ctx.provide("bot", record)
            if not bindings:
                await session.append("session.bot", record.authority.model_dump(mode="json"))
        await recover_delegations(workspace, session, config.subagents.max_output_chars)
        recovered.extend(await recover_session(session))
        if not any(event.type == "prompt.system" for event in session.events):
            instructions = config.bot.instructions if config.bot is not None else ""
            await session.append(
                "prompt.system",
                {"text": SYSTEM_PROMPT + ("\n" + instructions if instructions else "")},
            )

    async def model_setup(ctx: PluginContext) -> None:
        root_model = model or (
            MockModel() if config.model.adapter == "mock" else CompatibleModel(config.model)
        )
        ctx.on_dispose(root_model.close)
        ctx.provide("model", root_model)
        role_models: dict[str, ModelService] = {}
        model_ids: dict[str, str] = {}
        if config.subagents.enabled:
            for role in ("explorer", "reviewer"):
                role_config = getattr(config.subagents.roles, role).model
                if config.model.adapter == "mock":
                    role_model: ModelService = MockModel()
                    ctx.on_dispose(role_model.close)
                    model_id = "mock:mock"
                elif role_config is None:
                    role_model = root_model
                    model_id = f"{config.model.adapter}:{config.model.model.strip()}"
                else:
                    role_model = (
                        MockModel()
                        if role_config.adapter == "mock"
                        else CompatibleModel(role_config)
                    )
                    ctx.on_dispose(role_model.close)
                    model_id = (
                        "mock:mock"
                        if role_config.adapter == "mock"
                        else f"{role_config.adapter}:{role_config.model.strip()}"
                    )
                role_models[role] = role_model
                model_ids[role] = model_id
        ctx.provide("role_models", role_models)
        ctx.provide("model_ids", model_ids)

    async def permission_setup(ctx: PluginContext) -> None:
        ctx.provide("permission", PermissionPolicy(config.permissions, approval))

    async def tools_setup(ctx: PluginContext) -> None:
        excluded_env = [config.model.api_key_env]
        for role in ("explorer", "reviewer"):
            role_config = getattr(config.subagents.roles, role).model
            if role_config is not None:
                excluded_env.append(role_config.api_key_env)
        registry = build_file_tools(workspace, config.tools, excluded_env=excluded_env)
        if configure_tools is not None:
            configure_tools(registry)
        ctx.provide("tool_registry", registry)
        ctx.provide(
            "tools",
            ToolExecutor(
                registry,
                ctx.get("session"),
                ctx.get("permission"),
                runtime.events.emit,
                max_concurrency=config.subagents.max_concurrency,
                authority=authority,
                gate=gate,
            ),
        )

    async def context_setup(ctx: PluginContext) -> None:
        ctx.provide("context", context if context is not None else PassthroughContext())
        ctx.provide("preprocessor", IdentityPreprocessor())

    async def subagents_setup(ctx: PluginContext) -> None:
        async def child_session(child_id: str) -> SessionService:
            return await JsonlSession.open(workspace, child_id)

        manager = SubagentManager(
            ctx.get("model"),
            ctx.get("tool_registry"),
            child_session,
            config.subagents,
            runtime.events,
            role_models=ctx.get("role_models"),
            model_ids=ctx.get("model_ids"),
            authority=authority,
        )
        ctx.on_dispose(manager.close)
        ctx.provide("subagents", manager)
        manager.register(ctx.get("tool_registry"))

    async def agent_setup(ctx: PluginContext) -> None:
        agent = Agent(
            ctx.get("session"),
            ctx.get("model"),
            ctx.get("tools"),
            ctx.get("context"),
            ctx.get("preprocessor"),
            config.budget,
            runtime.events,
        )
        ctx.provide("agent", agent)
        ctx.on_dispose(agent.close)

    plugins = [
        ServicePlugin("session", session_setup),
        ServicePlugin("model", model_setup),
        ServicePlugin("permission", permission_setup),
        ServicePlugin("tools", tools_setup, ("session", "permission")),
        ServicePlugin("context", context_setup, ("session",)),
    ]
    try:
        for plugin in plugins:
            await runtime.install(plugin)
        # Resuming a bound session may have loaded its saved configuration.
        if config.subagents.enabled:
            await runtime.install(
                ServicePlugin("subagents", subagents_setup, ("session", "model", "tool_registry"))
            )
        await runtime.install(
            ServicePlugin(
                "agent", agent_setup, ("session", "model", "tools", "context", "preprocessor")
            )
        )
    except BaseException:
        try:
            await runtime.close()
        except Exception:
            pass
        raise
    return Harness(runtime, runtime.services["agent"], runtime.services["session"], recovered)
