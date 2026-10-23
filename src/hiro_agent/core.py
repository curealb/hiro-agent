"""Small plugin kernel with explicit service ownership and cleanup."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from hiro_agent.contracts import Observer
from hiro_agent.errors import HiroError

Disposer = Callable[[], Awaitable[None] | None]
log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def subscribe(self, observer: Observer) -> Disposer:
        self._observers.append(observer)

        def remove() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return remove

    def emit(self, name: str, data: dict[str, Any]) -> None:
        for observer in tuple(self._observers):
            try:
                observer(name, data)
            except Exception:
                # Observers cannot change execution policy or break persistence.
                log.warning("Runtime observer failed")


class PluginContext:
    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services
        self._cleanup: list[Disposer] = []
        self._disposed = False

    def get(self, name: str) -> Any:
        if name not in self._services:
            raise HiroError(f"服务不存在：{name}")
        return self._services[name]

    def provide(self, name: str, service: Any) -> None:
        if self._disposed:
            raise HiroError("插件已释放")
        if name in self._services:
            raise HiroError(f"服务重复注册：{name}")
        self._services[name] = service

        def remove() -> None:
            self._services.pop(name, None)

        self.on_dispose(remove)

    def on_dispose(self, disposer: Disposer) -> None:
        if self._disposed:
            raise HiroError("插件已释放")
        self._cleanup.append(disposer)

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors: list[Exception] = []
        for disposer in reversed(self._cleanup):
            try:
                result = disposer()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                errors.append(exc)
        self._cleanup.clear()
        if errors:
            raise ExceptionGroup("插件清理失败", errors)


class Plugin(Protocol):
    name: str
    requires: tuple[str, ...]

    async def setup(self, context: PluginContext) -> None: ...


class Runtime:
    def __init__(self) -> None:
        self.services: dict[str, Any] = {}
        self.events = EventBus()
        self._plugins: list[tuple[str, PluginContext]] = []
        self._closed = False

    async def install(self, plugin: Plugin) -> None:
        if self._closed:
            raise HiroError("Runtime 已关闭")
        if any(name == plugin.name for name, _ in self._plugins):
            raise HiroError(f"插件重复加载：{plugin.name}")
        missing = set(plugin.requires) - self.services.keys()
        if missing:
            raise HiroError(f"插件 {plugin.name} 缺少依赖：{', '.join(sorted(missing))}")
        context = PluginContext(self.services)
        try:
            await plugin.setup(context)
        except BaseException:
            try:
                await context.dispose()
            except Exception:
                log.warning("Failed plugin also reported cleanup errors")
            raise
        self._plugins.append((plugin.name, context))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        agent = self.services.get("agent")
        if agent is not None:
            try:
                await agent.close()
            except Exception as exc:
                errors.append(exc)
        for _, context in reversed(self._plugins):
            try:
                await context.dispose()
            except Exception as exc:
                errors.append(exc)
        self._plugins.clear()
        if errors:
            raise ExceptionGroup("Runtime 清理失败", errors)


class ServicePlugin:
    """A provider factory runs inside its plugin's owned lifecycle."""

    def __init__(
        self,
        name: str,
        setup: Callable[[PluginContext], Awaitable[None]],
        requires: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.requires = requires
        self._setup = setup

    async def setup(self, context: PluginContext) -> None:
        await self._setup(context)
