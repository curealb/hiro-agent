import pytest

from hiro_agent.core import EventBus, PluginContext, Runtime, ServicePlugin
from hiro_agent.errors import HiroError


async def test_dependencies_duplicates_and_owned_lifecycle():
    runtime = Runtime()
    cleaned = []

    async def provider(ctx):
        ctx.provide("first", object())
        ctx.on_dispose(lambda: cleaned.append("first"))

    async def consumer(ctx):
        assert ctx.get("first") is runtime.services["first"]
        ctx.provide("second", 2)
        ctx.on_dispose(lambda: cleaned.append("second"))

    with pytest.raises(HiroError, match="缺少依赖"):
        await runtime.install(ServicePlugin("second", consumer, ("first",)))
    await runtime.install(ServicePlugin("first", provider))
    with pytest.raises(HiroError, match="重复加载"):
        await runtime.install(ServicePlugin("first", provider))
    with pytest.raises(HiroError, match="重复注册"):
        await runtime.install(ServicePlugin("duplicate", provider))
    await runtime.install(ServicePlugin("second", consumer, ("first",)))
    await runtime.close()
    assert cleaned == ["second", "first"]
    assert runtime.services == {}
    await runtime.close()


async def test_setup_failure_disposes_resources_and_cleanup_continues():
    runtime = Runtime()
    cleaned = []

    async def broken(ctx):
        ctx.provide("temporary", 1)

        async def cleanup():
            cleaned.append("cleanup")

        ctx.on_dispose(cleanup)
        raise ValueError("setup failed")

    with pytest.raises(ValueError, match="setup failed"):
        await runtime.install(ServicePlugin("broken", broken))
    assert cleaned == ["cleanup"]
    assert runtime.services == {}

    async def good(ctx):
        ctx.provide("live", 1)
        ctx.on_dispose(lambda: cleaned.append("last"))

        def fail_cleanup():
            raise ValueError("cleanup error")

        ctx.on_dispose(fail_cleanup)

    await runtime.install(ServicePlugin("good", good))
    with pytest.raises(ExceptionGroup):
        await runtime.close()
    assert cleaned[-1] == "last"
    assert runtime.services == {}


def test_observer_failure_does_not_change_control_flow():
    bus = EventBus()
    seen = []

    def bad(name, data):
        raise ValueError("failed observer")

    remove = bus.subscribe(bad)
    bus.subscribe(lambda name, data: seen.append(name))
    bus.emit("first", {})
    remove()
    bus.emit("second", {})
    assert seen == ["first", "second"]


def test_disposed_context_cannot_register():
    context = PluginContext({})
    context._disposed = True
    with pytest.raises(HiroError):
        context.provide("service", 1)
