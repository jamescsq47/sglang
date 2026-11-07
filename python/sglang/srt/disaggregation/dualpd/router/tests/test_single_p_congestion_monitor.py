import asyncio
from unittest.mock import AsyncMock

from sglang.srt.disaggregation.dualpd.router.late_binding_router import LateBindingMiniLoadBalancer


def test_single_p_congestion_monitor_starts_once():
    async def run():
        router = LateBindingMiniLoadBalancer.__new__(LateBindingMiniLoadBalancer)
        router.dynamic_prefill_domains = False
        router._slow_congestion = object()
        router._prefill_pressure_path = None
        router._d2p_host_ledger = None
        wait = asyncio.Event()
        router._prefill_pressure_monitor_loop = AsyncMock(side_effect=wait.wait)
        router._ensure_prefill_pressure_monitor()
        task = router._prefill_pressure_task
        router._ensure_prefill_pressure_monitor()
        assert router._prefill_pressure_task is task
        await asyncio.sleep(0)
        router._prefill_pressure_monitor_loop.assert_awaited_once()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_single_p_without_feedback_does_not_start_monitor():
    router = LateBindingMiniLoadBalancer.__new__(LateBindingMiniLoadBalancer)
    router.dynamic_prefill_domains = False
    router._slow_congestion = None
    router._ensure_prefill_pressure_monitor()
    assert not hasattr(router, "_prefill_pressure_task")
