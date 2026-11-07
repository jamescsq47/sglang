import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.disaggregation.dualpd.router.late_binding_router import LateBindingMiniLoadBalancer
from sglang.srt.disaggregation.agentic_slow_congestion import SlowRecoveryCongestion


def test_global_q_reuses_one_ledger_read_excludes_slow_tools_and_domains():
    router = object.__new__(LateBindingMiniLoadBalancer)
    router.prefill_urls = ["p0", "p1"]
    entries = {
        "fast:0": dict(state="host_ready", arena_domain=0, recovery_domain=1, byte_size=10),
        "slow-tool:0": dict(state="host_ready", arena_domain=0, byte_size=20),
        "leased:0": dict(state="h2d_loading", arena_domain=1, byte_size=30,
                         recovery_claims={"0": {"phase": "leased"}}),
        "dma:0": dict(state="h2d_loading", arena_domain=1, byte_size=40,
                      recovery_claims={"0": {"phase": "io_inflight"}}),
        "evicted:0": dict(state="recompute_required", arena_domain=1, byte_size=50),
    }
    reads = []
    router._d2p_host_ledger = SimpleNamespace(
        snapshot_entries=lambda: reads.append(1) or entries
    )
    parents = frozenset({"fast:0", "leased:0", "dma:0", "evicted:0"})
    used, q = router._prefill_arena_bytes(parents)
    assert used == [30, 70]
    assert q == 2 and len(reads) == 1
    entries["fast:0"]["recovery_domain"] = 0
    assert router._prefill_arena_bytes(parents)[1] == 2
    entries["leased:0"].update(state="host_ready", recovery_claims={})
    assert router._prefill_arena_bytes(parents)[1] == 2
    assert router._prefill_arena_bytes() == [30, 70]


@pytest.mark.parametrize("cancel", [False, True])
def test_parent_reference_cleanup_including_early_resolve_failure(cancel):
    async def scenario():
        router = object.__new__(LateBindingMiniLoadBalancer)
        router._slow_congestion = SlowRecoveryCongestion(32, 8)
        # This unit test isolates dispatch ownership; monitor startup has its
        # own tests and requires a fully initialized Router filesystem.
        router._ensure_prefill_pressure_monitor = Mock()
        seen = []

        async def dispatch(*args):
            seen.append(dict(router._slow_congestion.parents))
            if cancel:
                raise asyncio.CancelledError()
            raise RuntimeError("resolve failed before original dispatch try")

        router._late_dispatch_with_metadata = dispatch
        request = {"bootstrap_room": 1, "sampling_params": {"custom_params": {
            "agentic_request_id": "agent", "agentic_generation": 1,
            "agentic_parent_generation": 0,
        }}}
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await router._late_dispatch(None, request, "p0", "generate", {})
        assert seen == [{"agent:0": 1}]
        assert not router._slow_congestion.parents
        router._ensure_prefill_pressure_monitor.assert_called_once()

    asyncio.run(scenario())
