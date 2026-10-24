from contextlib import contextmanager
from types import SimpleNamespace

from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
)


class _Authority:
    def __init__(self):
        self.held = False

    @contextmanager
    def native_guard(self, _owner):
        assert not self.held
        self.held = True
        try:
            yield
        finally:
            self.held = False


class _Pool:
    def __init__(self, authority, size, available):
        self.authority = authority
        self.size = size
        self.available = available

    def available_size(self):
        assert self.authority.held
        return self.available


class _Tree:
    def __init__(self, authority):
        self.authority = authority

    def supports_mamba(self):
        return True

    def is_tree_cache(self):
        return True

    def full_evictable_size(self):
        assert self.authority.held
        return 20

    def mamba_evictable_size(self):
        assert self.authority.held
        return 2


def test_agentic_mamba_usage_is_sampled_under_memory_authority_guard():
    authority = _Authority()
    scheduler = SimpleNamespace(
        agentic_p_memory_v2_bridge=None,
        agentic_d_memory_v2_bridge=SimpleNamespace(authority=authority),
        token_to_kv_pool_allocator=_Pool(authority, 1000, 600),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=_Pool(authority, 100, 60)
        ),
        tree_cache=_Tree(authority),
    )
    scheduler._agentic_memory_snapshot_guard = (
        SchedulerRuntimeCheckerMixin._agentic_memory_snapshot_guard.__get__(
            scheduler
        )
    )

    result = SchedulerRuntimeCheckerMixin._get_mamba_token_info(scheduler)

    assert result == (380, 38, 0.38, 0.38, 600, 20, 60, 2)
    assert not authority.held
