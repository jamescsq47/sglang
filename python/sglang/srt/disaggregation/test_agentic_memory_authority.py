import threading
import time

import torch

from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    LeasePhase,
    RequestGenerationAttempt,
)


class FakeAllocator:
    def __init__(self, size=1024, page_size=64):
        self.page_size = page_size
        self._free = list(range(size))
        self.live = set()

    def available_size(self):
        return len(self._free)

    def alloc(self, count):
        if count > len(self._free):
            return None
        values = self._free[:count]
        del self._free[:count]
        assert not self.live.intersection(values)
        self.live.update(values)
        return torch.tensor(values, dtype=torch.int64)

    def free(self, indices):
        values = [int(x) for x in indices.tolist()]
        assert set(values).issubset(self.live)
        self.live.difference_update(values)
        self._free.extend(values)


def key(n=0, attempt=0):
    return RequestGenerationAttempt(f"req-{n}", n, attempt)


def test_prefill_workset_rounds_parent_and_suffix_independently():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    lease = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=65, prompt_tokens=66
    )
    assert lease.parent_allocated_tokens == 128
    assert lease.prompt_allocated_tokens == 192
    assert len(lease.parent_indices) == 128
    assert len(lease.suffix_indices) == 64
    assert authority.available_tokens() == 1024 - 192


def test_capacity_sink_runs_once_after_release_and_outside_authority_lock():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    observed = []

    def on_capacity(available):
        # Re-entering the authority is safe because notification happens only
        # after the allocator transaction has released its lock.
        observed.append((available, authority.available_tokens()))

    authority.install_capacity_available_sink(on_capacity)
    lease = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    assert authority.request_release(lease.lease_id)
    assert authority.commit_release(lease.lease_id)
    assert observed == [(1024, 1024)]
    assert authority.commit_release(lease.lease_id)
    assert observed == [(1024, 1024)]


def test_state_failure_rolls_back_whole_workset():
    allocator = FakeAllocator()

    class NoState:
        def alloc(self, _count):
            return None

        def free(self, _indices):
            raise AssertionError("nothing was allocated")

    authority = AgenticMemoryAuthority(allocator, state_allocators=(NoState(),))
    assert (
        authority.reserve_prefill_workset(
            key(),
            owner="slow",
            parent_tokens=64,
            prompt_tokens=128,
            state_slot_counts=(1,),
        )
        is None
    )
    assert allocator.available_size() == 1024
    assert allocator.live == set()


def test_decode_credit_is_capacity_not_a_second_free_list():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    lease = authority.reserve_decode(
        key(), owner="p2d", prompt_tokens=65, decode_growth_tokens=64
    )
    assert len(allocator.live) == 128
    assert lease.growth_reserved_tokens == 64
    assert authority.available_tokens() == 1024 - 128 - 64

    assert authority.publish_ready(lease.lease_id, "decode-ready") is not None
    assert authority.adopt_ready(lease.lease_id)

    grown_pages = []

    def release_native(bound_lease):
        allocator.free(bound_lease.device_indices)
        for indices in grown_pages:
            allocator.free(indices)

    assert authority.install_release_handler(lease.lease_id, release_native)

    def alloc_one_page(raw):
        value = raw.alloc(64)
        grown_pages.append(value)
        return value

    grown = authority.run_decode_growth(lease.lease_id, alloc_one_page)
    assert len(grown) == 64
    assert authority.available_tokens() == 1024 - 192

    assert authority.finish_compute(lease.lease_id)
    assert authority.request_release(lease.lease_id)
    assert authority.commit_release(lease.lease_id)
    assert allocator.available_size() == 1024


def test_ready_queue_is_edge_driven_and_lease_is_adopted_without_allocation():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    lease = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    started = threading.Event()
    result = []

    def consumer():
        started.set()
        result.extend(
            authority.take_ready("prefill-ready", timeout=1.0, max_items=4)
        )

    thread = threading.Thread(target=consumer)
    thread.start()
    started.wait()
    time.sleep(0.01)
    before = allocator.available_size()
    authority.publish_ready(lease.lease_id, "prefill-ready")
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert [event.lease.lease_id for event in result] == [lease.lease_id]
    assert authority.adopt_ready(lease.lease_id)
    assert allocator.available_size() == before


def test_tp0_activation_takes_exact_ready_attempts_atomically():
    authority = AgenticMemoryAuthority(FakeAllocator())
    first = authority.reserve_prefill_workset(
        key(20), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    second = authority.reserve_prefill_workset(
        key(21), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    authority.publish_ready(first.lease_id, "prefill-ready")

    try:
        authority.take_ready_attempts(
            "prefill-ready", (first.key, second.key)
        )
    except RuntimeError as exc:
        assert "not staged" in str(exc)
    else:
        raise AssertionError("partial TP activation must fail closed")

    # The failed all-or-nothing take did not consume the first attempt.
    authority.publish_ready(second.lease_id, "prefill-ready")
    events = authority.take_ready_attempts(
        "prefill-ready", (second.key, first.key)
    )
    assert [event.lease.lease_id for event in events] == [
        second.lease_id,
        first.lease_id,
    ]


def test_release_waits_for_real_io_terminal_and_is_idempotent():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    lease = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    assert authority.begin_io(lease.lease_id, "dma-1")
    assert authority.request_release(lease.lease_id)
    assert not authority.commit_release(lease.lease_id)
    assert not authority.complete_io(lease.lease_id, "stale", success=True)
    assert authority.complete_io(lease.lease_id, "dma-1", success=False)
    assert authority.phase(lease.lease_id) is LeasePhase.RELEASE_PENDING
    assert authority.commit_release(lease.lease_id, reason="cancelled")
    assert authority.commit_release(lease.lease_id, reason="duplicate")
    assert allocator.available_size() == 1024


def test_ready_event_replay_is_idempotent_and_cancelled_event_is_skipped():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    lease = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    first = authority.publish_ready(lease.lease_id, "prefill-ready")
    assert authority.publish_ready(lease.lease_id, "prefill-ready") is first
    assert authority.request_release(lease.lease_id)
    assert authority.commit_release(lease.lease_id)
    assert authority.take_ready("prefill-ready", max_items=1) == ()


def test_attempt_identity_is_idempotent_but_shape_change_is_rejected():
    authority = AgenticMemoryAuthority(FakeAllocator())
    first = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    replay = authority.reserve_prefill_workset(
        key(), owner="direct", parent_tokens=64, prompt_tokens=128
    )
    assert replay is first
    try:
        authority.reserve_prefill_workset(
            key(), owner="direct", parent_tokens=64, prompt_tokens=192
        )
    except RuntimeError as exc:
        assert "another lease" in str(exc)
    else:
        raise AssertionError("shape-changing replay must fail")


def test_concurrent_reservations_never_duplicate_physical_indices():
    allocator = FakeAllocator(size=4096)
    authority = AgenticMemoryAuthority(allocator)
    leases = []
    lock = threading.Lock()

    def reserve(i):
        lease = authority.reserve_prefill_workset(
            key(i), owner="direct", parent_tokens=64, prompt_tokens=128
        )
        with lock:
            leases.append(lease)

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    all_indices = [
        int(value)
        for lease in leases
        for value in lease.device_indices.tolist()
    ]
    assert len(all_indices) == len(set(all_indices))
    for lease in leases:
        assert authority.request_release(lease.lease_id)
        assert authority.commit_release(lease.lease_id)
    assert allocator.available_size() == 4096


def test_native_guard_serializes_scheduler_and_controller_writers():
    allocator = FakeAllocator()
    authority = AgenticMemoryAuthority(allocator)
    scheduler_entered = threading.Event()
    scheduler_release = threading.Event()
    controller_finished = threading.Event()

    def scheduler_mutation():
        with authority.native_guard("scheduler-prepare"):
            assert authority.native_guard_held()
            authority.assert_native_guard()
            scheduler_entered.set()
            scheduler_release.wait(timeout=1.0)

    def controller_reservation():
        scheduler_entered.wait(timeout=1.0)
        authority.reserve_prefill_workset(
            key(99), owner="direct", parent_tokens=64, prompt_tokens=128
        )
        controller_finished.set()

    scheduler_thread = threading.Thread(target=scheduler_mutation)
    controller_thread = threading.Thread(target=controller_reservation)
    scheduler_thread.start()
    controller_thread.start()
    assert scheduler_entered.wait(timeout=1.0)
    time.sleep(0.02)
    assert not controller_finished.is_set()
    scheduler_release.set()
    scheduler_thread.join(timeout=1.0)
    controller_thread.join(timeout=1.0)
    assert not scheduler_thread.is_alive()
    assert not controller_thread.is_alive()
    assert controller_finished.is_set()


def test_native_guard_assertion_is_thread_local_and_reentrant():
    authority = AgenticMemoryAuthority(FakeAllocator())
    try:
        authority.assert_native_guard()
    except RuntimeError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("unguarded native mutation must fail")

    with authority.native_guard("outer"):
        authority.assert_native_guard()
        with authority.native_guard("inner"):
            authority.assert_native_guard()
        authority.assert_native_guard()
    assert not authority.native_guard_held()
