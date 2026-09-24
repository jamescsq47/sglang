from collections import deque
from contextlib import nullcontext
import queue
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation import agentic_multinode
from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey
from sglang.srt.disaggregation.agentic_multinode_runtime import (
    EndpointActivationTicket,
)
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler


class StopLoop(Exception):
    pass


class FakeRuntime:
    def __init__(self, *, p_bridge=None, d_bridge=None):
        self.p_memory_bridge = p_bridge
        self.d_memory_bridge = d_bridge
        self.submitted = []
        self.progress_count = 0
        self.close_count = 0

    def submit_request(self, req, *, is_retracted=False):
        self.submitted.append((req, is_retracted))

    def progress_nonblocking(self):
        self.progress_count += 1

    def is_idle(self):
        return True

    def take_activation_ticket(self, timeout=None):
        raise NotImplementedError

    def activate_staged(self, ticket):
        return None

    def confirm_scheduler_adopted(self, ticket):
        return None

    def close(self):
        self.close_count += 1


def test_runtime_factory_is_lazy_and_checks_contract(monkeypatch):
    from sglang.srt.disaggregation import agentic_multinode_composite

    runtime = FakeRuntime(p_bridge=object())
    monkeypatch.setattr(
        agentic_multinode_composite,
        "create_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    assert (
        agentic_multinode.create_agentic_multinode_runtime(
            object(), object()
        )
        is runtime
    )


def test_v2_running_status_does_not_construct_legacy_objects(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "TPGroupMailbox",
        lambda *_args, **_kwargs: pytest.fail("legacy mailbox constructed"),
    )
    monkeypatch.setattr(scheduler_module, "SessionController", lambda tree: ("s", tree))
    target = SimpleNamespace(
        agentic_multinode_config=object(),
        server_args=SimpleNamespace(page_size=64),
        tree_cache=object(),
    )
    target._init_common_running_status = lambda: Scheduler._init_common_running_status(
        target
    )

    Scheduler.init_running_status(target)

    assert target.agentic_p_workset_broker is None
    assert target.agentic_tp_direct_mailbox is None
    assert target.agentic_tp_host_mailbox is None
    assert target.running_batch.is_empty()


@pytest.mark.parametrize(
    ("mode", "p_bridge", "d_bridge", "bridge_attr"),
    [
        ("prefill", object(), None, "agentic_p_memory_v2_bridge"),
        ("decode", None, object(), "agentic_d_memory_v2_bridge"),
    ],
)
def test_v2_disaggregation_returns_before_legacy_queue_init(
    monkeypatch, mode, p_bridge, d_bridge, bridge_attr
):
    config = object()
    runtime = FakeRuntime(p_bridge=p_bridge, d_bridge=d_bridge)
    monkeypatch.setattr(
        agentic_multinode,
        "validate_multinode_runtime",
        lambda *_args: config,
    )
    monkeypatch.setattr(
        agentic_multinode,
        "create_agentic_multinode_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    target = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode=mode,
            disaggregation_transfer_backend="nixl",
        ),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
        draft_worker=None,
        spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
        agentic_multinode_runtime_v2=None,
        agentic_p_memory_v2_bridge=None,
        agentic_d_memory_v2_bridge=None,
    )

    Scheduler.init_disaggregation(target)

    assert target.agentic_multinode_config is config
    assert target.agentic_multinode_runtime_v2 is runtime
    assert getattr(target, bridge_attr) is (p_bridge if mode == "prefill" else d_bridge)
    assert not hasattr(target, "disagg_prefill_bootstrap_queue")
    assert not hasattr(target, "disagg_decode_prealloc_queue")


def test_v2_request_admission_never_touches_legacy_queues():
    runtime = FakeRuntime()
    req = object()
    target = SimpleNamespace(agentic_multinode_runtime_v2=runtime)

    Scheduler._add_request_to_queue(target, req, is_retracted=True)

    assert runtime.submitted == [(req, True)]


def test_v2_idle_check_never_reads_legacy_pd_queues():
    runtime = FakeRuntime()
    empty = SimpleNamespace(is_empty=lambda: True)
    target = SimpleNamespace(
        running_batch=empty,
        chunked_req=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None,
        cur_batch=None,
        enable_overlap=True,
        result_queue=[],
        pp_size=1,
        waiting_queue=[],
        agentic_multinode_runtime_v2=runtime,
    )

    assert Scheduler.is_fully_idle(target)
    assert Scheduler.is_fully_idle(target, for_health_check=True)


def test_v2_health_probe_never_enters_generation_lifecycle():
    target = SimpleNamespace(
        session_controller=SimpleNamespace(maybe_reap=lambda _now: None),
        agentic_multinode_runtime_v2=FakeRuntime(),
        return_health_check_ipcs=deque(),
        _request_dispatcher=lambda _req: pytest.fail("health probe was dispatched"),
        _check_pending_flush=lambda: None,
        agentic_host_staging_manager=None,
        _drain_agentic_kv_waiting_queue=lambda: None,
    )
    probe = SimpleNamespace(rid="HEALTH_CHECK_v2", http_worker_ipc="ipc")

    Scheduler.process_input_requests(target, [probe])

    assert list(target.return_health_check_ipcs) == ["ipc"]


def test_disabled_v2_preserves_native_request_admission():
    events = []
    req = SimpleNamespace(
        time_stats=SimpleNamespace(
            set_wait_queue_entry_time=lambda: events.append("queued")
        )
    )
    target = SimpleNamespace(
        agentic_multinode_runtime_v2=None,
        disaggregation_mode=DisaggregationMode.NULL,
        _set_or_validate_priority=lambda _req: True,
        _abort_on_queued_limit=lambda _req: False,
        _prefetch_kvcache=lambda _req: events.append("prefetch"),
        waiting_queue=[],
    )

    Scheduler._add_request_to_queue(target, req)

    assert target.waiting_queue == [req]
    assert events == ["prefetch", "queued"]


def test_v2_prefill_event_loop_only_progresses_runtime_and_ready_bridge(monkeypatch):
    runtime = FakeRuntime()
    calls = 0

    def recv_requests():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise StopLoop
        return ()

    target = SimpleNamespace(
        agentic_multinode_runtime_v2=runtime,
        recv_requests=recv_requests,
        process_input_requests=lambda _reqs: None,
        get_next_disagg_prefill_batch_to_run=lambda: None,
        self_check_during_idle=lambda: None,
        last_batch=None,
    )
    with pytest.raises(StopLoop):
        SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill(target)
    assert runtime.progress_count == 2


def test_v2_chunk_boundary_never_enters_legacy_kv_sender():
    bridge = object()
    req = SimpleNamespace(_agentic_p_workset_broker=bridge)
    calls = []
    target = SchedulerDisaggregationPrefillMixin()
    target.agentic_p_memory_v2_bridge = bridge
    target.send_kv_chunk = lambda *_args, **kwargs: calls.append(kwargs)

    sent = SchedulerDisaggregationPrefillMixin._send_legacy_prefill_chunk_if_needed(
        target, req, last_chunk=False, end_idx=128
    )

    assert not sent
    assert calls == []


def test_v1_chunk_boundary_preserves_legacy_kv_sender():
    calls = []
    req = SimpleNamespace()
    target = SchedulerDisaggregationPrefillMixin()
    target.agentic_p_memory_v2_bridge = None
    target.send_kv_chunk = lambda sent_req, **kwargs: calls.append(
        (sent_req, kwargs)
    )

    sent = SchedulerDisaggregationPrefillMixin._send_legacy_prefill_chunk_if_needed(
        target, req, last_chunk=False, end_idx=128
    )

    assert sent
    assert calls == [(req, {"last_chunk": False, "end_idx": 128})]


def test_v2_decode_event_loop_only_progresses_runtime_and_ready_bridge():
    runtime = FakeRuntime()
    calls = 0

    def recv_requests():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise StopLoop
        return ()

    target = SimpleNamespace(
        agentic_multinode_runtime_v2=runtime,
        recv_requests=recv_requests,
        process_input_requests=lambda _reqs: None,
        process_decode_queue=lambda: None,
        get_next_disagg_decode_batch_to_run=lambda: None,
        self_check_during_idle=lambda: None,
        last_batch=None,
    )
    with pytest.raises(StopLoop):
        SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode(target)
    assert runtime.progress_count == 2


def test_event_loop_always_closes_v2_runtime(monkeypatch):
    runtime = FakeRuntime()
    target = SimpleNamespace(
        device="cpu",
        device_module=SimpleNamespace(
            Stream=lambda priority: SimpleNamespace(synchronize=lambda: None),
            StreamContext=lambda _stream: nullcontext(),
        ),
        agentic_multinode_runtime_v2=runtime,
    )
    monkeypatch.setattr(
        scheduler_module,
        "dispatch_event_loop",
        lambda _scheduler: (_ for _ in ()).throw(StopLoop()),
    )
    with pytest.raises(StopLoop):
        Scheduler.run_event_loop(target)
    assert runtime.close_count == 1
    assert target.agentic_multinode_runtime_v2 is None


def test_tp8_activation_order_is_chosen_once_and_applied_on_every_rank():
    tickets = queue.Queue()
    tickets.put(
        EndpointActivationTicket(
            GenerationKey("run", "request", 3),
            7,
            "lease-7",
            "decode",
            4,
        )
    )
    applied = []
    activated = []
    confirmed = []

    class Runtime:
        config = SimpleNamespace(run_id="run")

        def __init__(self, rank):
            self.rank = rank
            self.d_memory_bridge = SimpleNamespace(
                activate_decode_attempts=self._activate_attempts
            )

        def take_activation_ticket(self, timeout=None):
            return tickets.get(timeout=timeout)

        def activate_staged(self, ticket):
            applied.append((self.rank, ticket.attempt, ticket.lease_id))

        def confirm_scheduler_adopted(self, ticket):
            confirmed.append((self.rank, ticket.attempt, ticket.lease_id))

        def _activate_attempts(self, attempts):
            activated.append((self.rank, tuple(attempts)))
            return (f"req-r{self.rank}",)

    schedulers = []
    for rank in range(8):
        scheduler = SimpleNamespace(
            tp_rank=rank,
            _AGENTIC_V2_ACTIVATION_KEY=Scheduler._AGENTIC_V2_ACTIVATION_KEY,
            disaggregation_mode=DisaggregationMode.DECODE,
            agentic_multinode_runtime_v2=Runtime(rank),
            waiting_queue=[],
        )
        schedulers.append(scheduler)

    control = Scheduler._agentic_v2_prepare_activation_control(schedulers[0])
    assert control is not None
    assert Scheduler._agentic_v2_prepare_activation_control(schedulers[1]) is None
    for scheduler in schedulers:
        Scheduler._agentic_v2_consume_activation_control(scheduler, control)

    assert applied == [(rank, 7, "lease-7") for rank in range(8)]
    assert confirmed == [(rank, 7, "lease-7") for rank in range(8)]
    assert [value[0] for value in activated] == list(range(8))
    assert [scheduler.waiting_queue for scheduler in schedulers] == [
        [f"req-r{rank}"] for rank in range(8)
    ]
