"""Startup-only wiring for the single-authority Prefill memory controller.

No model/transport policy lives here. Installation is allowed only before the
first request; native allocators lose their free lists before the actor can
receive work. The controller connection is distinct from model collectives and
the existing I/O notification client. Startup may wait; Forward never calls
these functions.
"""
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


def controller_requested(server_args):
    value = os.getenv("SGLANG_AGENTIC_P_WORKSET_CONTROLLER", "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("SGLANG_AGENTIC_P_WORKSET_CONTROLLER must be 0 or 1")
    return value in {"1", "true"} and server_args.disaggregation_mode == "prefill"


def validate_controller_configuration(scheduler):
    """Reject unconverted allocation paths rather than secretly falling back."""
    args = scheduler.server_args
    if args.disaggregation_mode != "prefill":
        raise ValueError("workset authority is a Prefill-only component")
    if not os.getenv("SGLANG_AGENTIC_CONTROL_ENDPOINT"):
        raise ValueError("workset authority requires the socket lifecycle backend")
    if not os.getenv("SGLANG_PD_P_READY_DIR"):
        raise ValueError("workset authority requires deferred full-snapshot P-ready delivery")
    if os.getenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "hysteresis") != "disabled":
        raise ValueError("complete worksets require disabled legacy P-ready HBM throttling")
    unsupported = ("enable_hierarchical_cache", "enable_hicache_storage",
                   "is_hybrid_swa", "enable_priority_preemption", "is_mixed_chunk",
                   "enable_lora")
    active = [name for name in unsupported if getattr(scheduler, name, False)]
    if (getattr(scheduler, "draft_worker", None) is not None
            or getattr(args, "pp_size", 1) != 1
            or getattr(args, "dp_size", 1) != 1
            or getattr(args, "attn_cp_size", 1) != 1):
        active.append("draft/PP/DP/CP")
    if active:
        raise ValueError("unconverted workset allocation paths: " + ", ".join(active))
    pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
    if pool is not None:
        if not getattr(scheduler.req_to_token_pool, "enable_mamba_extra_buffer", False):
            raise ValueError("hybrid worksets require preallocated tracking buffers")
        if not getattr(scheduler, "_agentic_mamba_prefill_admission", False):
            raise ValueError("hybrid worksets require request-owned Prefill state")
    for name in ("waiting_queue", "agentic_kv_waiting_queue"):
        if getattr(scheduler, name, ()):
            raise ValueError("workset authority must start before request ingress")


@dataclass
class EngineWorksetRuntime:
    broker: object
    runtime: object
    native_bridge: object
    client: object
    controller: object

    def check_health(self):
        self.runtime.check_health()
        self.native_bridge.check_health()


def install_controller(scheduler, broker):
    """Called once by Scheduler after pools exist, before I/O workers start."""
    import torch
    from sglang.srt.disaggregation.agentic_tp_events import TPEventClient
    from sglang.srt.disaggregation.agentic_workset_controller import WorksetController
    from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
    from sglang.srt.disaggregation.agentic_workset_native import NativeLastRefFreeAdapter
    from sglang.srt.disaggregation.agentic_workset_runtime import PWorksetRuntime
    from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator, TokenToKVPoolAllocator

    validate_controller_configuration(scheduler)
    allocator = scheduler.token_to_kv_pool_allocator
    if type(allocator) not in {PagedTokenToKVPoolAllocator, TokenToKVPoolAllocator}:
        raise ValueError("unconverted KV allocator wrapper; refusing a second authority")
    page_size = allocator.page_size
    pages = allocator.size // page_size
    mamba_pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
    slots = 0 if mamba_pool is None else mamba_pool.size
    if allocator.available_size() != pages * page_size:
        raise ValueError("live Attention pool cannot be imported as empty")
    if mamba_pool is not None and mamba_pool.available_size() != slots:
        raise ValueError("live Mamba pool cannot be imported as empty")
    endpoint = os.getenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT", "")
    run = os.getenv("SGLANG_AGENTIC_CONTROL_RUN_ID", "")
    token = os.getenv("SGLANG_AGENTIC_CONTROL_TOKEN", "")
    group = os.getenv("SGLANG_AGENTIC_CONTROL_GROUP_ID", "")
    if not all((endpoint, run, token, group)):
        raise ValueError("socket workset endpoint/run/token/group are required")
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    # Model initialization uses the native stream. Preparation on a new stream
    # must not race it. This is an initialization barrier, never a request wait.
    device = torch.device(allocator.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.current_stream(device).synchronize()
    broker.device = device
    client = TPEventClient((host, int(port)), run_id=run, token=token,
        group=group + ":workset", rank=scheduler.tp_rank, size=scheduler.tp_size)
    controller = runtime = bridge = None
    try:
        client.wait_ready()
        incarnation = run + ":" + group
        if scheduler.tp_rank == 0:
            controller = WorksetController(WorksetLedger(incarnation=incarnation,
                page_count=pages, page_size=page_size, mamba_slots=slots,
                tp_size=scheduler.tp_size))
        runtime = PWorksetRuntime(client, broker, controller=controller,
            device=device, page_capacity=pages, page_size=page_size,
            mamba_pool=mamba_pool, mamba_slots=slots, incarnation=incarnation,
            reference_fence=broker.reference_fence, dedicated_client=True)
        bridge = NativeLastRefFreeAdapter(incarnation=incarnation,
            rank=scheduler.tp_rank, page_size=page_size, counts=runtime.counts,
            on_ready=runtime.native_free)
        broker.attach_runtime(runtime, native_bridge=bridge)
        allocator.install_workset_adapter(bridge)
        if mamba_pool is not None:
            mamba_pool.install_workset_adapter(bridge)
        return EngineWorksetRuntime(broker, runtime, bridge, client, controller)
    except BaseException:
        # Startup has admitted no request and performed no transport. Never
        # reset a partially installed native pool and resume the old allocator.
        for component, method in ((bridge, "shutdown"), (runtime, "shutdown"),
                                  (controller, "shutdown"), (client, "close")):
            if component is not None:
                try:
                    getattr(component, method)()
                except BaseException:
                    logger.exception("Workset startup cleanup failed: %s", type(component).__name__)
        raise
