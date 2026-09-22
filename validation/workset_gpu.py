"""Actual CUDA descriptor/chunk/fence tests with the production TCP controller.

Uses tiny Attention/Mamba pools, not model weights, NCCL, NIXL or remote KV DMA.
The test driver freezes one exact compute selection over TCP; production must
do that at its native model-broadcast boundary. This does not prove scheduler
integration or two-node SWE correctness/performance.
"""
import argparse
from concurrent.futures import Future
import json
import multiprocessing as mp
import queue
import secrets
import time
import traceback
from types import SimpleNamespace as NS


def exact_header_complete(client, namespace, key, command_id):
    """A previous command's cached ACKs cannot authorize the next header."""
    if not command_id:
        return True
    entry = client.entry(namespace, key)
    return bool(entry and entry["command_id"] == command_id
                and set(entry["command_acks"]) == set(range(client.size)))


def worker(rank, size, address, run_id, token, rounds, timeout, results, finish_gate):
    import torch
    from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, EventKey
    from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
    from sglang.srt.disaggregation.agentic_workset_controller import WorksetController, WorksetIntent
    from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
    from sglang.srt.disaggregation.agentic_workset_runtime import PWorksetRuntime
    from sglang.srt.disaggregation.agentic_workset_tp import FenceProof
    from sglang.srt.mem_cache.memory_pool import MambaPool

    runtime = client = controller = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        forward = torch.cuda.Stream(device=device)
        # Exercise native initialization on real CUDA state, not a mock fence.
        mamba = MambaPool(size=32, spec_state_size=0, device=str(device),
            mamba_layer_ids=[0, 1], cache_params=NS(
                shape=NS(conv=[(4, 8)], temporal=(4, 8)),
                dtype=NS(conv=torch.float32, temporal=torch.float32)))
        req_pool = NS(enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
        broker = AgenticPWorksetLeaseBroker(64, state_allocators=(mamba,),
            mamba_req_to_token_pool=req_pool, reserve_mamba_checkpoint=True)
        client = TPEventClient(address, run_id=run_id, token=token,
                               group="P", rank=rank, size=size)
        client.wait_ready(timeout=timeout)
        if rank == 0:
            controller = WorksetController(WorksetLedger(incarnation=run_id,
                page_count=32, page_size=64, mamba_slots=32, tp_size=size))
        physical = {}
        observers = []

        def reference_fence(plan, scope):
            future = Future()
            observers.append((plan, scope, future))
            return future

        cache = torch.zeros((33 * 64, 8), device=device)
        # Test-only pool initialization boundary, before starting the actor.
        # A faster rank may publish immediately after creating its actor.
        torch.cuda.current_stream(device).synchronize()
        runtime = PWorksetRuntime(client, broker, controller=controller,
            device=device, page_capacity=32, page_size=64, mamba_slots=32,
            mamba_pool=mamba, incarnation=run_id, reference_fence=reference_fence,
            dedicated_client=True)
        verified = 0
        for turn in range(rounds):
            snapshot = f"request:g{turn}"
            key = EventKey(snapshot, "gpu-validation")
            future = None
            if rank == 0:
                future = runtime.request(WorksetIntent(snapshot, f"attempt:{turn}",
                    "fresh", 0, 321, checkpoint_slots=0, runtime_slots=5))
            deadline = time.monotonic() + timeout
            submitted = checked = cancel_sent = cleaned = False
            current_record = None
            while time.monotonic() < deadline:
                runtime.check_health()
                lease = broker.get(snapshot)
                if rank == 0 and future.done():
                    plan = future.result()
                    if plan.key in runtime.ready_cut().ready and not submitted:
                        client.publish_command("gpu-test-compute", key, {"snapshot": snapshot}, command_id=1)
                command = client.next_command("gpu-test-compute", key)
                if command is not None:
                    assert lease is not None, "group compute cut preceded prepared lease"
                    request = NS(origin_input_ids=list(range(321)), req_pool_idx=None,
                        prefix_indices=torch.empty(0, dtype=torch.int64, device=device),
                        mamba_pool_idx=None, mamba_ping_pong_track_buffer=None)
                    with torch.cuda.stream(forward):
                        lease.prepared_descriptor.wait_on(forward)
                        broker.handoff_fresh_to_req(snapshot, request, lease)
                        # Runtime+tracking+two rolling checkpoints were zeroed.
                        state = lease.prepared_descriptor.runtime_indices
                        state_sum = mamba.mamba_cache.temporal[:, state].abs().sum()
                        for conv in mamba.mamba_cache.conv:
                            state_sum = state_sum + conv[:, state].abs().sum()
                            conv.index_fill_(1, state, float(turn + rank + 1))
                        mamba.mamba_cache.temporal.index_fill_(1, state, float(turn + rank + 1))
                        chunks = [broker.consume_suffix(lease, count,
                                  final_prompt_chunk=(i == 2))
                                  for i, count in enumerate((128, 128, 65))]
                        indices = torch.cat(chunks)
                        cache.index_fill_(0, indices, float(rank + 1 + turn))
                        values = cache.index_select(0, indices).sum()
                        done = torch.cuda.Event()
                        done.record(forward)
                    current_record = {
                        "event": done, "unreferenced": False,
                        "keepalive": (request, lease, state, indices, chunks, state_sum, values),
                    }
                    physical[lease.controller_plan.key] = current_record
                    submitted = True
                    client.ack_command("gpu-test-compute", key, command[0])
                if submitted and not checked:
                    if current_record["event"].query():
                        *_, state_sum, values = current_record["keepalive"]
                        assert state_sum.item() == 0
                        assert values.item() == 321 * 8 * (rank + 1 + turn)
                        # The validation workload releases its final references;
                        # only now may its reference observer report quiet.
                        current_record["unreferenced"] = True
                        checked = True
                        verified += 1
                        client.report("gpu-test-verified", key, 1)
                # The controller is allowed to receive close while CUDA is
                # still in flight; this observer must NOT complete early.
                if rank == 0 and submitted and not cancel_sent and client.command_complete("gpu-test-compute", key):
                    runtime.cancel(plan.key)
                    cancel_sent = True
                for observed_plan, scope, fence in tuple(observers):
                    record = physical.get(observed_plan.key)
                    if (record and record["unreferenced"] and record["event"].query()
                            and not fence.done()):
                        fence.set_result(FenceProof(observed_plan.key, scope.sequence, True, True))
                if checked and not cleaned and runtime.counts().live_leases == 0:
                    assert runtime.counts().free_pages == 32
                    assert runtime.counts().free_mamba_slots == 32
                    client.report("gpu-test-cleaned", key, 1)
                    cleaned = True
                if rank == 0 and cancel_sent and controller.ledger.counts.live_leases == 0:
                    assert controller.ledger.counts.free_pages == 32
                    assert controller.ledger.counts.free_mamba_slots == 32
                    if (client.group_status("gpu-test-verified", key) == 1
                            and client.group_status("gpu-test-cleaned", key) == 1):
                        client.publish_receipt("gpu-test-round", key, 1)
                if client.receipt("gpu-test-round", key) == 1:
                    break
                time.sleep(0.002)  # Test driver only; not model scheduler code.
            else:
                raise TimeoutError(f"rank {rank}, round {turn} did not finish")
        client.flush(timeout=timeout)
        results.put({"rank": rank, "ok": True, "rounds": verified})
        if not finish_gate.wait(timeout):
            raise TimeoutError("test coordinator did not acknowledge all-rank cleanup")
    except BaseException:
        results.put({"rank": rank, "ok": False, "error": traceback.format_exc()})
        raise
    finally:
        if runtime is not None:
            runtime.shutdown()
        if controller is not None:
            controller.shutdown()
        if client is not None:
            client.close()


def facade_worker(rank, size, address, run_id, token, rounds, timeout, results, finish_gate):
    """Real facade/native-return/admission wiring; tiny CUDA buffers, no model."""
    import torch
    from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, EventKey
    from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
    from sglang.srt.disaggregation.agentic_workset_admission import FreshWorksetAdmission
    from sglang.srt.disaggregation.agentic_workset_broker import ControllerWorksetBroker
    from sglang.srt.disaggregation.agentic_workset_controller import WorksetController
    from sglang.srt.disaggregation.agentic_workset_ledger import WorksetLedger
    from sglang.srt.disaggregation.agentic_workset_native import NativeLastRefFreeAdapter
    from sglang.srt.disaggregation.agentic_workset_runtime import PWorksetRuntime
    from sglang.srt.disaggregation.agentic_mamba_prefill import fork_prefill_checkpoint
    from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from sglang.srt.mem_cache.common import preflight_controller_worksets, release_kv_cache

    runtime = bridge = controller = client = native_client = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        forward = torch.cuda.Stream(device=device)
        mamba = MambaPool(size=32, spec_state_size=0, device=str(device),
            mamba_layer_ids=[0, 1], cache_params=NS(
                shape=NS(conv=[(4, 8)], temporal=(4, 8)),
                dtype=NS(conv=torch.float32, temporal=torch.float32)))
        allocator = PagedTokenToKVPoolAllocator(32 * 64, 64, torch.float32, str(device), None, False)
        req_pool = NS(mamba_pool=mamba, enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
        tree = NS(token_to_kv_pool_allocator=allocator, req_to_token_pool=req_pool,
                  supports_mamba=lambda: True, page_size=64)
        broker = ControllerWorksetBroker(64, rank=rank, device=device,
            state_allocators=(mamba,), mamba_req_to_token_pool=req_pool, reserve_mamba_checkpoint=True)
        client = TPEventClient(address, run_id=run_id, token=token, group="P-runtime", rank=rank, size=size)
        native_client = TPEventClient(address, run_id=run_id, token=token, group="P-native", rank=rank, size=size)
        client.wait_ready(timeout=timeout)
        native_client.wait_ready(timeout=timeout)
        if rank == 0:
            controller = WorksetController(WorksetLedger(incarnation=run_id,
                page_count=32, page_size=64, mamba_slots=32, tp_size=size))
        cache = torch.zeros((33 * 64, 8), device=device)
        torch.cuda.current_stream(device).synchronize()  # test-only startup boundary
        runtime = PWorksetRuntime(client, broker, controller=controller, device=device,
            page_capacity=32, page_size=64, mamba_slots=32, mamba_pool=mamba,
            incarnation=run_id, reference_fence=broker.reference_fence, dedicated_client=True)
        bridge = NativeLastRefFreeAdapter(incarnation=run_id, rank=rank, page_size=64,
            counts=runtime.counts, on_ready=runtime.native_free)
        broker.attach_runtime(runtime, native_bridge=bridge)
        allocator.install_workset_adapter(bridge)
        mamba.install_workset_adapter(bridge)
        mailbox = SocketTPGroupMailbox("native-workset-admission", tp_rank=rank, tp_size=size, client=native_client)

        def cleanup(req, lease):
            if getattr(req, "_validation_cleaned", False):
                return True
            with torch.cuda.stream(forward):
                if lease is not None and lease.state == "handed":
                    assert broker.release_handed(lease.snapshot_id, lease, req=req)
                elif lease is not None and lease.state == "active":
                    assert broker.cancel_unstarted(lease.snapshot_id)
                indices = getattr(req, "_validation_native_indices", None)
                if indices is not None:
                    allocator.free(indices)
                    req._validation_native_indices = None
                release_kv_cache(req, tree, is_insert=False)
                checkpoint = getattr(req, "_validation_last_checkpoint", None)
                if checkpoint is not None:
                    mamba.free(checkpoint)  # last fake-Radix reference, exact rotation slot
                    req._validation_last_checkpoint = None
            req._validation_cleaned = True
            return True

        admission = FreshWorksetAdmission(broker, mailbox, rank=rank, tp_size=size, on_abort=cleanup)
        verified = 0
        for turn in range(rounds):
            req = NS(rid=f"req-{turn}", origin_input_ids=list(range(321)), req_pool_idx=None,
                prefix_indices=torch.empty(0, dtype=torch.int64, device=device),
                mamba_pool_idx=None, mamba_ping_pong_track_buffer=None,
                extra_key=f"agentic-v1:request:g{turn}")
            admission.register(req, f"request:{turn}", req.extra_key)
            event_key = EventKey(f"round:{turn}", "facade-validation")
            header_id = 0
            submitted = checked = cleaned = False
            result = None
            partial = bool(turn % 2)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                runtime.check_health()
                bridge.check_health()
                broker.check_health()
                if rank == 0:
                    header_done = exact_header_complete(native_client, "facade-native-header", event_key, header_id)
                    all_cleaned = (native_client.group_status("facade-cleaned", event_key) == 1
                                   and native_client.group_status("facade-verified", event_key) == 1)
                    if all_cleaned and header_done:
                        native_client.publish_receipt("facade-round", event_key, 1)
                    elif header_done:
                        header_id += 1
                        control = admission.build_control([req] if admission.contains(req) else [])
                        native_client.publish_command("facade-native-header", event_key, control, command_id=header_id)
                command = native_client.next_command("facade-native-header", event_key)
                if command is not None:
                    admission.apply_control(command[1], admission.req_by_rid())
                    native_client.ack_command("facade-native-header", event_key, command[0])
                if admission.is_committed(req) and not submitted:
                    lease = req._agentic_p_workset_lease
                    with torch.cuda.stream(forward):
                        lease.prepared_descriptor.wait_on(forward)
                        state = lease.prepared_descriptor.runtime_indices
                        zero = mamba.mamba_cache.temporal[:, state].abs().sum()
                        mamba.mamba_cache.temporal.index_fill_(1, state, float(turn + rank + 1))
                        previous, chunks, prefix = None, [], 0
                        for count in ((128,) if partial else (128, 128, 65)):
                            batch = NS(reqs=[req], prefix_lens=[prefix], extend_lens=[count],
                                       req_to_token_pool=req_pool, tree_cache=tree)
                            preflight_controller_worksets(batch)
                            chunk = broker.consume_suffix(lease, count, final_prompt_chunk=prefix + count == 321)
                            chunks.append(chunk)
                            req.prefix_indices = torch.cat(chunks)
                            req._agentic_workset_suffix_indices = lease.remaining_suffix_indices
                            checkpoint = fork_prefill_checkpoint(req, mamba, req.mamba_pool_idx.reshape(1))
                            if previous is not None:
                                mamba.free(previous)
                            previous = checkpoint
                            prefix += count
                        req._validation_last_checkpoint = previous
                        indices = torch.cat(chunks)
                        cache.index_fill_(0, indices, float(turn + rank + 1))
                        value = cache.index_select(0, indices).sum()
                        done = torch.cuda.Event()
                        done.record(forward)
                        req._validation_native_indices = indices if partial else lease.device_indices
                    result = done, zero, value, indices.numel()
                    submitted = True
                if submitted and not checked and result[0].query():
                    assert result[1].item() == 0
                    assert result[2].item() == result[3] * 8 * (turn + rank + 1)
                    verified += 1
                    checked = True
                    native_client.report("facade-verified", event_key, 1)
                    if partial:
                        admission.cancel(req)  # common next ABORT, not per-rank release
                    else:
                        cleanup(req, None)
                        admission.terminal(req)
                if checked and not cleaned and runtime.counts().live_leases == 0 and not admission.contains(req):
                    assert runtime.counts().free_pages == 32
                    assert runtime.counts().free_mamba_slots == 32
                    native_client.report("facade-cleaned", event_key, 1)
                    cleaned = True
                if native_client.receipt("facade-round", event_key) == 1:
                    break
                time.sleep(.002)  # bounded validation driver, never engine scheduling
            else:
                raise TimeoutError(f"facade rank {rank}, round {turn} did not finish")
        results.put({"rank": rank, "ok": True, "rounds": verified,
                     "partial_abort_rounds": rounds // 2, "facade": True})
        if not finish_gate.wait(timeout):
            raise TimeoutError("coordinator did not acknowledge facade cleanup")
    except BaseException:
        results.put({"rank": rank, "ok": False, "error": traceback.format_exc()})
        raise
    finally:
        if bridge is not None:
            bridge.shutdown()
        if runtime is not None:
            runtime.shutdown()
        if controller is not None:
            controller.shutdown()
        for connection in (native_client, client):
            if connection is not None:
                connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=[1, 2, 8], default=1)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--mode", choices=["component", "facade"], default="component",
                        help="facade adds actual native admission, free bridge and checkpoint rotation; neither mode runs a model")
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout <= 0:
        parser.error("positive rounds/timeout required")
    from sglang.srt.disaggregation.agentic_tp_events import TPEventServer
    run_id, token = "workset-gpu-" + secrets.token_hex(8), secrets.token_hex(24)
    server = TPEventServer(run_id, token)
    context = mp.get_context("spawn")
    results = context.Queue()
    finish_gate = context.Event()
    target = worker if args.mode == "component" else facade_worker
    children = [context.Process(target=target, args=(rank, args.tp, server.address,
                run_id, token, args.rounds, args.timeout, results, finish_gate)) for rank in range(args.tp)]
    reports = []
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + args.timeout * (args.rounds + 2)
        while len(reports) < args.tp and time.monotonic() < deadline:
            try:
                report = results.get(timeout=1)
                reports.append(report)
                if not report["ok"]:
                    break
            except queue.Empty:
                if any(child.exitcode not in (None, 0) for child in children):
                    break
        finish_gate.set()
        for child in children:
            child.join(timeout=5)
        ok = (len(reports) == args.tp and all(r["ok"] for r in reports)
              and all(child.exitcode == 0 for child in children))
        print(json.dumps({"test": "workset_cuda_component", "tp": args.tp,
            "mode": args.mode, "ok": ok, "ranks": reports, "model_test": False}, indent=2), flush=True)
        if not ok:
            raise SystemExit(1)
    finally:
        finish_gate.set()
        # Exact child objects only. No process-name kill, GPU reset or other jobs.
        for child in children:
            if child.pid and child.is_alive():
                child.terminate()
        for child in children:
            if child.pid:
                child.join(timeout=10)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=10)
        server.close()
        results.close()


if __name__ == "__main__":
    main()
