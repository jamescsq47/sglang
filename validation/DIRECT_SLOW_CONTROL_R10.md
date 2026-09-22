# R10: attempt identity and post-copy manifest completion

Status: implementation/review in progress. No GPU acceptance claimed.

Same c128 SWE500/Qwen3.5-122B TP8 per node, mem 0.8, Mamba ratio 0.5,
tool 2 s / Direct setup 1 s, both D2P paths, congestion recompute disabled.
Keep model/workload/order/Host capacities and P2D protocol unchanged.

## Bounded changes

1. Direct freezes snapshot/room/lease/claim attempt identity before handoff.
   Progress of an already-completed operation must never require reading the
   resource lease that native handoff legitimately transferred to the Req.
2. Existing Host worker completes the rank0 logical manifest fence after
   authoritative all-rank bound/CONSUMED. Exact-context ACK precedes rank0
   phase3; native COMMIT consumes the proof instead of starting another RPC.
   Native BIND/COMMIT/ADMIT and all physical DMA fences remain unchanged.
3. Qwen multi-node launcher explicitly sets model workers' Triton cache under
   node-local `/tmp/dualpd-multinode/compiler-cache/<engine>/triton` rather
   than default NFS home. Both nodes' backing mounts checked ext4, with over
   400 GiB free. Content-addressed compiler cache persists across runs; no
   lifecycle data or control records use it. Other launchers remain opt-out.

## Ownership/cancellation and gate

No policy changes to routing, capacity, eviction, reuse, tool/setup deadlines
or source release. Direct unknown RPC outcomes still retain ownership; timeout
is never a DMA fence. Slow context cancellation invalidates old callbacks and
retains existing physical retirement. TP1 uses its established path.

Before GPU execution, require exact-attempt/stale callback/partial-rank tests,
the full CPU gate and independent GO. Evaluate Direct setup success and
post-copy group handoff/slot recovery, not merely process liveness or smoke.
The compiler-cache change is an additional variable: do not attribute all
throughput changes exclusively to control-path edits.

## Acceptance mapping before GPU

1. Unique owner: immutable Direct identity survives handoff; Host manifest
   finalization requires the same claim/lease/read epoch and full binder ACKs.
2. P2D Direct release: unchanged.
3. P2D Host durable release: unchanged.
4. D2P Host durable/source release: unchanged; real-ledger TP2/8 tests prove
   source release and prune cannot erase unfinished P handoff evidence.
5. Progress: manifest RPC runs on existing workers, scheduler consumes proof;
   no new threads, blocking scheduler RPC or NFS runtime coordination.
6. TP: native allocation/BIND/COMMIT/ADMIT and physical fences remain. Rank
   lag, cancellation, old completion and delayed ACK tests retain ownership.
7. Reuse: no new recompute/eviction trigger or page/checkpoint changes.
8. Gate: full CPU regression and independent review below, then GPU smoke and
   monitored workload. Passing CPU tests is not performance acceptance.

The broader regression exposed one stale D cleanup fixture: it supplied only
snapshot/chunks to a now-full local-copy progress entry point. The test now
executes the real `_cleanup_write` fence, then verifies ABORTING writer ACK
only after cleanup. Production D behavior is unchanged; all 36 tests in that
file pass, and the file is now included in the standard multi-node gate.

New Direct start-log diagnostics split intent-to-plan, plan-to-service,
service-to-grant, grant-to-receipt, receipt echo and receipt-to-claim. They use
local monotonic times, no per-tick logging or extra synchronization. Missing
stages are -1 and excluded from distributions, not interpreted as zero latency.

## Frozen gate and launch

Full `validation/check_multinode_cpu.sh`: **1323 passed, 2 GPU-only skipped**;
launcher/diagnostics **86 passed**. Raw log `/tmp/dualpd-r10-cpu-gate.log`.
Independent `control_audit`: **GO** after 65 Direct, 130 Host and 54 launcher
checks. Both repositories pass diff-check. This is permission for the GPU
validation, not performance acceptance.

Launch: `qwen35-122b-swe500-tp8-c128-socket-full-r10`, using
`bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --run-dir
/homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-full-r10`.
Both nodes' GPU compute process lists and experiment listener ports were empty
before launch. Smoke and workload outcomes remain pending.

### Outcome: improved Slow handoff; Direct objective still unmet

Direct/Slow two-turn token equality smoke passed; c128 workload began at
07:04:47 UTC. Stopped via verified coordinator PID 176970 SIGTERM at about
07:11 UTC because Direct setup success remained poor. No P/D traceback or the
R9 lease-identity exception was observed. Owned coordinator exited through
cleanup; both nodes' GPU process lists, run-labelled containers and coordinator
were checked absent. This is an interrupted diagnostic, not a final benchmark.

Excluding smoke: P Direct starts 563, completions 462; Slow selected 541,
copy-complete 537, group-release 537. D unique fallback 627, Host offers 626,
D2H complete 624, source release 623; the outstanding source-write/release
items were interrupted by shutdown, not counted as completed. D2H wall mean
156 ms. The final P log window ended at 07:11:19; completed Slow copy-to-P-group
release mean 1.315 s/P90 2 s, all-rank copy-to-first-release 1.216 s versus R8
about 2.2 s. This is a diagnostic cohort, not a controlled throughput ablation.

In that last window Direct arrival-to-start still averaged 611 ms/P90 1008 ms.
New measurements: intent-to-plan 263 ms, plan-to-service 27 ms,
service-to-grant 4.7 ms, grant-to-receipt 4 ms, receipt echo 55 ms,
receipt-to-claim 1.3 ms, claim 98 ms, metadata 3 ms. The physical allocation
itself is not the dominant measured wait. Failed attempts absent from start
logs must be examined separately; successful-start distributions alone cannot
attribute all fallback.

Metrics sample interval 07:08:21.617–07:10:15.286 (113.669 s) gave Decode
463.65 tokens/s, P Forward 74.95%, D Forward 81.90% per physical GPU. Earlier
startup intervals were lower; none constitutes a 300+1200 s performance result.
The new local Triton cache was cold: 20 s P rank0 CPU profile had 94/347 main
samples in compiler stacks and 25/347 in cache stacks. It does not prove the
remaining native-boundary waits are all GPU compute or all filesystem I/O.

Next root cause: socket TP Direct cap counts unmaterialized active grants as
`active - receives`. A completed receive counts zero, but native handoff removes
it while the TP active record awaits group5 cleanup, making it count one again.
The existing local_admitted fact proves the workset has already transferred to
Req and must exclude this bookkeeping-only state from admission slots. Keep
pending claims and unfenced cancellations counted. Independent read-only review
confirmed the accounting bug; its contribution to fallback remains unquantified.
