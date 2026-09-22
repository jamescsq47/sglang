# R4: explicit Host terminal admission (2026-09-21)

## Diagnosis and scope

R3 had requests with D Host writes/release completed but no P restore selection,
followed by 30-minute HTTP timeouts. This does not prove all such requests had
been evicted: the stopped broker's live state was not captured. Independently,
the code contained a reproducible terminal progress cycle: TP selects only
`snapshot_ready`, while async preparation requires selection before consuming
FAILED/RECOMPUTE_REQUIRED. Retaining terminal receipts alone did not fix this.

The later P SIGKILL remains a separate, unexplained failure. This change does
not claim to fix SIGKILL, all 94 observed waits, bandwidth, or full-run throughput.

## Ownership transition

An authoritative Host FAILED/RECOMPUTE_REQUIRED receipt is NOT new DMA work.
Rank0 freezes the waiter rid, request-generation and fallback reason, and uses
the existing native Host command/attempt and report channel:

`terminal_prepare -> all ranks locally drained (6) -> terminal_admit -> clear (7)`.

- Metadata-only terminal contexts are bounded by the existing Host pipeline
  depth, separately from physical lane occupancy. No new threshold or I/O lane.
- Cleanup runs on the existing Host worker; it reuses real remote/CUDA fences
  and allocator retirement. Only scheduler-owned Radix rollback stays in the
  scheduler. Both pending-bind ACK and fully bound states are handled.
- A sticky terminal load cannot restart DMA from a stale follower cache.
- Ordinary async fallback results cannot act as terminal cleanup receipts or
  admit TP ranks independently. Followers use the leader's reason at admission.
- Exact rid/attempt filtering prevents a retry from consuming an old ACK.
- Cancellation drains preparations and deletes completed terminal receipts;
  shutdown and unfenced transfers retain existing fail-closed behavior.
- FAILED compact receipts now survive broker metadata prune just like eviction
  receipts, after the same all-rank source-release/claim fences. No Host bytes
  or workset are retained in the receipt; existing generation bounds apply.
- TP1 async uses the same worker cleanup without a TP handshake. Legacy
  synchronous operation keeps synchronous progress. Normal Direct/Slow/P2D
  routing, limits, congestion policy and model parameters are unchanged.

## Acceptance checklist

1. Unique owner: only explicit terminal records authorize fallback; no inferred
   recompute for missing metadata or a slow scheduler.
2. P2D Direct release: unchanged.
3. P2D Host durable release: unchanged.
4. D2P Host durable/source release: unchanged.
5. Progress: terminal consumption needs neither a free H2D lane nor new HBM;
   worker handles cleanup, no new NFS or scheduler RPC.
6. TP: leader commands, all-rank readiness and exact identities before admit.
7. Reuse: only explicit failed/evicted parents recompute, with distinct reasons.
8. Gate: CPU fault tests and independent audit required before GPU execution.

## CPU validation entry point

```bash
cd /homes/siqic/dualpd/sglang
DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python \
  bash validation/check_multinode_cpu.sh
```

The gate disables CUDA visibility. Focused tests cover TP1/2/8, full lanes,
stale follower states, pending DMA/leases/CPU preparation, partial/full bind,
cancellation, retry identity and terminal receipt pruning. GPU/full SWE results
must be recorded separately; none are implied by these CPU tests.

Final validation: **1152 passed, 2 GPU-only tests skipped**; launcher suite
**75 passed**. Independent reviewer `terminal_review`: **GO**, including the
final exact-rid partial-bind regression. `git diff --check` passed. No GPU
processes were started or remote runtime code changed during this repair.
