# DualPD multi-node TP8 rebuild

This branch starts from `pd_mamba`.  The prior `pd_multi_node` control plane is
preserved only on `pd_multi_node_failed_20260924` and is not a compatibility
target.

## Runtime rules

1. One in-memory lifecycle record exists per request-generation.  It contains
   only the current attempt, physical owner and terminal result.  Rank progress
   is short-lived attempt state and is retired after the group transaction.
2. TP rank zero is the only route/path/lane/attempt decision maker.  Followers
   execute the immutable command for their shard and report a real local fence.
3. Runtime control uses persistent TCP connections.  No NFS, `/dev/shm`
   mailbox, marker directory, directory scan, inotify watcher or periodic file
   polling participates in a request lifecycle.
4. Every rank has one physical memory authority.  Agentic transfers reserve an
   immutable lease through that authority.  Scheduler code consumes a ready
   lease and cannot independently allocate or release the same pages.
5. `D2P_DIRECT`, `D2P_HOST`, `P2D_DIRECT` and `P2D_HOST` have independent
   bounded queues and lane pools.  Blocking one queue cannot block progress in
   another queue or a model Forward loop.
6. Host payload lives in source-node DRAM.  Source HBM is released after all
   ranks report the source D2H durable fence.  Source Host is released only
   after all target ranks report their remote READ fence and group ownership is
   committed.
7. P receives a full parent-plus-suffix workset before publishing
   `prefill-ready`.  D receives the complete input plus configured Decode growth
   credit before publishing `decode-ready`.
8. A timeout requests cancellation; it is never a DMA fence.  Posted work keeps
   its source and destination storage until every rank reports success or a
   physically drained failure.

## Intended transaction

```text
rank0 selects one generation and attempt
  -> all ranks PREPARE the exact lease/descriptors
  -> all ranks PREPARED
  -> rank0 broadcasts START
  -> all ranks report the real DMA fence
  -> all ranks prepare the target bind and report BOUND
  -> rank0 broadcasts HANDOFF
  -> all ranks RELEASED
  -> source ranks release old HBM; target leases remain staged/invisible
  -> every rank arms the same activation and reports ACTIVATION_READY
  -> target TP0 receives one immutable activation ticket
  -> SGLang native TP broadcast applies that ticket on every target rank
  -> target ranks publish ready, adopt into the native scheduler queue, then report ACTIVATED
  -> rank0 commits ownership, finalizes and retires the short attempt
```

Direct performs this transaction between source and destination HBM.  Slow
performs it twice: source HBM to source Host, then source Host to destination
HBM.  The two transactions have distinct attempt IDs.

The rank-zero policy actor correlates two independent events: a completed D
parent snapshot and the arrival of its child request at P.  If both are present
inside the tool threshold and a complete P workset can be reserved, it proposes
Direct.  Otherwise it stores the parent in source-local Host DRAM.  A rejected
Direct PREPARE is retained as policy state and is explicitly retried as Host;
an intent is never silently dropped.  No scheduler rank independently drains a
rank-local ready queue: only the TP0 activation ticket defines visibility and
ordering.

## Current acceptance boundary

The CPU/fake-DMA suite proves the control, allocator and error-path invariants.
Remote deployment must still pass a real TP8 Attention+Mamba digest smoke and
GPU/NIXL fault injection before this branch is treated as production-ready.
The first launcher intentionally supports one P TP group and one D TP group;
multi-P/multi-D routing is a later topology extension, not hidden in this
transaction protocol.

V2 deliberately forbids SGLang's native Decode retract path: native retract
would release pages behind the single memory authority and corrupt its lease.
The configured Decode growth credit must therefore be sized so the acceptance
run never reaches retract.  A future request-level mid-Decode Host spill must
be implemented through the same authority before this restriction can be
removed.

## Activation gates

- TP1/2/8 tests cover reordered and duplicate reports, one-rank failure,
  cancellation after post, stale attempts and disconnects.
- Allocator tests prove no page/state-slot double ownership and full rollback
  after partial TP reservation.
- Four-queue tests prove independent progress and fence-based completion.
- Runtime tests fail on any request-path file access or directory polling.
- A two-turn TP8 GPU smoke compares all Attention and Mamba shard digests with
  full recompute before a c128 workload is allowed.
