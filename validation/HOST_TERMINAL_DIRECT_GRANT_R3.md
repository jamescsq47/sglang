# r3: terminal Host notification and partial Direct grant recovery

## Failed r2 evidence

Run `qwen35-122b-swe500-tp8-c128-socket-full-r2` was stopped via its
owned coordinator SIGTERM. Both nodes' GPU process lists were empty afterwards.
The launcher retained logs and cleanup status; 18 completed episodes are partial
results, not a full500 performance or correctness acceptance.

At approximately 23:25 UTC, a cross-log census (not a global atomic snapshot)
accounted for 128 active agents: 43 awaiting a slow parent with no remaining Host
record, 25 awaiting an explicitly recompute-required Host parent, 24 Host-ready,
3 H2D-loading, 2 Host-writing, 1 HBM-ready, 4 Direct, 23 downstream inference,
and 3 between completed inference and the next model call. Two smoke references
were excluded. The missing Host records alone do not prove all 43 were evicted.

One Direct parent `d4511c41477e4b32bfd3e8e96a977897:23` had seven completed
follower receives, no rank0 receiver, and a source terminal-abort proof at
23:17:23. Local completed receives skip transport polling, so neither those
receivers nor the missing leader receive initiated cancellation of the grant.

## Scope and lifecycle

- Direct: rank0 observes the exact source terminal proof at the active group
  grant, even with no receiver. Only matching DIRECT_LOADING ownership requests
  the existing native rollback. Every rank must still acknowledge rollback and
  the exact physical IO fence before claim/workset release. P_RECEIVED/CONSUMED
  and old-claim proofs cannot be cancelled by this path. TP1 is unchanged.
- Host: an authoritative RECOMPUTE_REQUIRED outcome must reach the waiter even
  before any Host routing result exists, and must not disappear during metadata
  pruning before the consumer is informed. The existing eviction eligibility,
  all-shard source release and in-flight IO exclusion must remain unchanged.
- Success, failed transport, timeout, cancellation and shutdown continue using
  existing owner/CAS and DMA-fence rules. No timeout authorizes data/page reuse.
- No change to P→D routing, allocator limits, Host sizes, Slow IO concurrency,
  tool/setup thresholds, sampling, workload order or congestion policy.

## Acceptance gate

1. Unique owner: terminal notification is not ownership and cannot free data.
2. P→D Direct release: unchanged.
3. P→D Host durable/source release: unchanged.
4. D→P Host durable/source release: unchanged; eviction notification follows
   the existing authoritative terminal transition.
5. Independent progress: no new NFS access, global scans or scheduler network
   waits. Direct checks bounded active grants on the existing memory mirror.
6. TP atomicity: rank0 decides; all-rank rollback remains mandatory.
7. Parent correctness: eviction recompute must be explicit and measured,
   never silent reuse of missing data; no new recompute policy.
8. Tests plus independent GO required before identical c128 full500 restart.

Direct regressions currently pass 22 cases, including a receipt=1 TP8 group
whose leader never started and seven peers already completed. The real grant
progress initiates the real native abort, all ranks acknowledge rollback,
claim CAS succeeds, and occupied Direct slots return from four to three.
Host now keeps a compact, run-bounded RECOMPUTE_REQUIRED receipt after physical
release and detail pruning. Both Router entry and Direct-outcome watcher consult
the existing Host mirror independently of routing records. A previously submitted
P can still read the same terminal receipt; no consumer-ACK RPC was added.

Frozen-source CPU gate: 1128 passed, 2 GPU-dependent skipped (25.86 seconds),
plus 75 launcher unit tests passed. Host-focused gate: 156 passed, including
9 real TCP Router/broker cases and TP1/2/8 late-consumer coverage. Direct-focused
gate: 22 passed. Independent frozen-source audit: 69 targeted tests passed
(18.22 seconds), all eight invariants reviewed, GO for identical c128 full500
engineering rerun. This does not establish throughput acceptance or prove that
every historical missing Host record was caused by eviction.

New run: `qwen35-122b-swe500-tp8-c128-socket-full-r3`, coordinator PID 647528.
Both TP8 model/Host prewarms and startup Direct/Slow two-turn token-consistency
checks passed. The same c128 full500 workload is starting under the owned
launcher. Long-run terminal-waiter/grant-age validation and final results are
still pending; the startup check alone does not cover capacity eviction.
