# TP event control migration (2026-09-20)

Status: implementation in progress; NOT a GPU acceptance result.

## Scope

Replace runtime filesystem coordination, including node-local mailbox files,
with messages and in-memory state. The existing file backend remains the
single-node compatibility baseline until the complete new backend passes the
cutover gate. Do not launch the existing multi-node file backend as a test of
this change. No new scheduling, eviction, timeout, or recomputation policy is
authorized by this migration.

The current file users are not limited to TPGroupMailbox: Host staging ledger,
metadata store and local Direct exclusion claims, early arrival/tool/final/route
markers, P-ready/accepted/scheduled records, remote Host descriptors and read
receipts, pressure/reservations must also migrate. Merely moving their paths to
`/tmp` would break communication between nodes; it is not the proposed solution.

## Ownership plan, before editing

* TP rank0 remains the only logical decision maker. Followers receive an
  attempt-scoped command and report local preparation, DMA fence, bind, or
  cleanup completion. A report is not itself permission to change the group's
  path or release another rank's storage.
* Run, TP group, request-generation and transfer attempt scope every operation.
  Exact claim/lease/export IDs are retained by existing transitions. Repeated
  messages are idempotent; stale attempts cannot complete a current attempt.
* P_HBM -> D_HBM: preserve all-shard Direct completion before P release.
* P_HBM -> P2D_HOST: preserve all-shard Host durable before P release; D binding
  completes the later Host -> D_HBM handoff. No host eviction is introduced.
* D_HBM -> P_HBM or D2P_HOST: preserve the complete workset and physical fence
  conditions. Failure of one shard requests cancellation, not a fake DMA fence.
* D2P_HOST -> P_HBM: preserve pin, complete workset allocation, all-shard load
  and Radix bind. CONSUMED is NOT permission to discard recovery ownership before
  scheduler handoff; retain the handoff receipt through delayed acknowledgments.
* Timeout, capacity failure and cancellation use existing rollback states.
  Disconnect/overflow fails closed: no source free, destination reuse or new
  ownership based on missing messages. Restart requires a new run epoch after
  the old GPU processes have actually stopped; no silent broker reset.
* Shutdown stops admission and drains/aborts I/O with its real fences before
  coordinator shutdown. This change does not authorize killing other GPU jobs.

## Architecture

1. In-memory authoritative transition cores reuse existing lifecycle methods.
   No remote lock is held around arbitrary Python callbacks or GPU work.
2. Persistent message connections carry commands and status changes. Local
   mirrors and completion queues serve scheduler reads; no synchronous network
   queries or scans on each scheduler tick. Bounded queues fail explicitly,
   rather than silently dropping ownership messages.
3. Group-local rank status and group-to-group ownership receipts are distinct.
   Rank0 publishes a group receipt only after all physical shards are safe.
   No background collective is added to the model's NCCL group.
4. KV data continues over existing NIXL paths; Host arenas stay in source-local
   CPU DRAM. Control messages do not serialize tensor payloads.
5. Static configs/model loading and post-run reports are separate from runtime
   control. Live control may not use NFS or periodic directory scans.

## Cutover gate

Before GPU testing, require:

- CPU process tests at TP=1/2/8, duplicate/reordered/stale messages, late join,
  cancellation, partial completion, disconnect, queue overflow and delayed
  scheduler handoff; no control filesystem operations.
- Existing lifecycle regression tests and a full call-site audit of both
  directions, Direct/Slow, Router and harness. All eight design acceptance
  criteria must be mapped to evidence, not assumed from unit coverage.
- Fault injection that rejects legacy control-file accesses in the new mode.
  A partial mailbox-only conversion must fail this gate.
- Independent audit GO. Only then a10/a11 smoke, followed by the requested
  workload. Preserve TP=1 defaults until its parity tests pass.

Current acceptance status: none of the eight end-to-end criteria is newly
certified by this plan alone. Test reports below must distinguish primitives,
integrated control, CPU two-node tests and actual GPU experiments.

## Review-discovered lease replay defect

The existing `attach_d2p_recovery_lease_rank` accepted a repeated matching
lease ID but reset its phase from `io_inflight` to `leased`. A later ordinary
cancellation could then incorrectly consider the destination safe to free.
The minimal shared transition fix makes matching-lease attach a no-op success,
preserving its existing phase; first attachment requires `pinned`. This does
not change admission or TP=1 policy. It is needed before enabling message retry.
The original physical owner and in-flight DMA fence remain authoritative.

## 2026-09-20 component validation (not engine acceptance)

- New TP event transport: independent review GO; 19 CPU tests, including real
  TP2/TP8 subprocesses, disconnect, queue backpressure, malformed rank identity,
  stale attempt, sticky failure, ordered commands and rollback fences.
- New control record core: atomic owner/revision CAS, generation-wide exclusion
  keys, exact last-request retry receipts and event cursor overflow tests.
- New Host state core: inherits existing transitions, replaces only persistence
  and notifications. Retired-generation tombstones prevent delayed offers from
  resurrecting already freed KV. Detailed metadata pruning waits for scheduler
  handoff and source-release receipts. The component is not wired to workers.
- Final full CPU gate: **977 passed, 2 skipped** (25.11 s), plus **72 launcher
  unittest tests passed**. This is regression coverage, not proof that a live
  engine has stopped using its legacy control files.
- Actual a10/a11 CPU protocol smoke passed TP2 and TP8, eight distinct attempts
  each (1.133 s and 0.848 s including SSH/process startup). These are control
  smoke durations, NOT KV bandwidth, model latency or throughput results.
  Rank0 ran on a10; follower processes ran on a11. The module source was sent
  over SSH stdin and held in memory, with no shared control files.

Reproduce the CPU two-node smoke:

```bash
DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python \
  bash validation/check_tp_events_two_node.sh \
  --peer a11 --listen 10.0.1.170 --peer-python /usr/bin/python3 --tp 8
```

### Remaining cutover work (must not be reported as completed)

The new cores are standalone. Production still constructs TPGroupMailbox,
SharedHostStagingLedger and the file-backed APIs listed above. Their command
dispatch/completion adapters, Router/harness records, remote Host descriptors
and launcher service lifetime must be connected to the new backend together.
Group-local completion subscriptions must not be confused with cross-group
P2D receiver receipts. Existing `complete_*_rank` APIs also require an
attempt/claim/lease guard at the new network dispatch boundary.

There is currently **no integrated no-NFS engine/GPU GO**. No model experiment
was launched in this component-validation round. TP=1 scheduling/routing
defaults have not been switched. Only the lease-attach idempotence fix above
changes an existing shared runtime transition.

Final independent component review of the initial standalone stage: **GO**, **68 targeted tests passed**
(Host core 35, record core 14, TP event transport 19). The Host broker boundary
now has `apply_recovery_event(loaded/bound/handed)`, which checks exact owner,
claim, lease, read epoch and physical rank inside the same in-memory transaction
as the inherited state transition. Network handlers must use this boundary,
not directly expose the unguarded legacy completion methods. This component GO
does not change the remaining integration/GPU NO-GO above.

## Production adapter integration (2026-09-20, subsequent work)

The standalone-only description above is historical. The following production
entry points now select the persistent message backend when
`SGLANG_AGENTIC_CONTROL_ENDPOINT` is set:

- Router arrival/route and P-ready/accepted/scheduled records; independent
  change queues replace directory scans. Publication ACKs are awaited outside
  the asyncio loop. Cancellation drains the exact pending write before cleanup.
- Host ledger transitions and request-generation metadata use the in-memory
  broker. Scheduler queries use pushed local mirrors; mutations on the
  scheduler are submitted and later committed from a retained Future.
- RemoteHost export/read/receipt cleanup validates the broker's authoritative
  Host entry and exact attempt. Descriptor control no longer uses filesystem
  locks; NIXL KV payload transfer is unchanged.
- TP socket adapter retains rank0-only decisions and per-rank physical fences.
  Direct PREPARE pushes metadata to followers; followers no longer discover
  requests independently from Router arrival records.
- P-ready admission is removed only after every D rank acknowledges. The
  cross-group receiver receipt is an observer subscription, not permission for
  P to delete D's authoritative TP state.
- Host registration prewarm starts/completes through broker notifications.
  Arenas themselves remain source-local memfd DRAM. The launcher starts the
  broker before workers and retains it if GPU worker shutdown is incomplete.

Safety checks added: stale attempt rejection, no lease downgrade on retry,
no generation resurrection after cleanup, failed publication poisons authority,
synchronous RPC forbidden on the running scheduler, and no global cache lock
held while initializing an unrelated subscription.

Current launcher scope is one logical P TP group and one logical D TP group,
each contained in its own node. TP=1 compatibility stays behind the legacy
default; the new backend supports the same attempt/fence API for TP=1/2/8.
This migration does not change Direct thresholds, recompute policy, workset
size, batching, Host capacity, or any physical ownership release condition.

Actual latest a10/a11 CPU TP8 message smoke: **8 attempts passed**, 1.214 seconds
including SSH/process startup; still **not a GPU or bandwidth measurement**.
Full integration regression and independent call-site review are in progress.
`socket_control_engine` remains false until they return GO. No GPU experiment
has been started with a partial control-plane migration.

### Integrated cutover gate result

Latest complete regression: **1077 passed, 2 GPU-only skipped** (28.51s),
plus **75 launcher tests passed**. Six real-broker integration tests are
included: file operations are forbidden; fake fenced DMA exercises TP2/8
Host success/cancellation, authoritative cleanup, and stale lease/epoch ACKs.

Independent whole-control review: **GO for bounded a10/a11 GPU engineering
smoke only**, not formal throughput acceptance. The capability flag is now
enabled for that test, with `hardware_verified=false` and explicit smoke-required
status. Eight-invariant check before launching:

1. Unique owner: exact claim/lease/export guards and generation tombstones;
   stale completion tests pass.
2. P→D Direct release: original all-D-rank receipt remains; source observer
   cannot clear authoritative receiver state.
3. P→Host release: all-shard durable ACK remains independent of D capacity.
4. D→Host release: all-shard durable fence remains; no optimistic source free.
5. Progress: migrated control files, nonblocking scheduler mirror/submit,
   retained completion Futures; real overlap still requires GPU observation.
6. TP: rank0 decisions, exact PREPARE/Direct/Host/retire attempt identity,
   followers report actual shard completion.
7. Reuse: data bytes/layout/recompute policy unchanged; exact outputs and
   actual reuse must still be checked on the GPU.
8. Gate: tests and independent audit completed before model launch. Any
   lifecycle/Forward anomaly stops smoke; do not continue into performance.

### First GPU smoke: stopped at Slow handoff, not accepted

Run: `qwen35-122b-swe500-tp8-c128-socket-control-r1` (synthetic engineering
smoke only; no SWE workload was started). a10=P and a11=D both TP8,
Qwen3.5-122B-A10B BF16, static memory0.8, Mamba ratio0.5.

- Both models and Host prewarm started successfully. Runtime control namespace
  directory was absent; no shared filesystem control was created.
- Direct two-turn case passed:8192 cached parent tokens; exact generated token
  IDs matched the chunk-matched full-recompute reference; all8 Direct ranks
  and P→D source releases were observed.
- Slow case: all8 D2H durable and source release, then all8 Host→P copy and
  group-release logs completed at21:25:11 UTC. Prefill then stopped progressing.
- Nonblocking stack samples prove a TP admission divergence, not a bandwidth
  wait: TP0 was in `forward_extend/gdn_backend.py:399`; TP3 was already waiting
  for the next native `recv_requests` broadcast. TP3/4/6 still held handed
  worksets8256tokens; other ranks had consumed theirs.
- Cause: new asynchronous `handed` ACK was checked after the common Host
  commit command. ACK-ready ranks became runnable while ACK-pending ranks
  returned to the waiting queue. The original synchronous operation could not
  yield there. It must be staged before a final rank0 all-ready admit command.
- Coordinator was sent SIGTERM; owned supervisors stopped workers, then broker.
  Both nodes' GPU process lists were empty afterward. No GPU reset or external
  process termination was used. No formal throughput is reported.

The integration capability is disabled again pending delayed-ACK regression,
independent re-audit, and another GPU smoke. No policy/timeout adjustment is
proposed; the fix is a rank-consistent consumer handoff on the existing command
list, with no blocking RPC or new model collective.

### Slow handoff fix before second canary

The existing TP Host command list now separates local handoff preparation from
the final group decision: status3 -> commit/preparation; status4 means the
exact handed ACK completed on this rank; all-rank status4 -> rank0 admit;
status5 -> admitted or cancelled-and-quiescent. The final admit does not query
the broker or depend on per-rank mirror freshness. TP1/legacy behavior remains
unchanged. Replaceable scheduler state reports retain their legacy semantics;
physical transfer progress/failure fences remain sticky and attempt-scoped.

Also deferred the endpoint-mode grammar-error source release through the
existing P→D all-rank cleanup path, instead of letting each rank decide from
its own async cancellation ACK.

New regressions explicitly stagger one or multiple handed ACKs at TP2/8,
test cancellation after handoff-ready but before admit, and forbid RPC in final
admit. Full gate: **1090 passed, 2 GPU skipped; 75 launcher tests passed**.
Independent focused review: **94 passed, GO to repeat bounded forced-Slow
canary**, not formal performance/evaluation acceptance.

### Second GPU smoke: Direct and D→Host→P passed

Run: `qwen35-122b-swe500-tp8-c128-socket-control-r2`, 2026-09-20,
same a10/a11 TP8 model/configuration. `c128` is the configured future workload
concurrency; this synthetic smoke is sequential, **not a c128 load test**.

- Direct and forced-Slow two-turn cases both passed: each reported8192 cached
  parent tokens and exact generated token IDs matching its chunk-matched full
  recompute reference. All8 ranks reported the corresponding path commit.
- Slow all8 H2D completions were followed by incremental Prefill and all8
  P→D releases, rather than the divergent Forward seen in r1. Final P progress
  reports had no active/handed workset leases.
- D2H wall47–60ms/rank and first-use remote H2D wall2.30–2.40s/rank are
  diagnostic timings, not steady-state link bandwidth measurements.
- `completion.json` reports success. The coordinator stopped both model
  groups before the control broker; GPU process lists were empty on both
  nodes. Already-exited smoke/no-workload cleanup notices were harmless.
- `/run/dualpd-control/<run>` did not exist. Runtime coordination uses the
  message service; node-local log files are observation, not coordination.

Next bounded test uses a fresh identity and diagnostic D capacity12288tokens
to force real P→Host→D staging. Full SWE/c128 and performance acceptance remain
unverified; no formal throughput is claimed from this smoke.

### Forced P→Host→D GPU smoke: passed

Run: `qwen35-122b-swe500-tp8-c128-socket-control-r4-p2d-host`.
The previous r3 attempt stopped before model startup because port23904 failed
the conservative availability check immediately after r2 cleanup. No listener
or GPU worker remained on inspection. A fresh r4 identity started normally;
no timeout, port-safety, or engine policy was changed.

- Diagnostic D capacity12288tokens makes two8192token prompts unable to fit
  concurrently; this triggers real P→D Host staging, not an injected receipt.
- `p2d:826008051372469578`: queued8shards = D2H complete8 = P HBM release8 =
  D H2D complete8. All8 P releases occurred21:44:09 before D H2D completion at
  21:44:10. The staged request output matched the direct reference exactly.
- The subsequent D→P Direct and forced-Slow cases also passed, each with
  8192cached parent tokens and exact output matches. Slow entry was consumed.
- Coordinator completion succeeded, both groups drained to0requests and both
  GPU process lists were empty after cleanup. No formal SWE workload ran.
- First P→D D2H elapsed~4.95s despite GPU23–43ms: follower Host extent
  registration logged4.27–4.31s, rank0~0.03ms. This is an observed startup/
  registration latency to investigate, not a measured PCIe bandwidth limit.
  Remote H2D~1.16–1.17s is likewise first-use diagnostic timing.

The runtime capability records the bounded TP8 smoke evidence, while retaining
`hardware_verified=false` and `safe_to_launch_full_pipeline=false` for broader
topologies/full workload acceptance. Passing these probes does not establish
c128 throughput or long-run absence of stalls.

### First-use P→D worker CUDA device correction

Read-only follow-up confirmed that the model KV pool stores unindexed `cuda`.
Startup prewarm resolves it on the owning rank, but the P→D D2H/H2D worker
previously prepared Host mappings before entering its explicit CUDA stream.
A fresh worker thread can therefore select default GPU0 rather than its rank.
The registration timing also includes CUDA-device entry and registry-lock wait;
the4.3s observation alone did not prove a cache miss or slow PCIe.

The minimal correction is exactly one `torch.cuda.set_device(stream.device)`
at each of the two worker entries. Constructors already create those streams
on the concrete owner GPU. No pool-global device, queue, ownership, fence,
route, or TP decision changes. TP1 on a nonzero GPU also selects its actual
owner, not default0. Four fake-device ordering tests cover both directions at
rank0/rank3 and are included in `check_multinode_cpu.sh`.

Gate: existing full suite1090passed/2skipped +75launcher passed; scoped suite
including the four new cases47passed/2real-CUDA deselected. Independent audit
GO for a fresh forced-P2DHost probe. Timing improvement remains unproven until
the real r5 probe completes.

Repeated r4 `p2d_host_release` log lines were independently checked: concurrent
cleanup callers can log after an already-removed record. Actual extent release
checks exact snapshot identity, deregistration is guarded by a per-snapshot
lock/closed bit, and ledger ACKs are set-valued. This was not a double free;
count `(snapshot, rank)` rather than raw log lines. No lifecycle change made.

### Worker-device fix GPU validation: passed

Run: `qwen35-122b-swe500-tp8-c128-socket-control-r5-p2d-device`, same diagnostic
settings as r4. All three cases passed again: real P→Host→D staging and both
D→P Direct/Slow, exact token output matches throughout. Each reverse case
restored8192parent tokens. Source-release counts remained all8/8ranks.

- First P→Host registration preparation: all ranks0.028–0.048ms, versus
  follower4.27–4.31s in r4.
- First P→Host D2H elapsed:720–748ms/rank, versus~4950ms; actual GPU copy
  remained35–43ms. Residual first-use export/control work is not a steady-state
  bandwidth measurement.
- All8 P HBM releases occurred21:56:00. D resumed this staged request after
  capacity was available and completed remote H2D at21:56:05 (~1120ms transfer
  worker elapsed). Thus P did not keep its HBM while D was unavailable.
- Both model groups drained and stopped; both GPU process lists were empty.
  No NFS control directory was created. Full SWE/c128 load has not run.

Independent review allows a monitored c128 evaluation of this exact topology
after the passing probes; that is permission to test concurrency, not evidence
that full-workload throughput/correctness has already passed.

### R13 Slow native BIND/handoff fusion (CPU validation, GPU pending)

TCP TP>1 now performs local `handoff_to_req` at the existing scheduler-owned
BIND boundary, **before** submitting that rank's bound receipt. Background
workers still never modify Req/Radix/allocator state. Bound receipts therefore
prove both local Radix installation and successful Req ownership. No phase3
is published on this path: workers keep phase2 until the existing handed ACK
finishes, and rank0's existing native phase4 ADMIT remains the only permission
to enter Forward. This removes the intermediate native COMMIT revisit, not
the all-rank physical completion or final admission barrier. TP1 and legacy
non-TCP paths retain their previous chain.

Every rank must observe exact owner/claim/lease/read-epoch, all binder ACKs,
and CONSUMED before closing any Host record or submitting handed. The exact
job retains this proven fact because source release and ledger pruning can
finish before the final handed reply is observed. Rank0 additionally waits
for its existing asynchronous logical manifest completion. Cancellation
invalidates the job and drains pending receipts; errors retain ownership.

A partial local-handoff failure cannot publish a bound ACK and consequently
cannot consume the complete Host source. Existing group retry is reused:
binding leases use `abort_bind`; pre-admission handed leases use
`release_handed`, native no-reqslot Mamba cleanup, and the same TP retirement
transaction. Radix drops only unreferenced parent branches; shared prefixes
remain. HTTP cancellation enters this same rollback before generic cleanup.

CPU regressions use real TP2/8 workset brokers, authoritative in-memory Host
ledger, real Radix shared-prefix references, and CPU Mamba runtime allocation.
They inject failure before/after the last local handoff, delayed bound/handed
ACKs, cancellation, duplicate rollback, old-attempt replies and source-prune
before final reply observation. They prove completion without a COMMIT command
and no runnable rank before common ADMIT. These do not prove CUDA performance;
independent audit and the full CPU gate are required before a fresh GPU run.
