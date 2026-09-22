# P workset controller refactor

Status: implementation in progress. Opt-in model Scheduler integration is now
present; CPU and CUDA component gates are being rerun before model smoke.
No full-model performance claim.

## TP workset control-history retirement (2026-09-21)

c128 r3 failed at 18:32:03 UTC after ~585 business seconds: the shared TP
service reached 100,000 entries. Workset grant/prepared/decision/fence records
were never retired; ordinary clear also retains replay tombstones. This was
control history, not exhausted Host or KV capacity. r3 is not a passed long run.

After the existing exact all-rank FREE execution ACK, rank0 now asynchronously
retires that lease's four workset event namespaces. The server verifies the
original grant and final decision/ACKs, deletes indexed records and reduces the
live entry count. Exact retired-version intervals reject delayed messages;
unfinished versions remain gaps. Client mirrors and dedup caches retire too.
No timeout, scheduler wait, physical release or routing rule changes. Generic
UUID mailbox tombstones, observed receipts and executor replay fingerprints
retain their original semantics; this is not universal control-store GC.
An authenticated read-only `system.tp_stats` reports compact counts for the run.

Checks 1–4 and 7: physical ownership and existing DMA/reference fences unchanged.
Check 5: bounded delta-driven cleanup stays in the background; check 6: missing
any final FREE execution ACK prevents reclamation; check 8: independent audit GO
and 109 focused tests, including >100,000 historical entries with capacity 8,
late-message rejection and TP1/2/8 real runtime cycles with capacity 24.
Broader regression and the new c128 run are recorded separately; no new full
model performance result is claimed here.

## c128 ingress repair (2026-09-21)

The first c128 launch failed at ingress after Direct/Slow model smoke passed.
Neutral cancellation/abort reports generated 2048 rank-zero updates plus 256
ACKs at TP8; a real TCP test reproduced bounded-queue overflow with a delayed
writer. `queue.Full` also stored an empty failure string, causing repeated
teardown. The live log lacked the original exception type, so the overflowing
peer cannot be identified retrospectively.

Ingress now retains metadata without sending neutral reports. Any exact-attempt
negative report requests rank-zero ABORT; COMMIT and abort retirement still
require all ranks. Server failure is nonempty and latched once. Queue capacity,
physical ownership, CUDA fences, DMA paths and transfer policy are unchanged.
Acceptance checks 1–4 and 7 retain their existing ownership/release boundaries;
5 removes ingress-only messages without blocking sends; 6 keeps ordered common
decisions and full-rank completion; 8 passed 1811 CPU + 91 launcher tests and
independent 62-test review/GO. c128 r2 uses tool=1s and policy recompute disabled.

r2 passed ingress but real workset grant/preparation also overflowed the
dedicated group's pending queue (`Full`, before useful measurement). Server
outbound cumulative progress/command-ACK/rollback mirrors now coalesce only
within the same identity and command ID; non-cumulative updates break merging.
Connection ACKs remain after preceding states; queue capacity stays 1024 and
distinct excess events still fail closed. Actual TP8/c128 CPU actor tests cover
two complete grant/prepare/cancel/free/reuse waves and delayed final-rank fences.

## Current integration (2026-09-21)

`SGLANG_AGENTIC_P_WORKSET_CONTROLLER=1` installs one CPU page/Mamba-slot
authority before request ingress. The native allocator's allocation methods
are then disabled, rather than leaving two writers. The launcher exposes
`--p-workset-controller`; default-off preserves the existing serving path.

- Direct/Slow, fresh and recompute submit complete workset intents to that
  authority. Dedicated socket TP commands install identical plans; descriptor
  preparation runs on a background CUDA stream using the actual rank device.
- Native ingress records request identity on all ranks without allocating.
  Rank0 selects fresh/recompute work. Prepare/ACK/commit admits the exact ready
  lease to native compute; followers never independently choose allocations.
- Prefill consumes private suffix pages; Radix last-reference returns and two
  Mamba checkpoint slots retain exact lease ownership. Native returns are
  event-fenced and asynchronously mirrored, not GPU free-list reads on Forward.
- HTTP abort stops future chunks but retains the current workset until the
  existing overlap result fence; P->D sources still use their transport fence.
- Parent Radix binding is still a native commit, and **D source release still
  waits for the existing CONSUMED boundary**. Moving it to an irrevocable
  all-rank receive commit remains pending; this integration does not claim
  complete end-to-end scheduler independence yet.

Final CPU integration gate: 1806 passed, 2 skipped, then 91 launcher tests
passed (`/tmp/dualpd-workset-model-final-gate-r3.log`). Independent startup,
registration, cancellation and lifecycle audit gave scoped model-smoke GO.

The actual-broker CUDA driver (`--mode facade`) passed TP1, TP2 and TP8
eight-cycle tests. The first TP8 attempt failed because its test-only header driver reused
a stale ACK when publishing the next command; it is not a passed TP8 result.
After checking the exact command ID, TP8 rerun passed (log suffix `tp8-r2`).
These remain component tests, never model/throughput results. Older sections
below record the staged implementation history, not current activation status.

### Two-node model smoke

Run: `slime/runs/dualpd/qwen35-122b-workset-controller-smoke-r1`.
a10=P, a11=D, Qwen3.5-122B-A10B, TP8 per group, mem 0.8, Mamba ratio 0.5.
Controller enabled; existing TCP control and transport policy unchanged.
Both Direct and Slow two-turn cases passed: 8192/8192 aligned parent tokens
reused, all eight ranks committed the same path, and output tokens exactly
matched full recompute. Slow's eight source Host shards were released.
Coordinator exited 0 and both nodes had no remaining GPU compute processes.
This is synthetic deterministic functionality, not SWE sampling/evaluation or
300+1200s performance. Model cancellation remains CPU/component-tested, not a
real model abort test in this run.

Separate `qwen35-122b-workset-controller-p2d-smoke-r1` also passed. A diagnostic
12k-token D pool caused real capacity backpressure: all eight P shards staged
the 8192-token snapshot in P->D Host, then D recovered it. Output exactly
matched direct recompute. Both reverse-path cases passed again. Coordinator
exited 0; both nodes' GPU compute-process lists were empty after owned cleanup.
These diagnostic capacities are not a performance comparison configuration.

## 2026-09-21 CUDA component gate

The ordered TP log now includes cancel, subset-close/return and whole free. The
runtime actor connects that log to descriptor preparation and the legacy lease
handoff; native last-reference returns and two-slot Mamba checkpoint rotation
have guarded adapters. These are opt-in components, not yet the live model's
allocation path.

- CPU gate: 1660 passed, 2 skipped; launcher gate: 90 passed
  (`/tmp/dualpd-workset-integration-final-gate.log`).
- Independent review: scoped GO for the tiny-pool CUDA ownership test only.
- `bash validation/check_workset_gpu.sh --tp N --rounds 8`: N=1,2,8 all passed
  on a10 A100 GPUs. Each rank completed eight real CUDA allocation/descriptor,
  chunk-consumption, event-fenced retirement and same-address reuse cycles.
  Mamba state was poisoned between cycles to check reinitialization.
- Logs: `/tmp/dualpd-workset-gpu-tp1.log`, `...-tp2.log`, `...-tp8.log`.
  `model_test:false` is explicit. No model weights, NIXL KV payload, network
  throughput or SWE evaluation were tested. All test children exited; the
  subsequent compute-process query was empty.
- Still required: integrate native request ingress/compute cut/last-reference
  cleanup, audit the complete model path, then execute model GPU tests.

## Ownership boundary

The P memory controller is the sole allocator, not an additional allocator next
to Scheduler. Rank 0 owns one CPU page/slot ledger and issues ordered, immutable
full-workset grants. Direct, Slow, fresh and recompute use the same authority.
Followers install exact grants; they do not choose pages from independent free
lists. Attention and Mamba resources are committed atomically. A CPU grant is
not a CUDA-ready event or a transfer completion.

Scheduler consumes a TP-wide committed ready set, selects compute batches and
uses preallocated suffix slices. It must not fall back to native KV allocation.
The existing model collective remains separate from the controller connection.
Computational admission retains the native policy, with no Direct/Slow/New
priority or new watermark. The entire waiting queue is not preallocated at once.

## Transitions and fences

- Capacity unavailable: keep an allocation intent, no partial Attention/Mamba
  ownership; retry on resource return rather than scanning the waiting set.
- Granted: pages and slots are unavailable to all other requests. Per-rank
  descriptor creation, Mamba initialization and stream events must finish before
  the original Direct/Slow worker can use the grant.
- Received: all ranks report actual complete parent transfer. P controller owns
  a non-revocable destination and retains it through compute handoff. Only this
  state permits D source release without waiting for Scheduler's Radix bind.
  After this ownership commit, a later bind failure cannot revert the manifest
  to D-owned `DIRECT_READY`: D is allowed to have freed its copy. Retry the P
  handoff from the retained P lease or fail the request explicitly instead.
- Prefill: consume only this lease's suffix; chunks do not allocate again.
- P→D complete / P→Host durable: release the generation's private resources and
  its shared-prefix references after the existing physical fence. A lease cannot
  free pages already donated to Radix or used by another generation.
- Cancel/timeout: close the exact attempt to new operations first. Unknown or
  in-flight CUDA/transport work retains resources until all-rank quiescence.
  Old attempt messages never reclaim a newer lease.
- Disconnect/shutdown: fail closed and retain ownership, not empty-ledger restart
  or time-based resource reuse. No NFS or file-based TP coordination is added.

## Migration inventory

1. Move workset ownership classes out of Scheduler without changing behavior.
2. Introduce pure CPU ledger and ordered controller protocol with fault tests.
3. Normalize fresh/recompute to complete physical leases, including Mamba runtime,
   tracking and output checkpoints. Existing code only physically preallocates
   Direct/Slow; fresh requests currently allocate native chunks lazily.
4. Migrate all Attention/Mamba allocation/free sites, including Radix duplicate
   release, chunk checkpoint replacement, partial cancellation and P→D release.
   Native `free_group`, backup/restore and CUDA free-list state cannot remain a
   second independently writable allocator.
5. Move actual grant preparation and Direct/Slow selection to the background
   controller. Original native header carries an immutable ready-set cut, not a
   directive to perform allocation. No independently sampled free capacity may
   cause ranks to choose different compute batches.
6. Gate runtime activation on the absence of legacy allocator fallbacks. Audit
   and test before TP1/TP8 GPU experiments.

## Acceptance / current evidence

The eight project invariants remain unchanged. Mechanical extraction does not
change ownership, source release or transport behavior (criteria 1–4, 6–7).
Criterion 5, actual compute/transfer allocation independence, is **not yet
achieved** by extraction or standalone CPU tests. Criterion 8 requires a separate
audit GO and full lifecycle/fault gate before running GPUs.

No benchmark improvement is claimed by this document. The last R15 run had a
valid sampled performance window but later failed when a P process exited with
SIGKILL; the cause of that kill was not established. It is not a successful
500-task evaluation.

## Implemented components (not runtime activation)

- `agentic_workset.py`: extracted old lease/broker implementation, retaining
  Scheduler import aliases and existing Direct/Slow semantics. Added explicit
  fresh/recompute full-prompt handoff without inventing a received parent.
- `agentic_workset_ledger.py`: CPU-only exact address authority and all-rank
  fenced partial returns. This is not another active allocator alongside the
  native one: no runtime code instantiates it yet.
- `agentic_workset_device.py`: materializes exact CPU page/slot plans and a
  preparation event on an explicit background CUDA stream, without reading back
  native CUDA free lists. Native Mamba initialization was extracted into the
  same `initialize_slots` operation; native allocation semantics are unchanged.
- `TPEventClient.wait_commands`: event-driven command consumer for a background
  controller. It neither executes nor ACKs commands automatically.
- Opt-in `subscribe_updates/drain_update_keys`: bounded, coalesced changed-key
  notifications. The TP executor need not scan all historical grants to discover
  cancellation or all-rank readiness. Existing unsubscribed users are unchanged.
- `agentic_workset_controller.py`: single background ledger writer, pending
  intents retried only on actual resource return, cancellation-before-arrival
  protection, bounded queues and fail-closed shutdown. This component owns CPU
  address decisions, not the runtime native allocator yet.
- `agentic_workset_tp.py`: exact grant installation / asynchronous preparation
  protocol, kept separate from model collectives. It currently supports grant
  preparation, not the complete cancel/partial-return/free decision log and not
  runtime compute admission. There is deliberately no launch switch enabling
  this incomplete migration.
- Tests include fresh partial-chunk cancellation, atomic Attention/Mamba
  reservation failure, stale attempts, partial shared-prefix returns, exact
  descriptor preparation and socket-disconnect wakeup.

The fresh handoff currently requires no existing prefix/runtime owner. Shared
prefix matching must subsequently adopt/release exact ranges; callers may not
silently disable prefix sharing to work around this precondition. Likewise two
output Mamba checkpoints are a rotation budget, not proof that an arbitrarily
shared old checkpoint can always be reclaimed. Runtime activation requires a
safe checkpoint return/adoption policy rather than hidden native `alloc()`.

### Independent component audit

An independent reviewer found a fresh-runtime handoff exception hazard: publishing
`req.mamba_pool_idx` before constructing the tracking descriptor could leave both
Req and the broker apparently owning the same slots. The helper now constructs
and validates descriptors first, then publishes the ownership fields together;
fault injection covers the failed construction and cleanup.

The reviewer passed 87 CPU component tests (fresh 20, ledger 36, device 10,
command wait 5, controller 16) after that correction. This is component-level
GO only. The later TP executor and incremental-update additions require their
own final audit. That final independent audit also returned **component GO**:
56 TP-executor/update/original-transport tests passed, with exact descriptor
identity, physical preparation fences, cancellation retention and incremental
active-set progress checked. It explicitly withheld runtime/GPU GO. No CUDA or
model experiment has been run for this refactor.

Final root CPU regression after adding the TP executor and update subscriptions:
`validation/check_multinode_cpu.sh` — **1583 passed, 2 skipped**, followed by
**90 launcher tests passed** (2026-09-21). Log:
`/tmp/dualpd-workset-final-gate.log`. These results exercise components and the
unchanged legacy runtime contracts; they are not evidence that the new controller
already owns the live engine's allocator or improves throughput.

### Mamba checkpoint rotation audit

The aligned SWE harness uses `agentic-v1:{request_id}:g{generation}` namespaces.
Direct/Host restoration inserts under the next Req's namespace. With exactly
one active Req/attempt, two checkpoint slots can rotate: after inserting and
pinning the new checkpoint, the old checkpoint loses its Mamba reference even
if Attention ancestor pages remain referenced.

This is not guaranteed by the `request_owned_mamba` flag alone. Legacy
trajectory-only keys and raw/duplicate same-generation submissions still exist
in compatibility inputs. A second Req can keep the old checkpoint pinned;
the reviewer reproduced this with the actual reference/release methods.

Before runtime activation, enforce generation identity and one active attempt
for this request-owned hybrid path. Retire the old checkpoint **back into its
original lease's spare slot budget**, after its last compute/copy fence, rather
than returning it to the global pool and trying to allocate a replacement later.
The latter would let an unrelated grant steal the promised state capacity.
Other serving modes retain their existing behavior; do not silently disable
prefix sharing or rewrite user keys to hide this integration requirement.
