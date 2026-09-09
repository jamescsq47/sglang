# Agentic PD Qwen3.5 port status

## Isolation

- Clean runtime clone: `/homes/siqic/anaconda3/envs/pd_mamba_baseline`
- Development worktree: `/homes/siqic/sglang-agentic-mamba`
- Branch: `codex/agentic-pd-mamba-v2`, based on upstream SGLang `v0.5.14`
- The installed `pd_mamba` runtime has not been modified while the existing
  6P:2D experiment is alive.

## Implemented

- Request-generation lifecycle, ownership manifests, TP mailbox, early claim,
  shared-host staging, and P2D staging were imported as independent modules.
- P workset leasing is separated from the old 0.5.10 Scheduler.
- Workset allocation is atomic across attention pages and one Qwen3.5 Mamba
  state slot; allocation failure rolls back both resources.
- Reverse NIXL registration uses SGLang 0.5.14 `setup_state_kv_args`, so the
  wire layout includes Qwen3.5 temporal/conv state.
- Reverse send/receive helpers pass Mamba source/destination indices through
  native NIXL state transfer.
- Request-generation manifests identify the complete component set, state
  byte size, checkpoint token position, TP size, and layout hash.
- A single shared-memory extent layout for attention KV plus Mamba state is
  implemented and CPU-tested.
- The old Decode lifecycle manager is isolated in `agentic_decode_manager.py`
  and imports on 0.5.14; direct candidates now describe/send composite state.
- Scheduler creates a Mamba-aware workset broker and can select the isolated
  Decode manager only when `SGLANG_AGENTIC_KV_LIFECYCLE=1`. Baseline defaults
  remain unchanged.

## Verified

- `test_agentic_kv_lifecycle.py` plus `test_agentic_mamba.py`: 59 passed.
- Independent portions of `test_agentic_tp.py`: 51 tests observed passing
  before the first 20 unported Scheduler/queue integration failures.
- Python compile/import checks pass for the new hybrid snapshot, transfer,
  direct runtime, workset, and Decode manager modules.

## Current validation status (2026-09-02)

The composite attention-KV plus Mamba-state path has progressed beyond the
original porting checklist: focused CPU tests pass and the physical TP=2 GPU
path transfers and decodes correctly during the healthy prefix of a live run.
It is still **NO-GO for a reported full500 result** because the long-running
SWE-bench validation exposed a control-plane lifecycle failure after successful
early traffic.

Failed run retained for diagnosis:

`/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-4p4d-c128-full500-drainable-20260902-r54`

The run used the collocate-aligned workload: SWE-bench Verified 500 source
rows, c128, 8K per turn, 64 turns, OpenAI structured tools, `glm45` reasoning
parser and `qwen3_coder` tool parser. It is therefore not a harness mismatch.

Observed failure boundary:

- the healthy prefix exercised Qwen3.5 attention KV and Mamba state on TP=2;
- the primary trigger was a Chat Completions metadata-schema mismatch on the
  P->D Host fallback: P had already moved a snapshot to Host staging, while the
  Host snapshot id was written under `sampling_params.custom_params` and was
  discarded by the Chat request schema, so D incorrectly constructed a native
  NIXL receiver and waited for input that P would never send;
- later, P-ready generations were cancelled or timed out while D admission or
  P->D transfer was pending;
- D could remain empty with ample KV capacity while requests stayed in
  prealloc/`WaitingForInput`;
- a consumed `.ready` file was treated as absent even after readiness had been
  latched, and failed generation results/control receipts could poison retries;
- GPUs eventually idled while HTTP retries repeatedly returned the cached 500.

This is currently classified as a **TP=2 P->D control/lifecycle bug exposed by
the long SWE-bench workload**, not evidence of corrupted Mamba tensor contents
and not a SWE-bench harness failure.

## Active repair target

1. Make TP0 admission use the latched P-ready fact after the marker is consumed;
   the marker is an edge notification, not continuing ownership.
2. Put P->D Host metadata in the protocol-correct envelope: top-level
   `custom_params` for Chat Completions and
   `sampling_params.custom_params` for Generate.
3. Give cancellation an explicit destination cleanup path so an aborted P
   producer cannot leave a D preallocation or receiver generation alive.
4. Do not cache a failed Router singleflight result as a completed generation;
   a retry must create a fresh wire attempt while preserving the same logical
   request-generation identity.
5. Preserve physical ownership/fencing: no stale generation may clear or reuse
   pages belonging to an unfinished DMA, and retries may not manufacture a
   terminal fence.
6. Add focused schema, latch, cancellation, retry and stale-mailbox tests; then run the
   full lifecycle/capacity/cancellation/TP/fault suite and obtain an independent
   audit GO before the next GPU run.
7. Validate with a bounded TP=2 smoke and a long stability run before rerunning
   the complete 500-task collocate-aligned experiment.

## Porting rule after r54

The source of truth for serving behavior is the already validated `pd`
implementation at `/homes/siqic/sglang-agentic` (SGLang 0.5.10.post1).  This
v0.5.14/Qwen3.5 branch must not redesign its Router, queueing, timeout, retry,
P-ready, Direct/Slow, or request-generation ownership state machines.

The port should copy those implementations wherever the v0.5.14 interface is
unchanged and otherwise make the smallest mechanical API adaptation.  New
logic is restricted to representing Qwen3.5's complete physical snapshot:

- attention KV pages;
- Mamba temporal state;
- Mamba convolution state;
- their page/state indices, layout metadata, TP shard metadata, transfer,
  restore, release, and rollback.

Current alignment facts:

- `agentic_early_claim.py`, `agentic_tp.py`, and `agentic_tp_control.py` are
  already byte-for-byte identical to `pd`;
- the branch's original copy point was `pd` commit `54807a9913`; the later
  validated `pd` lifecycle commits `c3e40b1935` and `c549c0e005` have now
  been three-way merged into `agentic_host_staging.py`, the isolated
  `agentic_decode_manager.py`, `agentic_prefill_pressure.py`, the workset
  broker, Scheduler wiring, and their TP tests;
- intentional differences from `pd` in those modules are restricted to
  v0.5.14 API/import/layout adaptation and complete attention+Mamba snapshot
  allocation, transfer, recovery, release, and rollback;
- the non-`pd` experiment that treated `finish_reason=length` as a reusable
  tool generation has been removed; length/abort/terminal handling now follows
  the validated `pd` implementation exactly;
- the attempted new Router abort/cleanup state machine was withdrawn; it is
  not part of this port;
- `pd_mamba` imports the worktree through
  `00_agentic_mamba_source.pth`; existing compiled extensions remain supplied
  by the environment because an editable wheel build requires an unavailable
  Rust compiler;
- current combined source-path validation is `347 passed` across Mamba,
  request-generation lifecycle, cancellation/capacity, and TP tests, plus
  `92 passed` for the shared late-binding Router and `21 passed` for the
  targeted NIXL cleanup/Prefill-adder regressions;
- the focused suite now includes hybrid Attention+Mamba TP=2 fault injection
  for rank-local state allocation rollback, Direct state-handle failure,
  D-to-P Slow H2D state failure, D-to-P D2H state failure, and P-to-D Host
  state failure, plus a two-rank Attention/conv/temporal success checksum;
- AST comparison against the validated `pd` source shows 167 Host-staging
  functions and 50 Decode-lifecycle functions are semantically identical.
  Every differing lifecycle function is being held to the narrower porting
  boundary: v0.5.14 import/API changes or complete Mamba-state
  allocate/transfer/restore/release handling; no `pd` lifecycle function is
  missing from the port;
- independent audit approved a bounded TP=2 GPU smoke.  The first physical
  2P:2D/TP=2 smoke (`/tmp/pd-persist/qwen35-27b-tp2-agentic-kv-2p2d-c2-bounded-smoke-r56`)
  exercised repeated complete Direct transfers and the Shared Host fallback:
  D-to-Host reached 5.7--11.9 GiB/s per observed rank transfer and Host-to-P
  reached 32.9--40.0 GiB/s.  TP release occurred only after the complete
  Direct or Host durability fence, with no leaked CUDA process after cleanup;
- that smoke exposed the Mamba-only stop-boundary case where Decode's sampled
  stop token advanced the recurrent checkpoint to 1024/9920 while the reusable
  logical prefix excluded it and contained 1023/9919 tokens.  The port now
  selects the immediately preceding normal-mode ping-pong checkpoint
  (960/9856), transfers matching Attention and Mamba state, and lets P
  recompute the at-most-63-token tail.  Older-state selection is rejected in
  lazy/single-slot mode or whenever the exact retained checkpoint cannot be
  proven.  Direct and Shared Host use the same manifest checkpoint token count;
- the second bounded TP=2 GPU smoke
  (`/tmp/pd-persist/qwen35-27b-tp2-agentic-kv-2p2d-c2-bounded-smoke-r57`)
  completed all four source-ordered SWE-bench tasks with zero request failures.
  It exercised 150 Direct rank sends, 258 D2H rank completions, 258 H2D rank
  completions, and 382 TP release commits, with zero `recompute_required` and
  zero `token_digest_mismatch` events.  The 1024/1023 stop-boundary case was
  physically hit and correctly published a 960-token snapshot; P reused the
  960-token prefix and recomputed the 63-token page tail.  For every observed
  Direct rank transfer, the P-side Attention page digests and conv/temporal
  state digests matched D exactly.  The run shut down cleanly with no remaining
  GPU process.  Its c2 plus GPU-to-CPU hashing throughput is diagnostic only,
  not a performance result;
- Shared Host H2D now has the same opt-in post-fence Attention/Mamba digest
  observability as Direct.  The log point is after complete Attention and state
  DMA fences and before scheduler bind/ownership handoff; it is disabled in
  normal runs and does not modify transfer or lifecycle semantics.  A forced
  Slow-path TP=2 GPU smoke must verify these new destination digests before a
  long/full500 run;
- live smoke r59 exposed one final Mamba-only porting omission: the P-side
  reverse NIXL receiver runtime had been created without `req_to_token_pool`,
  so its descriptor layout did not register Qwen3.5 conv/temporal destination
  buffers.  The call now passes the same HybridReqToTokenPool already used by
  the P workset broker.  D sender and P receiver therefore build their state
  descriptors through the same native `setup_state_kv_args` path.  No Router,
  queue, timeout, ownership, or release behavior was changed;
- the follow-up normal-threshold TP=2 diagnostic
  (`/tmp/pd-persist/qwen35-27b-tp2-agentic-kv-direct-host-digest-smoke-r60`)
  verified 22 logical Direct snapshots: all 44 rank-local Attention digests
  and all 44 rank-local conv/temporal digests matched D source to P destination.
  Four fully comparable Shared Host snapshots likewise matched all 8 rank-local
  Attention and all 8 rank-local Mamba digests.  One additional Host snapshot
  completed while the diagnostic was being interrupted and had no retained D
  source digest, so it is unclassified rather than a mismatch.  The diagnostic
  was intentionally interrupted after sufficient path coverage because full
  GPU-to-CPU hashing reduces Decode to roughly 15 token/s;
- token-digest mismatches remain fail-closed full-Prefill events and are never
  bypassed to claim reuse.

## Full500 r65 failure and recovery protocol alignment (2026-09-02)

The first full-load attempt after r64,
`/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-2p2d-c128-full500-r65`,
failed after roughly 16 minutes with `Slow workset lease lost Host claim`.  It
proved that this branch still had the old local-owner-only Shared-Host recovery
protocol while the Router was assigning completed snapshots across logical P
domains.  The repair mechanically aligns the control plane with the current
validated implementation while retaining hybrid Attention+Mamba snapshots:

- Host placement and recovery ownership are separate.  The storage P owns the
  physical tmpfs extent; the assigned recovery P opens a lazy foreign mapping
  and owns only the recovery claim/workset.
- Router recovery assignments use renewable, reservation-scoped leases.
  Parent-digest failures before claim use an exact
  `(prefill_domain, reservation_id)` CAS, so a foreign recovery P can fail its
  own assignment without impersonating the storage owner.
- A failed in-flight TP recovery becomes `ABORTING`; each rank publishes an
  exact `{claim_id, lease_id}` receipt only after its Attention+Mamba DMA fence
  and workset release.  `FAILED` is visible only after all ranks drain.  Exact
  receipts make retries idempotent if the ledger commit succeeded but its
  reply was lost.
- Foreign mappings never release the physical arena extent.  The original
  storage owner releases that extent only after terminal `CONSUMED`/`FAILED`.
  Every normal, abort, spill, and terminal cleanup path retains the exact
  record and retries if `_release_record()` fails.
- Shared-Host placement now uses effective free bytes plus atomic byte
  reservations; recovery selection remains independent and Router-owned.

The current complete in-tree lifecycle regression is `281 passed` across
`test_agentic_mamba.py` and `test_agentic_tp.py`, including injected
post-commit ACK loss, TP-rank abort ordering, foreign assignment failure, lazy
hybrid mapping, storage-owner terminal release, and release-failure retry.
The next step is an independently audited TP=2 GPU smoke before replacing r65
with a fresh full500 run.

## Full-load r67 pre-I/O lease race and repair (2026-09-02)

The 128-request sustained smoke r66 crossed the r65 failure point and exercised
Direct, Shared Host, pressure eviction, and 20--42 GiB/s Host-to-P recovery for
17 minutes without a fatal error.  The subsequent full500 attempt r67 ran for
about 25 minutes before exposing a narrower TP race on snapshot
`83d1182f8fe046c19affa503dc56a2e5:5`:

- the Shared-Host ledger had pinned the same logical recovery claim with rank
  leases 2594/2597 in the pre-I/O `leased` phase;
- before any H2D was published, native TP allocation-plan reconciliation
  retired the still-`active` rank-local worksets and recreated the same
  owner/shape with newer physical lease ids (rank0 2598);
- the exact ledger check correctly rejected the newer lease, but the old code
  treated this recoverable pre-I/O race as a scheduler-fatal ownership loss.

The root repair changes the ordering in `gate_request`: the broker now moves a
physical lease from `active` to `io_reserved` **before** publishing its exact id
to the Shared-Host ledger.  `io_reserved` is an allocator ownership fence, so a
stale TP plan can no longer silently free or recreate the destination pages.
If the currently installed TP plan had already omitted that still-active lease,
`begin_io_attempt` refuses the pin and lets the allocator retire it before any
lease id is published; once pinned, TP0's following control plan necessarily
retains the exact lease.
For a pre-existing ledger claim left by an older process/control attempt, an
exact CAS may rebind only the same owner, claim, rank, and pre-I/O `leased`
phase from the observed old id to the current already-reserved id.  It cannot
replace an `io_inflight` or handed lease.  Failure cleanup first cancels the
unstarted I/O reservation, then releases the exact workset and rolls back the
exact Host claim.

Regression coverage now explicitly proves that an `io_reserved` lease omitted
by a stale TP plan fails closed without allocator reuse, and that compatibility
rebind changes only the exact rank/old id while refusing stale CAS and
`io_inflight` replacement.  Current validation is `284 passed` for
`test_agentic_mamba.py + test_agentic_tp.py`, `93 passed` for the shared Router,
and `21 passed` for NIXL cleanup/Prefill-adder tests.  r67 is invalid and was
cleanly stopped; its logs remain under
`/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-2p2d-c128-full500-r67`.

## Full500 r68 TP Host-visibility race and repair (2026-09-02)

r68 ran for about 46 minutes and crossed both earlier failure windows, but it
then exposed an older TP visibility race on snapshot
`017b79a735554f9283a70d178a8ff3c7:53`.  Both D ranks had committed their Host
shards, yet P rank0 observed `HOST_READY`, adopted its local shard, and changed
the shared ledger to `H2D_LOADING` while claiming recovery.  P rank1 had not
yet polled the level-trigger: its local shard therefore remained in `active`
for 27 minutes and it never joined recovery.  Rank0 subsequently entered a
different model-forward path, so the TP group diverged and NCCL all-reduce
sequence 226331 timed out after 600 seconds.  This was not a bandwidth stall or
lease replacement; r68 is invalid and its logs are retained under
`/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-2p2d-c128-full500-r68`.

The recovery claim is now a TP group barrier.  A partial rank claim records the
exact logical owner and fences pressure eviction, but deliberately leaves the
ledger in `HOST_READY` so every P rank can observe and adopt its local extent.
Only the complete rank set transitions the generation to `H2D_LOADING`.
`gate_request` then re-reads this group state and cannot request, allocate, or
pin a P-HBM workset before the transition.  Idempotent claims preserve any
existing lease/phase fields instead of overwriting them.  Abort cancellation
also accepts the partial-claim `HOST_READY` state, while eviction rejects it
because `recovery_owner/recovery_claims` are already present.

Regression coverage explicitly reproduces the late-peer window, proves a
partial claim remains visible but cannot be evicted, proves no workset intent
exists before the group barrier, and retains the TP abort lifecycle tests.
Current validation is `286 passed` for
`test_agentic_mamba.py + test_agentic_tp.py`, `93 passed` for the shared Router,
and `21 passed` for NIXL cleanup/Prefill-adder tests.

## Full500 r69 stopped: data paths valid, lifecycle/performance invalid (2026-09-03)

The follow-up full500 run
`/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-2p2d-c128-full500-r69`
was stopped at the user's request after roughly 75 minutes because its steady
progress was far below the expected rate.  The run crossed r68's TP claim
barrier and exercised both reverse Direct and Shared-Host restore, including
rank-local Host-to-HBM transfers around 20--26 GiB/s, without NCCL divergence
or a Mamba-state mismatch.  It therefore validates broad path reachability,
not end-to-end performance or lifecycle correctness.

Two concrete blocking defects were observed:

- Snapshot `132edb2dc24d4b9297d63555bfa89203:46` completed Shared-Host H2D on
  both TP ranks, but the final `handed` ledger CAS lost lifecycle ownership.
  `gate_request` then retried without backoff or a terminal recovery action.
  P0 emitted 1,361,734 `shared_host_handoff_retry` exceptions from 23:35:45
  through 00:05:25; its log grew to 6.89 million lines.  This request never
  became runnable and the busy retry materially perturbed P0's control plane.
- Five non-shutdown parent generations reached the Router's route timeout.
  Four had completed D-to-Host snapshots that remained unavailable to the
  child request instead of being assigned, restored, or converted to
  recomputation.  The fifth,
  `e13299f465aa45a0b5dc27ab4db25be2:54`, was explicitly classified by D as
  `final_skip ... output_kind=terminal`, but the OpenEnv agent immediately
  issued generation 55 using it as the parent.  Because D intentionally wrote
  no reusable snapshot for a purported terminal output, the Router waited for
  a route that could never exist and retried every route-timeout interval.
  Terminal classification is therefore not aligned with the authoritative
  agent continuation decision.

At the same time, physical P0 HBM and its SGLang execution queue were often
nearly empty while Router shadow accounting retained tens of parent requests
and hundreds of thousands of pending tokens.  P1 performed substantially more
Prefill, and logical-D running counts fell from roughly 48--50 early in the
run to 2--11 late in the run.  Shared-Host pressure also caused hundreds of
snapshot evictions and corresponding full-Prefill fallbacks.  These are
control/lifecycle stalls, not evidence of insufficient raw Direct or Host-H2D
bandwidth.

Do not use r69 for throughput or accuracy comparison.  Before another full
run, a bounded smoke must prove: (1) handed-CAS failure leaves the retry loop
through an explicit rollback/reclaim path with backoff; (2) every Host-ready
parent reaches restore or recompute within a short bound; and (3) only the
agent/harness's authoritative completion decision permits `final_skip`.
The run was cleanly stopped, all eight GPUs and dedicated ports were released,
its 128 OpenEnv containers were removed, and its logs were retained.

## Required correctness assertions for audit

- A snapshot becomes visible only after attention KV and Mamba state complete
  on every TP rank.
- D releases neither attention pages nor `mamba_pool_idx` before the complete
  Direct/Host fence.
- P releases neither Host state nor the manifest before attention and state
  are bound to the live request-generation.
- A failed/cancelled partial transfer releases or quarantines all components;
  it never exposes an attention-only prefix.
- Restored generation output matches colocated output token-for-token, except
  for at most the native page/chunk tail recomputation.
