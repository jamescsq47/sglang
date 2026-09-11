# TP1 request-owned Mamba H2D decoupling (R11 evaluated)

R9 reference: commit9262db4fa3, validation/SWE_QWEN35_9B_R9.md. This is an
opt-in engineering change, not a throughput claim. Set
`SGLANG_AGENTIC_KV_P_H2D_DECOUPLED=true`; it is effective only for TP1 Mamba
with request-owned checkpoints. Dense, TP>1 and default-off paths are unchanged.

## Changed mechanism

1. Keep original Host recovery claim and atomic full parent+suffix Attention
   plus Mamba lease. Space allocation/free stays on the scheduler safe boundary.
2. Immediately after broker service, advance only already-selected Slow
   snapshots to start granted H2D or bind completed H2D. No second fresh-request
   admission scan, new class priority, or background allocator/Radix writer.
3. After final composite Attention+Mamba event and broker quiescence, release
   physical lane, provided no DMA references or CPU prefetch remain. Keep Host,
   destination lease and resident admission until original Radix/handoff or
   fenced failure cleanup. Occupancy counting excludes recycled transport lanes.
4. Resident reservations are bounded to2×physical lanes (4 copying/preparing +
   at most4 additional completed/preparing worksets for R10). This is an explicit
   bounded staging-budget change, not a pure same-budget timing ablation. No
   fixed HBM pool is allocated; ordinary full-workset allocation still applies.
5. Keep the existing synchronous page-index mirror. Although Slow does not
   use Direct page metadata, this also fences scheduler-stream allocator-index
   production before another thread consumes it. Removing it requires a separate
   explicit CUDA stream dependency and is not part of this experiment.
6. Per-snapshot timing logs selection→grant, I/O→composite fence, fence→handoff.
   The labels are stage wall times, not proof of pure hardware DMA duration.

## Ownership / failure map

Affected transition: D2P_HOST_OWNED → fenced composite H2D → P_HBM_OWNED.
Lane recycling is not a KV ownership transition. Host remains authoritative
through failed ledger ACKs/failed binding, and is released only at the original
successful bind/handoff boundary. Resident workset retains both KV pools.
Transport/CPU prefetch failure holds the physical lane until existing drain;
capacity failure retains Host; cancellation uses original exact-lease cleanup;
shutdown still drains all DMA users before reclaiming arenas. No timeout or
fallback/recomputation/routing/harness policy is changed. TP keeps original code.

Acceptance criteria1–4/6/7: ownership/CAS, P2D releases, D2H releases, TP and
stable-prefix reuse unchanged. Criterion5 is the target: remove unnecessary
Forward-boundary waits, not bypass necessary fence/allocation serialization.
Criterion8: CPU lifecycle/fault tests and independent GO required before GPU.

## Planned comparison

R9-aligned finite500 SWE Verified, Qwen3.5-9B TP1 2P:6D c500, 4 H2D lanes/P,
Q32/32 recorded with congestion/fixed-failure recompute off, same source order,
unchanged harness/model/temperature/static memory/page64/checkpoint mode.
Report full evaluation accuracy/time/length plus the same300–1500s window.
This finite dataset run does not satisfy replenished closed-loop performance
acceptance; do not relabel it as a steady-state benchmark.

## Prelaunch validation

2026-09-11:597 agentic lifecycle/fault/TP tests plus4 native Mamba cache tests
passed (601 total). Independent agent `/root/audit_h2d_lane_decoupling` reviewed
the final diff and independently passed15 new regression cases: **GO** for
the scoped TP1 opt-in finite500 comparison. Audit-required narrow lifecycle-race
handling was added; removal of the allocator-index readiness fence was reverted.
All eight criteria above were checked. GPU outcomes are still pending.

## R10 integration failure and correction

R10 was stopped during early evaluation and is NOT a performance result. The
initial implementation hooked only `Scheduler.get_next_batch_to_run`, but PD
uses `get_next_disagg_prefill_batch_to_run`. Additionally physical occupancy
included prestart lane owners while `_agentic_io_kind` treated them as NEW,
so a full4-lane pool rejected its own already-granted owners (4 active broker
leases,0 H2D loads). This was an integration test coverage gap.

The corrected shortcut is in the shared `_agentic_service_p_workset_leases`
wrapper reached by both entrypoints. Already-selected prestart Slow owners
are classified as active and can progress without acquiring a new-slot budget.
Added tests execute the actual PD entry with the real broker and Host gate:
four preselected requests become four start-allowed loads in that boundary,
while a fifth metadata-only request remains unselected. A second test exercises
the ordinary drain with all4 lanes already reserved.17 focused tests pass.
The existing bootstrap queue may still add a scheduler boundary after binding;
this change does not claim bind→Forward always occurs in the same tick.

Corrected prelaunch:603 tests passed (599 agentic +4 native Mamba cache).
Independent auditor re-traced the actual normal/overlap PD entrypoints,
independently passed all17 focused regressions, and returned fresh GO forR11.
R10 supervisor, watcher, owned GPU workers and run-labelled containers all exited
before relaunch; the co-tenant onGPU7 was untouched.

R11 completed all500 tasks. See SWE_QWEN35_9B_R11.md: physical-lane occupancy
and recovery waiting decreased, but middle-window D throughput fell1.59% and
P Forward fraction fell4.18 percentage points. No throughput acceptance claim;
keep the opt-in default off. All500 task prefix-accounting checks and complete
Host/Direct lifecycle count conservation passed; owned GPU workers exited.
