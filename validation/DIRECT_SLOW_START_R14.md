# R14: overlap authorization with preparation; consume ready events directly

Status: implementation frozen; full CPU gate and independent actual-code GO passed.
No GPU acceptance. R13 stopped safely before edits; retain its Slow BIND/handoff
improvement. No model, deadline, capacity, concurrency or recompute policy change.

## Evidence

R13 post-warmup arrivals 08:44:25--08:45:25: 92/183 reached Direct native ADMIT,
56 fell back before any rank started and 35 after a start. This remains about
50%, despite zero allocator misses. An independent 30-second sample found
successful starts waiting 241 ms intent-to-native-plan versus 459 ms for
started fallbacks. Plan-to-service was only about 30 ms; physical allocation
about 4--5 ms. Do not attribute all these delays to network or pretend that
moving service 30 ms earlier removes the full native iteration wait.

The selected Direct intent currently consumes one of four admission slots
while waiting for native allocation and a subsequent receipt-1 round trip.
Existing receipt-1 only grants logical execution permission: it is not a
physical lease or a completion fence. Overlap that notification with allocation.

Slow already reduced all-copy-to-group-handoff to about 0.9 s, but retains a
second native START after all ranks have prepared. During R13's trailing
120-second sample, remote Host read calls averaged about 297 ms while the
load-to-loaded-ACK wall average was 1473 ms. These timers overlap other stages;
they are not an additive end-to-end partition or proof that all remaining time
is scheduler delay. The extra prepare-ready-to-native-START edge is explicit
in the implementation and can be removed without moving allocator ownership.

## Lifecycle contract

Direct, socket TP only:

- Rank0 selects an intent under the unchanged four-slot cap, registers its
  existing active context (lease may be absent), then publishes the existing
  exact-attempt receipt-1. No additional grant protocol or collective.
- Each rank still waits for its own real workset from the original rank0-frozen
  native allocator plan and executes the original claim/receiver sequence.
  Receipt alone never starts DMA. Negative/terminal receipts and cancellation
  tombstones take precedence; repeated queue visits cannot resurrect permission.
- The active context retains pre-grant cancellation ownership. A late native
  allocation or claim must retire through the existing all-rank fence protocol;
  inability to publish authorization is not permission to discard that owner.
- D reports successful send submission on the same worker visit using its
  existing setup mailbox, rather than deferring discovery to another visit.
  Failed/partial submission is not success; reporting failure must not change
  the physical send outcome or force-release source pages.

Slow, socket TP only:

- Native PREPARE authorizes one exact restore attempt. After the unchanged
  native full-workset grant, exact local claim/attach and all-rank prepared
  receipts, the I/O worker can set the existing start permission and read.
- H2D_LOADING alone is not proof: the first rank already sets that state.
  Validate owner, claim, epoch, TP size and each rank's prepared lease identity.
- No second scheduler START is required, but all-rank physical completion,
  native BIND, handed receipts and native ADMIT remain mandatory.
- Cancellation drains posted DMA and outstanding callbacks before release;
  partial preparation/IO failure cannot yield runnable partial TP work.

TP1 and non-socket legacy execution retain their original gates.

## Eight requirements and verification

1. Unique ownership: exact generation/claim/attempt/lease tests on stale commands.
2. P2D Direct release unchanged.
3. P2D Host durable/source release unchanged.
4. D2P Host durable/source release unchanged; no timer replaces a DMA fence.
5. Existing workers and pushed metadata; no new blocking RPC, scan, filesystem
   control or allocator thread. Remove redundant serial edges, not safety facts.
6. Real TP2/TP8 delayed-grant, delayed-prepare, partial-post and cancel regressions;
   TP1/legacy cases remain in the CPU gate.
7. No new recompute or eviction decisions and no shortened KV/checkpoint lengths.
8. Full CPU gate and independent actual-code GO before the same-config R14 run.

Runtime comparison must use settled fast-arrival cohorts with Direct success
measured at `early_direct_admit`, not reversible `early_direct_group_complete`.
R13 had one all-receive-then-cancelled example: it safely rolled back before
Slow, not a final Direct success. The offline helper now reports that category
separately. R12/R13 post-warmup cohort counts above are unchanged under the
corrected final-admission criterion.

Inspect both improvements, group ownership convergence, pending-age tails,
Forward/running/Attention and Mamba capacity, and output tokens. Stop renewed
control/lifecycle failures; smoke and CPU pass alone do not prove performance.

## Pre-run validation (2026-09-21)

The frozen implementation passed `validation/check_multinode_cpu.sh`:
**1443 passed, 2 GPU-only skipped**, plus **89 launcher/diagnostic tests**.
Raw output: `/tmp/dualpd-r14-cpu-gate-final.log`.
Direct author separately ran 510 tests (23 new early-authorization cases);
Host author ran 71 focused and 421 compatibility tests. Both repositories pass
`git diff --check`. These suites overlap and must not be summed as unique tests.

Pre-launch checks found no compute processes on a10/a11 and no listeners on
23900--23905. Independent `control_audit` returned GO after 199 lifecycle/fault
tests across seven files, plus nine offline parser tests. It verified the
post-authorization room-change boundary in the actual claim path, unknown
publication cancellation ownership, same-visit posted report without duplicate
DMA, and exact all-rank prepared claims before worker START. All eight design
criteria passed code/CPU review. Benchmark configuration stays the R13
configuration; GPU performance and long-run lifecycle acceptance remain open.

## Interrupted R14 diagnostic outcome

Run `qwen35-122b-swe500-tp8-c128-socket-full-r14` passed both prewarm and
Direct/Slow two-turn token-equality checks. SWE500 c128 started at
**09:13:42.525 UTC**. After 300-second business warmup, metrics at Unix time
1789982348.1277735 recorded P/D execution 1894.4223821258543 /
2108.9790483403226 (eight-rank sums), D output 152961. At 1789982454.6749692,
P/D execution was 2531.003964309692 / 2816.1961686596906, D output 205045:
106.547 seconds, **488.8 logical output tokens/s**, P Forward **74.7%**,
D Forward **83.0%**. This is not a completed 1200-second benchmark.

Settled post-warmup fast arrivals 09:19:10--09:20:10: **163 total, 61 final
Direct ADMIT, 63 fallback without any rank start, 39 fallback after a start**,
zero pending/conflicting outcomes. Success **37.42%**, not an improvement
over R13's 50.27% diagnostic cohort. Allocation misses remained zero in the
periodic workset counters; zero misses does not by itself prove every arrival
had immediately available capacity.

Trailing 120-second sample ending 09:21:06 had 206 complete Slow TP groups:
all-copy-to-all-P-handoff **0.888 s**. Per-rank TP0 selected-to-copy was 1.307 s;
loaded-ACK-to-bound-queue 549 ms remains a large post-copy component. New
prepared-to-start / start-to-submit / submit-to-copy means were 98 / 23 / 332
ms. These are distinct completed cohorts and not an additive end-to-end
partition. No incomplete/inverted released group was present at that sample.

For the example `81918b416d234598880691805777ac44:6`, two rank starts were
safely rejected when their leases were already retiring. All eight D ranks
released through Host at 09:17:10; P selected it at 09:17:50 and all P ranks
completed handoff at 09:17:52. It did not remain orphaned, but about 40 seconds
of pre-selection waiting is not explained by its roughly two-second selected
pipeline. Investigate logical restore occupancy and Direct pre-start queues;
do not substitute further small DMA changes for this admission bottleneck.

Stopped only verified coordinator PID 957860 via SIGTERM at **09:21:17 UTC**
because performance acceptance was unmet. It exited; cleanup.json exists;
both nodes' GPU compute-process queries were empty before further edits.
The coordinator's signal-15 failure artifact is intentional termination, not
an engine crash. No production files were changed during the experiment.
