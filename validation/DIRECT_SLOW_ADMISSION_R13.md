# R13: transport slot ownership and one-boundary Slow handoff

Status: frozen implementation passed CPU regression and independent audit.
No GPU acceptance. R12 was stopped and owned GPU cleanup verified before edits.

## Evidence and changes

R12's settled post-warmup cohort had 86 Direct completions out of 171 fast
arrivals. Started failures spent 435 ms before their allocator intent, versus
53 ms for successes. Claim RPC differed by only about 7 ms. Three concrete
no-start generations still appeared as active Direct leases after D had
already durably offloaded their snapshots and released its source pages.
The actual admission routine reproduces this: a grant's refreshed manifest
is SLOW_FALLBACK, so it cancels the unstarted owner and drops the arrival;
the broker nonetheless counts its active, retiring lease against the four
transport slots until the later native allocator retirement.

The Direct change separates those two resource lifetimes. A pending intent
or active unposted lease whose existing cancellation/retirement tombstones
already forbid begin_io_attempt does not occupy a transport admission slot.
Its real pages remain unavailable until the unchanged all-rank retirement.
Pending claim (io_reserved), posted I/O, receivers and active group grants
retain their existing accounting and physical-fence requirements. No cap,
deadline, HBM reservation, routing or TP1 policy changes.

R12 Slow last-300s all-copy-to-all-P-handoff averaged 1.412 seconds, including
564 ms loaded-to-bind and 717 ms bound-ACK-to-release. The Slow change moves
local workset handoff to the existing scheduler BIND boundary, before its
bound ACK. The same worker then confirms exact all-rank CONSUMED, finalizes
the manifest, closes local Host mappings and sends handed ACKs. Rank0's final
native ADMIT remains mandatory. This removes the intermediate native COMMIT
round; it does not eliminate the first Radix-safe scheduler boundary.

## Ownership and failure contract

- BIND may transfer local suffix/runtime ownership to the waiting Req but
  never makes it runnable. Every rank must successfully bind **and hand off**
  before its bound receipt can contribute to CONSUMED.
- All followers, not just rank0, check the pushed exact owner/claim/read-epoch
  and all-rank binder set before local close/handed. Missing or delayed proof
  means wait, not success or forced release.
- Partial binding/handed failure retains Host and rolls back the matching
  local owner. Binding uses abort_bind; handed uses release_handed plus native
  pre-admission Mamba cleanup. HTTP cancellation takes the same path. Actual
  suffix pages still retire through the original native TP transaction.
- Old callbacks and cancellation cannot publish success for a newer attempt.
  No timeout is interpreted as a DMA fence. Shutdown uses owned cleanup.

## Eight acceptance requirements

1. Unique generation/lease/attempt ownership preserved; test partial handoff.
2. P2D Direct release unchanged.
3. P2D Host durability and source release unchanged.
4. D2P source durability/release unchanged; Host completion still requires
   exact all-rank physical/read/bind facts.
5. No new thread, synchronous scheduler RPC, scan or file control. Existing
   I/O workers progress metadata; scheduler owns allocator/Radix/Req changes.
6. TP2/8 failure, delayed rank and cancellation tests must prove no early
   Forward; TP1 and legacy completion remain unchanged.
7. No new recompute/eviction trigger or checkpoint/length policy.
8. Relevant lifecycle/fault tests, full CPU gate and independent GO must pass
   before a same-configuration c128 GPU experiment.

Runtime acceptance must inspect settled Direct arrival cohorts (including
never-started failures), Slow all-shard copy-to-handoff/slot recycling,
outstanding-age and cancellation ownership, plus wall-clock Decode, Forward,
running and Attention/Mamba occupancy. Smoke or a green CPU suite alone is
not performance acceptance. Keep the R12 model, data order, memory fractions,
tool 2s / setup 1s, Host capacities and disabled congestion recompute fixed.

## Interrupted R13 outcome

Both nodes passed prewarm and the Direct/Slow two-turn token-equality smoke.
The c128 SWE500 workload started at **08:39:11.719 UTC**. After 300 seconds
of warmup, the baseline at Unix time 1789980263.4664547 had P execution
1999.3739189453122, D execution 1992.975879191398 (eight-rank sums), and
D output 147995. The sample at 1789980359.2342327 had P execution
2580.9726213073736, D execution 2631.281999873159, and D output 196667:
95.768 seconds, **508.2 logical output tokens/s**, P Forward **75.9%** and
D Forward **83.3%**. This is NOT a completed 1200-second measurement.

Settled arrivals 08:44:25--08:45:25: **183 eligible, 92 Direct complete,
56 fallbacks without any rank start, 35 after a rank start**, no pending or
conflicting outcomes. Direct success **50.27%**, essentially unchanged from
the R12 diagnostic cohort, so runtime acceptance remains unmet.

At 08:46:17 the trailing 120-second Slow group sample had 164 complete groups:
all-eight-copy to all-eight P handoff release **0.896 s**, versus R12's 1.412 s
(different diagnostic windows, not a matched final benchmark). No incomplete
or inverted release cohorts. Three selected copies were pending at ages 1--2s;
one copied generation awaited release at age <1s (log resolution). Loaded ACK
to bind queue still averaged 490 ms; bound ACK to release fell to 207 ms.

Stopped only the verified R13 coordinator PID 773935 via SIGTERM around
08:47 UTC. It exited, run cleanup.json exists, and both GPU compute-process
queries returned empty before further changes. Retain the Slow improvement;
investigate the remaining Direct native-admission/deadline delay rather than
marking the run successful or widening its one-second setup deadline.

## Final pre-run gate (2026-09-21)

Full `validation/check_multinode_cpu.sh` completed after the last production
edit: **1393 passed, 2 GPU-only skipped**, plus **88 launcher/diagnostic tests**.
Raw log: `/tmp/dualpd-r13-cpu-gate-final.log` (08:33 UTC). Both repositories
pass `git diff --check`. Independent `control_audit` returned **GO**, rerunning
170 tests across eight focused files. The audit found and independently
reproduced a cancelled-context entry after Radix bind; installing rollback
identity before validation and taking the existing rollback on failure now
has real TP2/TP8 regression coverage. Mamba Req checkpoints and another live
Radix-owned checkpoint are also tested for exact, non-duplicated cleanup.

Next run: `qwen35-122b-swe500-tp8-c128-socket-full-r13`, existing Bash launcher,
same R12 settings. Both a10/a11 GPU compute-process lists and ports
23900--23905 were empty before launch. CPU tests and audit do not establish
the runtime performance objective; retain and inspect full workload outcomes.
