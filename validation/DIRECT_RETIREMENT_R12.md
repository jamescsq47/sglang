# R12: preserve unposted-versus-posted Direct ownership on cancellation

Status: CPU/audit passed; GPU diagnostic stopped with Direct objective unmet.
No full-run performance acceptance. See interrupted outcome below.

R11 improved a post-warmup Direct arrival cohort but exposed an unretired
Direct reservation which also blocked the same generation's Slow recovery.
The exact snapshot and stop evidence are in `DIRECT_ADMISSION_R11.md`.

## Root cause and required correction

A pending async claim owns an `io_reserved` workset before any receiver data
transfer is submitted. Native TP retirement currently changes both reserved
and posted I/O to `release_pending`. Later claim cancellation only recognizes
`io_reserved`, cannot cancel the lease, yet discards its claim context. No
receiver exists to report a DMA fence, so this reservation cannot retire.

Preserve the physical distinction between reservation and submitted I/O.
Cancellation intent must not erase the only proof that no write was posted.
Settled claim cleanup must retain its context until exact-attempt retirement
is accepted. Reuse the existing broker and TP group retirement, not another
timeout or parallel state machine. A grant tuple with no lease must not hide
a matching lease materialized meanwhile by the broker.

The chosen implementation keeps one physical state rather than adding a
second "DMA posted" flag. Native TP retirement and exact release of a planned
`io_reserved` lease only set the existing retirement-intent maps, preserving
`io_reserved` until its exact claim callback drains. `mark_io_inflight` refuses
new publication after such an intent. Posted `io_inflight` still becomes
`release_pending` and requires the original physical completion fence.
The TP1 no-plan release behavior is unchanged. Socket TP unstarted cancellation
uses the broker's actual owner-scoped lease, not a possibly stale cached tuple.

## Lifecycle/acceptance gate

1. Unique ownership: exact request-generation, lease and I/O attempt identify
   cancellation. Stale callbacks cannot affect a newer Slow lease.
2. P2D Direct release and 3. P2D Host release remain unchanged.
4. D2P Host/source release remains unchanged. Slow may start only after the
   old Direct owner's actual group retirement; do not bypass that gate.
5. Cancellation uses existing asynchronous claim completion and native
   allocator retirement. No new blocking RPC, scanning, thread or NFS control.
6. TP1/TP2/TP8 tests cover reservation-before-submit cancellation, submitted
   DMA, one delayed rank, delayed claim result, repeated retire, and late
   attempt callbacks. Posted DMA still needs its true completion fence.
7. No new recompute, eviction, deadline, capacity or transfer-length policy.
8. Reproduce with the real broker, run lifecycle/fault regression and obtain
   independent GO before a same-config c128 GPU validation.

On success an unposted reservation follows the existing safe group release;
on pending/unknown claim outcome it stays owned until resolved; posted I/O
stays owned through its physical terminal fence. Cancellation and shutdown
must never infer completion from elapsed time. GPU run and result pending.

## Frozen CPU gate and independent audit

The real multi-broker TP2/TP8 regression also exposed commit-then-abort in the
same native envelope re-opening a retirement. Successful all-rank commit now
marks the frozen allocation plan retired immediately (with or without a local
lease), instead of waiting for physical free. Existing service marking remains
idempotent; a new allocation epoch retains the existing reset semantics.
Repeated cancellation before and after physical free cannot resurrect the old
plan; a subsequent Slow lease for the same generation remains valid.

Final `validation/check_multinode_cpu.sh`: **1360 passed, 2 GPU-only skipped**;
launcher/diagnostic suite **88 passed**. Raw log:
`/tmp/dualpd-r12-cpu-gate-final.log`. The gate now includes
`test_agentic_direct_retirement.py` (28 new cases). Independent `control_audit`
returned **GO**, independently running 124 lifecycle cases and 8 offline
diagnostic cases. The new real multi-broker tests use one leader plan, delayed
claim and DMA on different ranks, the native producer/consumer and group ACK
reduction; they are not merely one broker parameterized with a TP size.

No GPU performance claim follows from those checks. The next same-config run
is `qwen35-122b-swe500-tp8-c128-socket-full-r12`, through the existing
`tools/dualpd/qwen35_multinode.sh run --concurrency 128 --run-dir ...` launcher.
Verify Direct outcomes including never-started attempts and pending work,
Slow copy-to-group handoff, absence of stranded cancellation leases, plus
P/D Forward, running, KV utilization and throughput. Stop on renewed lifecycle
stall rather than widening deadlines or silently recomputing.

## R12 running (not yet accepted)

Both nodes passed Host prewarm. Direct/Slow two-turn token-equality smoke
passed at 07:59 UTC; SWE500 c128 workload loaded at 07:59:22.169 UTC.
The coordinator is live in exec session 39003. Configuration comparison with
R11 differs only in run/output identifiers and authentication token.

After at least 300 seconds of business warmup, the metric baseline was sampled
at Unix time **1789977885.0748413** (08:04:45.075 UTC): P execution counter
2032.3168688125604, D execution 2083.5663641233446 (sums over 8 physical ranks),
D output counter 157476 (logical group tokens). A complete 1200-second window
must end no earlier than 08:24:45.075 UTC; neither early counters nor smoke
constitute final performance acceptance. Earlier 201.925-second diagnostic
interval gave 519.94 output tokens/s, P Forward 79.31%, D Forward 82.96%.

## Interrupted diagnostic outcome

Stopped the verified R12 coordinator PID 500446 with SIGTERM at 08:12 UTC.
The coordinator exited, cleanup.json exists, and GPU compute-process queries
on both a10 and a11 returned empty. No production source was edited during
the run. This is not a completed 300+1200-second benchmark or SWE500 result.

The post-warmup sample at 1789978354.5342326 had P execution counter
4989.5040036087, D execution 5169.91538488388, and D output 374258.
Compared with the baseline above, the 469.459-second interval yielded about
461.8 output tokens/s, P Forward 78.7%, and D Forward 82.2% per physical GPU.

The settled fast-arrival cohort 08:04:45--08:05:45 contained 171 generations:
86 Direct completions, 64 fallbacks with no rank start, and 21 fallbacks after
a rank start. No pending or conflicting outcomes remained in that cohort.
Thus Direct acceptance remains unmet, despite the cancellation repair.

Last-300-second diagnostic at 08:11:41: all-eight-copy to all-eight P handoff
release averaged 1.412 s; rank-copy skew 0.235 s. Loaded ACK to bound queue
averaged 564 ms, bound ACK to release 717 ms. These P handoff timestamps do
not measure the remote source Host extent's physical-free time. There were
no incomplete all-rank release cohorts in that sample. Slow improved versus
R4, but still waits across multiple native scheduler boundaries.

Direct successful-start timing cannot explain never-started requests by
itself. The settled cohort shows arrival-to-intent averaging 53 ms for
successes versus 435 ms for started fallbacks; intent-to-plan was 243 versus
350 ms. Claim RPC differed by only about 7 ms. A sample with zero active DMA
and all four admission slots occupied motivates auditing logical slot
ownership and cancellation retirement before optimizing small RPC latency.
