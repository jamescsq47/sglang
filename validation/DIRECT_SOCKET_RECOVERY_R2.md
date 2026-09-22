# Socket TP Direct recovery, 2026-09-20

## Observed failure and scope

The c128 full500 run `qwen35-122b-swe500-tp8-c128-socket-full-r1`
was stopped through its owned coordinator after analysis. Both a10 and a11
GPU process lists were empty afterwards. It is a failed progress/performance
diagnostic, not a full SWE accuracy result.

Last Direct start: 22:08:17 UTC. Four partially started generation grants
remained, while Host recovery continued. A native TP rollback removed seven
receivers, but rank0 waited for those deleted receivers to report through a
second destination-abort barrier. In addition, P's setup timer required full
DMA completion, rather than sender setup, before disarming.

Only socket TP Direct changes in this round. Slow scheduling, worker counts,
buffers, source-order workload, sampling, memory ratio and thresholds stay
unchanged (tool 2 seconds, setup 1 second, congestion recompute disabled).

## Ownership and failure plan

- D owns the parent until the existing all-shard Direct completion or Host
  durable transition. P holds an exact workset lease throughout any posted IO.
- D rank0 owns the arrival-relative setup deadline until every D shard reports
  successful send submission. A pessimistic `sent=True` before a raising send
  is NOT such an acknowledgement. Once every send is posted, that deadline
  does not constrain DMA duration. Existing transfer-error/source fences stay.
- The posted bit uses a separate typed namespace on the existing persistent
  TP connection, scoped to the immutable Direct wire-room attempt. It carries
  no ownership authority, no tensor data, no model collective, and no files.
- A D no-future-write proof marks only the matching P IO attempt quiescent.
  The ordinary native rollback then removes Radix references and retires the
  local receive. All ranks, including never-started ranks, acknowledge that
  SAME rollback before rank0 returns the claim and clears the grant.
- Failed authoritative claim release retains the grant for retry; it must not
  publish a false terminal receipt. Missing network messages never free pages.
- TP1 and the existing non-socket TP compatibility branch remain unchanged.
- Shutdown continues through the owned launcher; no broad process kills.

## Eight acceptance criteria

1. Unique owner: existing lifecycle CAS and attempt/lease matching retained.
2. P→D Direct release: no code changes in that path.
3. P→D Host release: no code changes in that path.
4. D→P Host release: existing durable fence/fallback unchanged.
5. Independent progress: Direct cancellation must return its IO slots; bounded
   event reports run on Direct workers, no new scheduler RPC or Host wait.
6. TP atomicity: only rank0 times out/decides; all-rank rollback still required.
7. Parent correctness: no new recompute or eviction; real IO fences unchanged.
8. Gate: CPU fault tests and independent audit required before GPU restart.

## Validation so far

- Final targeted Direct recovery regressions: 11 passed.
- Final full CPU gate: 1105 passed, 2 GPU-dependent tests skipped; launcher 75 passed.
- New regressions cover TP2/TP8 posted-vs-completed, a missing follower after
  rank0 sends, partial failed launch, common cancellation without the second
  barrier, stale attempt rejection, and claim-release failure retry.
- Integrated regression executes native abort, actual receiver cleanup, the
  group claim return and grant clear: seven followers finish before rank0;
  the optional eighth shard never started; Direct slot count returns 4 to 3.
- Independent final rerun: 11 passed (14.26 s), GO for identical c128 GPU
  engineering validation with Slow parameters unchanged. New run:
  `qwen35-122b-swe500-tp8-c128-socket-full-r2`. GPU workload validation pending. CPU success
  alone does not establish Direct liveness or performance under c128 load.

## Initial GPU observation (2026-09-20 23:00 UTC)

- Both startup two-turn Direct and Slow token-consistency checks passed.
- The c128 workload loaded all 500 SWE-bench Verified samples at 22:57:14.
- Through 23:00:00, rank0 logs contain 326 distinct Direct-complete snapshots
  and 268 distinct fallback snapshots (startup smoke included; not final path
  percentages). Direct starts and completions continue after cancellation
  bursts, unlike r1's loss of new starts after its first seconds.
- Cancellation logs show all-rank rollback completion and subsequent new
  Direct transfers. Setup timeouts still occur; this is not a claim that the
  timeout rate or long-run performance is acceptable yet.
- Full500 remains running. Slow configuration is unchanged. No final grader,
  throughput or all-snapshot correctness result is available at this point.
