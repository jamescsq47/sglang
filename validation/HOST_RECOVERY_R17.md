# R17: incremental Host discovery and bounded TP recovery overlap

R16 was stopped through its run-owned supervisors for a control-path fix. It is
not a completed 500-instance result. Both GPU process sets exited cleanly.

## Before editing: ownership map

The bottleneck is D2P_HOST_OWNED -> fenced H2D -> P_HBM_OWNED. Incremental
notifications only invalidate cached metadata; they never grant ownership.
Snapshot selection, Host pin, complete workset allocation, per-rank physical
fences, group bind/commit and Host release retain their existing CAS protocol.

Full ledger reconciliation/pruning belongs to the notification worker, not the
physical progress loop. A startup scan plus periodic reconciliation recovers
missed notifications or interrupted publishers. Notification errors must not
cancel or invent a snapshot. TP physical-lane reuse may occur only after its
local DMA/futures are quiescent; logical resident/workset/Host ownership remains
bounded and pinned until the existing all-rank handoff or fenced cancellation.

Capacity failure leaves data in Host. Timeout/cancellation never frees active
DMA memory. Shutdown uses the run-owned supervisors; no PID-name cleanup or GPU
reset. TP rank0 remains the sole group admission decision maker. No workload,
memory fraction, recompute, routing or transfer deadline change is intended.

## Acceptance gate

1. Unique owner: unchanged authoritative snapshot CAS, notifications not grants.
2. P2D Direct source release: unchanged.
3. P2D Host source release: unchanged.
4. D2P Host source release: unchanged durable/all-rank fence.
5. Progress separation: targeted metadata updates and off-thread reconciliation.
6. TP atomicity: group bind/commit retained; no rank-local path selection.
7. Parent reuse: no new fallback/recompute or eviction policy.
8. CPU lifecycle/fault tests and independent GO required before R17 GPU run.

## Implementation and CPU validation

- Multi-node Host manifests publish 65-byte digest notifications to eight
  append journals. Readers tail bounded batches, coalesce snapshot IDs and
  read only affected manifests. No global lifecycle lock was added.
- Shared-control startup/backstop scans and prune run on the ledger watcher;
  the I/O progress worker consumes an ordered queue of snapshots/deltas.
  Local inotify/TP1 default discovery remains unchanged.
- TP>1 may recycle a quiescent physical H2D lane before group bind/commit;
  rank0 counts all-rank-copy-complete contexts separately. Physical occupancy
  remains four; at most eight logical residents can overlap, each retaining
  its original full workset and Host claim. This applies to TP>1 generally,
  not only the two-node launcher. TP1 async/event mode is not enabled for TP8.
- CPU gate: 853 passed, 2 skipped; 62 launcher tests passed. Additional
  authoritative-commit/journal-error injection passed. Independent reviewer
  ran 421 focused tests covering TP, watchers and TP1 event/async compatibility.
  Tests include delayed scan, bounded resident backlog, cancellation, non-quiesced
  copies, stale delta filtering, concurrent notification publishers, partial
  journals and no lock-wait in notification readers.
- Existing per-snapshot CAS/get operations still access the shared control
  store. This change removes historical scans from I/O progress; it does not
  claim all network filesystem latency is eliminated.

GPU performance and complete 500-instance correctness remain unmeasured until
the new run completes.

Independent audit returned GO after the 421-test suite and three extra fault/
event tests. Authorized full-run command (a10=P, a11=D, TP8 on both):

```bash
bash /homes/siqic/dualpd/slime/tools/dualpd/qwen35_multinode.sh run \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r17 \
  --concurrency 128
```

Same R16 model/dataset/source order/memory/sampling settings; congestion
recompute remains disabled. The launcher waits for all-rank Host prewarm and
Direct/Slow correctness smoke before starting the 500-instance evaluator.

## GPU-found missing admission edge; R18 correction

R17 passed all-rank prewarm and Direct/Slow correctness smoke, then started SWE500
at 02:57:30 UTC. At 02:59:13 four Slow intents acquired lanes and full worksets
on all ranks, but had not constructed `loads` yet. `_agentic_io_kind` recognized
such selected intents only for TP1's decoupled mode. After all four lanes were
owned, the scheduler misclassified their continuation as NEW I/O, denied its
own already-granted intents, and no H2D started. KV remained safely pinned but
this is a liveness failure, not bandwidth or capacity exhaustion. R17 stopped
via run-owned supervisors; it is not an accepted throughput/evaluation result.

Correction: include the narrow TP lane-overlap capability in the existing
selected-intent classification. No new admission priority, physical lane or
ownership transition. The added cross-function regression drives the real
`_drain_agentic_kv_waiting_queue` and `_agentic_io_kind` with four prestart
reservations and zero load objects for TP2/TP8, ensuring all four continue.
Re-audit and regression must pass before the R18 rerun.

R18 gate: 856 passed / 2 skipped plus 62 launcher tests. Independent reviewer
ran 82 targeted tests including the actual scheduler-drain regression and
returned GO. Both R17 GPU groups exited cleanly. R18 uses the same launch
command/configuration with the run-directory suffix changed from r17 to r18.

R18 did not enter evaluation: startup `/model_info` returned a transient
`http.client.BadStatusLine`, which escaped the launcher's readiness retry.
Supervisors cleaned both groups; no GPU compute processes remained. Launcher
readiness now treats HTTP protocol exceptions as not-ready, under the same
deadline and process-liveness checks. Two tests cover retry-to-success and
deadline failure; all 64 launcher tests pass. No engine change for the R19
retry, which retains the independently audited R18 admission correction.

R19 reached 16/16 prewarm, but the interactive tool session was interrupted
before smoke/workload launch and its outer coordinator disappeared. Both
run-owned supervisors were still reachable and were explicitly stopped;
no SWE500 results were collected. R20 runs the same audited code/configuration
with its coordinator detached via setsid/nohup and an explicit coordinator log,
so conversation interruption does not terminate experiment orchestration.

## R20 interim observation (2026-09-19 03:24:52 UTC; not final acceptance)

The unchanged audited retry passed all 16 ranks' prewarm and Direct/Slow
correctness smoke. SWE500 started at 03:15:20 UTC, c128. At this checkpoint,
883 logical Host snapshots had been selected and 879 had completed the full
eight-rank handoff. Four outstanding selected snapshots were approximately
one to two seconds old; the R17 prestart-lane deadlock has not recurred.
Three episodes finished with verifier results (two resolved), far too few to
estimate final accuracy. The detached coordinator is still running.

This is liveness improvement, not proof that the performance problem is solved.
At approximately 03:23, 97 HOST_READY snapshots occupied about 98.3 GiB; the
oldest entry's last-update age was about 58 seconds. Those waiting *before*
selection must not be confused with the short selected-to-handoff latency.
An approximately 72-second metric window showed P Forward about 51%, D about
92%, and D output about 450 tokens/s for the logical TP8 group. It is an interim
window, not completed-run throughput. A small nonblocking stack sample still
found main-thread per-snapshot shared-file locks, lifecycle marker operations
and TP synchronization. These remain candidates for measured optimization;
the sample is not a statistically reliable stall-time attribution. No engine
or workload parameters were changed during this run.
