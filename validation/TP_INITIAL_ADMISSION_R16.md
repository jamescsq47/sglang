# R16: independent ordinary admission alongside TP restore progress

R15 diagnosis: at 01:59 UTC, 78/128 agents were still in their initial generation,
23 for over 30 minutes. P used about 10% Attention KV, while a continuously active
Host recovery pipeline selected only parent-bearing requests in the scheduler's
forced-TP branch. Initial requests never reached native Prefill admission.

Scope: TP>1 Prefill metadata admission only. No DMA, ownership, workset, routing,
timeouts, model, or memory-policy changes. TP1 retains its previous path.

Affected transition: an initial request owns only metadata, then enters native
Prefill admission and obtains its normal P HBM allocation. TP0 selects at most
the existing admission-batch limit of no-parent wire request IDs on its current
queue and carries them on the existing request/control broadcast. Restore
commands continue in their previous order with unchanged physical lane limits;
they no longer exclude the independently bounded ordinary list. Rank-local
async gate-complete flags do not qualify a parent for this list. An absent,
cancelled, already-consumed or replaced attempt is not recreated.

Capacity failure still waits in native Prefill admission. Timeout, cancellation,
shutdown and every in-flight parent transfer retain their existing handlers.
No ownership is transferred or released by this metadata selection. Native
compute limits still apply after admission; no additional I/O or collective.

Acceptance checks: (1) unique ownership unchanged; (2,3) P Direct/Host release
unchanged; (4) D Host release unchanged; (5) restore control no longer gates
ordinary compute admission; (6) one leader-authored ordered request list, no
rank-local selection; (7) all parent fences, pins and reuse semantics unchanged;
(8) CPU regression and independent audit must pass before GPU restart.

Tests cover continuous restore commands, exhausted restore admission budget,
missing commanded parent, follower queue-order skew, cancelled/retried wire IDs,
TP2/TP8 and TP1 unchanged behavior. GPU rerun uses the same r15 c128 settings,
including SWE congestion recompute disabled and Mamba memory ratio 0.5.

Validation before launch: full CPU gate 842 passed / 2 skipped plus 62 launcher
tests. Independent reviewer reproduced that full gate and an additional 44
targeted tests, checked the eight invariants, and returned GO. `git diff --check`
passed. R15 run-owned supervisors stopped P/D/workload; both nodes have no
remaining GPU compute processes, and the run-labelled Docker containers exited.
The already-finished smoke supervisor's refused stop connection is not a live
process. GPU performance/long-run correctness remains to be measured in R16.

Reproduce on a10:
```bash
bash /homes/siqic/dualpd/slime/tools/dualpd/qwen35_multinode.sh run \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r16 \
  --concurrency 128
```

R16 launched after GO; normalized configuration equals R15 (only run ID/output
paths differ). Both engines completed all eight prewarm reports. Direct and
Slow two-turn correctness smoke passed before SWE500 began at 02:19:55 UTC.

Early regression observation at 02:23 UTC: all initial 128 agents completed
their first P->D handoff and submitted a later generation; no initial request
remained. Router initial submission -> P release was mean 10.17s, p90 15s,
max 16s (log timestamps have one-second resolution). This verifies early
ordinary-admission progress, not final performance or completed SWE accuracy.
Two short metric windows had P Forward 65.0%/63.9%, D 93.9%/88.6%; D running
remained low, so residual recovery/throughput limits still need observation.
No TP/Router exception found in these early logs. Full 500-instance run is
still in progress; do not label this as final throughput acceptance.

Final status: R16 was deliberately stopped through run-owned supervisors before
500-instance completion to repair Host recovery control. The coordinator's
`SWE evaluation failed` record is the expected interrupted workload result,
not a new spontaneous engine crash. Both nodes' GPU workers exited cleanly.

At 02:27:30 UTC 105/128 active agents had durable Host KV but were not selected
for P restore; only 3 were preparing/loading and 1 waiting for bind. Over
02:30–02:35, all-rank selected→H2D ACK averaged 1.908s and selected→commit
2.772s, completing ~1.11 logical restores/s. Nonblocking CPU samples implicated
repeated full NFS Host ledger scans on the same thread advancing I/O. The
8-rank physical lanes also remained occupied through scheduler bind/commit.
This motivates R17; these short diagnostic windows are not formal results.

One Direct start raced a release-pending workset. It safely fell back and all
8 Host shards were subsequently restored/committed; do not claim zero boundary
errors or complete evaluation accuracy for R16.
