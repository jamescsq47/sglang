# R23 stalled handoff: lifetime-based manifest retirement

## Evidence and root cause

R23 is not an accepted performance result. The TP8 Direct and Slow smoke passed
with 8192/8192 reusable parent tokens and exact output equality, but the c128
SWE500 run stalled under load. P and D Forward became intermittent while their
attention KV occupancy remained low. The P log recorded 360 instances of
`Slow handoff lost lifecycle ownership` for
`3672f82102ea428fa37b51659966cc5c:0` before shutdown.

The existing protocol marks a Host ledger entry CONSUMED once every rank has
loaded and Radix-bound its shard. This permits the source to release Host data.
It does not mean every destination scheduler has accepted its workset: that
happens at the subsequent TP COMMIT/handoff. The old prune rule removed a
CONSUMED entry after five seconds and source release without checking remaining
destination recovery claims. NFS delays lengthened that interval. The exact
snapshot manifest was absent, and later rank handoffs could never validate their
claim against the deleted entry. Extending the TTL or accepting missing entries
as success would hide the race rather than repair it.

R23 was stopped through run-owned supervisors; both a10 and a11 subsequently
reported no GPU compute processes. Raw logs and control records remain intact.
The root coordinator received duplicate termination while shutting down; remaining
live supervisors were explicitly contacted through their authenticated run-scoped
control sockets. No PID-pattern kill, GPU reset or unrelated process cleanup was
used. Future shutdown should send one termination and verify owned cleanup.

## Ownership plan

This change affects the final D2P_HOST_OWNED → P_HBM_OWNED handoff metadata.
The existing all-rank physical completion, Radix binding and TP COMMIT remain
authoritative; the source Host/KV release boundary is not delayed or weakened.
Only the small manifest must outlive all of its destination consumers.

- CONSUMED with recovery claims is retained until every expected rank has the
  matching claim, an attached lease, and the final handed acknowledgment.
- Apply the same predicate to the initial prune candidates and under the
  per-entry lock, including relay entries. Time alone never authorizes deletion.
- Make handed the last ledger-dependent operation, after retryable Host
  cleanup. A failed cleanup retains the request/load carrier and the unhanded
  claim; it cannot be pruned. Retry remains idempotent.
- Keep legacy entries without recovery claims and explicit terminal cleanup
  behavior, source-release fencing, cancellation and failed-transfer handling.
- Do not add a new timeout, thread, phase, admission cap, recompute policy or
  change TP1 scheduling/transport policy.

## Acceptance review

1. Unique ownership: unchanged physical bind/COMMIT; metadata retained until used.
2. P→D Direct release: unchanged.
3. P→D Host release: unchanged.
4. D→P durable/source release: unchanged; extra retention is metadata only.
5. Progress: no added polling or barriers; remove the permanent missing-record retry.
6. TP: require the full expected rank set; no rank-local route decisions.
7. Reuse: no new recompute, missing-record success or relaxed DMA fence.
8. Required gate: delayed-handoff/prune races, cleanup retries, TP1/2/8 and
   cancellation regressions, followed by independent review before GPU rerun.

## Remaining performance limitation

R23 also had scheduler/control threads waiting in NFS file operations. An observed
Host recovery took 94–177 seconds end to end; those figures are not network DMA
time. This lifecycle repair removes a permanent failure amplified by that delay;
it does not establish that all filesystem/control latency has been eliminated.
Do not report startup, smoke correctness, or process liveness as a sustained
performance pass. Validation results will be recorded after execution.
