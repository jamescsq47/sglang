# P→D control-plane audit / r14 plan

Scope: the multi-node TP asynchronous transfer path. No changes to DMA,
routing, Host capacity, lane count, workset allocation or TP1 policy.

Confirmed risks:

- P scheduler repeats destination receipt reads and Host `group_claimed`
  reads even when its background sender already owns those checks.
- D background polling publishes rank reports on NFS while holding the
  receiver lifecycle lock also needed by the Decode scheduler.
- D scheduler reduces those NFS reports and publishes destination receipts.

Plan: keep receiver lookup/poll/teardown protected by the existing lifecycle
lock. Move multi-node TP rank-report publication, group reduction and receipt
publication to the existing background progress worker AFTER leaving that
lock. Expose a local immutable `(status, cancel_requested)` result only after
the existing destination receipt has been persisted. The scheduler broadcasts
that cached result through the same native TP control message. P background
mode uses local sender reports, not redundant remote confirmation reads.

Ownership mapping / failure contract:

- Direct remains `P_HBM_OWNED` until actual receiver fences and all-rank
  completion; a missing/failed publication yields no terminal cached result.
- Host remains `P2D_HOST_OWNED` until the existing all-rank loader commit;
  the sender's durable release condition is unchanged.
- Failure is terminal only after every receiver physically finishes or aborts;
  mixed failure + in-flight remains Transferring with cancellation requested.
- Cancellation still follows the existing group command and DMA drain.
- A blocked receipt delays that transfer but does not hold the receiver lock.
  No new timeout or fabricated completion is introduced.
- On process shutdown pending publication cannot release the source; existing
  fail-closed ownership/cleanup applies. No extra worker is introduced.

Eight criteria: ownership and all three source release fences unchanged;
control I/O removed from the targeted scheduler/lock paths; same rank0 native
broadcast and full-shard reductions; no recompute/data layout changes;
fault tests and independent GO required. This does not claim elimination of
all NFS paths (Host ownership CAS and readiness/admission still need audit).

## Host preparation / abort boundary

Also confirmed: D `submit` performed ledger validation/claim while holding
the receiver state lock, and `abort` synchronously published ledger failure
under the Decode queue's lifetime lock. Multi-node TP>1 now queues immutable
destination indices and their CUDA producer event; the existing load worker
calls the factored, unchanged grant validation/claim sequence. Queued means
submitted-but-NOT-drained. Cancellation merely records local intent; the
existing completion worker publishes the shared abort intent. Only the load
worker/no-I/O proof or physical DMA fence can acknowledge target-page drain.
Exceptions retain the original failure/quarantine path. TP1 and non-remote
loaders keep synchronous preparation, using the same validation helper.

Metadata setup can fail AFTER Host bind has queued H2D (for example unlinking
the P-ready marker). Its error handler must not free target pages immediately
after `abort()`: asynchronous abort is only intent. For these remote TP Host
receivers, retain the metadata/indices and enter the normal transfer queue to
wait for the existing all-shard physical failure/drain boundary.

The success path already does Host completion/remote release ACK in background
workers, without holding the scheduler's receiver lock; it remains unchanged.

## Known remaining boundary (not silently changed)

P Host offer/claim and final `prepare_scheduler_release` arbitration still
share the staging manager lock and a ledger CAS. Replacing that operation with
per-rank asynchronous authorization alone would allow TP ranks to remove their
inflight request on different scheduler iterations. A later change must fold
release-ready into the all-rank terminal decision, including Host-won and
failed-native races. This round deliberately does not introduce that protocol
change or claim every P→D control wait is eliminated.

TP8 still consumes P-ready markers synchronously in preallocation. Its Host
`bind` now only queues local work, but subsequent marker operations have not
been moved. The metadata-error retention branch above is defensive: today's
async metadata queue is TP1-only whereas remote async Host control is TP>1;
the test covers that failure interface, not a claim that TP8 metadata setup
has entirely moved to background workers.

## Validation status

- Final CPU regression: **817 passed, 2 skipped**, plus **62 launcher tests**.
- Twelve new P→D control tests cover blocked NFS publication/prepare, terminal
  persistence failure, immutable terminal after remote cleanup, peer DMA not
  yet drained, scheduler cache-only consumption, queued cancellation,
  invalid grant/CAS rejection, and metadata failure after Host bind.
- TP1 lifecycle/fault regressions remain in the full gate.
- Independent audit: final **GO**, including an independent rerun of all
  twelve new control tests. GO covers this increment, not the two remaining
  synchronous P release-CAS / TP8 ready-marker boundaries.
- No GPU run launched: a10 still has the prior NFS-blocked scheduler retaining
  GPU0 memory. This is not a c128 performance result.
