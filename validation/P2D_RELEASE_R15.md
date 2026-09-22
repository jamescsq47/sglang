# R15: background P release authorization and D ready-marker maintenance

Scope: multi-node TP>1 only; TP1 and single-node behavior unchanged.
No routing, capacity, recompute, workset or DMA policy changes.

Before editing: the affected handoff is P_HBM_OWNED -> D_HBM_OWNED or
P2D_HOST_OWNED. Physical success remains destination receipt or complete Host
durability, never submission alone. Existing Host/native CAS remains necessary
to reject a late Host offer; perform it in the transfer consumer, cache its
result once, then report a separate all-rank release-ready barrier. Scheduler
only consumes that barrier and performs local allocator/Radix release. Failure
uses existing cancel/fence semantics; timeout, insufficient capacity, NFS error
and shutdown keep ownership until proven safe. No new retry-to-recompute rule.
Authorized transfer outcome is immutable even when cancel marks Host terminal.
Before authorization, an already-winning Host copy may still complete after a
native failure (the existing behavior); a failed physical attempt alone is not
permission to discard Host ownership.

Producer watch registration becomes an in-memory queue in this scope so it
cannot wait on the manager lock held by background ledger arbitration. The
existing offer worker discards queued registrations after terminal authorization.

D ready discovery uses its existing background cache. Successful admission
enqueues ACK cleanup; only background rank0 unlinks after every rank ACK.
Partial file errors retry without recreating already-written ACKs. Metadata
submission and GPU allocation retain their existing TP scheduling boundary.

Eight acceptance checks (design): (1) original ownership CAS retained;
(2,3) release after Direct completion/Host durable, no waiting for D after Host;
(4) D->P unchanged; (5) these ledger/marker operations leave scheduler;
(6) separate all-rank release-ready barrier plus native TP broadcast;
(7) no KV reuse/recompute changes; (8) fault tests and independent audit before GPU.

Validation: final full CPU gate 831 passed / 2 skipped and 62 launcher tests
(`/tmp/dualpd-r15-cpu-gate-final.log`). Independent reviewer reproduced the
preceding full gate (828 plus 62) and returned GO. Extended targeted R15
tests: 14 passed, including TP2/TP8 release barriers, real consumer exception
followed by peer failure convergence, partial marker cleanup failure/retry,
Host-winning late arbitration and producer registration after cancellation.
The review found one failure exit missing the physical terminal report; fixed
by publishing it before release authorization, never after release-ready.

No GPU performance claim. Existing a10 PID 1754686 still retains a thread in
NFS kernel wait (`nfs_lookup_revalidate`) with SIGKILL already pending, so no
GPU experiment was restarted. Launcher/parameters unchanged.

Reproduce CPU gate:
```bash
DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python \
  bash validation/check_multinode_cpu.sh
```
