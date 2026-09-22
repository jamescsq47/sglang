# r13: local TP reports; cross-node ownership unchanged

## Evidence / scope

The r12b P TP0 scheduler waited in `publish_local` on the mailbox cache lock.
Its P→D progress thread held that lock in `Path.stat` on NFS
(`nfs_lookup_revalidate`). Other P ranks waited in the native model broadcast.
The watchdog terminated the run. This is a control-plane stall, not HBM
exhaustion or proof of inadequate RDMA bandwidth.

Before the final stall, 313 complete Slow restores had about 30.4 seconds from
all-rank D2H completion to P selection, versus 2.73 seconds from selection to
group release. These log timestamps have one-second resolution. The four
logical restore slots include preparation, H2D, bind, commit and clear; local
reports reduce control waiting in that chain, without increasing physical
workset/lane limits. Remaining shared-ledger prepare/commit costs are not
claimed to be eliminated by this change.

## Ownership and error behavior

No physical state transition changes. Host stays owner until all-rank H2D,
bind and existing ledger commit; Direct/source HBM stays pinned until its
existing physical fence and all-rank confirmation. Only the location of
intra-engine TP report files changes, selected explicitly by the multi-node
launcher. TP1 and launchers without the override retain previous behavior.

Local allowlist: `d2p-direct`, `d2p-direct-abort-p`, `d2p-host:<domain>`,
`p-workset-retire`, `p2d-sender`, `p2d-cleanup`, `p2d-admission`.
The previous D-only abort override remains as-is.
`p2d-receiver` MUST remain shared: D publishes the logical receipt, P observes
it and performs terminal cleanup. Unknown namespaces stay shared. Direct
arrival/abort messages, Host grants, source-release acknowledgments and
lifecycle ledgers are still cross-node shared records.

- Success: same all-rank reduction and same physical release fence.
- Failure / timeout: same negative/terminal status and existing abort protocol;
  no fake success or forced release is introduced.
- Capacity failure: unchanged full-workset allocator and Host pin rules.
- Cancellation: same rollback and all-rank cleanup; group-local reports are
  engine-scoped, so an old P's cancellation cannot clear the new P's reports.
- Shutdown: no directory deletion or active-page reclaim is added. Each run
  uses a distinct directory; do not reuse an active run or clear its reports.
- A TP group spanning multiple hosts rejects the local override (`nnodes != 1`).

## Eight-criterion review

1. Unique ownership: unchanged ledger/CAS; no extra owner/cache introduced.
2. P→D Direct release: unchanged, destination receipt stays shared.
3. P→D Host release: unchanged durable fence and sender reduction.
4. D→P Host source release: unchanged complete-shard durable commit.
5. Progress: removes NFS from intra-engine reports, including the demonstrated
   shared-lock stall. Cross-node ledgers can still stall and need measurement.
6. TP atomicity: same rank0 decisions, all-rank minimum and failure ordering.
7. Reuse: no data layout, page alignment, eviction or recompute changes.
8. Gate: new mailbox scope/failure/cleanup tests plus full CPU lifecycle gate
   and independent audit are required before GPU rerun.

## Next experiment

Qwen3.5-122B-A10B, a10=P/a11=D, each TP8/EP1, static memory .8,
Mamba/attention ratio .5, congestion recompute disabled, SWE Verified 500,
openai_tools, sampling .6/.95/20. Only workload concurrency changes to 128;
Host startup prewarm and Direct/Slow correctness smoke precede workload.

```bash
cd /homes/siqic/dualpd/slime
bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r13
```

GPU ownership checks must pass; a defunct prior scheduler retaining GPU memory
is not an allowed co-tenant and must not be bypassed.

Gate result: 805 passed / 2 skipped, 62 launcher tests passed; independent
review reran 18 relevant tests and returned code GO. GPU launch is currently
NO-GO: r12b PID1754686/TID1759850 remains in kernel NFS wait with SIGKILL
pending and retains about65GiB on a10 GPU0. No c128 result exists yet.
