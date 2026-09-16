# Cross-node agentic PD: implementation boundary and ownership plan

This document records the original integration plan in `dualpd/sglang` and
`dualpd/slime`. The attachment points below are now implemented; wording in the
plan describes their intended invariants, not remaining unimplemented code.
It does **not** report a completed remote inference experiment. Current launch
instructions and limitations are in `MULTINODE.md` and Slime's matching guide.

## Deployment contract

- A logical P or D TP group is wholly inside one physical host. Match TP sizes
  across the two groups; TP=8 means eight P GPUs on one node and eight D GPUs on
  another, not one TP collective split between P and D.
- TP rank0 decides the logical route/attempt once; rank i reads source shard i.
  All ranks must agree on model/layout, aligned parent tokens and attempt ID.
- Direct reuses stock NIXL GPU-to-GPU transport. Cross-host UCX/GPUDirect support
  must be verified on the user's machines, not inferred from NCCL logs.
- Slow is source GPU -> **source-node DRAM** -> remote target GPU, both for
  D→P and P→D. A successful source D2H fence frees source HBM while Host owns KV.
  DRAM must remain registered, pinned against eviction, and process-alive until
  the entire consumer group commits or cancellation is physically fenced.
- Full P workset admission remains parent KV + new prompt suffix. This workset
  rule, the late-binding D router, and the two independent I/O queues do not
  change because of the network transport.
- V1 uses an explicitly validated shared POSIX directory **only for metadata**.
  This is a prototype bridge, not a low-latency distributed control service.
  O_EXCL, hard-link publication, flock, rename and cross-client read visibility
  need remote validation. Remote writes cannot rely on local inotify events.
  All clocks must be synchronized: existing marker age/deadline logic uses
  wall-clock timestamps exchanged across machines.
- Actual KV must never be placed on NFS, and `/proc/PID/fd` addresses must never
  be interpreted on a different node. A PID and NUMA ID are node-local, not
  cluster-wide identities.

## What is implemented in this first development stage

1. `agentic_multinode.py`: explicit run/node/engine identity, matching TP
   configuration and unchanged default single-node mode. Capability reporting
   separates completed code integration from unverified hardware execution.
2. `agentic_early_claim.py`: opt-in bounded shared-control polling in existing
   watcher call sites. No new polling in Forward; existing background consumers
   request authoritative metadata refresh. Default inotify behavior is retained.
3. `agentic_remote_host.py`: source DRAM export, remote NIXL READ into registered
   VRAM, descriptor validation, complete-shard coverage, TP-group receipt checks,
   and fence-safe failure/cancellation. This is a transport library; its caller
   must supply real Arena pins, workset leases and authoritative lifecycle CAS.
4. Slime `tools/dualpd/multinode.sh`: remote configuration/planning and metadata
   preflight entry. A hardware-independent plan is not a runnable full method.

## Engine integration contract (now attached through opt-in adapters)

### D→P source-local Host allocation

Today `AgenticDHostStagingClient` obtains a grant from a P-managed arena and
opens that arena's process-local path. Multi-node mode must instead request a
source D-local extent, publish its owner node/engine/rank, and reuse the existing
source D2H pipeline. Only all-rank durable completion may publish HOST_READY
and permit the existing D source release. Keep local mode untouched.

### Recovery of remote Host snapshots

`AgenticPHostStagingManager._import_remote_host_record` and the P→D receiver
must select local mmap or a remote descriptor by explicit owner-node identity.
For the remote case, pin Host and allocate the complete destination workset
under the same request-generation lifecycle before posting NIXL READ.
P→D additionally retains its existing auxiliary/first-token metadata semantics.
Attention + Mamba models must include all state at the same temporal checkpoint;
transferring only the attention tensor is not a valid snapshot.

The received transport completion is not by itself a successful engine handoff:
the destination must perform the existing safe allocator/Radix bind and TP
commit. Only then may the source release the exported DRAM extent. If one rank
fails, retain all source shards, drain every submitted receiver handle, roll back
destination ownership together, and retry or enter the existing explicit failure
semantics. Do not free by elapsed time, missing heartbeat or process PID alone.

### Source lifetime and cleanup

An exporter must retain the mapping and registration even if a control ACK is
lost. ACK retry must use the same run/snapshot/export/attempt identity and be
idempotent. Restarting a source process invalidates exported addresses; it is
not a transparent retry. Unknown DMA completion requires fail-closed quarantine,
not immediate arena reuse. Source-side registration and metadata preparation
belong on independent background workers, never scheduler/Forward.

## Eight invariant checkpoints

| Invariant | Required cross-node proof |
|---|---|
| Unique owner | Existing snapshot CAS plus node-qualified extent/export and attempt IDs |
| P→D Direct release | Reuse existing NIXL terminal fence and all-rank receiver commit |
| P→D Host release | All source D2H shards durable; no dependency on D capacity |
| D→P Host release | All source D2H shards durable; no dependency on tool/P readiness |
| Independent progress | Separate I/O workers; no network call in Forward/allocator hot path |
| TP atomicity | Identical group attempt; no release on partial/duplicate/stale shard receipts |
| Parent correctness | Complete aligned shard + hybrid state, not just byte-count completion |
| Audit gate | CPU fault tests then independent audit, then user-run GPU correctness tests |

The current CPU tests exercise the new boundaries, not these complete
end-to-end proofs. GPU/UCX tests, throughput measurements, and the final
300-second warmup + 1200-second measurement remain remote acceptance work.
