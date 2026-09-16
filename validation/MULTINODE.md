# Experimental multi-node Agentic PD

Status: engine attachment implemented; remote GPU/RDMA acceptance is pending.
Configuration validation and CPU tests do **not** certify end-to-end hardware
correctness or performance. Capability reporting distinguishes code integration
from hardware verification. Follow Slime `tools/dualpd/MULTINODE.md` for the
user-run sequence; do not start with a full throughput benchmark.

## Scope and ownership

- One complete TP group stays within one host. P and D have matching TP sizes
  (1, 2, 4 or 8), DP=1, PP=1, with the same model, dtype, page and shard layout.
  TP=8 means eight P GPUs on a P host and eight D GPUs on a D host, not eight
  GPUs split between the two roles. A TP group spanning hosts is outside V1.
- Direct reuses existing NIXL GPU-to-GPU transport. NIXL network capability
  does not prove the selected UCX backend uses GDR on the remote hardware.
- Slow's intended path is source GPU → source-local CPU DRAM → remote GPU.
  Shared filesystem access must never be used to transfer KV bytes.
- Source-local D2H completion makes Host durable and permits releasing source
  HBM. The Host extent remains pinned during remote reads and is released only
  after all TP shard fences and the receiving ownership commit.
- A timeout never substitutes for a physical DMA fence. Partial TP transfer,
  failed launch, lost ACK and cancellation must retain the old owner or reach
  explicit fail-closed/recompute state. Never independently reroute a TP shard.

## Configuration interface

`python/sglang/srt/disaggregation/agentic_multinode.py` is a dependency-light,
explicit validator, not an import-time engine patch. It does not set variables
or create directories. Server argument validation and scheduler initialization
call its runtime guards. Native HiCache/offload, hybrid pools and unsupported
layouts fail explicitly. The launcher currently supports one logical P group
and one or more D groups, with dense Qwen3 models only.

Example P environment (D uses another NODE_ID/ENGINE_ID/HOST_IP and ROLE=decode):

```bash
export SGLANG_AGENTIC_MULTINODE_ENABLED=1
export SGLANG_AGENTIC_MULTINODE_RUN_ID=trial-001
export SGLANG_AGENTIC_MULTINODE_NODE_ID=node-p
export SGLANG_AGENTIC_MULTINODE_ENGINE_ID=engine-p
export SGLANG_AGENTIC_MULTINODE_ROLE=prefill
export SGLANG_AGENTIC_MULTINODE_HOST_IP=10.20.1.2
export SGLANG_AGENTIC_MULTINODE_TP_SIZE=8
export SGLANG_AGENTIC_MULTINODE_PEER_TP_SIZE=8
export SGLANG_AGENTIC_MULTINODE_CONTROL_ROOT=/shared/dualpd-control
export SGLANG_HOST_IP=10.20.1.2
export SGLANG_AGENTIC_KV_ENGINE_ID=engine-p
export SGLANG_AGENTIC_KV_TP_SIZE=8
export SGLANG_PD_P_READY_DIR=/shared/dualpd-control/trial-001
python python/sglang/srt/disaggregation/agentic_multinode.py --check-env
```

This command checks configuration only. All peers and Router use the same run
namespace; engine identities are unique across the cluster and identical within
one TP group. Local GPU ordinal, hostname shared by containers, and PID are not
cluster identities. DP_SIZE/PP_SIZE/ENGINE_NNODES default to 1. Actual engine
arguments must still be checked separately. PREFILL_DOMAIN must consistently
identify a logical P group; it need not equal a physical node ID.

Use a routable IPv4 address, not 0.0.0.0 as the advertised address. Listener
`--host 0.0.0.0` is different from advertised `SGLANG_HOST_IP`. Permit HTTP
forward/reverse bootstrap ports and dynamically allocated ZMQ ports between
peers. Device/rail selection is a deployment setting, not inferred from IP.
Reverse bootstrap IPv6 formatting is not audited, so this V1 rejects IPv6.

The existing single-node default is unchanged when ENABLED is absent/false.
Opt-in P_HOST_ASYNC_PREPARE, P_HOST_EVENT_PROGRESS, NUMA_HOST_POOL and
P2D_PREBIND ablation are outside this remote scope and rejected. Generic Direct
and Slow I/O queues must remain independent; no new priorities are introduced.

## Shared POSIX control bridge: requirements, not a KV store

The smallest compatibility bridge retains the existing small control files on
an explicitly shared POSIX filesystem. This is an experiment bridge, not a
claim that NFS metadata latency is sufficient for production. A path string
outside `/dev/shm` is not proof of shared visibility or distributed locking.

Required semantics across hosts: exclusive create (O_EXCL), atomic rename,
mutually exclusive flock, coherent reads after committed writes, and bounded
visibility. P/D lifecycle claims, manifests, P-ready, Direct arrival/ACK and
cross-role TP receipts must share the same namespace. Some TP reports are local,
but others deliberately exchange P↔D receipts; do not split these accidentally.

Synchronize P/D/Router wall clocks before testing and record measured skew.
Existing arrival markers use wall-clock age for expiry; an unsynchronized clock
can falsely age out a valid Direct request. Monotonic clocks remain process-local
and cannot be directly compared across machines.

Remote mutations generally do not trigger local inotify. Opt-in control polling
has a default interval of 0.1 seconds, configurable with
`SGLANG_AGENTIC_MULTINODE_CONTROL_POLL_INTERVAL` in [0.05, 1.0]. This only supplies
the watcher interval; it does not create a remote metadata service. Polling and
potentially slow filesystem work must not run on Forward submission threads.

### Two-host acceptance procedure before a GPU run

Run a bounded probe using the **same two containers/users/mounts and exact
control root** used by P and D, not only two shells on one host:

1. Coordinator creates a fresh unpredictable probe directory under CONTROL_ROOT
   and writes a nonce. The other host must read the same bytes and write its own
   nonce; coordinator verifies that nonce. Record identities and timestamps.
2. Release two participants simultaneously to O_CREAT|O_EXCL the same claim.
   Exactly one may succeed. Both must read that winner's complete identity.
   Repeat with different filenames and swapped initiation order.
3. A holds flock(LOCK_EX) on a stable inode. B's LOCK_EX|LOCK_NB must fail while
   A holds it, then succeed after A explicitly releases it. Repeat swapping A/B.
   Never delete and recreate the locked inode during this test.
4. Writer alternates distinct JSON payloads using write+flush+atomic rename.
   Peer reads in a bounded polling loop: every observed file must be a complete
   old or new payload, never truncated/mixed; final version must become visible.
5. Record visibility latency distribution; fail on missing visibility or any
   duplicate winner/lock violation. Stop probe participants via their exact PIDs
   and remove only the probe directory after both report stopped.

Slime's `tools/dualpd/multinode.sh` supplies `fs-publish`, `fs-verify`,
`fs-lock-hold` and `fs-lock-probe` for initial visibility/exclusive-create and
cross-host lock checks. Follow its README on the two actual machines. These
bounded probes do not yet implement the repeated simultaneous-create/rename
stress or measure a visibility latency distribution described above. They have
not been run remotely. A successful single-host unit test cannot certify remote
flock/NFS behavior. If semantics fail, do not launch the experiment; use an
authoritative transactional control service.

## Remaining runtime acceptance

The attached adapters use source-local Host descriptors, the existing ledger
pin/workset ownership, paged GPU scatter mapping and all-rank completion.
Independent CPU review is not remote acceptance. Verify cross-host filesystem
semantics, Direct and Host recovery, cancellation and source lifetime on real
machines. GPU correctness requires comparing restored KV or deterministic
outputs, not only aggregate throughput. Performance remains a separate
300-second warmup + 1200-second measurement acceptance gate.
