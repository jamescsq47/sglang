# Remote Host data plane — engine-integrated, hardware acceptance pending

`agentic_remote_host.py` is an isolated NIXL adapter for:

```text
source GPU HBM -> source-node CPU Arena -> RDMA READ -> target GPU HBM
                 existing local D2H       this module
```

It is wired through `agentic_remote_host_engine.py` into
`AgenticPHostStagingManager`, `AgenticDHostStagingClient`, and
`AgenticPToDHostLoadManager` behind the multi-node opt-in. Existing same-node
paths remain unchanged. CPU
tests verify API shapes/state handling against a fake NIXL agent, not RDMA hardware
correctness, throughput, GPU visibility, or actual TP=8 inference.

## Small adapter API

1. Use an I/O-worker-owned `RemoteHostTransport(nixl_agent)` with UCX backend.
   If sharing an agent, every caller must use its same lifecycle lock. Do not do
   NIXL memory registration, metadata loading or connection work on Forward.
2. The source finishes **local D2H**, publishes Host durable across every TP shard,
   and pins the Arena extent in the authoritative snapshot lifecycle. Source HBM
   can then be released by the existing lifecycle. `export(...)` registers the
   source DRAM address with NIXL and retains a Python mapping reference. This
   reference is NOT an Arena eviction pin: the lifecycle pin is also mandatory.
3. Rank0 creates one recovery `read_id`. Source `claim(read_id)` yields a JSON
   `HostShard`: snapshot ID, random export epoch, TP rank/size, token count, layout
   fingerprint, registered address/size, and partial NIXL connection metadata.
   Metadata travels over a trusted control backend (the experimental shared-POSIX
   bridge, or a future authenticated RPC service). KV bytes never travel through
   that metadata channel; a source memfd path is not a remote memory descriptor.
4. Target first owns a complete parent+suffix workset. Its VRAM pool is already
   NIXL-registered. `prepare_read(...)` checks shard rank, TP size, agreed layout
   and complete source byte coverage, then creates a NIXL `READ` with **local
   VRAM descriptors first, remote DRAM descriptors second**. `start()` and `poll()`
   only run on the I/O worker. GPU target pages cannot be reused while pending.
5. `poll()` returns a receipt only after NIXL reports `DONE` and handle release
   succeeds. Rank0 gathers every rank's receipt using `group_ack()`; partial
   groups, mixed generations, mismatched token counts/layout, duplicate ranks,
   stale export epochs and mixed recovery attempts are rejected.
6. The existing lifecycle accepts/binds all target worksets and commits target
   ownership. Only then may it issue `committed_read_id` and call each source's
   `release_after_group_ack(..., committed_read_id=...)`. Successful deregistration
   permits the caller to remove the pin and release its Arena extent. A mere DMA
   receipt is not this application ownership commit.

`layout_fingerprint()` hashes a canonical caller-provided schema. Derive this from
the actual model pools (dtype, layer/head dimensions, page size, model revision,
hybrid state layout), not a model name alone. The helper `mha_read_spans()` maps the
existing `[K/V, layer, token, head, dim]` Host layout into coalesced runs of paged
GPU token slots. It covers **MHA attention KV only**. A Mamba/other hybrid adapter
must also describe and transfer every recurrent/conv state component; do not
silently use the MHA helper on hybrid snapshots.

## Failure and cancellation

- `transfer()` can submit then throw. The handle is stored before calling it; an
  exception yields `UNKNOWN`, not permission to free either buffer.
- NIXL `ERR` and wall-clock timeout are not fences. `drain_failure()` calls
  `release_xfer_handle()`; installed NIXL 1.3.2 specifies that an active handle is
  cancelled, or cancellation raises and the handle remains allocated. Until that
  call succeeds, destination pages and source registration stay pinned.
- Rank0 gathers all successful/drained receipts for a cancelled attempt, then
  invokes `unclaim_after_group_cancel()`. This retains Host data and permits a new
  recovery attempt. A partially successful TP group is never committed.
- If metadata publication fails and deregistration also fails, the transport
  quarantines the registration and mapping. Unknown/partial registration cannot
  be released automatically. Explicit cleanup retries only known unpublished
  registrations; process teardown is the final boundary for unknown ones.
- No expiry, best-effort source release, automatic recompute or crash recovery is
  introduced. Lost ACKs retain source storage; release is idempotent. The control
  service must fence old reader processes before replacing a recovery epoch.
- The transport owns a bounded READ registry (default 128 entries) so dropping a
  caller reference cannot lose a PREPARED/PROC/UNKNOWN native handle. Repeating
  `prepare_read()` for one export/read epoch returns the same object only when
  source and destination descriptors match. Fenced completions remain as replay
  guards until the control plane retires the attempt and calls `retire_read()`;
  that call refuses any unfenced handle. Registry capacity failure is explicit,
  never an eviction of active I/O. Caller still owns/pins the destination workset.

## Engine attachment points

| Path | Existing local attachment | Implemented remote adapter |
|---|---|---|
| P→D write | `p2d_host_staging.AgenticPToDHostStagingManager.try_submit()` | Export already source-local durable extent; publish per-rank NIXL grant. |
| P→D read | `AgenticPToDHostLoadManager._worker()` | Network grant selects `RemoteHostRead`, not `SharedMHAHostSnapshot(path=...)`; preserve group completion/bind. |
| D→P write | `AgenticDHostStagingClient` | Allocate source-D-local Arena instead of mmap of a P-owned extent, then reuse existing D2H worker and durable fences. |
| D→P read | `AgenticPHostStagingManager._import_remote_host_record()` | Create remote descriptor instead of opening another machine's `/proc/PID/fd/FD`; use the same workset admission/completion protocol. |

`/proc/PID/fd/FD` and mmap remain node-local constructs; local inotify does not
discover remote writes. The V1 shared metadata bridge requires verified
cross-client filesystem locks/CAS visibility and bounded polling; a future RPC
backend can replace this bridge. Neither makes process memory paths globally
addressable. Equal P/D TP size and one full TP group per
node are the initial intended topology; shard conversion/TP resharding is not
implemented. For TP=8, rank i reads the matching source rank i's complete shard.

## Local validation

`test_agentic_remote_host.py`: 23 CPU-only tests covering TP8 all-shard commit,
partial failure cancellation, layout/token mismatch, mixed attempt IDs, malformed
spans, ambiguous post, error-not-fence, duplicate/stale ACK, metadata-registration
cleanup quarantine, successful idempotent release, paged MHA run coalescing,
READ-handle retention across caller GC, duplicate-read suppression and bounded
registry retirement after a physical fence.

No GPU experiment was run. See `check_multinode_cpu.sh` for the current regression
suite, including engine attachment and lifecycle fault tests. The launcher gate
checks integrated capabilities, not hardware readiness. Remote smoke, pressure,
cancellation and TP8 tests remain required before reporting inference results.
