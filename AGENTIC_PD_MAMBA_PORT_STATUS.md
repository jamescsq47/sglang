# Agentic PD Qwen3.5 port status

## Isolation

- Clean runtime clone: `/homes/siqic/anaconda3/envs/pd_mamba_baseline`
- Development worktree: `/homes/siqic/sglang-agentic-mamba`
- Branch: `codex/agentic-pd-mamba-v2`, based on upstream SGLang `v0.5.14`
- The installed `pd_mamba` runtime has not been modified while the existing
  6P:2D experiment is alive.

## Implemented

- Request-generation lifecycle, ownership manifests, TP mailbox, early claim,
  shared-host staging, and P2D staging were imported as independent modules.
- P workset leasing is separated from the old 0.5.10 Scheduler.
- Workset allocation is atomic across attention pages and one Qwen3.5 Mamba
  state slot; allocation failure rolls back both resources.
- Reverse NIXL registration uses SGLang 0.5.14 `setup_state_kv_args`, so the
  wire layout includes Qwen3.5 temporal/conv state.
- Reverse send/receive helpers pass Mamba source/destination indices through
  native NIXL state transfer.
- Request-generation manifests identify the complete component set, state
  byte size, checkpoint token position, TP size, and layout hash.
- A single shared-memory extent layout for attention KV plus Mamba state is
  implemented and CPU-tested.
- The old Decode lifecycle manager is isolated in `agentic_decode_manager.py`
  and imports on 0.5.14; direct candidates now describe/send composite state.
- Scheduler creates a Mamba-aware workset broker and can select the isolated
  Decode manager only when `SGLANG_AGENTIC_KV_LIFECYCLE=1`. Baseline defaults
  remain unchanged.

## Verified

- `test_agentic_kv_lifecycle.py` plus `test_agentic_mamba.py`: 59 passed.
- Independent portions of `test_agentic_tp.py`: 51 tests observed passing
  before the first 20 unported Scheduler/queue integration failures.
- Python compile/import checks pass for the new hybrid snapshot, transfer,
  direct runtime, workset, and Decode manager modules.

## Not yet GO

The branch must not be used for a GPU experiment yet. The following integration
work is incomplete:

1. Port the asynchronous P Direct receive/bind loop into 0.5.14 Scheduler
   boundaries without copying the old Scheduler.
2. Make the shared Host arena create/load one composite extent in production;
   the data structure is tested, but the old manager still instantiates the
   MHA-only extent in its hot path.
3. Include the Mamba state fence in D2H completion, Host-ready publication,
   H2D completion, cancellation, and source release.
4. Port P2D completion/Decode prealloc progress to the new 0.5.14 queues.
5. Run the complete lifecycle/capacity/cancellation/TP/fault suite.
6. Obtain an independent audit GO.
7. Only after GO, install the worktree into `pd_mamba` and run the SWE-bench
   collocate-aligned PD experiment (300 s warmup + 1200 s measurement).

## Required correctness assertions for audit

- A snapshot becomes visible only after attention KV and Mamba state complete
  on every TP rank.
- D releases neither attention pages nor `mamba_pool_idx` before the complete
  Direct/Host fence.
- P releases neither Host state nor the manifest before attention and state
  are bound to the live request-generation.
- A failed/cancelled partial transfer releases or quarantines all components;
  it never exposes an attention-only prefix.
- Restored generation output matches colocated output token-for-token, except
  for at most the native page/chunk tail recomputation.
