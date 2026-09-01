# Agentic PD experiment log

This file is the durable index for the `codex/agentic-pd` development branch.
Every GPU experiment must append one entry and be committed and pushed even
when the run is stopped or fails. Raw artifacts remain in the Slime run
directory; this log records the reproducible configuration, artifact path,
outcome, and the code issue exposed by the run.

## Entry template

- UTC time:
- Commit tested:
- Status: `completed`, `terminated`, or `failed`
- Model/workload/topology/concurrency:
- Warmup/measurement:
- Configuration:
- Slime artifact path:
- Result:
- Problem found:
- Follow-up:

## 2026-08-26 recovery checkpoint

- Status: validation only; no GPU experiment was started.
- Base: `b28547619a` (`pd-native-hicache-mooncake-isolation-complete`).
- Recovered the uncommitted request-generation staging implementation after
  the host filesystem filled and truncated two source files.
- Preserved the split Decode progress domains for P-to-D, D-to-P Direct,
  D-to-P Slow, preallocation control, and preallocation metadata.
- Made P-to-D Host rank claim and physical grant one atomic ledger mutation.
- Made duplicate TP control offers monotonic so an owned generation cannot
  regress to `OFFERED`.
- Added the in-lock TP=1 priority recheck so a queued P-to-D legacy NIXL
  sender yields when a D-to-P Direct request arrives while it waits for the
  shared control boundary.
- Validation: SGLang agentic/TP suite `146 passed`; Slime router and async
  progress suite `90 passed`.
- Independent code audit: GO. It confirmed the TP=1 priority handoff and the
  TP1/TP2 Host ledger ownership transitions. Remaining non-blocking risk:
  sustained Direct traffic has no strict fairness bound for an unsubmitted
  legacy P-to-D sender; monitor oldest-active age during the GPU run.

## 2026-08-26 TP1 recovered-state short validation (`r39`)

- UTC time: 2026-08-26 17:53--18:08.
- Commit tested: `e71ed426f6`.
- Status: `completed`.
- Model/workload/topology/concurrency: Qwen3-8B, fixed source-order pure
  BrowseComp, TP=1, 4P:4D, closed-loop c512.
- Warmup/measurement: 60.65 s / 180.00 s.
- Configuration: native HiCache/Mooncake disabled; D target KV fraction 1.0;
  2 s tool/Direct windows; 128 GiB per-P D-to-P arena; 32 GiB per-P P-to-D
  arena; max Prefill inflight 12; 24 P-to-D consumers.
- Slime artifact path:
  `slime/examples/pd/runs-host/new-method/smoke-unified-workset-tp1-4p4d-c512-20260826-r39-recovered`.
- Result: 314 agents completed during measurement; Decode 3611.48 token/s
  total (902.87 token/s/D); D Forward 96.21% per GPU. All 164 completed later
  turns fully reused their page-aligned parent prefix: 891,136/891,136 parent
  tokens, 100.00% reuse, and zero extra Prefill tokens from parent-KV loss.
  The old `unconfirmed_tool_recompute`, `parent_snapshot_terminal_fallback`,
  Host-capacity timeout, and shared-Host ready timeout paths did not occur.
  Runtime observed 480 completed D-to-P Direct receives and 524 completed
  D-to-P Shared-Host D2H copies; the latter were predominantly snapshots whose
  application/tool result had not arrived within the 2 s fast window.
- Problem found: the shortened warmup is not performance-comparable to the
  300+1200 s reference. The measurement completion set averaged only 1.52
  model calls per agent, versus about 3.55 in the 4660 token/s reference, so
  the run remained dominated by initial-Prefill/transient trajectory mix.
  Shutdown also left the experiment Router alive until it was explicitly
  terminated; no GPU worker was orphaned.
- Follow-up: audit all remaining transient-failure-to-recompute branches,
  then run the canonical 300+1200 s validation. Fix process ownership in the
  launcher separately from the serving state machine.

## 2026-08-28 TP1 decoupled P-ready Host staging formal validation

- UTC time: 2026-08-28.
- Commit tested: `6f4aa6c19a`.
- Status: `completed`.
- Model/workload/topology/concurrency: Qwen3-8B, fixed source-order n680 pure
  BrowseComp, TP=1, 4P:4D, closed-loop c512.
- Warmup/measurement: 301.1 s / 1200.0 s.
- Configuration: native SGLang HiCache/Mooncake disabled; request-generation
  D-to-P Direct and Shared Host paths enabled; decoupled P-ready Host
  preparation and FIFO D-admission stages; 32 GiB P-to-D Shared Host Arena per
  P; 128 GiB D-to-P Shared Host Arena per P.
- Slime artifact path:
  `slime/examples/pd/runs-host/new-method/formal-qwen3-8b-tp1-browsecomp-4p4d-p2d-decoupled-w300-m1200-20260828-r1`.
  Durable result summary is in Slime commit `18b0fc5` on
  `agent/agentic-pd-experiments`.
- Result: 2,653 agents completed with zero failures (2.211 agents/s). Decode
  throughput was 4,557.62 token/s total (1,139.41 token/s/D); average D Forward
  was 98.66% per GPU. Prefill compute throughput was 30,221.24 token/s and
  average P Forward was 78.81% per GPU. All 6,389 completed later turns reused
  100.00% of their page-aligned parent KV, with zero extra Prefill tokens from
  parent-KV loss. P-to-D Host staging and P-HBM release counters matched on
  every P except for one copy still live at the abrupt measurement cutoff.
- Data-path behavior: 6,239 settled D-to-P Direct transfers and 2,281 settled
  fallbacks, for 73.2% Direct success. Of 2,244 Host-ready fallback snapshots,
  2,055 had completed Host-to-P recovery by cutoff; 189 remained pending among
  the 512 still-active closed-loop requests. No Mooncake path, eviction, or
  recompute fallback was used.
- Problem found: no lifecycle or parent-KV correctness failure. The formal
  Decode result is about 2.2% below the earlier approximately 4,660 token/s
  reference and varies by source-order trajectory phase; D-to-P Direct success
  remains the main optimization opportunity. Router graceful shutdown can take
  longer than 120 s while c512 HTTP requests are active, although supervised
  cleanup left no orphan worker or GPU allocation.
- Follow-up: treat this as the stage checkpoint for the decoupled P-ready Host
  pipeline. Optimize Direct admission and shutdown latency without weakening
  request-generation ownership, FIFO P-to-D commit, or strict parent-KV reuse.
