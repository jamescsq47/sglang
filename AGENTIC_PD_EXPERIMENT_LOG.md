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
