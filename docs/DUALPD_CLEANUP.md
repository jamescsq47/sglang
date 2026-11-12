# Engine boundaries and validation

Installation, model/data preparation and experiment commands live in the
[DPD application README](https://github.com/jamescsq47/DualPD#installation).
Use its Python 3.12 environment and the engine commit pinned by its submodule.
Native baselines use a separate engine checkout and environment; never replace
them with this modified fork.

## Component boundaries

The application owns workload loading, tool execution, model HTTP requests,
grading, launch orchestration and resource sampling. The engine owns routing,
admission, KV transport, allocator integration and Scheduler progress.

Engine extension sources are in `python/sglang/srt/disaggregation/dualpd/`:

| Component | Responsibility |
| --- | --- |
| `router/` | Request routing, admission and local/remote control-plane coordination |
| `agentic_host_*` | D-to-P Host staging, ledger, registration, buffers and DMA helpers |
| `p2d_host_*` | P-to-D snapshot transport and protocol declarations |
| Workset and Scheduler integration | Complete-workset admission, staging and binding |
| `agentic_client.py` | Application ACK adapter for the existing local marker protocol |
| `agentic_host_observation.py` | Opt-in, read-only Host arena counters |

Public imports remain `sglang.srt.disaggregation.agentic_*` and
`sglang.srt.disaggregation.p2d_host_staging`. Do not import a second copy via
`sglang.srt.disaggregation.dualpd.*`; that creates distinct module identities.
Local shared-arena and event-driven TCP/TP execution are both retained.

## Compatibility and safety

Runtime selection, Direct/Host policies, arena sizing, NUMA/registration controls,
ablation switches and fault diagnostics remain supported. A default-off switch
is not necessarily obsolete. `PD_DRAM_TRACE_DIR` enables read-only Host counters;
the observer does not own allocator capacity, DMA fences or transfer decisions.

Transfers require complete snapshots/worksets. Source HBM and Host extents must
remain owned until the existing durable completion and physical DMA fences.
Claim, commit, handoff and release remain coordinated across TP ranks. A timeout
is not a DMA fence. Cancellation and failure must converge before buffer reuse.
Direct transfer, Host recovery, tool execution and model Forward progress must
remain decoupled; shared radix prefixes retain reference-safe lifetimes.

## Validation scope

Extension tests remain alongside the engine in `dualpd/tests/` and
`dualpd/router/tests/`. Run applicable tests in the installed engine environment;
some import native kernels and require CUDA libraries even during collection.
Missing dependencies or failed collection must not be counted as passing tests.
The [validation helpers](../validation/README.md) are optional diagnostics, not
a substitute for end-to-end experiments.

The latest application/engine boundary changes passed scoped CPU contracts and
independent CPU review. These checks do not establish GPU correctness,
throughput or task accuracy for the current checkout. BrowseComp acceptance
requires the complete 300-second warmup plus 1200-second steady-state window;
SWE-bench comparisons use the recorded T450 window and matching harness.

`dualpd-source-map.json`, `router-source-map.json` and
`tools/verify_dualpd_*.py` retain provenance and exact equivalence checks for
particular historical extraction stages. Later reviewed changes, including
Host observation registration, can invalidate those frozen comparisons. They
are not a current-release test suite and are not expected to certify the current
tree. Do not weaken their historical assertions to manufacture a passing gate.
Older development narratives and internal experiment records are available in
Git history rather than presented as current installation or validation steps.
