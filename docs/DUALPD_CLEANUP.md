# First engine cleanup: scope and compatibility

Reference: engine commit `bc67aa0814364f2c7ce3df3f83f9d5ff013c5b5f`.
The original `sglang-src/pd` tree and baseline environments are not modified.

## Implemented

- Extract the Host control ledger from `agentic_host_staging.py` into
  `agentic_host_ledger.py`. Nine top-level state/constants/class nodes, including
  the CAS ownership sentinel, retain identical ASTs. The remaining DMA, arena,
  registration, transport worker and manager bodies are unchanged.
- Preserve the old import locations as aliases to the same class, enum and
  state-table objects, not duplicate implementations. New ledger code has no
  Torch, NumPy or native kernel dependency of its own.
- Move the 41 existing flat test files into `dualpd/tests/`; fix four cross-test
  imports and one source-file lookup. Keep Router tests in `router/tests/`.
- Remove `memory_authority_v2_enabled` and its assertion-only test. No runtime
  code read it. The similarly named `SGLANG_AGENTIC_MEMORY_AUTHORITY_V2` export
  is removed from the maintained multi-node launcher. Actual activation remains
  `SGLANG_AGENTIC_MULTINODE_ENABLED` with its existing validation and defaults.
- Remove `control_poll_interval`, an uncalled compatibility hook that always
  returned `None`. The actual directory watchers and event-driven transport
  controllers remain intact.

## Switch policy: retained, not inferred obsolete

| Category | Examples | Decision |
|---|---|---|
| Runtime selection | `SGLANG_AGENTIC_MULTINODE_ENABLED`, `SGLANG_AGENTIC_KV_LIFECYCLE`, `SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY` | Preserve local/TCP selection and validation |
| Ablations | `SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE`, `SGLANG_PD_ABLATION_P2D_PREBIND` | Preserve, including TP restrictions |
| Direct/Slow policy | Fast-tool/admission thresholds, slow-congestion recompute, high/low queue thresholds | Preserve values and scope; do not unify local/TCP defaults |
| Memory/transport | Arena sizes, NUMA placement, Relay, lanes, registration windows, startup prewarm | Preserve; default-off does not mean dead |
| Compatibility/fallback | Historical Mamba Router, registered-DMA fallback, bounce buffers, legacy manifest migration | Preserve supported replay and physical safety paths |
| Correctness/diagnostics | Token hashes, digest logging, TP and Mamba checks | Preserve |

The verifier inventories all literal SGLang environment identifiers across the
extension and both Router variants. Only the proven no-op memory-authority
helper selector disappears. Native `environ.py`, scheduler, model and allocator
integration files are unchanged in this cleanup.

## Ownership and failure mapping

There are no new states, paths, deadlines, thread queues or transfer decisions.
Physical owners, complete-workset admission, DMA fences, durable commits,
failure/timeout/cancel/shutdown transitions, TP atomicity and parent reuse are
the same implementation. All eight design invariants remain mandatory.

The ledger classes now report `agentic_host_ledger` as their `__module__`.
Old pickle class/enum references resolve through the historical aliases; new
references require the new module. These classes are not serialized in the
durable JSON ledger. Rebinding a *module attribute* such as an old private
sentinel is not a supported monkeypatch contract after extraction; existing
instance-method patches and shared state-table identity remain valid. Actual
DMA/arena monkeypatch sites remain in `agentic_host_staging` unchanged.

## Verification and remaining work

`python tools/verify_dualpd_cleanup.py` reconstructs the original module AST
boundary and rejects any change to ledger, DMA/worker or runtime behavior
beyond the two named dead helpers. `dualpd-source-map.json` retains original
hashes and records current hashes for the nine reviewed source/test edits.
The frontend snapshot verifier also checks every mapped file hash.

The reproducible CPU gate is `../DualPD/.venv/bin/python
../DualPD/scripts/check_cpu.py`; its evidence is stored under
`../DualPD/tests/reports/`. Additional tests cover alias identity, old/new pickle
paths, replay-after-claim and isolated ledger imports.

Some broader tests import native kernels requiring `libcuda.so.1`; collecting
all tests on the login host therefore fails for those modules. Do not count
those tests as passes, stub the CUDA library or weaken their assertions.
This cleanup does not claim a new GPU correctness or throughput measurement.

The large Scheduler body and remaining Host transport code are not fully
decoupled yet. Those are separate follow-ups requiring their own lifecycle
tests; this pass does not delete reachable branches to meet a line-count goal.
