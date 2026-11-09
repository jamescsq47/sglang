# Staged engine cleanup: scope and compatibility

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
DMA/arena monkeypatch sites remained in `agentic_host_staging` at the ledger-only
stage. See the later DMA ABI boundary below for its precise private-hook scope.

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
The ledger-only code at `4cde672cc` subsequently passed the actual BrowseComp
baseline on node2700: 300.98 s warmup, 1200.00 s measurement, 11265.01 Decode
token/s, 75367.90 Prefill token/s, parent reuse100%, 6704 completed agents and
zero failures. GPU/port/FUSE cleanup passed and all scoped OOM deltas were0.
This is not performance acceptance of the later changes described below.

The large Scheduler body and remaining Host transport code are not fully
decoupled yet. Those are separate follow-ups requiring their own lifecycle
tests; this pass does not delete reachable branches to meet a line-count goal.

## Next stage: Host DMA ABI (performance gate pending)

`agentic_host_dma.py` owns the six CUDA ABI types/helpers and four process-local
handle/cache/lock definitions formerly inline in Host staging. Their complete
ASTs remain identical, including the Ampere fallback, versioned eight-argument
`cuMemcpyBatchAsync_v2` ABI, conservative stream ordering, descriptor keepalive
and distinction between unavailable-before-submit and failure-after-submit.
The historical helper imports in Host staging are aliases to these same
objects. Native callers and the indexed-copy helper continue using the existing
interfaces. No actual feature or configuration switch is removed.

The mutable runtime/driver cache globals now belong only to the DMA module.
ABI-internal test patches should target that module; rebinding a historical
helper alias does not forward into another module's globals. Existing indexed
descriptor, arena and snapshot hooks remain at their original caller boundary.

Ownership/failure mapping: this leaf has no snapshot, ledger, capacity or TP
policy. The caller remains the owner until its existing physical CUDA fence and
durable commit. Success returns the same descriptor keepalive; pre-submit
unavailability returns the same fallback indication; post-submit errors raise
and must drain/quarantine through the unchanged caller fence. Timeout,
cancellation and shutdown are still handled by existing managers. No I/O queue
or Forward scheduling order changes, and TP still has its existing group owner.
Thus invariants1–4 and6–7 have no state/branch changes,5 has no new waits, and8
requires the new ABI fault tests, existing lifecycle suite, independent GO and
the same real BrowseComp comparison before proceeding.

The AST verifier now reconstructs ten DMA nodes as well as the nine ledger
nodes and all remaining Host worker bodies. New CPU checks cover alias identity,
Ampere/no-driver fallback, library lifetime/ABI shape, zero/mismatched arrays,
descriptor contents/lifetime, and both single and batch submission errors.
The rest of Host arena/registration/workers, Router, Scheduler, configuration
and maintained/archival launcher organization remain outstanding stages.
