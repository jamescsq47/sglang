# DualPD engine fork

## Organized release (2026-11-09)

The `main` branch publishes the cleaned standalone DualPD engine fork. This
release reorganizes Router, Host transport/ledger and Scheduler integration into
focused components while preserving public imports, runtime switches and
experiment semantics. It does not replace or update the existing baseline
branches or environments.

The companion application is [DualPD](https://github.com/jamescsq47/DualPD),
which pins this engine as `3rdparty/sglang`. Its
[README](https://github.com/jamescsq47/DualPD#installation) provides installation
and experiment commands. Historical validation material is retained in the
[pre-release cleanup handoff](https://github.com/jamescsq47/DualPD/blob/61e3342fc006920af5553961697dfc5c255e0735/docs/CLEANUP_HANDOFF.md),
not distributed in the current application source tree.
The final BrowseComp regression retained the 300 s warmup + 1200 s measurement
window; Decode throughput differed by -0.23% from its comparison baseline.
This is not a new SWE-Bench accuracy validation.

## Layout and compatibility

This checkout preserves SGLang's normal package and server entrypoints. The
custom extension sources are grouped in
`python/sglang/srt/disaggregation/dualpd/`, with focused tests under `tests/`
and Router tests under `router/tests/`.

Public imports remain `sglang.srt.disaggregation.agentic_*` and
`sglang.srt.disaggregation.p2d_host_staging`. The parent package extends its
module search path to the extension directory, so the existing scheduler,
launchers and Python monkeypatches refer to the same canonical module objects.
Do not import the files using a second `disaggregation.dualpd.agentic_*` name.

The extension has two supported runtime paths: the existing local shared-arena
pipeline and the event-driven TCP/TP runtime. Both are needed by the retained
experiments. Their policy defaults are not silently unified during extraction.
The original extraction preserved core implementations byte-for-byte.
`dualpd-source-map.json` records source revision, original paths/hashes and
reviewed current hashes. The first internal cleanup separates the CUDA-free
Host ledger, retains historical imports as class-identical aliases, and removes
two uncalled helpers. Its AST-equivalence gate is
`python tools/verify_dualpd_cleanup.py`; see `docs/DUALPD_CLEANUP.md` for the exact
scope, retained switches and compatibility boundary.

The original integration cleanup was a type-only HostKVCache import in
`hicache_storage.py`, allowing metadata/control-plane modules to load on CPU
hosts without importing CUDA kernels solely for type annotations.

See `docs/DUALPD_CLEANUP.md` for the engine validation scope. Performance and
GPU-correctness claims still require actual GPU runs. Existing baseline Python
environments must remain pointed at their separately selected baseline checkout.
