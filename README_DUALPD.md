# DualPD engine fork

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

See the sibling `../DualPD/docs/EXTRACTION.md` for validation. Performance and
GPU-correctness claims still require actual GPU runs. Existing baseline Python
environments must remain pointed at their separately selected baseline checkout.
