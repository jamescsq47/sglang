# DualPD engine fork

This checkout preserves SGLang's normal package and server entrypoints. The
custom extension sources and their focused tests are grouped in
`python/sglang/srt/disaggregation/dualpd/`.

Public imports remain `sglang.srt.disaggregation.agentic_*` and
`sglang.srt.disaggregation.p2d_host_staging`. The parent package extends its
module search path to the extension directory, so the existing scheduler,
launchers and Python monkeypatches refer to the same canonical module objects.
Do not import the files using a second `disaggregation.dualpd.agentic_*` name.

The extension has two supported runtime paths: the existing local shared-arena
pipeline and the event-driven TCP/TP runtime. Both are needed by the retained
experiments. Their policy defaults are not silently unified during extraction.
The core protocol, DMA, allocator and ownership code is preserved byte-for-byte;
`dualpd-source-map.json` records source revision, original paths and SHA-256.

The only integration cleanup is a type-only HostKVCache import in
`hicache_storage.py`, allowing metadata/control-plane modules to load on CPU
hosts without importing CUDA kernels solely for type annotations.

See the sibling `../DualPD/docs/EXTRACTION.md` for validation. Performance and
GPU-correctness claims still require actual GPU runs. Existing baseline Python
environments must remain pointed at their separately selected baseline checkout.
