# DPD engine fork

## Standalone engine

The `main` branch publishes the cleaned standalone DPD engine fork. This
release reorganizes Router, Host transport/ledger and Scheduler integration into
focused components while preserving public imports, runtime switches and
experiment semantics. It does not replace or update the existing baseline
branches or environments.

The companion application is [DPD](https://github.com/jamescsq47/DualPD),
which pins this engine as `3rdparty/sglang`. Its
[README](https://github.com/jamescsq47/DualPD#installation) provides installation
and experiment commands. The public application tree contains source code and
workload configuration, not internal experiment records. Engine cleanup does
not constitute new throughput or SWE-bench accuracy validation.

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
Use the companion application's Python 3.12 environment and pinned submodule
revision. See [engine boundaries and validation](docs/DUALPD_CLEANUP.md) for
current component responsibilities, retained switches and validation limits.
Source maps and extraction-specific AST tools are historical development
artifacts, not current release acceptance checks.

Performance and GPU-correctness claims still require actual GPU runs. Existing
baseline Python environments must remain pointed at their separately selected
baseline checkout.
