# Agentic Router implementations

These are engine services, not benchmark harnesses. They depend on SGLang and
sglang-router, never on DualPD or Slime. DualPD configures and launches them.

| Module | Purpose |
|---|---|
| `late_binding_router.py` | Local control-ledger routing, P-ready admission and late-bound Decode |
| `launch_late_binding_router.py` | Local Router service startup/shutdown |
| `global_pd_router.py` | TCP/group-protocol Router and callback HTTP endpoints |
| `legacy_mamba_router.py` | Explicit historical Mamba compatibility startup/adapter |
| `legacy_mamba_pinned.py` | Immutable matched old Router; only used by that historical launcher |
| `tests/` | Routing, admission, congestion and standalone-dependency tests |

Current service entries:

```bash
python -m sglang.srt.disaggregation.dualpd.router.launch_late_binding_router --help
python -m sglang.srt.disaggregation.dualpd.router.global_pd_router --help
```

The package initializer must stay lazy: global TCP callbacks register routes on
the shared mini-lb application. Do not import both service implementations into
it. Do not merge the local and TCP policy defaults; they support different
control protocols. Existing KV ownership/fences, admission, cancellation and
shutdown behavior are unchanged by the move. See `router-source-map.json` at
the repository root for content provenance.
