# Agentic Router implementations

These are engine services, not benchmark harnesses. They depend on SGLang and
sglang-router, never on DPD or Slime. DPD configures and launches them.

| Module | Purpose |
|---|---|
| `late_binding_router.py` | Local control-ledger routing, P-ready admission and late-bound Decode |
| `local_capacity.py` | Capacity sampling, shadow credits and Decode reservations (33 methods) |
| `local_admission.py` | Admission gate, ledger/P-ready coordination (38 methods) |
| `local_http.py` | HTTP attempts, cancellation joins, deduplication and dispatch (27 methods) |
| `local_types.py` / `local_config.py` | Shared records/logger and unchanged dynamic parsing helpers |
| `launch_late_binding_router.py` | Local Router service startup/shutdown |
| `global_pd_router.py` | TCP/group-protocol Router and callback HTTP endpoints |
| `tcp_capacity.py` | TCP causal load epochs, shadow reservations and its distinct record |
| `tcp_http.py` | TCP backend HTTP and generation retry lifetime, not local policy |
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

## Local Router organization in progress

The concrete local Router keeps its constructor and public identity. Capacity
and shadow reservations, admission/ledger/P-ready coordination, and backend HTTP
attempt/dedup/dispatch each become one method-only component. They share the
same Router instance, locks, queues and records; components add no constructors,
tasks or policy priorities. Data records and parsing helpers each have one
canonical definition, reexported at the old entry point. TCP and pinned legacy
policies are not merged with local behavior.

Ownership mapping: Router shadow credits/markers are not physical KV ownership.
The original backend completion/ledger commits still authorize P/D/Host release.
Success, capacity wait, timeout, client cancellation/retry and shutdown retain
their exact existing branches, physical-operation joins and reservation rollback.
Thus invariants1–4 and6–7 keep their transitions; invariant5 preserves queue/
Forward independence and network offload; invariant8 requires all lifecycle/
fault tests, independent audit and actual BrowseComp before acceptance.

Source validation must reconstruct all original methods/data nodes in their
original order, reject duplicates/additions and validate the exact MRO. The old
overlay file entry must retain its private alias and native overlay; helpers
use relative imports through a file-as-package spec. Logger and data aliases
must match both canonical and file-entry contexts. Only defining-module private
test hooks may be retargeted, with assertions unchanged. Full performance is
pending, and TCP/Scheduler/configuration/launcher/final stages remain required.

The reconstruction verifier additionally checks runtime global bindings in each
component: reproducing a method body alone does not prove its imports exist.
Mutation tests remove a reservation record import, remove parsing imports and
redirect an import to the wrong owner; all must fail validation. Annotation
resolution, canonical aliases, old pickle paths and a fresh-process historical
overlay entry are checked separately. These checks do not establish a pure
control plane: local records still import the existing Host state definitions;
removing that native dependency is a later, explicit stage.

## TCP Router organization in progress

The concrete TCP root keeps eight methods: construction/close, provisional P
selection, request-group annotations, room parsing and P-ready callback/waiting.
Its capacity component has12 methods and owns the private Decode reservation
record; its HTTP component has9 methods. There is still only one instance and
one set of counters, queues, tasks and epochs. Neither helper imports the root,
registers callbacks or creates a session/task on import. The three decorated
callbacks, main and -m guard remain unchanged at the original service entry;
the original aiohttp and send_application_final patch sites remain available.

TCP capacity charges are hints, never physical ownership. Original TP memory
authorities and DMA/ledger fences still authorize handoff and release. Exact
method bodies preserve successful handoff, no-capacity waiting, causal sample
retirement, timeout, cancellation/shielding, backend settlement and shutdown.
Thus invariants1–4/6–7 have no new transition;5 has no new queue, blocking work
or scheduling priority;8 still requires fault tests, independent GO and actual
performance validation. The local frozen performance case does not exercise
the TCP model path and must not be presented as such. Local and pinned policies
are unchanged in this stage. TCP performance acceptance is still pending.
