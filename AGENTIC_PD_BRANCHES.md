# Agentic PD branch policy

This repository tracks the SGLang runtime used by the agentic PD experiments in
`slime/examples/pd`.

## Branches

- `pd_baseline`: frozen experiment baseline. It is based on upstream
  `v0.5.10.post1` (`7c35342c10e201899e22fe2972d40e60da19ff3e`) plus the Decode
  KV-offload lifetime correctness fix already present in the `pd_baseline`
  Conda environment. Do not develop directly on this branch.
- `pd`: stable integration branch for the modified agentic PD runtime. Only
  merge changes here after the focused tests and an end-to-end experiment pass.
- `pd_node_a`: active development branch for this node. Commit work here first,
  then merge it into `pd` after validation.

For another machine, create a separate branch such as `pd_node_b` from `pd`.
Do not let multiple machines write to the same node branch.

## Environment mapping

- Conda environment `pd_baseline` should install `pd_baseline`.
- Conda environment `pd` should normally install the current node branch in
  editable mode:

  ```bash
  conda activate pd
  python -m pip install -e ./python --no-deps
  ```

The `pd` branches pin the editable package metadata to `0.5.10.post1`. This
avoids setuptools-scm incorrectly selecting a newer upstream tag that happens
to exist in the clone. Update this pin only when creating a new baseline.

Before a baseline experiment, verify the branch and source location:

```bash
python -c 'import importlib.metadata as m, pathlib, sglang; print(m.version("sglang"), pathlib.Path(sglang.__file__).resolve())'
git branch --show-current
git status --short
```

## Integration workflow

```bash
git switch pd_node_a
# edit, test, and commit
git switch pd
git merge --no-ff pd_node_a
git push origin pd pd_node_a
```

The `pd_baseline` ref is intentionally frozen. If the upstream SGLang version
must change, create a new versioned baseline branch instead of moving it.

## Current agentic runtime scope

The initial `pd` snapshot mirrors the modified source currently installed in
the `pd` Conda environment. It adds the request lifecycle, early claim,
direct-transfer, host-staging, Mooncake admission, P-ready buffering/routing,
and related scheduler/metrics plumbing. The initial snapshot contains 18
modified and 6 new `python/sglang` source files relative to `pd_baseline`.
