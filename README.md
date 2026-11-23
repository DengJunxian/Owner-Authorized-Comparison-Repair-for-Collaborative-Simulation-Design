# Simulation experiment and data-processing code

This repository contains the software and experiment configurations for the
CSCWD study on owner-authorized comparison repair in collaborative simulation.
It is a code-only snapshot: the manuscript, its figures and PDFs, raw experiment
results, local audit records and author-review material are not included.

## Contents

- `src/policytwin/`: the synthetic simulator, comparison-contract
  implementation, typed evidence, owner policies, revision planning and
  supporting data-processing modules. The package also contains optional
  simulator components that are not part of every CSCWD experiment.
- `scripts/cscwd2027*`: experiment design, execution, analysis and audit code
  for the separate CSCWD studies.
- `configs/cscwd2027/`: study definitions, schedules and selected original
  freeze manifests. Configurations are retained without changing their values.
- `tests/`: experiment and shared engineering tests.
- `analysis/presentation/` and `ccfa-workfiles/figures/revision-workload/`:
  code that derives presentation tables and plots from recorded results.

## Environment and focused tests

The project targets Python 3.12 and uses the dependency versions in `uv.lock`.
With uv 0.7.17 installed:

```sh
uv sync --frozen --group dev
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_cscwd_revision_workload.py \
  tests/test_cscwd_consent_projection.py \
  tests/test_cscwd_owner_policy.py \
  tests/test_cscwd_negotiated_repair.py \
  tests/test_cscwd_typed_revision.py
```

These tests exercise synthetic engineering fixtures; they are not a rerun of
the reported study. They passed in the source-export environment before upload.

## Evidence boundary

Recorded measurements and simulation capsules were intentionally not uploaded.
Consequently, analyses and publication tests that read `results/`, `paper/`,
prior review archives or full freeze snapshots cannot run from this checkout
alone. The included historical freeze manifests refer to those omitted files;
their presence here is not a claim that the archived studies were independently
reproduced. Do not alter a frozen run or reuse its output directory to make a
verification pass.

The plot renderer additionally depends on Matplotlib, a local CCFA plotting
theme and the Times New Roman font. It requires the original structured timing
records and prior aggregate, none of which is included in this code-only export.
