# Shift-Aware Risk Control

Shift-Aware Risk Control is a reproducible Python toolkit for evaluating selective
prediction and risk-control methods when data distributions vary across contexts.
It provides an end-to-end workflow from prediction caches and behavioral summaries
to calibrated policies, diagnostics, tables, figures, and machine-readable run
manifests.

The repository includes:

- a self-contained synthetic benchmark that runs on CPU;
- global, context-transfer, and context-weighted risk-control methods;
- simultaneous-validity, source-certificate, transfer-penalty, and data-leakage
  audits;
- optional CIFAR-10-C, CIFAR-100-C, WILDS, and Office-Home workflows; and
- deterministic configuration, seed handling, and environment snapshots for each
  pipeline run.

## Quick start

Python 3.10 through 3.14 is supported. The default workflow is CPU-only and does
not download external datasets.

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/full_pipeline.py --config configs/main.yaml --stage configured
```

The pipeline writes generated data beneath `outputs/` and records the resolved
configuration, Git state, platform details, package versions, stage status, and
timings in `outputs/pipeline_runs/local_main/manifests/`.

For a fully locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --frozen
uv run pytest -q
uv run python scripts/full_pipeline.py --config configs/main.yaml --stage configured
```

## Common workflows

Run only the synthetic cache and evaluation:

```bash
python scripts/full_pipeline.py --config configs/main.yaml --stage synthetic
```

Regenerate tables and figures from available runs:

```bash
python scripts/full_pipeline.py --config configs/main.yaml --stage artifacts
```

Use the broader benchmark artifact selection after the corresponding runs exist:

```bash
python scripts/make_artifacts.py --config configs/artifacts_benchmarks.yaml
```

External benchmarks are opt-in because they require additional dependencies,
storage, or manually obtained data. Install the relevant extras first:

```bash
python -m pip install -e ".[dev,vision]"  # CIFAR-C and Office-Home
python -m pip install -e ".[dev,wilds]"   # WILDS-backed datasets
```

Examples:

```bash
python scripts/build_cifar_c_cache.py --config configs/cifar10c_resnet18.yaml --download
python scripts/run_cifar_c_experiment.py --config configs/experiments/cifar10c_resnet18_main.yaml

python scripts/build_wilds_cache.py --config configs/wilds_camelyon17.yaml --download
python scripts/run_wilds_experiment.py --config configs/experiments/wilds_camelyon17_main.yaml
```

Downloads occur only when explicitly requested. See [Dataset provenance and
usage](docs/datasets.md) before running an external benchmark.

## Repository layout

```text
configs/                Versioned experiment and pipeline configurations
scripts/                Command-line workflows
src/contextual_crc/     Library implementation
tests/                  Unit and integration tests
outputs/                Ignored generated caches, runs, and artifacts
docs/                   Reproduction and dataset guidance
```

The installable distribution is named `shift-aware-risk-control`; the import
package remains `contextual_crc`.

## Reproducibility contract

- Seeds and experiment parameters live in versioned YAML files.
- Generated results are never treated as source inputs unless their path is
  explicitly configured.
- Evaluation workflows validate cache schemas and stable example identifiers.
- Pipeline manifests record enough environment and Git metadata to diagnose run
  differences.
- The lightweight workflow is exercised by continuous integration across the
  supported Python range.

Exact floating-point results can still vary by platform, dependency build, or GPU
backend. Use `uv.lock`, retain each run manifest, and prefer CPU execution when
bit-level comparability matters. See [Reproducibility guide](docs/reproducibility.md)
for the complete protocol.

## Development

Before opening a change, run:

```bash
python -m ruff check src scripts tests
python -m pytest -q
python -m build
```

Contribution guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The source code is available under the [Apache License 2.0](LICENSE). External
datasets and pretrained model weights retain their own terms and are not included
in this repository.
