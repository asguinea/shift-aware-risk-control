# Reproducibility guide

This guide separates the self-contained verification path from larger external
benchmarks. Run commands from the repository root.

## 1. Record the source revision

Use a clean checkout and record the commit:

```bash
git status --short
git rev-parse HEAD
```

Avoid comparing runs produced from a dirty working tree. Every pipeline manifest
records the revision and whether local changes were present.

## 2. Create the environment

The locked path is preferred for exact dependency resolution:

```bash
uv sync --extra dev --frozen
uv run python --version
uv run python -m pip check
```

The standard Python path is:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```

## 3. Verify the implementation

```bash
python -m ruff check src scripts tests
python -m pytest -q
python scripts/full_pipeline.py --config configs/main.yaml --stage smoke
```

All commands must exit with status zero before an experiment is considered valid.

## 4. Run the self-contained workflow

```bash
python scripts/full_pipeline.py \
  --config configs/main.yaml \
  --stage configured \
  --run-name local_main
```

This creates a synthetic prediction cache, runs the synthetic evaluation, and
generates tables and figures. It performs no network downloads.

Inspect these files after completion:

- `outputs/pipeline_runs/local_main/manifests/pipeline_manifest.json`
- `outputs/pipeline_runs/local_main/manifests/requirements_snapshot.txt`
- `outputs/pipeline_runs/local_main/configs/resolved_config.yaml`
- `outputs/runs/synthetic_main/run_manifest.json`
- `outputs/artifacts/artifact_manifest.json`

The pipeline manifest must report zero failed stages. Keep the manifest and
resolved configuration with any exported result bundle.

## 5. Run external benchmarks

External data is not stored in this repository. Follow
[Dataset provenance and usage](datasets.md), install the matching optional
dependencies, and run cache generation before evaluation.

Use explicit flags for network access and explicit `--overwrite` flags when a
generated directory should be replaced. Cache validators reject incomplete or
schema-incompatible data before an evaluation begins.

After all desired runs are available, generate the broader artifact set:

```bash
python scripts/make_artifacts.py --config configs/artifacts_benchmarks.yaml
```

## 6. Compare runs

Compare, at minimum:

1. Git commit and dirty-tree status.
2. Resolved YAML configuration.
3. Python, operating system, and processor metadata.
4. Selected dependency versions and the full requirements snapshot.
5. Dataset cache metadata and example counts.
6. Seed, split identifiers, and stage statuses.

CPU runs with the same locked environment are the most portable reference.
GPU kernels and third-party model weights may introduce nondeterminism or change
over time; record device and weight metadata whenever those backends are used.
