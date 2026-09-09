# Contributing

Contributions that improve correctness, reproducibility, documentation, or
benchmark coverage are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create a focused branch, keep generated data outside version control, and add or
update tests for behavioral changes.

## Required checks

```bash
python -m ruff check src scripts tests
python -m pytest -q
python -m build
```

Changes to an experiment must also document the seed, dataset version, split,
configuration, and expected output location. Do not add raw datasets, pretrained
weights, credentials, private paths, or generated result directories.

## Reports and proposals

Bug reports should include a minimal reproduction, operating system, Python
version, dependency versions, configuration, and the relevant manifest excerpt.
For substantial changes, open a discussion before investing in an implementation
so interfaces and evaluation criteria can be agreed first.
