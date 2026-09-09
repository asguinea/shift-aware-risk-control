from __future__ import annotations

from pathlib import Path


RUN_SUBDIRS = ("logs", "cache", "results", "figures", "tables", "configs")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_run_dir(output_root: str | Path, run_name: str) -> Path:
    """Create a run directory and standard output subdirectories."""
    run_dir = ensure_dir(Path(output_root) / run_name)
    for subdir in RUN_SUBDIRS:
        ensure_dir(run_dir / subdir)
    return run_dir
