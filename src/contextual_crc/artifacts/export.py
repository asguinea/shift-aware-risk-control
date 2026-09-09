from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from contextual_crc.utils.paths import ensure_dir


RESULT_FILENAMES = {
    "method_comparison": "method_comparison.csv",
    "ct_crc_windows": "ct_crc_windows.csv",
    "cw_crc_windows": "cw_crc_windows.csv",
    "context_smoothness": "context_smoothness.csv",
    "bound_diagnostics": "bound_diagnostics.json",
    "ct_crc_summary": "ct_crc_summary.json",
    "cw_crc_summary": "cw_crc_summary.json",
    "manifest": "run_manifest.json",
    "online_trace": "online_trace.csv",
}


def discover_run_dirs(
    run_dirs: Sequence[str | Path],
    *,
    include_missing: bool = False,
) -> list[Path]:
    """Return requested run directories, optionally preserving missing paths."""
    discovered: list[Path] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if path.exists() or include_missing:
            discovered.append(path)
    return discovered


def load_run_outputs(run_dir: str | Path) -> dict[str, Any]:
    """Load available result files from one experiment run directory."""
    run_path = Path(run_dir)
    results_dir = run_path / "results"
    manifest = _read_json(results_dir / RESULT_FILENAMES["manifest"])
    run_name = str(manifest.get("run_name") or run_path.name) if manifest else run_path.name
    dataset = _infer_dataset(run_name, manifest)

    return {
        "run_dir": run_path,
        "run_name": run_name,
        "dataset": dataset,
        "missing": not run_path.exists(),
        "valid": _is_valid_run_dir(run_path),
        "manifest": manifest,
        "method_comparison": _read_csv(results_dir / RESULT_FILENAMES["method_comparison"]),
        "ct_crc_windows": _read_csv(results_dir / RESULT_FILENAMES["ct_crc_windows"]),
        "cw_crc_windows": _read_csv(results_dir / RESULT_FILENAMES["cw_crc_windows"]),
        "context_smoothness": _read_csv(results_dir / RESULT_FILENAMES["context_smoothness"]),
        "bound_diagnostics": _read_json(results_dir / RESULT_FILENAMES["bound_diagnostics"]),
        "ct_crc_summary": _read_json(results_dir / RESULT_FILENAMES["ct_crc_summary"]),
        "cw_crc_summary": _read_json(results_dir / RESULT_FILENAMES["cw_crc_summary"]),
        "online_trace": _read_csv(results_dir / RESULT_FILENAMES["online_trace"]),
    }


def load_many_run_outputs(
    run_dirs: Sequence[str | Path],
    *,
    include_missing: bool = False,
) -> list[dict[str, Any]]:
    """Load outputs for many runs, raising only when no usable runs are found."""
    discovered = discover_run_dirs(run_dirs, include_missing=include_missing)
    outputs = [load_run_outputs(run_dir) for run_dir in discovered]
    valid_outputs = [output for output in outputs if bool(output.get("valid"))]
    if not valid_outputs:
        raise ValueError(
            "No valid experiment run directories found. Expected results/run_manifest.json "
            "or results/method_comparison.csv in at least one run directory."
        )
    return outputs


def write_json(obj: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON file with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(obj)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def ensure_artifact_dirs(output_root: str | Path) -> dict[str, Path]:
    """Create and return artifact output directories."""
    root = ensure_dir(output_root)
    return {
        "root": root,
        "figures": ensure_dir(root / "figures"),
        "tables": ensure_dir(root / "tables"),
    }


def artifact_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_run_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    results_dir = path / "results"
    return (
        (results_dir / RESULT_FILENAMES["manifest"]).is_file()
        or (results_dir / RESULT_FILENAMES["method_comparison"]).is_file()
        or (results_dir / RESULT_FILENAMES["online_trace"]).is_file()
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _infer_dataset(run_name: str, manifest: dict[str, Any] | None) -> str:
    if manifest and manifest.get("dataset"):
        return str(manifest["dataset"])
    if manifest and manifest.get("schema_dataset"):
        return str(manifest["schema_dataset"])
    return run_name


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
