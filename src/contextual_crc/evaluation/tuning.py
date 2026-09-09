from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from contextual_crc.evaluation.cifar_c_experiment import run_cifar_c_experiment
from contextual_crc.utils.paths import ensure_dir


DEFAULT_CIFAR_CW_SWEEP: dict[str, Sequence[Any]] = {
    "cw_crc.target_effective_sample_size": [3, 4, 6, 8, 12],
    "cw_crc.bound_margin": [0.0, 0.005, 0.01, 0.02, 0.03],
    "cw_crc.bound_multiplier": [1.0, 1.1, 1.25],
    "cw_crc.bandwidth": [None],
}


def set_nested(config: dict, dotted_key: str, value: Any) -> dict:
    """Set a nested dictionary value addressed by a dot path."""
    if not dotted_key:
        raise ValueError("dotted_key must be non-empty")
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value
    return config


def make_config_variants(
    base_config: dict,
    sweep: Mapping[str, Sequence[Any]],
) -> list[dict]:
    """Return deep-copied configs for every deterministic sweep-grid combination."""
    items = [(key, list(values)) for key, values in sweep.items()]
    if not items:
        return [deepcopy(base_config)]
    for key, values in items:
        if not values:
            raise ValueError(f"sweep values for {key!r} must be non-empty")

    variants: list[dict] = []
    for combination in itertools.product(*(values for _, values in items)):
        variant = deepcopy(base_config)
        for (key, _), value in zip(items, combination, strict=True):
            set_nested(variant, key, value)
        variants.append(variant)
    return variants


def compute_selection_score(
    *,
    mean_risk: float,
    mean_coverage: float,
    bound_underestimation_rate: float,
    risk_budget: float,
    min_coverage: float = 0.05,
    underestimation_penalty: float = 2.0,
    coverage_reward: float = 0.25,
) -> float:
    """Score CW-CRC sweep candidates; lower scores are better."""
    risk = _finite_or_default(mean_risk, default=1.0)
    coverage = max(0.0, _finite_or_default(mean_coverage, default=0.0))
    underestimation = max(0.0, _finite_or_default(bound_underestimation_rate, default=1.0))
    budget = _finite_or_default(risk_budget, default=0.1)
    required_coverage = max(0.0, _finite_or_default(min_coverage, default=0.05))

    excess_risk = max(0.0, risk - budget)
    low_coverage = max(0.0, required_coverage - coverage)
    degenerate_penalty = 1.0 if coverage <= 1e-12 else 0.0
    score = (
        risk
        + 5.0 * excess_risk
        + underestimation_penalty * underestimation
        + 5.0 * low_coverage
        + degenerate_penalty
        - coverage_reward * min(coverage, 1.0)
    )
    return float(score) if np.isfinite(score) else float("inf")


def compute_officehome_selection_score(
    *,
    mean_risk: float,
    mean_coverage: float,
    bound_underestimation_rate: float,
    global_risk: float | None = None,
    metadata_risk: float | None = None,
    risk_budget: float = 0.10,
    min_coverage: float = 0.50,
    underestimation_penalty: float = 4.0,
    low_coverage_penalty: float = 2.0,
    risk_reward_weight: float = 1.0,
) -> float:
    """Score OfficeHome bound-conservatism candidates; lower scores are better."""
    risk = max(0.0, _finite_or_default(mean_risk, default=1.0))
    coverage = max(0.0, _finite_or_default(mean_coverage, default=0.0))
    underestimation = max(0.0, _finite_or_default(bound_underestimation_rate, default=1.0))
    budget = max(0.0, _finite_or_default(risk_budget, default=0.1))
    required_coverage = max(0.0, _finite_or_default(min_coverage, default=0.5))

    reference_risks = [
        _finite_or_default(value, default=np.nan)
        for value in (global_risk, metadata_risk)
        if value is not None
    ]
    reference_risks = [value for value in reference_risks if np.isfinite(value)]
    reference_risk = min(reference_risks) if reference_risks else None
    risk_improvement = max(0.0, reference_risk - risk) if reference_risk is not None else 0.0
    risk_regression = max(0.0, risk - reference_risk) if reference_risk is not None else 0.0

    excess_budget_risk = max(0.0, risk - budget)
    low_coverage = max(0.0, required_coverage - coverage)
    degenerate_penalty = 10.0 if coverage <= 1e-12 else 0.0
    score = (
        risk
        + 2.0 * excess_budget_risk
        + underestimation_penalty * underestimation
        + low_coverage_penalty * low_coverage
        + 2.0 * risk_regression
        + degenerate_penalty
        - risk_reward_weight * risk_improvement
    )
    return float(score) if np.isfinite(score) else float("inf")


def summarize_sweep_results(
    result_rows: Sequence[Mapping[str, Any]],
    *,
    sort_by: str = "score",
) -> pd.DataFrame:
    """Convert sweep rows to a sorted DataFrame, robust to sparse row keys."""
    frame = pd.DataFrame([dict(row) for row in result_rows])
    if frame.empty:
        return frame
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, kind="mergesort", na_position="last")
    return frame.reset_index(drop=True)


def run_cifar_cwcrc_sweep(
    *,
    base_config: dict,
    output_dir: str | Path,
    sweep: Mapping[str, Sequence[Any]] | None = None,
    overwrite: bool = False,
    max_runs: int | None = None,
    dataset_name: str | None = None,
    run_experiment: Callable[..., Path] = run_cifar_c_experiment,
) -> Path:
    """Run a CIFAR CW-CRC sweep and write summary artifacts."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Sweep output directory exists and is not empty: {output_path}")
    ensure_dir(output_path)
    runs_root = ensure_dir(output_path / "runs")

    sweep_grid = dict(sweep or DEFAULT_CIFAR_CW_SWEEP)
    variants = make_config_variants(base_config, sweep_grid)
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive when provided")
        variants = variants[: int(max_runs)]

    rows: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    risk_budget = float(base_config.get("risk_control", {}).get("risk_budget", 0.1))
    for index, variant in enumerate(variants):
        run_name = f"cw_sweep_{index:03d}"
        variant.setdefault("paths", {})
        variant["paths"]["output_root"] = str(runs_root)
        variant["paths"]["run_name"] = run_name
        run_dir = run_experiment(variant, overwrite=True, run_name_override=run_name)
        method_frame = pd.read_csv(Path(run_dir) / "results" / "method_comparison.csv")
        params = _variant_parameters(variant, sweep_grid)
        variant_records.append({"index": index, "run_name": run_name, "params": params})
        for method in ("cw_crc", "ct_crc"):
            method_rows = method_frame[method_frame["method"] == method]
            if method_rows.empty:
                continue
            metric_row = method_rows.iloc[0].to_dict()
            row = {
                "variant_index": index,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "dataset": dataset_name or _dataset_from_config(base_config),
                "method": method,
                **params,
                **metric_row,
            }
            if method == "cw_crc":
                row["score"] = compute_selection_score(
                    mean_risk=float(row.get("mean_risk", np.nan)),
                    mean_coverage=float(row.get("mean_coverage", np.nan)),
                    bound_underestimation_rate=float(row.get("bound_underestimation_rate", np.nan)),
                    risk_budget=risk_budget,
                )
            rows.append(row)

    summary = summarize_sweep_results(rows, sort_by="score")
    summary.to_csv(output_path / "sweep_results.csv", index=False)
    _write_best_config(output_path / "best_config.yaml", variants, summary)
    _write_json(
        {
            "num_variants": len(variants),
            "sweep": {key: list(values) for key, values in sweep_grid.items()},
            "variants": variant_records,
            "dataset": dataset_name or _dataset_from_config(base_config),
        },
        output_path / "sweep_manifest.json",
    )
    return output_path


def _variant_parameters(config: dict, sweep: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    return {key: _get_nested(config, key) for key in sweep}


def _get_nested(config: dict, dotted_key: str) -> Any:
    cursor: Any = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _write_best_config(path: Path, variants: Sequence[dict], summary: pd.DataFrame) -> None:
    if summary.empty or "method" not in summary.columns:
        best = variants[0] if variants else {}
    else:
        candidates = summary[summary["method"] == "cw_crc"] if "method" in summary.columns else summary
        if candidates.empty:
            candidates = summary
        best_index = int(candidates.iloc[0].get("variant_index", 0))
        best = variants[best_index] if 0 <= best_index < len(variants) else {}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best, handle, sort_keys=False)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _dataset_from_config(config: dict) -> str:
    cache_path = str(config.get("paths", {}).get("cache_dir", "cifar"))
    if "cifar100" in cache_path.lower():
        return "cifar100c"
    if "cifar10" in cache_path.lower():
        return "cifar10c"
    return str(config.get("paths", {}).get("run_name", "cifar"))


def _finite_or_default(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
