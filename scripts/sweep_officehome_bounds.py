#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.officehome_experiment import run_officehome_experiment
from contextual_crc.evaluation.tuning import (
    compute_officehome_selection_score,
    make_config_variants,
)
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.paths import ensure_dir


DEFAULT_CT_SWEEP: dict[str, Sequence[Any]] = {
    "ct_crc.bound_margin": [0.03, 0.05, 0.08, 0.10, 0.15],
    "ct_crc.bound_multiplier": [1.25, 1.5, 2.0, 3.0],
    "ct_crc.l_hat_quantile": [0.95, 1.0],
}

DEFAULT_CW_SWEEP: dict[str, Sequence[Any]] = {
    "cw_crc.bound_margin": [0.03, 0.05, 0.08, 0.10, 0.15],
    "cw_crc.bound_multiplier": [1.0, 1.25, 1.5, 2.0],
    "cw_crc.target_effective_sample_size": [4, 6, 8, 12],
}

PARAMETER_COLUMNS = (
    "ct_crc.bound_margin",
    "ct_crc.bound_multiplier",
    "ct_crc.l_hat_quantile",
    "cw_crc.bound_margin",
    "cw_crc.bound_multiplier",
    "cw_crc.target_effective_sample_size",
)

METHOD_ROWS = ("global_crc", "metadata_stratified_context", "ct_crc", "cw_crc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep OfficeHome CT/CW bound-conservatism settings.")
    parser.add_argument("--config", required=True, help="Path to OfficeHome experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing sweep directory.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional deterministic cap on sweep variants.")
    parser.add_argument(
        "--method",
        choices=("ct_crc", "cw_crc", "both"),
        default="both",
        help="Which method family to sweep.",
    )
    parser.add_argument("--risk-budget", type=float, default=None, help="Optional risk budget override.")
    parser.add_argument("--min-coverage", type=float, default=0.50, help="Minimum useful coverage for scoring.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    output_dir = run_officehome_bound_sweep(
        base_config=config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_runs=args.max_runs,
        method=args.method,
        risk_budget=args.risk_budget,
        min_coverage=args.min_coverage,
        config_path=args.config,
    )
    print(f"OfficeHome bound sweep written to: {output_dir}")
    return 0


def run_officehome_bound_sweep(
    *,
    base_config: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
    max_runs: int | None = None,
    method: str = "both",
    risk_budget: float | None = None,
    min_coverage: float = 0.50,
    config_path: str | Path | None = None,
) -> Path:
    """Run separate OfficeHome CT/CW bound sweeps and write summary files."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Sweep output directory exists and is not empty: {output_path}")
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    ensure_dir(output_path)
    runs_root = ensure_dir(output_path / "runs")

    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive when provided")
    if method not in {"ct_crc", "cw_crc", "both"}:
        raise ValueError("method must be one of: ct_crc, cw_crc, both")

    resolved_base = deepcopy(base_config)
    if risk_budget is not None:
        resolved_base.setdefault("risk_control", {})["risk_budget"] = float(risk_budget)
    scoring_risk_budget = float(resolved_base.get("risk_control", {}).get("risk_budget", 0.10))

    variants = _build_sweep_variants(resolved_base, method=method)
    if max_runs is not None:
        variants = variants[: int(max_runs)]

    rows: list[dict[str, Any]] = []
    variant_manifest: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        sweep_method = variant["sweep_method"]
        variant_config = variant["config"]
        run_name = f"officehome_sweep_{variant_index:03d}"
        run_config = deepcopy(variant_config)
        run_config.setdefault("paths", {})
        run_config["paths"]["output_root"] = str(runs_root)
        run_config["paths"]["run_name"] = run_name
        run_dir = run_officehome_experiment(run_config, overwrite=True, run_name_override=run_name)
        method_frame = pd.read_csv(Path(run_dir) / "results" / "method_comparison.csv")
        global_risk = _metric_for_method(method_frame, "global_crc", "mean_risk")
        metadata_risk = _metric_for_method(method_frame, "metadata_stratified_context", "mean_risk")
        params = _parameter_values(variant_config)

        variant_manifest.append(
            {
                "variant_index": variant_index,
                "sweep_method": sweep_method,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "parameters": params,
            }
        )
        rows.extend(
            _rows_for_variant(
                variant_index=variant_index,
                sweep_method=sweep_method,
                run_name=run_name,
                run_dir=Path(run_dir),
                method_frame=method_frame,
                params=params,
                global_risk=global_risk,
                metadata_risk=metadata_risk,
                risk_budget=scoring_risk_budget,
                min_coverage=min_coverage,
            )
        )

    results = _summarize_officehome_sweep(rows)
    results.to_csv(output_path / "sweep_results.csv", index=False)
    _write_best_config(output_path / "best_ct_config.yaml", variants, results, sweep_method="ct_crc")
    _write_best_config(output_path / "best_cw_config.yaml", variants, results, sweep_method="cw_crc")
    _write_json(
        {
            "config_path": str(config_path) if config_path is not None else None,
            "output_dir": str(output_path),
            "method": method,
            "max_runs": max_runs,
            "num_variants": len(variants),
            "risk_budget": scoring_risk_budget,
            "min_coverage": float(min_coverage),
            "ct_sweep": {key: list(values) for key, values in DEFAULT_CT_SWEEP.items()},
            "cw_sweep": {key: list(values) for key, values in DEFAULT_CW_SWEEP.items()},
            "variants": variant_manifest,
        },
        output_path / "sweep_manifest.json",
    )
    return output_path


def _build_sweep_variants(base_config: dict[str, Any], *, method: str) -> list[dict[str, Any]]:
    ct_variants = [
        {"sweep_method": "ct_crc", "config": config}
        for config in make_config_variants(base_config, DEFAULT_CT_SWEEP)
    ]
    cw_variants = [
        {"sweep_method": "cw_crc", "config": config}
        for config in make_config_variants(base_config, DEFAULT_CW_SWEEP)
    ]
    if method == "ct_crc":
        return ct_variants
    if method == "cw_crc":
        return cw_variants

    variants: list[dict[str, Any]] = []
    max_len = max(len(ct_variants), len(cw_variants))
    for index in range(max_len):
        if index < len(ct_variants):
            variants.append(ct_variants[index])
        if index < len(cw_variants):
            variants.append(cw_variants[index])
    return variants


def _rows_for_variant(
    *,
    variant_index: int,
    sweep_method: str,
    run_name: str,
    run_dir: Path,
    method_frame: pd.DataFrame,
    params: dict[str, Any],
    global_risk: float | None,
    metadata_risk: float | None,
    risk_budget: float,
    min_coverage: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ROWS:
        method_rows = method_frame[method_frame["method"] == method]
        if method_rows.empty:
            continue
        metrics = method_rows.iloc[0].to_dict()
        score = np.nan
        if method == sweep_method:
            score = compute_officehome_selection_score(
                mean_risk=_float_or_nan(metrics.get("mean_risk")),
                mean_coverage=_float_or_nan(metrics.get("mean_coverage")),
                bound_underestimation_rate=_float_or_nan(metrics.get("bound_underestimation_rate")),
                global_risk=global_risk,
                metadata_risk=metadata_risk,
                risk_budget=risk_budget,
                min_coverage=min_coverage,
            )
        rows.append(
            {
                "variant_index": variant_index,
                "sweep_method": sweep_method,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "method": method,
                **params,
                "mean_risk": metrics.get("mean_risk", np.nan),
                "mean_coverage": metrics.get("mean_coverage", np.nan),
                "violation_rate": metrics.get("violation_rate", np.nan),
                "mean_bound": metrics.get("mean_bound", np.nan),
                "bound_underestimation_rate": metrics.get("bound_underestimation_rate", np.nan),
                "global_risk": global_risk,
                "metadata_risk": metadata_risk,
                "score": score,
            }
        )
    return rows


def _summarize_officehome_sweep(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    sort_columns = [column for column in ("score", "variant_index", "method") if column in frame.columns]
    return frame.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)


def _write_best_config(
    path: Path,
    variants: Sequence[dict[str, Any]],
    results: pd.DataFrame,
    *,
    sweep_method: str,
) -> None:
    if results.empty or "score" not in results.columns:
        return
    candidates = results[
        (results["sweep_method"] == sweep_method)
        & (results["method"] == sweep_method)
        & np.isfinite(pd.to_numeric(results["score"], errors="coerce"))
    ]
    if candidates.empty:
        return
    best_index = int(candidates.iloc[0]["variant_index"])
    for variant_index, variant in enumerate(variants):
        if variant_index == best_index:
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(variant["config"], handle, sort_keys=False)
            return


def _parameter_values(config: dict[str, Any]) -> dict[str, Any]:
    return {column: _get_nested(config, column) for column in PARAMETER_COLUMNS}


def _get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _metric_for_method(frame: pd.DataFrame, method: str, column: str) -> float | None:
    rows = frame[frame["method"] == method]
    if rows.empty or column not in rows.columns:
        return None
    value = _float_or_nan(rows.iloc[0][column])
    return value if np.isfinite(value) else None


def _float_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


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


if __name__ == "__main__":
    raise SystemExit(main())
