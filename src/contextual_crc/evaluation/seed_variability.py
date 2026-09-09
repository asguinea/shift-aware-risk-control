from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.metrics import (
    bound_underestimation_rate,
    mean_bound_gap,
    safe_mean,
    safe_std,
    violation_rate,
)
from contextual_crc.utils.paths import ensure_dir


DEFAULT_SEEDS = (0, 1, 2)

METHOD_ORDER = (
    "global_crc",
    "metadata_stratified_crc",
    "ct_crc",
    "cw_crc",
    "oracle_context_crc",
)

METHOD_DISPLAY_NAMES = {
    "global_crc": "Global CRC",
    "metadata_stratified_crc": "Metadata-stratified",
    "ct_crc": "CT-CRC",
    "cw_crc": "CW-CRC",
    "oracle_context_crc": "Oracle context",
}

DATASET_DISPLAY_NAMES = {
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "synthetic_contextual_shift": "Synthetic",
    "synthetic_stronger_contextual_shift": "Synthetic",
    "officehome": "OfficeHome",
    "wilds_camelyon17": "Camelyon17-WILDS",
    "wilds_fmow": "FMoW-WILDS",
}

DATASET_ORDER = (
    "cifar10c",
    "cifar100c",
    "synthetic_stronger_contextual_shift",
    "wilds_camelyon17",
    "officehome",
    "wilds_fmow",
)

METRIC_COLUMNS = (
    "dataset",
    "dataset_display",
    "run_dir",
    "seed",
    "method",
    "method_display",
    "risk",
    "coverage",
    "violation_rate",
    "bound_underestimation",
    "mean_bound_gap",
    "n_windows",
    "n_examples",
    "notes",
)

SMOOTHNESS_COLUMNS = (
    "dataset",
    "dataset_display",
    "run_dir",
    "seed",
    "n_windows",
    "n_pairs",
    "spearman_rho",
    "pearson_r",
    "notes",
)


@dataclass(frozen=True)
class SeedVariabilityRunInputs:
    dataset: str
    run_dir: Path
    risk_budget: float
    base_windows: pd.DataFrame
    method_windows: dict[str, pd.DataFrame]
    smoothness_pairs: pd.DataFrame | None


def normalize_seed_list(values: Sequence[int | str] | None = None) -> list[int]:
    """Normalize CLI/config seed values, using the compact default when absent."""
    if values is None or len(values) == 0:
        return list(DEFAULT_SEEDS)
    seeds: list[int] = []
    for value in values:
        seed = int(value)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def resample_window_ids(window_ids: Sequence[str], seed: int, size: int | None = None) -> list[str]:
    """Bootstrap window ids reproducibly for one seed."""
    ids = [str(value) for value in window_ids]
    if not ids:
        return []
    sample_size = len(ids) if size is None else int(size)
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(np.asarray(ids, dtype=object), size=sample_size, replace=True)
    return [str(value) for value in sampled.tolist()]


def summarize_seed_metrics(
    metrics: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("dataset", "dataset_display", "method", "method_display"),
) -> pd.DataFrame:
    """Aggregate per-seed method metrics into mean/std appendix summaries."""
    if metrics.empty:
        return pd.DataFrame(columns=_metric_summary_columns(group_columns))
    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby(list(group_columns), sort=False, dropna=False)
    for keys, group in grouped:
        key_values = _as_tuple(keys)
        row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row.update(
            {
                "mean_risk": safe_mean(group["risk"]),
                "std_risk": safe_std(group["risk"]),
                "mean_coverage": safe_mean(group["coverage"]),
                "std_coverage": safe_std(group["coverage"]),
                "mean_violation_rate": safe_mean(group["violation_rate"]),
                "std_violation_rate": safe_std(group["violation_rate"]),
                "mean_bound_underestimation": safe_mean(group["bound_underestimation"]),
                "std_bound_underestimation": safe_std(group["bound_underestimation"]),
                "mean_bound_gap": safe_mean(group["mean_bound_gap"]),
                "std_bound_gap": safe_std(group["mean_bound_gap"]),
                "n_seeds": float(group["seed"].nunique()),
                "mean_n_windows": safe_mean(group["n_windows"]),
                "mean_n_examples": safe_mean(group["n_examples"]),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_dataset_rank"] = summary["dataset"].map(_dataset_rank) if "dataset" in summary.columns else 0
    summary["_method_rank"] = summary["method"].map(_method_rank) if "method" in summary.columns else 0
    return summary.sort_values(["_dataset_rank", "dataset", "_method_rank"], kind="mergesort").drop(
        columns=["_dataset_rank", "_method_rank"]
    ).reset_index(drop=True)


def summarize_seed_smoothness(
    smoothness: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("dataset", "dataset_display"),
) -> pd.DataFrame:
    """Aggregate per-seed smoothness correlations into mean/std summaries."""
    if smoothness.empty:
        return pd.DataFrame(columns=_smoothness_summary_columns(group_columns))
    rows: list[dict[str, Any]] = []
    grouped = smoothness.groupby(list(group_columns), sort=False, dropna=False)
    for keys, group in grouped:
        key_values = _as_tuple(keys)
        row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row.update(
            {
                "mean_smoothness_rho": safe_mean(group["spearman_rho"]),
                "std_smoothness_rho": safe_std(group["spearman_rho"]),
                "mean_pearson_r": safe_mean(group["pearson_r"]),
                "std_pearson_r": safe_std(group["pearson_r"]),
                "n_seeds": float(group["seed"].nunique()),
                "mean_n_windows": safe_mean(group["n_windows"]),
                "mean_n_pairs": safe_mean(group["n_pairs"]),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_dataset_rank"] = summary["dataset"].map(_dataset_rank) if "dataset" in summary.columns else 0
    return summary.sort_values(["_dataset_rank", "dataset"], kind="mergesort").drop(columns=["_dataset_rank"]).reset_index(
        drop=True
    )


def load_seed_variability_inputs(run_dir: str | Path) -> SeedVariabilityRunInputs:
    """Load saved outputs needed for the seeded window-bootstrap diagnostic."""
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "results" / "run_manifest.json")
    dataset = str(manifest.get("dataset") or manifest.get("schema_dataset") or run_path.name)
    risk_budget = _risk_budget_from_config(run_path)

    ct_windows = _load_method_window_file(run_path / "results" / "ct_crc_windows.csv", "ct_crc")
    cw_windows = _load_method_window_file(run_path / "results" / "cw_crc_windows.csv", "cw_crc")
    base_windows = _base_windows(ct_windows, cw_windows)
    method_windows: dict[str, pd.DataFrame] = {}
    if not ct_windows.empty:
        method_windows["ct_crc"] = ct_windows
    if not cw_windows.empty:
        method_windows["cw_crc"] = cw_windows

    context_specs = {
        "global_crc": run_path / "results" / "global_by_context.csv",
        "metadata_stratified_crc": run_path / "results" / "stratified_by_context.csv",
        "oracle_context_crc": run_path / "results" / "oracle_by_context.csv",
    }
    for method, path in context_specs.items():
        context_frame = _load_context_method_file(path, method)
        if not context_frame.empty and not base_windows.empty:
            method_windows[method] = _expand_context_metrics_to_windows(context_frame, base_windows, method)

    smoothness_pairs = _load_smoothness_pairs(run_path / "results" / "context_smoothness.csv")
    return SeedVariabilityRunInputs(
        dataset=dataset,
        run_dir=run_path,
        risk_budget=risk_budget,
        base_windows=base_windows,
        method_windows=method_windows,
        smoothness_pairs=smoothness_pairs,
    )


def compute_seed_variability_for_run(
    run_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-seed method and smoothness diagnostics for one saved run."""
    inputs = load_seed_variability_inputs(run_dir)
    return compute_seed_variability_from_inputs(inputs, seeds=seeds)


def compute_seed_variability_from_inputs(
    inputs: SeedVariabilityRunInputs,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seeded window-bootstrap metrics from already-loaded run inputs."""
    normalized_seeds = normalize_seed_list(seeds)
    window_ids = [str(value) for value in inputs.base_windows["window_id"].tolist()] if not inputs.base_windows.empty else []
    metric_rows: list[dict[str, Any]] = []
    smoothness_rows: list[dict[str, Any]] = []

    for seed in normalized_seeds:
        sampled_ids = resample_window_ids(window_ids, seed)
        for method in METHOD_ORDER:
            method_frame = inputs.method_windows.get(method)
            if method_frame is None or method_frame.empty:
                metric_rows.append(_unavailable_metric_row(inputs, seed, method, "method outputs unavailable"))
                continue
            sampled = _take_sampled_windows(method_frame, sampled_ids)
            metric_rows.append(_summarize_sampled_method(inputs, seed, method, sampled))
        smoothness_rows.append(_summarize_sampled_smoothness(inputs, seed, sampled_ids))

    metrics = pd.DataFrame(metric_rows)
    smoothness = pd.DataFrame(smoothness_rows)
    for column in METRIC_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    for column in SMOOTHNESS_COLUMNS:
        if column not in smoothness.columns:
            smoothness[column] = np.nan
    metrics = metrics.loc[:, list(METRIC_COLUMNS)]
    smoothness = smoothness.loc[:, list(SMOOTHNESS_COLUMNS)]
    metrics["_dataset_rank"] = metrics["dataset"].map(_dataset_rank)
    metrics["_method_rank"] = metrics["method"].map(_method_rank)
    metrics = metrics.sort_values(["_dataset_rank", "dataset", "seed", "_method_rank"], kind="mergesort").drop(
        columns=["_dataset_rank", "_method_rank"]
    )
    smoothness["_dataset_rank"] = smoothness["dataset"].map(_dataset_rank)
    smoothness = smoothness.sort_values(["_dataset_rank", "dataset", "seed"], kind="mergesort").drop(
        columns=["_dataset_rank"]
    )
    return metrics.reset_index(drop=True), smoothness.reset_index(drop=True)


def run_seed_variability(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Run seeded variability diagnostics over saved default experiment outputs."""
    output_path = ensure_dir(output_dir)
    metrics_path = output_path / "seed_variability_metrics.csv"
    smoothness_path = output_path / "seed_variability_smoothness.csv"
    if not overwrite and (metrics_path.exists() or smoothness_path.exists()):
        raise FileExistsError("seed variability outputs already exist; pass overwrite=True to replace them")

    metric_frames: list[pd.DataFrame] = []
    smoothness_frames: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, Any]] = []
    normalized_seeds = normalize_seed_list(seeds)
    for run_dir in run_dirs:
        metrics, smoothness = compute_seed_variability_for_run(run_dir, seeds=normalized_seeds)
        metric_frames.append(metrics)
        smoothness_frames.append(smoothness)
        dataset = str(metrics["dataset"].iloc[0]) if not metrics.empty else Path(run_dir).name
        manifest_runs.append(
            {
                "dataset": dataset,
                "run_dir": str(run_dir),
                "n_metric_rows": int(len(metrics)),
                "n_smoothness_rows": int(len(smoothness)),
            }
        )

    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(columns=METRIC_COLUMNS)
    smoothness_all = (
        pd.concat(smoothness_frames, ignore_index=True) if smoothness_frames else pd.DataFrame(columns=SMOOTHNESS_COLUMNS)
    )
    metrics_all.to_csv(metrics_path, index=False)
    smoothness_all.to_csv(smoothness_path, index=False)
    summarize_seed_metrics(metrics_all).to_csv(output_path / "seed_variability_metrics_summary.csv", index=False)
    summarize_seed_smoothness(smoothness_all).to_csv(output_path / "seed_variability_smoothness_summary.csv", index=False)
    _write_json(
        {
            "run_dirs": [str(path) for path in run_dirs],
            "seeds": normalized_seeds,
            "metrics_csv": str(metrics_path),
            "smoothness_csv": str(smoothness_path),
            "runs": manifest_runs,
            "notes": "Seed variability uses a seeded window bootstrap from saved outputs; it does not rerun model inference or recalibrate methods.",
        },
        output_path / "seed_variability_manifest.json",
    )
    return metrics_path, smoothness_path


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset).replace("_", " "))


def display_method(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method), str(method).replace("_", " "))


def _summarize_sampled_method(
    inputs: SeedVariabilityRunInputs,
    seed: int,
    method: str,
    sampled: pd.DataFrame,
) -> dict[str, Any]:
    if sampled.empty:
        return _unavailable_metric_row(inputs, seed, method, "no sampled windows with finite method outputs")
    risk = pd.to_numeric(sampled["risk"], errors="coerce")
    coverage = pd.to_numeric(sampled["coverage"], errors="coerce")
    bounds = pd.to_numeric(sampled["upper_bound"], errors="coerce") if "upper_bound" in sampled.columns else None
    n_examples = pd.to_numeric(sampled.get("n_examples"), errors="coerce") if "n_examples" in sampled.columns else None
    finite_risk = risk[np.isfinite(risk)]
    finite_coverage = coverage[np.isfinite(coverage)]
    return {
        "dataset": inputs.dataset,
        "dataset_display": display_dataset(inputs.dataset),
        "run_dir": str(inputs.run_dir),
        "seed": int(seed),
        "method": method,
        "method_display": display_method(method),
        "risk": safe_mean(finite_risk),
        "coverage": safe_mean(finite_coverage),
        "violation_rate": violation_rate(finite_risk, inputs.risk_budget),
        "bound_underestimation": bound_underestimation_rate(risk, bounds) if bounds is not None else float("nan"),
        "mean_bound_gap": mean_bound_gap(risk, bounds) if bounds is not None else float("nan"),
        "n_windows": float(len(finite_risk)),
        "n_examples": _safe_sum(n_examples) if n_examples is not None else float("nan"),
        "notes": "seeded window bootstrap from saved outputs; thresholds/bounds are not recalibrated",
    }


def _summarize_sampled_smoothness(
    inputs: SeedVariabilityRunInputs,
    seed: int,
    sampled_ids: Sequence[str],
) -> dict[str, Any]:
    base = {
        "dataset": inputs.dataset,
        "dataset_display": display_dataset(inputs.dataset),
        "run_dir": str(inputs.run_dir),
        "seed": int(seed),
    }
    if inputs.smoothness_pairs is None or inputs.smoothness_pairs.empty:
        return {
            **base,
            "n_windows": 0.0,
            "n_pairs": 0.0,
            "spearman_rho": float("nan"),
            "pearson_r": float("nan"),
            "notes": "context_smoothness.csv unavailable",
        }
    unique_ids = {str(value) for value in sampled_ids}
    pairs = inputs.smoothness_pairs[
        inputs.smoothness_pairs["window_id_i"].astype(str).isin(unique_ids)
        & inputs.smoothness_pairs["window_id_j"].astype(str).isin(unique_ids)
    ].copy()
    distances = pd.to_numeric(pairs.get("distance"), errors="coerce")
    risk_diffs = pd.to_numeric(pairs.get("risk_difference"), errors="coerce")
    finite = np.isfinite(distances) & np.isfinite(risk_diffs)
    distances = distances[finite]
    risk_diffs = risk_diffs[finite]
    return {
        **base,
        "n_windows": float(len(unique_ids)),
        "n_pairs": float(len(distances)),
        "spearman_rho": _spearman(distances, risk_diffs),
        "pearson_r": _pearson(distances, risk_diffs),
        "notes": "seeded window bootstrap from saved pairwise smoothness diagnostics",
    }


def _unavailable_metric_row(
    inputs: SeedVariabilityRunInputs,
    seed: int,
    method: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "dataset": inputs.dataset,
        "dataset_display": display_dataset(inputs.dataset),
        "run_dir": str(inputs.run_dir),
        "seed": int(seed),
        "method": method,
        "method_display": display_method(method),
        "risk": float("nan"),
        "coverage": float("nan"),
        "violation_rate": float("nan"),
        "bound_underestimation": float("nan"),
        "mean_bound_gap": float("nan"),
        "n_windows": 0.0,
        "n_examples": float("nan"),
        "notes": notes,
    }


def _load_method_window_file(path: Path, method: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if "window_id" not in frame.columns:
        return pd.DataFrame()
    risk_col = "realized_risk" if "realized_risk" in frame.columns else "mean_loss"
    if risk_col not in frame.columns or "coverage" not in frame.columns:
        return pd.DataFrame()
    bound_col = "weighted_risk_upper_bound" if method == "cw_crc" else "transferred_upper_bound"
    output = pd.DataFrame(
        {
            "window_id": frame["window_id"].astype(str),
            "context_id": frame.get("context_id", pd.Series(["unknown"] * len(frame))).astype(str),
            "risk": pd.to_numeric(frame[risk_col], errors="coerce"),
            "coverage": pd.to_numeric(frame["coverage"], errors="coerce"),
            "n_examples": _numeric_or_nan(frame, "n_examples"),
            "upper_bound": _numeric_or_nan(frame, bound_col),
        }
    )
    return output


def _load_context_method_file(path: Path, method: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if "context_id" not in frame.columns:
        return pd.DataFrame()
    risk_col = "mean_loss"
    if risk_col not in frame.columns or "coverage" not in frame.columns:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "context_id": frame["context_id"].astype(str),
            "risk": pd.to_numeric(frame[risk_col], errors="coerce"),
            "coverage": pd.to_numeric(frame["coverage"], errors="coerce"),
            "upper_bound": _numeric_or_nan(frame, "risk_upper_bound"),
            "source_method": method,
        }
    )


def _expand_context_metrics_to_windows(
    context_frame: pd.DataFrame,
    base_windows: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if context_frame.empty or base_windows.empty:
        return pd.DataFrame()
    merged = base_windows[["window_id", "context_id", "n_examples"]].merge(
        context_frame[["context_id", "risk", "coverage", "upper_bound"]],
        on="context_id",
        how="left",
    )
    merged["method"] = method
    return merged


def _base_windows(ct_windows: pd.DataFrame, cw_windows: pd.DataFrame) -> pd.DataFrame:
    base = ct_windows if not ct_windows.empty else cw_windows
    if base.empty:
        return pd.DataFrame(columns=["window_id", "context_id", "n_examples"])
    columns = ["window_id", "context_id", "n_examples"]
    for column in columns:
        if column not in base.columns:
            base[column] = np.nan
    return base[columns].copy().reset_index(drop=True)


def _take_sampled_windows(frame: pd.DataFrame, sampled_ids: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not sampled_ids:
        return pd.DataFrame(columns=frame.columns)
    keyed = frame.set_index("window_id", drop=False)
    rows = []
    for window_id in sampled_ids:
        if window_id not in keyed.index:
            continue
        row = keyed.loc[window_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_smoothness_pairs(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None
    required = {"window_id_i", "window_id_j", "distance", "risk_difference"}
    if not required.issubset(set(frame.columns)):
        return None
    return frame.loc[:, ["window_id_i", "window_id_j", "distance", "risk_difference"]].copy()


def _risk_budget_from_config(run_path: Path) -> float:
    config_path = run_path / "configs" / "resolved_config.yaml"
    if not config_path.is_file():
        return 0.10
    try:
        config = load_config(config_path)
    except (OSError, ValueError):
        return 0.10
    try:
        return float(config.get("risk_control", {}).get("risk_budget", 0.10))
    except (TypeError, ValueError):
        return 0.10


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_sum(values: Any) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.sum(numeric)) if numeric.size else float("nan")


def _numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(x_rank, y_rank)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_array = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("correlation inputs must have matching lengths")
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if x_array.size < 2 or float(np.std(x_array)) == 0.0 or float(np.std(y_array)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _metric_summary_columns(group_columns: Sequence[str]) -> list[str]:
    return [
        *group_columns,
        "mean_risk",
        "std_risk",
        "mean_coverage",
        "std_coverage",
        "mean_violation_rate",
        "std_violation_rate",
        "mean_bound_underestimation",
        "std_bound_underestimation",
        "mean_bound_gap",
        "std_bound_gap",
        "n_seeds",
        "mean_n_windows",
        "mean_n_examples",
    ]


def _smoothness_summary_columns(group_columns: Sequence[str]) -> list[str]:
    return [
        *group_columns,
        "mean_smoothness_rho",
        "std_smoothness_rho",
        "mean_pearson_r",
        "std_pearson_r",
        "n_seeds",
        "mean_n_windows",
        "mean_n_pairs",
    ]


def _method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(str(method))
    except ValueError:
        return len(METHOD_ORDER)


def _dataset_rank(dataset: str) -> int:
    try:
        return DATASET_ORDER.index(str(dataset))
    except ValueError:
        return len(DATASET_ORDER)
