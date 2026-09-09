from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_cache_root = Path(tempfile.gettempdir()) / "contextual_crc_matplotlib"
(_cache_root / "mpl").mkdir(parents=True, exist_ok=True)
(_cache_root / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contextual_crc.artifacts.make_tables import dataframe_to_latex_tabular
from contextual_crc.cache.io import cache_exists, load_prediction_cache
from contextual_crc.config.loader import load_config
from contextual_crc.risk_control.policies import candidate_thresholds
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


RISK_BUDGET = 0.10

FRONTIER_COLUMNS = (
    "dataset",
    "run_name",
    "method",
    "threshold",
    "coverage",
    "accepted_error_risk",
    "n_units",
    "n_accepted",
    "n_accepted_errors",
    "selected_operating_point",
    "unit",
    "reconstruction_status",
)

MATCHED_COVERAGE_COLUMNS = (
    "dataset",
    "method",
    "target_coverage",
    "achieved_coverage",
    "interpolated_or_nearest",
    "risk_at_coverage",
    "delta_risk_vs_global_at_same_coverage",
    "delta_risk_vs_metadata_at_same_coverage",
)

MATCHED_RISK_COLUMNS = (
    "dataset",
    "method",
    "target_risk",
    "achieved_risk",
    "coverage_at_risk",
    "delta_coverage_vs_global_at_same_risk",
    "delta_coverage_vs_metadata_at_same_risk",
)

METHOD_ORDER = (
    "global_crc",
    "metadata_stratified",
    "ct_crc",
    "cw_crc",
    "oracle_context_crc",
)

METHOD_DISPLAY_NAMES = {
    "global_crc": "Global CRC",
    "metadata_stratified": "Metadata-stratified",
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

SELECTED_TARGET_METHODS = (
    "global_crc",
    "metadata_stratified",
    "ct_crc",
    "cw_crc",
)

FIXED_COVERAGE_TARGETS = (0.10, 0.25, 0.50, 0.75)


@dataclass(frozen=True)
class MethodMargins:
    method: str
    margins: np.ndarray
    errors: np.ndarray
    selected_available: bool = True
    selected_coverage: float | None = None
    selected_risk: float | None = None
    unit: str = "window_example"
    status: str = "reconstructed"


def compute_risk_coverage_curve(
    scores: Sequence[float],
    incorrect: Sequence[bool | int | float],
    *,
    selected_threshold: float | None = None,
    accept_if: str = "greater_equal",
) -> pd.DataFrame:
    """Compute accepted-error risk and coverage while sweeping score thresholds."""
    score_array = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    incorrect_array = pd.to_numeric(pd.Series(incorrect), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(score_array) & np.isfinite(incorrect_array)
    score_array = score_array[mask]
    incorrect_array = incorrect_array[mask] > 0.0
    if score_array.size == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "coverage",
                "accepted_error_risk",
                "n_units",
                "n_accepted",
                "n_accepted_errors",
                "selected_operating_point",
            ]
        )

    thresholds = candidate_thresholds(score_array, accept_if=accept_if, include_extremes=True)
    if selected_threshold is not None and np.isfinite(float(selected_threshold)):
        thresholds = np.unique(np.concatenate([thresholds, np.asarray([float(selected_threshold)])]))
        if accept_if == "less_equal":
            thresholds = thresholds[::-1]

    rows: list[dict[str, Any]] = []
    n_units = float(score_array.size)
    for threshold in thresholds:
        if accept_if == "greater_equal":
            accepted = score_array >= float(threshold)
        elif accept_if == "less_equal":
            accepted = score_array <= float(threshold)
        else:
            raise ValueError("accept_if must be 'greater_equal' or 'less_equal'")
        accepted_errors = accepted & incorrect_array
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(np.mean(accepted)),
                "accepted_error_risk": float(np.mean(accepted_errors)),
                "n_units": n_units,
                "n_accepted": float(np.sum(accepted)),
                "n_accepted_errors": float(np.sum(accepted_errors)),
                "selected_operating_point": bool(
                    selected_threshold is not None and float(threshold) == float(selected_threshold)
                ),
            }
        )

    curve = pd.DataFrame(rows)
    return _deduplicate_curve(curve)


def matched_coverage_point(curve: pd.DataFrame, target_coverage: float) -> dict[str, Any] | None:
    """Return risk at a requested coverage using interpolation when possible."""
    prepared = _prepare_curve_for_matching(curve, x_col="coverage", y_col="accepted_error_risk")
    if prepared.empty or not np.isfinite(float(target_coverage)):
        return None

    x = prepared["coverage"].to_numpy(dtype=float)
    y = prepared["accepted_error_risk"].to_numpy(dtype=float)
    target = float(target_coverage)
    exact = np.flatnonzero(np.isclose(x, target, atol=1e-12, rtol=0.0))
    if exact.size:
        idx = int(exact[0])
        return {
            "achieved_coverage": float(x[idx]),
            "risk_at_coverage": float(y[idx]),
            "interpolated_or_nearest": "exact",
        }
    if x[0] <= target <= x[-1] and len(x) >= 2:
        return {
            "achieved_coverage": target,
            "risk_at_coverage": float(np.interp(target, x, y)),
            "interpolated_or_nearest": "interpolated",
        }
    idx = int(np.argmin(np.abs(x - target)))
    return {
        "achieved_coverage": float(x[idx]),
        "risk_at_coverage": float(y[idx]),
        "interpolated_or_nearest": "nearest",
    }


def matched_risk_point(curve: pd.DataFrame, target_risk: float) -> dict[str, Any] | None:
    """Return coverage at a requested accepted-error risk."""
    prepared = _prepare_curve_for_matching(curve, x_col="accepted_error_risk", y_col="coverage")
    if prepared.empty or not np.isfinite(float(target_risk)):
        return None

    risks = prepared["accepted_error_risk"].to_numpy(dtype=float)
    coverage = prepared["coverage"].to_numpy(dtype=float)
    target = float(target_risk)
    exact = np.flatnonzero(np.isclose(risks, target, atol=1e-12, rtol=0.0))
    if exact.size:
        idx = int(exact[-1])
        return {
            "achieved_risk": float(risks[idx]),
            "coverage_at_risk": float(coverage[idx]),
            "interpolated_or_nearest": "exact",
        }
    if risks[0] <= target <= risks[-1] and len(risks) >= 2:
        return {
            "achieved_risk": target,
            "coverage_at_risk": float(np.interp(target, risks, coverage)),
            "interpolated_or_nearest": "interpolated",
        }
    idx = int(np.argmin(np.abs(risks - target)))
    return {
        "achieved_risk": float(risks[idx]),
        "coverage_at_risk": float(coverage[idx]),
        "interpolated_or_nearest": "nearest",
    }


def build_risk_coverage_frontiers(run_outputs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Reconstruct risk-coverage frontiers for all runs with enough cached detail."""
    frames: list[pd.DataFrame] = []
    for output in run_outputs:
        run_frame = _build_run_frontiers(output)
        if not run_frame.empty:
            frames.append(run_frame)
    if not frames:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[:, list(FRONTIER_COLUMNS)].sort_values(
        ["dataset", "method", "coverage", "threshold"],
        kind="mergesort",
    ).reset_index(drop=True)


def make_matched_coverage_table(frontiers: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all methods at shared per-dataset coverage targets."""
    if frontiers.empty:
        return pd.DataFrame(columns=MATCHED_COVERAGE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for dataset in _ordered_values(frontiers["dataset"], preferred=DATASET_ORDER):
        dataset_frame = frontiers[frontiers["dataset"] == dataset]
        curves = _curves_by_method(dataset_frame)
        targets = _coverage_targets(curves)
        for target in targets:
            global_point = matched_coverage_point(curves.get("global_crc", pd.DataFrame()), target)
            metadata_point = matched_coverage_point(curves.get("metadata_stratified", pd.DataFrame()), target)
            for method in _ordered_methods(curves):
                point = matched_coverage_point(curves[method], target)
                if point is None:
                    continue
                risk = point["risk_at_coverage"]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "target_coverage": float(target),
                        "achieved_coverage": point["achieved_coverage"],
                        "interpolated_or_nearest": point["interpolated_or_nearest"],
                        "risk_at_coverage": risk,
                        "delta_risk_vs_global_at_same_coverage": _delta(
                            risk,
                            None if global_point is None else global_point["risk_at_coverage"],
                        ),
                        "delta_risk_vs_metadata_at_same_coverage": _delta(
                            risk,
                            None if metadata_point is None else metadata_point["risk_at_coverage"],
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=MATCHED_COVERAGE_COLUMNS)
    return pd.DataFrame(rows).loc[:, list(MATCHED_COVERAGE_COLUMNS)].reset_index(drop=True)


def make_matched_risk_table(frontiers: pd.DataFrame) -> pd.DataFrame:
    """Evaluate coverage for all methods at shared per-dataset risk targets."""
    if frontiers.empty:
        return pd.DataFrame(columns=MATCHED_RISK_COLUMNS)

    rows: list[dict[str, Any]] = []
    for dataset in _ordered_values(frontiers["dataset"], preferred=DATASET_ORDER):
        dataset_frame = frontiers[frontiers["dataset"] == dataset]
        curves = _curves_by_method(dataset_frame)
        targets = _risk_targets(curves)
        for target in targets:
            global_point = matched_risk_point(curves.get("global_crc", pd.DataFrame()), target)
            metadata_point = matched_risk_point(curves.get("metadata_stratified", pd.DataFrame()), target)
            for method in _ordered_methods(curves):
                point = matched_risk_point(curves[method], target)
                if point is None:
                    continue
                coverage = point["coverage_at_risk"]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "target_risk": float(target),
                        "achieved_risk": point["achieved_risk"],
                        "coverage_at_risk": coverage,
                        "delta_coverage_vs_global_at_same_risk": _delta(
                            coverage,
                            None if global_point is None else global_point["coverage_at_risk"],
                        ),
                        "delta_coverage_vs_metadata_at_same_risk": _delta(
                            coverage,
                            None if metadata_point is None else metadata_point["coverage_at_risk"],
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=MATCHED_RISK_COLUMNS)
    return pd.DataFrame(rows).loc[:, list(MATCHED_RISK_COLUMNS)].reset_index(drop=True)


def write_risk_coverage_artifacts(
    run_outputs: Sequence[Mapping[str, Any]],
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
    *,
    figure_format: str = "pdf",
    precision: int = 3,
) -> dict[str, str]:
    """Write frontier, matched-coverage, matched-risk tables and the PDF figure."""
    tables_dir = ensure_dir(output_tables_dir)
    figures_dir = ensure_dir(output_figures_dir)
    frontiers = build_risk_coverage_frontiers(run_outputs)
    matched_coverage = make_matched_coverage_table(frontiers)
    matched_risk = make_matched_risk_table(frontiers)

    frontier_csv = tables_dir / "risk_coverage_frontiers.csv"
    matched_coverage_csv = tables_dir / "matched_coverage.csv"
    matched_coverage_tex = tables_dir / "matched_coverage.tex"
    matched_risk_csv = tables_dir / "matched_risk.csv"
    matched_risk_tex = tables_dir / "matched_risk.tex"
    figure_path = figures_dir / f"risk_coverage_frontiers.{figure_format.lstrip('.')}"

    frontiers.to_csv(frontier_csv, index=False)
    matched_coverage.to_csv(matched_coverage_csv, index=False)
    matched_risk.to_csv(matched_risk_csv, index=False)
    matched_coverage_tex.write_text(
        dataframe_to_latex_tabular(
            matched_coverage,
            caption="Matched-Coverage Risk",
            label="tab:matched-coverage",
            precision=precision,
        ),
        encoding="utf-8",
    )
    matched_risk_tex.write_text(
        dataframe_to_latex_tabular(
            matched_risk,
            caption="Matched-Risk Coverage",
            label="tab:matched-risk",
            precision=precision,
        ),
        encoding="utf-8",
    )
    plot_risk_coverage_frontiers(frontiers, figure_path)

    return {
        "risk_coverage_frontiers_csv": str(frontier_csv),
        "matched_coverage_csv": str(matched_coverage_csv),
        "matched_coverage_tex": str(matched_coverage_tex),
        "matched_risk_csv": str(matched_risk_csv),
        "matched_risk_tex": str(matched_risk_tex),
        "risk_coverage_frontiers": str(figure_path),
    }


def plot_risk_coverage_frontiers(frontiers: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot one risk-coverage panel per dataset."""
    output = Path(output_path)
    ensure_dir(output.parent)
    if frontiers.empty:
        return _placeholder(output, "No reconstructable risk-coverage frontier data")

    datasets = _ordered_values(frontiers["dataset"], preferred=DATASET_ORDER)
    n_rows, n_cols = _panel_grid(len(datasets))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 3.6 * n_cols), max(3.4, 3.0 * n_rows)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    for panel_index, dataset in enumerate(datasets):
        row_index, col_index = divmod(panel_index, n_cols)
        ax = axes[row_index, col_index]
        dataset_frame = frontiers[frontiers["dataset"] == dataset]
        for method in _ordered_methods(_curves_by_method(dataset_frame)):
            method_frame = dataset_frame[dataset_frame["method"] == method]
            curve = _plot_curve(method_frame)
            if curve.empty:
                continue
            label = METHOD_DISPLAY_NAMES.get(method, method.replace("_", " "))
            (line,) = ax.plot(
                curve["coverage"],
                curve["accepted_error_risk"],
                linewidth=1.7,
                label=label,
            )
            legend_handles.setdefault(label, line)
            selected = method_frame[method_frame["selected_operating_point"].astype(bool)]
            if not selected.empty:
                selected_point = selected.iloc[0]
                ax.scatter(
                    [selected_point["coverage"]],
                    [selected_point["accepted_error_risk"]],
                    s=28,
                    marker="o",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=4,
                    color=line.get_color(),
                )
        ax.axhline(RISK_BUDGET, linestyle="--", linewidth=1.0, alpha=0.7, color="0.35")
        ax.set_title(_display_dataset(dataset), fontsize=11)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Accepted-error risk")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=-0.01)
        ax.grid(alpha=0.3)

    for empty_index in range(len(datasets), n_rows * n_cols):
        row_index, col_index = divmod(empty_index, n_cols)
        axes[row_index, col_index].set_axis_off()
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(5, max(1, len(legend_handles))),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.16, hspace=0.34, wspace=0.26)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _build_run_frontiers(output: Mapping[str, Any]) -> pd.DataFrame:
    run_dir = Path(output.get("run_dir", ""))
    run_name = str(output.get("run_name") or run_dir.name or "unknown")
    dataset = str(output.get("dataset") or run_name)
    details = _load_run_details(run_dir)
    if details is None:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)

    examples, windows, config = details
    occurrences = _evaluation_occurrences(examples, windows, config)
    if occurrences.empty:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)

    methods = _method_margins(run_dir, occurrences)
    frames: list[pd.DataFrame] = []
    for method in methods:
        curve = compute_risk_coverage_curve(
            method.margins,
            method.errors,
            selected_threshold=0.0 if method.selected_available else None,
            accept_if="greater_equal",
        )
        if curve.empty:
            continue
        curve = _mark_selected_from_outputs(
            curve,
            selected_coverage=method.selected_coverage,
            selected_risk=method.selected_risk,
        )
        curve["dataset"] = dataset
        curve["run_name"] = run_name
        curve["method"] = method.method
        curve["unit"] = method.unit
        curve["reconstruction_status"] = method.status
        frames.append(curve)
    if not frames:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[:, list(FRONTIER_COLUMNS)]


def _load_run_details(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    config_path = run_dir / "configs" / "resolved_config.yaml"
    windows_path = run_dir / "intermediate" / "windows.parquet"
    if not config_path.is_file() or not windows_path.is_file():
        return None
    try:
        config = load_config(config_path)
    except (OSError, ValueError):
        return None
    cache_dir = _cache_dir(config, run_dir)
    if not cache_exists(cache_dir):
        return None
    try:
        examples, _, _ = load_prediction_cache(cache_dir)
        windows = load_windows(windows_path)
    except (OSError, ValueError, ImportError):
        return None
    return examples, windows, config


def _cache_dir(config: Mapping[str, Any], run_dir: Path) -> Path:
    raw = str(config.get("paths", {}).get("cache_dir", ""))
    if raw:
        path = Path(raw)
        if path.is_absolute():
            return path
        repo_candidate = Path.cwd() / path
        if cache_exists(repo_candidate):
            return repo_candidate
        run_candidate = run_dir / path
        if cache_exists(run_candidate):
            return run_candidate
        return repo_candidate
    manifest_path = run_dir / "results" / "run_manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict) and manifest.get("cache_dir"):
            return Path(str(manifest["cache_dir"]))
    return run_dir / "cache"


def _evaluation_occurrences(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if windows.empty or "example_indices" not in windows.columns:
        return pd.DataFrame()
    evaluation_contexts = _evaluation_contexts(config, examples)
    evaluation_windows = windows[windows["context_id"].astype(str).isin(evaluation_contexts)].copy()
    if evaluation_windows.empty:
        return pd.DataFrame()

    risk_config = config.get("risk_control", {})
    score_col = str(risk_config.get("score_column", "confidence"))
    required = {score_col, "label", "pred", "context_id"}
    if not required.issubset(examples.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, window in evaluation_windows.iterrows():
        for index in window["example_indices"]:
            example_index = int(index)
            if example_index < 0 or example_index >= len(examples):
                continue
            example = examples.iloc[example_index]
            rows.append(
                {
                    "window_id": window["window_id"],
                    "window_context_id": str(window.get("context_id", "")),
                    "example_index": example_index,
                    "context_id": str(example.get("context_id", "")),
                    "score": example.get(score_col),
                    "incorrect": int(example["pred"]) != int(example["label"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame.dropna(subset=["score"]).reset_index(drop=True)


def _evaluation_contexts(config: Mapping[str, Any], examples: pd.DataFrame) -> set[str]:
    raw = config.get("splits", {}).get("evaluation_contexts", "all")
    if raw == "all" or raw == ["all"]:
        return {str(value) for value in examples["context_id"].drop_duplicates().tolist()}
    return {str(value) for value in raw}


def _method_margins(run_dir: Path, occurrences: pd.DataFrame) -> list[MethodMargins]:
    methods: list[MethodMargins] = []
    global_frame = _read_csv(run_dir / "results" / "global_by_context.csv")
    global_threshold = _global_threshold(global_frame)
    if global_threshold is not None:
        selected = _weighted_selected_metrics(global_frame)
        methods.append(
            MethodMargins(
                method="global_crc",
                margins=_margin(occurrences["score"], global_threshold),
                errors=occurrences["incorrect"].to_numpy(dtype=bool),
                selected_coverage=selected[0],
                selected_risk=selected[1],
            )
        )

    metadata_frame = _read_csv(run_dir / "results" / "stratified_by_context.csv")
    metadata_thresholds = _thresholds_by_key(metadata_frame, "context_id")
    if metadata_thresholds:
        margins = _mapped_margins(occurrences, metadata_thresholds, key_col="context_id")
        if margins is not None:
            selected = _weighted_selected_metrics(metadata_frame)
            methods.append(
                MethodMargins(
                    method="metadata_stratified",
                    margins=margins[0],
                    errors=margins[1],
                    selected_coverage=selected[0],
                    selected_risk=selected[1],
                )
            )

    oracle_frame = _read_csv(run_dir / "results" / "oracle_by_context.csv")
    oracle_thresholds = _thresholds_by_key(oracle_frame, "context_id")
    if oracle_thresholds:
        margins = _mapped_margins(occurrences, oracle_thresholds, key_col="context_id")
        if margins is not None:
            selected = _weighted_selected_metrics(oracle_frame)
            methods.append(
                MethodMargins(
                    method="oracle_context_crc",
                    margins=margins[0],
                    errors=margins[1],
                    selected_coverage=selected[0],
                    selected_risk=selected[1],
                    unit="eligible_window_example",
                )
            )

    ct_frame = _read_csv(run_dir / "results" / "ct_crc_windows.csv")
    ct_thresholds = _thresholds_by_key(ct_frame, "window_id")
    if ct_thresholds:
        margins = _mapped_margins(occurrences, ct_thresholds, key_col="window_id")
        if margins is not None:
            selected = _window_selected_metrics(ct_frame)
            methods.append(
                MethodMargins(
                    method="ct_crc",
                    margins=margins[0],
                    errors=margins[1],
                    selected_coverage=selected[0],
                    selected_risk=selected[1],
                )
            )

    cw_frame = _read_csv(run_dir / "results" / "cw_crc_windows.csv")
    cw_thresholds = _thresholds_by_key(cw_frame, "window_id")
    if cw_thresholds:
        margins = _mapped_margins(occurrences, cw_thresholds, key_col="window_id")
        if margins is not None:
            selected = _window_selected_metrics(cw_frame)
            methods.append(
                MethodMargins(
                    method="cw_crc",
                    margins=margins[0],
                    errors=margins[1],
                    selected_coverage=selected[0],
                    selected_risk=selected[1],
                )
            )

    return methods


def _global_threshold(frame: pd.DataFrame | None) -> float | None:
    if frame is None or "threshold" not in frame.columns:
        return None
    thresholds = pd.to_numeric(frame["threshold"], errors="coerce").dropna()
    if thresholds.empty:
        return None
    return float(thresholds.iloc[0])


def _thresholds_by_key(frame: pd.DataFrame | None, key_col: str) -> dict[str, float]:
    if frame is None or key_col not in frame.columns or "threshold" not in frame.columns:
        return {}
    working = frame[[key_col, "threshold"]].copy()
    working["threshold"] = pd.to_numeric(working["threshold"], errors="coerce")
    working = working.dropna(subset=[key_col, "threshold"])
    return {str(row[key_col]): float(row["threshold"]) for _, row in working.iterrows()}


def _mapped_margins(
    occurrences: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    key_col: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    selected = occurrences[occurrences[key_col].astype(str).isin(thresholds)].copy()
    if selected.empty:
        return None
    threshold_array = selected[key_col].astype(str).map(thresholds).to_numpy(dtype=float)
    return _margin(selected["score"], threshold_array), selected["incorrect"].to_numpy(dtype=bool)


def _margin(scores: Any, thresholds: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float) - np.asarray(
        thresholds,
        dtype=float,
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None


def _weighted_selected_metrics(frame: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if frame is None or frame.empty or not {"n", "coverage", "mean_loss"}.issubset(frame.columns):
        return None, None
    working = frame[["n", "coverage", "mean_loss"]].copy()
    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["n", "coverage", "mean_loss"])
    working = working[working["n"] > 0.0]
    if working.empty:
        return None, None
    total = float(working["n"].sum())
    coverage = float(np.sum(working["coverage"].to_numpy(dtype=float) * working["n"].to_numpy(dtype=float)) / total)
    risk = float(np.sum(working["mean_loss"].to_numpy(dtype=float) * working["n"].to_numpy(dtype=float)) / total)
    return coverage, risk


def _window_selected_metrics(frame: pd.DataFrame | None) -> tuple[float | None, float | None]:
    if frame is None or frame.empty or "coverage" not in frame.columns:
        return None, None
    risk_col = "realized_risk" if "realized_risk" in frame.columns else "mean_loss"
    if risk_col not in frame.columns:
        return None, None
    coverage = pd.to_numeric(frame["coverage"], errors="coerce").dropna()
    risk = pd.to_numeric(frame[risk_col], errors="coerce").dropna()
    if coverage.empty or risk.empty:
        return None, None
    return float(coverage.mean()), float(risk.mean())


def _mark_selected_from_outputs(
    curve: pd.DataFrame,
    *,
    selected_coverage: float | None,
    selected_risk: float | None,
) -> pd.DataFrame:
    if curve.empty:
        return curve
    coverage = _finite_float(selected_coverage)
    risk = _finite_float(selected_risk)
    if coverage is None and risk is None:
        return curve

    working = curve.copy()
    working["selected_operating_point"] = False
    distances = np.zeros(len(working), dtype=float)
    if coverage is not None:
        distances += np.abs(pd.to_numeric(working["coverage"], errors="coerce").to_numpy(dtype=float) - coverage)
    if risk is not None:
        distances += np.abs(
            pd.to_numeric(working["accepted_error_risk"], errors="coerce").to_numpy(dtype=float) - risk
        )
    distances[~np.isfinite(distances)] = np.inf
    if np.isfinite(distances).any():
        selected_index = int(np.argmin(distances))
        working.loc[working.index[selected_index], "selected_operating_point"] = True
    return working


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _deduplicate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return curve
    sort_columns = ["coverage", "accepted_error_risk", "threshold"]
    working = curve.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for coverage, group in working.groupby("coverage", sort=True):
        selected = group[group["selected_operating_point"].astype(bool)]
        representative = selected.iloc[0] if not selected.empty else group.iloc[0]
        rows.append(
            {
                "threshold": float(representative["threshold"]),
                "coverage": float(coverage),
                "accepted_error_risk": float(group["accepted_error_risk"].min()),
                "n_units": float(representative["n_units"]),
                "n_accepted": float(representative["n_accepted"]),
                "n_accepted_errors": float(group["n_accepted_errors"].min()),
                "selected_operating_point": bool(group["selected_operating_point"].any()),
            }
        )
    grouped = pd.DataFrame(rows)
    return grouped[
        [
            "threshold",
            "coverage",
            "accepted_error_risk",
            "n_units",
            "n_accepted",
            "n_accepted_errors",
            "selected_operating_point",
        ]
    ]


def _prepare_curve_for_matching(curve: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    if curve is None or curve.empty or x_col not in curve.columns or y_col not in curve.columns:
        return pd.DataFrame(columns=[x_col, y_col])
    working = curve[[x_col, y_col]].copy()
    working[x_col] = pd.to_numeric(working[x_col], errors="coerce")
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")
    working = working.dropna()
    if working.empty:
        return pd.DataFrame(columns=[x_col, y_col])
    if y_col == "accepted_error_risk":
        grouped = working.groupby(x_col, as_index=False)[y_col].min()
    else:
        grouped = working.groupby(x_col, as_index=False)[y_col].max()
    grouped = grouped.sort_values(x_col, kind="mergesort").reset_index(drop=True)
    if y_col == "accepted_error_risk":
        grouped[y_col] = np.maximum.accumulate(grouped[y_col].to_numpy(dtype=float))
    elif y_col == "coverage":
        grouped[y_col] = np.maximum.accumulate(grouped[y_col].to_numpy(dtype=float))
    return grouped


def _coverage_targets(curves: Mapping[str, pd.DataFrame]) -> list[float]:
    targets: list[float] = []
    for method in SELECTED_TARGET_METHODS:
        selected = _selected_point(curves.get(method, pd.DataFrame()))
        if selected is not None:
            targets.append(float(selected["coverage"]))
    targets.extend(FIXED_COVERAGE_TARGETS)
    return _unique_targets(targets, lower=0.0, upper=1.0)


def _risk_targets(curves: Mapping[str, pd.DataFrame]) -> list[float]:
    targets = [RISK_BUDGET]
    for method in SELECTED_TARGET_METHODS:
        selected = _selected_point(curves.get(method, pd.DataFrame()))
        if selected is not None:
            targets.append(float(selected["accepted_error_risk"]))
    return _unique_targets(targets, lower=0.0, upper=1.0)


def _selected_point(curve: pd.DataFrame) -> pd.Series | None:
    if curve is None or curve.empty or "selected_operating_point" not in curve.columns:
        return None
    selected = curve[curve["selected_operating_point"].astype(bool)]
    if selected.empty:
        return None
    return selected.iloc[0]


def _unique_targets(values: Sequence[float], *, lower: float, upper: float) -> list[float]:
    rounded: dict[float, float] = {}
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < lower - 1e-12 or numeric > upper + 1e-12:
            continue
        rounded[round(numeric, 12)] = min(max(numeric, lower), upper)
    return [rounded[key] for key in sorted(rounded)]


def _curves_by_method(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    if frame.empty or "method" not in frame.columns:
        return curves
    for method, group in frame.groupby("method", sort=False):
        curves[str(method)] = group.copy()
    return curves


def _ordered_methods(curves: Mapping[str, pd.DataFrame]) -> list[str]:
    observed = set(curves.keys())
    ordered = [method for method in METHOD_ORDER if method in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _ordered_values(values: pd.Series, *, preferred: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _delta(value: Any, reference: Any) -> float:
    if value is None or reference is None:
        return float("nan")
    try:
        numeric = float(value)
        baseline = float(reference)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(numeric) or not np.isfinite(baseline):
        return float("nan")
    return numeric - baseline


def _plot_curve(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["coverage", "accepted_error_risk"])
    curve = _prepare_curve_for_matching(frame, x_col="coverage", y_col="accepted_error_risk")
    return curve.dropna(subset=["coverage", "accepted_error_risk"])


def _panel_grid(n_panels: int) -> tuple[int, int]:
    n_panels = max(int(n_panels), 1)
    if n_panels == 1:
        return 1, 1
    if n_panels == 2:
        return 1, 2
    if n_panels <= 4:
        return 2, 2
    return int(np.ceil(n_panels / 3)), 3


def _display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset).replace("_", " "))


def _placeholder(output_path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
