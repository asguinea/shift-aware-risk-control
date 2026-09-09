from __future__ import annotations

import os
import tempfile
from pathlib import Path

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
from contextual_crc.evaluation.seed_variability import (
    DATASET_ORDER,
    METHOD_ORDER,
    METRIC_COLUMNS,
    SMOOTHNESS_COLUMNS,
    display_dataset,
    display_method,
    summarize_seed_metrics,
    summarize_seed_smoothness,
)
from contextual_crc.utils.paths import ensure_dir


DEFAULT_METRICS_CSV = Path("outputs/runs/seed_variability/seed_variability_metrics.csv")
DEFAULT_SMOOTHNESS_CSV = Path("outputs/runs/seed_variability/seed_variability_smoothness.csv")


def load_seed_variability_tables(
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    smoothness_csv: str | Path = DEFAULT_SMOOTHNESS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw seed variability tables, returning empty frames when absent."""
    metrics_path = Path(metrics_csv)
    smoothness_path = Path(smoothness_csv)
    metrics = _read_csv_or_empty(metrics_path, METRIC_COLUMNS)
    smoothness = _read_csv_or_empty(smoothness_path, SMOOTHNESS_COLUMNS)
    return metrics, smoothness


def write_seed_variability_artifacts(
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
    *,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    smoothness_csv: str | Path = DEFAULT_SMOOTHNESS_CSV,
    figure_format: str = "pdf",
    precision: int = 3,
) -> dict[str, str]:
    """Write appendix-ready seed variability summaries and figure if raw outputs exist."""
    raw_metrics, raw_smoothness = load_seed_variability_tables(metrics_csv, smoothness_csv)
    if raw_metrics.empty and raw_smoothness.empty:
        return {}

    tables_dir = ensure_dir(output_tables_dir)
    figures_dir = ensure_dir(output_figures_dir)
    metric_summary = summarize_seed_metrics(raw_metrics)
    smoothness_summary = summarize_seed_smoothness(raw_smoothness)

    metrics_path = tables_dir / "seed_variability_metrics.csv"
    metrics_tex_path = tables_dir / "seed_variability_metrics.tex"
    smoothness_path = tables_dir / "seed_variability_smoothness.csv"
    smoothness_tex_path = tables_dir / "seed_variability_smoothness.tex"
    figure_path = figures_dir / f"seed_variability.{figure_format.lstrip('.')}"

    metric_summary.to_csv(metrics_path, index=False)
    smoothness_summary.to_csv(smoothness_path, index=False)
    metrics_tex_path.write_text(
        dataframe_to_latex_tabular(
            metric_summary,
            caption="Seed and Window-Bootstrap Variability",
            label="tab:seed-variability-metrics",
            precision=precision,
        ),
        encoding="utf-8",
    )
    smoothness_tex_path.write_text(
        dataframe_to_latex_tabular(
            smoothness_summary,
            caption="Seed and Window-Bootstrap Smoothness Variability",
            label="tab:seed-variability-smoothness",
            precision=precision,
        ),
        encoding="utf-8",
    )
    plot_seed_variability(metric_summary, figure_path)
    return {
        "seed_variability_metrics_csv": str(metrics_path),
        "seed_variability_metrics_tex": str(metrics_tex_path),
        "seed_variability_smoothness_csv": str(smoothness_path),
        "seed_variability_smoothness_tex": str(smoothness_tex_path),
        "seed_variability_figure": str(figure_path),
    }


def plot_seed_variability(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot mean +/- std risk and coverage by method and dataset."""
    output = Path(output_path)
    ensure_dir(output.parent)
    if summary.empty:
        return _placeholder(output, "No seed variability data")

    frame = summary.copy()
    for column in ("mean_risk", "std_risk", "mean_coverage", "std_coverage"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    datasets = _ordered_values(frame["dataset"], DATASET_ORDER)
    n_rows = 2
    n_cols = max(1, len(datasets))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 3.2 * n_cols), 5.8),
        squeeze=False,
        sharey=False,
        constrained_layout=True,
    )
    metric_specs = (
        ("mean_risk", "std_risk", "Risk"),
        ("mean_coverage", "std_coverage", "Coverage"),
    )
    for col_index, dataset in enumerate(datasets):
        subset = frame[frame["dataset"] == dataset].copy()
        subset["method_rank"] = subset["method"].map(_method_rank)
        subset = subset.sort_values("method_rank", kind="mergesort")
        x = np.arange(len(subset), dtype=float)
        for row_index, (mean_col, std_col, ylabel) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            means = pd.to_numeric(subset[mean_col], errors="coerce").to_numpy(dtype=float)
            stds = pd.to_numeric(subset[std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.bar(x, means, yerr=stds, color="tab:blue" if row_index == 0 else "tab:green", alpha=0.82, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([_short_method_label(value) for value in subset["method"]], rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0.0)
            if row_index == 0:
                ax.axhline(0.10, linestyle="--", linewidth=0.9, color="0.35", alpha=0.7)
                ax.set_title(display_dataset(dataset), fontsize=10)
            if col_index == 0:
                ax.set_ylabel(ylabel)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _read_csv_or_empty(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, list(columns)]


def _ordered_values(values: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _short_method_label(method: str) -> str:
    labels = {
        "global_crc": "Global",
        "metadata_stratified_crc": "Metadata",
        "ct_crc": "CT",
        "cw_crc": "CW",
        "oracle_context_crc": "Oracle",
    }
    return labels.get(str(method), display_method(str(method)))


def _method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(str(method))
    except ValueError:
        return len(METHOD_ORDER)


def _placeholder(output_path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
