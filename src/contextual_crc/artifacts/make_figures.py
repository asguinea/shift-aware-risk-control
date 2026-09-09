from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

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
from matplotlib.figure import Figure

from contextual_crc.artifacts.make_tables import (
    make_bound_diagnostics_table,
    make_label_efficiency_table,
    make_main_results_table,
)
from contextual_crc.utils.paths import ensure_dir

DATASET_DISPLAY_NAMES = {
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "synthetic_contextual_shift": "Synthetic",
    "synthetic_stronger_contextual_shift": "Synthetic",
    "officehome": "OfficeHome",
    "wilds_camelyon17": "Camelyon17-WILDS",
    "wilds_fmow": "FMoW-WILDS",
}

METHOD_DISPLAY_NAMES = {
    "ct_crc": "CT-CRC",
    "cw_crc": "CW-CRC",
    "global_crc": "Global CRC",
    "metadata_stratified": "Metadata-stratified",
    "metadata_stratified_context": "Metadata-stratified",
    "metadata_stratified_crc": "Metadata-stratified",
    "metadata_stratified_corruption": "Metadata-stratified corruption",
    "metadata_stratified_severity": "Metadata-stratified severity",
    "oracle_context_crc": "Oracle context",
}

PREFERRED_DATASET_ORDER = (
    "cifar10c",
    "cifar100c",
    "synthetic_stronger_contextual_shift",
    "wilds_camelyon17",
    "officehome",
    "wilds_fmow",
)

PREFERRED_METHOD_ORDER = (
    "global_crc",
    "metadata_stratified_context",
    "metadata_stratified_crc",
    "ct_crc",
    "cw_crc",
    "oracle_context_crc",
)

METHOD_COMPARISON_METHOD_ORDER = (
    "global_crc",
    "metadata_stratified",
    "ct_crc",
    "cw_crc",
    "oracle_context_crc",
)

METHOD_COMPARISON_FALLBACKS = {
    "global_crc": ("global_crc",),
    "metadata_stratified": (
        "metadata_stratified_context",
        "metadata_stratified_corruption",
        "metadata_stratified_crc",
        "metadata_stratified_severity",
    ),
    "ct_crc": ("ct_crc",),
    "cw_crc": ("cw_crc",),
    "oracle_context_crc": ("oracle_context_crc",),
}


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save and close a matplotlib figure."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_method_comparison(
    main_results: pd.DataFrame,
    output_path: str | Path,
    *,
    metric: str = "mean_risk",
) -> Path:
    """Plot a readable grouped bar chart of key method performance."""
    if main_results.empty or metric not in main_results.columns:
        return _placeholder(output_path, "No method comparison data")

    frame = _select_method_comparison_rows(main_results, metric=metric)
    if frame.empty:
        return _placeholder(output_path, f"No finite {metric} data")

    datasets = _ordered_values(frame["dataset"], preferred=PREFERRED_DATASET_ORDER)
    methods = [method for method in METHOD_COMPARISON_METHOD_ORDER if method in set(frame["plot_method"])]
    pivot = frame.pivot_table(index="dataset", columns="plot_method", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=datasets, columns=methods)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(datasets)), 4.0))
    x = np.arange(len(datasets), dtype=float)
    width = min(0.82 / max(len(methods), 1), 0.16)
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods, strict=True):
        bars = ax.bar(
            x + offset,
            pivot[method].to_numpy(dtype=float),
            width=width,
            label=_display_method(method),
        )
        _annotate_bars(ax, bars)
    if metric == "mean_risk":
        ax.axhline(0.10, linestyle="--", linewidth=1.0, alpha=0.7, label="Risk budget")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_dataset(dataset) for dataset in datasets])
    ax.set_ylabel("Mean risk" if metric == "mean_risk" else metric.replace("_", " ").title())
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=min(3, max(1, len(methods))))
    return save_figure(fig, output_path)


def plot_context_smoothness(
    run_outputs: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot pairwise context distance against absolute risk difference."""
    rows: list[pd.DataFrame] = []
    for output in run_outputs:
        frame = output.get("context_smoothness")
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        dataset = str(output.get("dataset") or output.get("run_name") or "unknown")
        if "dataset" not in working.columns:
            working["dataset"] = dataset
        else:
            working["dataset"] = working["dataset"].fillna(dataset).astype(str)
        working["run_name"] = str(output.get("run_name") or dataset)
        rows.append(working)
    if not rows:
        return _placeholder(output_path, "No context smoothness data")

    combined = pd.concat(rows, ignore_index=True)
    distance_col = _first_column(
        combined,
        ("distance", "context_distance", "pairwise_distance", "mean_pairwise_distance"),
    )
    risk_diff_col = _first_column(
        combined,
        ("risk_difference", "abs_risk_diff", "risk_diff", "value_difference", "mean_value_difference"),
    )
    if distance_col is None or risk_diff_col is None:
        return _placeholder(output_path, "No distance/risk-difference columns")

    combined[distance_col] = pd.to_numeric(combined[distance_col], errors="coerce")
    combined[risk_diff_col] = pd.to_numeric(combined[risk_diff_col], errors="coerce").abs()
    combined = combined.dropna(subset=[distance_col, risk_diff_col])
    if combined.empty:
        return _placeholder(output_path, "No finite context smoothness data")

    datasets = _ordered_values(combined["dataset"], preferred=PREFERRED_DATASET_ORDER)
    n_rows, n_columns, figsize = _context_smoothness_grid(len(datasets))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figsize,
        squeeze=False,
        sharey=False,
    )
    for panel_index, dataset in enumerate(datasets):
        row_index, column_index = divmod(panel_index, n_columns)
        ax = axes[row_index, column_index]
        group = combined[combined["dataset"] == dataset]
        ax.scatter(
            group[distance_col],
            group[risk_diff_col],
            alpha=0.75,
            s=24,
        )
        _plot_binned_smoothness_line(ax, group, x_col=distance_col, y_col=risk_diff_col)
        ax.set_title(_display_dataset(dataset), fontsize=12, pad=8)
        if row_index == n_rows - 1 or panel_index + n_columns >= len(datasets):
            ax.set_xlabel("Pairwise embedding distance", fontsize=10)
        if column_index == 0:
            ax.set_ylabel("Absolute risk difference", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)
        annotation = _smoothness_annotation(group, x_col=distance_col, y_col=risk_diff_col)
        if annotation:
            ax.text(
                0.04,
                0.96,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.85"},
            )
    for empty_index in range(len(datasets), n_rows * n_columns):
        row_index, column_index = divmod(empty_index, n_columns)
        axes[row_index, column_index].set_axis_off()
    fig.subplots_adjust(
        hspace=0.42 if n_rows > 1 else 0.25,
        wspace=0.30,
    )
    return save_figure(fig, output_path)


def plot_online_shift_trace(
    run_outputs: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot deployment-window novelty, risk/bounds, and coverage."""
    selected = _select_online_trace_output(run_outputs)
    if selected is None:
        return _placeholder(output_path, "No online trace data")

    dedicated_trace = selected.get("online_trace")
    if dedicated_trace is not None and not dedicated_trace.empty:
        return _plot_dedicated_online_trace(
            dedicated_trace,
            output_path,
            selected=selected,
        )

    ct_windows = selected.get("ct_crc_windows")
    cw_windows = selected.get("cw_crc_windows")
    primary_windows = ct_windows if ct_windows is not None and not ct_windows.empty else cw_windows
    if primary_windows is None or primary_windows.empty:
        return _placeholder(output_path, "No online trace data")

    dataset = str(selected.get("dataset") or selected.get("run_name") or "run")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.4),
        sharex=True,
    )
    fig.suptitle(f"Online trace: {_display_dataset(dataset)}", fontsize=12)

    _plot_novelty_panel(axes[0], primary_windows)
    _plot_risk_bound_panel(axes[1], ct_windows, cw_windows)
    _plot_coverage_panel(axes[2], ct_windows, cw_windows)
    _add_context_change_lines(axes, primary_windows)

    axes[2].set_xlabel("Window index")
    for ax in axes:
        ax.grid(alpha=0.3)
    return save_figure(fig, output_path)


def _plot_dedicated_online_trace(
    trace: pd.DataFrame,
    output_path: str | Path,
    *,
    selected: dict[str, Any],
) -> Path:
    frame = trace.copy()
    if "trace_index" not in frame.columns:
        frame["trace_index"] = np.arange(len(frame), dtype=int)
    frame["trace_index"] = pd.to_numeric(frame["trace_index"], errors="coerce")
    frame = frame.dropna(subset=["trace_index"]).sort_values("trace_index", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        return _placeholder(output_path, "No online trace data")

    title = _dedicated_trace_title(selected, frame)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.7), sharex=True)
    fig.suptitle(title, fontsize=12)
    _plot_dedicated_novelty_severity_panel(axes[0], frame)
    _plot_dedicated_risk_bound_panel(axes[1], frame)
    _plot_dedicated_coverage_panel(axes[2], frame)
    _add_severity_change_lines(axes, frame)
    axes[2].set_xlabel("Ordered deployment window")
    for ax in axes:
        ax.grid(alpha=0.3)
    return save_figure(fig, output_path)


def plot_bound_diagnostics(
    bound_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot bound underestimation and bound-gap diagnostics by dataset."""
    primary_metric = "bound_underestimation_rate"
    if bound_table.empty or primary_metric not in bound_table.columns:
        return _placeholder(output_path, "No bound diagnostics data")

    columns = ["dataset", "method", primary_metric]
    include_gap = "mean_bound_gap" in bound_table.columns
    if include_gap:
        columns.append("mean_bound_gap")
    frame = bound_table[columns].copy()
    for column in (primary_metric, "mean_bound_gap"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[primary_metric])
    if frame.empty:
        return _placeholder(output_path, f"No finite {primary_metric} data")

    gap_frame = (
        frame.dropna(subset=["mean_bound_gap"])
        if "mean_bound_gap" in frame.columns
        else pd.DataFrame()
    )
    has_gap = not gap_frame.empty
    fig, axes = plt.subplots(
        2 if has_gap else 1,
        1,
        figsize=(7.0, 5.8 if has_gap else 3.8),
        squeeze=False,
        sharex=has_gap,
    )
    _plot_grouped_bound_metric(
        axes[0, 0],
        frame,
        metric=primary_metric,
        ylabel="Bound underestimation rate",
        show_xlabel=not has_gap,
    )
    if has_gap:
        _plot_grouped_bound_metric(
            axes[1, 0],
            gap_frame,
            metric="mean_bound_gap",
            ylabel="Mean bound gap",
            show_xlabel=True,
        )
    return save_figure(fig, output_path)


def plot_label_efficiency(
    label_efficiency: pd.DataFrame,
    output_path: str | Path,
    *,
    metric: str = "mean_risk",
) -> Path:
    """Plot risk and coverage label-efficiency curves by method and dataset."""
    del metric
    metric_columns = ("mean_risk", "mean_coverage")
    if label_efficiency.empty or any(column not in label_efficiency.columns for column in metric_columns):
        return _placeholder(output_path, "No label-efficiency data")
    x_col = _first_column(label_efficiency, ("budget_windows_per_context", "actual_labeled_examples"))
    if x_col is None:
        return _placeholder(output_path, "No label budget column")

    frame = label_efficiency[["dataset", "method", x_col, *metric_columns]].copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["dataset", "method", x_col])
    frame = frame[frame[list(metric_columns)].notna().any(axis=1)]
    if frame.empty:
        return _placeholder(output_path, "No finite label-efficiency data")

    datasets = _ordered_values(frame["dataset"], preferred=PREFERRED_DATASET_ORDER)
    fig, axes = plt.subplots(
        2,
        len(datasets),
        figsize=(max(5.5, 3.8 * len(datasets)), 5.8),
        squeeze=False,
        sharex="col",
    )
    legend_handles: dict[str, Any] = {}
    panels = (
        ("mean_risk", "Mean risk"),
        ("mean_coverage", "Mean coverage"),
    )
    for column_index, dataset in enumerate(datasets):
        subset = frame[frame["dataset"] == dataset]
        methods = _ordered_values(subset["method"], preferred=PREFERRED_METHOD_ORDER)
        axes[0, column_index].set_title(_display_dataset(dataset), fontsize=11)
        for row_index, (metric_column, y_label) in enumerate(panels):
            ax = axes[row_index, column_index]
            if metric_column == "mean_risk":
                ax.axhline(0.10, linestyle="--", linewidth=1.0, alpha=0.7)
            for method in methods:
                curve = _label_efficiency_curve(
                    subset[subset["method"] == method],
                    x_col=x_col,
                    y_col=metric_column,
                )
                if curve.empty:
                    continue
                label = _display_method(method)
                (line,) = ax.plot(
                    curve[x_col],
                    curve[metric_column],
                    marker="o",
                    linewidth=1.5,
                    markersize=4,
                    label=label,
                )
                legend_handles.setdefault(label, line)
            if column_index == 0:
                ax.set_ylabel(y_label)
            if row_index == 1:
                ax.set_xlabel(_budget_axis_label(x_col))
                ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(4, max(1, len(legend_handles))),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.16, hspace=0.25, wspace=0.25)
    return save_figure(fig, output_path)


def plot_label_efficiency_risk_only(
    label_efficiency: pd.DataFrame,
    output_path: str | Path,
    *,
    metric: str = "mean_risk",
) -> Path:
    """Plot the legacy risk-only label-efficiency view for debugging."""
    if label_efficiency.empty or metric not in label_efficiency.columns:
        return _placeholder(output_path, "No label-efficiency data")
    x_col = _first_column(label_efficiency, ("budget_windows_per_context", "actual_labeled_examples"))
    if x_col is None:
        return _placeholder(output_path, "No label budget column")

    frame = label_efficiency[["dataset", "method", x_col, metric]].copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["dataset", "method", x_col, metric])
    if frame.empty:
        return _placeholder(output_path, f"No finite {metric} label-efficiency data")

    datasets = _ordered_values(frame["dataset"], preferred=PREFERRED_DATASET_ORDER)
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(5.0, 3.8 * len(datasets)), 3.6),
        squeeze=False,
    )
    legend_handles: dict[str, Any] = {}
    for ax, dataset in zip(axes[0], datasets, strict=True):
        subset = frame[frame["dataset"] == dataset]
        for method in _ordered_values(subset["method"], preferred=PREFERRED_METHOD_ORDER):
            curve = _label_efficiency_curve(
                subset[subset["method"] == method],
                x_col=x_col,
                y_col=metric,
            )
            if curve.empty:
                continue
            label = _display_method(method)
            (line,) = ax.plot(curve[x_col], curve[metric], marker="o", label=label)
            legend_handles.setdefault(label, line)
        ax.axhline(0.10, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(_display_dataset(dataset))
        ax.set_xlabel(_budget_axis_label(x_col))
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, max(1, len(legend_handles))),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.20, wspace=0.25)
    return save_figure(fig, output_path)


def write_figures(
    run_outputs: Sequence[dict[str, Any]],
    output_figures_dir: str | Path,
    *,
    figure_format: str = "pdf",
    precision: int = 3,
    label_efficiency_frames: Sequence[pd.DataFrame] | None = None,
    make_label_efficiency: bool = False,
) -> dict[str, str]:
    """Write all configured figures."""
    del precision
    figures_dir = ensure_dir(output_figures_dir)
    suffix = figure_format.lstrip(".")
    main_results = make_main_results_table(run_outputs)
    bound_table = make_bound_diagnostics_table(run_outputs)
    written = {
        "method_comparison": plot_method_comparison(
            main_results,
            figures_dir / f"method_comparison.{suffix}",
        ),
        "context_smoothness": plot_context_smoothness(
            run_outputs,
            figures_dir / f"context_smoothness.{suffix}",
        ),
        "online_shift_trace": plot_online_shift_trace(
            run_outputs,
            figures_dir / f"online_shift_trace.{suffix}",
        ),
        "bound_diagnostics": plot_bound_diagnostics(
            bound_table,
            figures_dir / f"bound_diagnostics.{suffix}",
        ),
    }
    if make_label_efficiency:
        label_efficiency = make_label_efficiency_table(label_efficiency_frames)
        written["label_efficiency"] = plot_label_efficiency(
            label_efficiency,
            figures_dir / f"label_efficiency.{suffix}",
        )
        written["label_efficiency_risk_only"] = plot_label_efficiency_risk_only(
            label_efficiency,
            figures_dir / f"label_efficiency_risk_only.{suffix}",
        )
    return {name: str(path) for name, path in written.items()}


def _placeholder(output_path: str | Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    return save_figure(fig, output_path)


def _first_column(frame: pd.DataFrame | None, candidates: tuple[str, ...]) -> str | None:
    if frame is None:
        return None
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _select_method_comparison_rows(main_results: pd.DataFrame, *, metric: str = "mean_risk") -> pd.DataFrame:
    output_columns = ["dataset", "method", "plot_method", "method_display", metric]
    if main_results.empty or metric not in main_results.columns:
        return pd.DataFrame(columns=output_columns)

    frame = main_results[["dataset", "method", metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["dataset", "method", metric])
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, Any]] = []
    datasets = _ordered_values(frame["dataset"], preferred=PREFERRED_DATASET_ORDER)
    for dataset in datasets:
        subset = frame[frame["dataset"] == dataset]
        for plot_method in METHOD_COMPARISON_METHOD_ORDER:
            selected = _first_matching_method_row(
                subset,
                METHOD_COMPARISON_FALLBACKS[plot_method],
            )
            if selected is None:
                continue
            row = selected.to_dict()
            row["plot_method"] = plot_method
            row["method_display"] = _display_method(plot_method)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows).loc[:, output_columns].reset_index(drop=True)


def _first_matching_method_row(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for method in candidates:
        rows = frame[frame["method"] == method]
        if not rows.empty:
            return rows.iloc[0]
    return None


def _select_online_trace_output(run_outputs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, output in enumerate(run_outputs):
        online_trace = output.get("online_trace")
        if online_trace is not None and not online_trace.empty:
            candidates.append((-1, index, output))
            continue
        ct_windows = output.get("ct_crc_windows")
        cw_windows = output.get("cw_crc_windows")
        has_windows = (ct_windows is not None and not ct_windows.empty) or (
            cw_windows is not None and not cw_windows.empty
        )
        if has_windows:
            candidates.append((_online_trace_dataset_rank(output), index, output))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def _online_trace_dataset_rank(output: dict[str, Any]) -> int:
    dataset = str(output.get("dataset") or output.get("run_name") or "").lower()
    preferred = {
        "cifar10c": 0,
        "cifar100c": 1,
        "synthetic_stronger_contextual_shift": 2,
    }
    if dataset in preferred:
        return preferred[dataset]
    for key, rank in preferred.items():
        if key in dataset:
            return rank
    return len(preferred)


def _dedicated_trace_title(selected: dict[str, Any], frame: pd.DataFrame) -> str:
    manifest = selected.get("manifest") or {}
    dataset = _display_dataset(manifest.get("dataset") or selected.get("dataset") or selected.get("run_name") or "run")
    corruption = manifest.get("corruption_type")
    if corruption is None and "corruption_type" in frame.columns and not frame["corruption_type"].empty:
        corruption = frame["corruption_type"].dropna().astype(str).iloc[0]
    if corruption:
        return f"Online severity trace: {dataset} {_compact_context_label(str(corruption)).title()}"
    return f"Online severity trace: {dataset}"


def _plot_dedicated_novelty_severity_panel(ax: Any, frame: pd.DataFrame) -> None:
    x = pd.to_numeric(frame["trace_index"], errors="coerce").to_numpy(dtype=float)
    novelty = pd.to_numeric(frame.get("novelty_distance"), errors="coerce") if "novelty_distance" in frame else None
    severity = pd.to_numeric(frame.get("severity"), errors="coerce") if "severity" in frame else None
    plotted = False
    if novelty is not None and np.isfinite(novelty.to_numpy(dtype=float)).any():
        ax.plot(x, novelty.to_numpy(dtype=float), label="Novelty distance", color="tab:blue")
        ax.set_ylabel("Novelty distance")
        plotted = True
    if severity is not None and np.isfinite(severity.to_numpy(dtype=float)).any():
        if plotted:
            twin = ax.twinx()
            twin.step(x, severity.to_numpy(dtype=float), where="mid", label="Severity", color="tab:gray", alpha=0.7)
            twin.set_ylabel("Severity")
            twin.set_yticks(sorted(severity.dropna().astype(int).unique()))
            lines, labels = ax.get_legend_handles_labels()
            twin_lines, twin_labels = twin.get_legend_handles_labels()
            ax.legend(lines + twin_lines, labels + twin_labels, fontsize=8, loc="upper left")
        else:
            ax.step(x, severity.to_numpy(dtype=float), where="mid", label="Severity", color="tab:gray")
            ax.set_ylabel("Severity")
            ax.set_yticks(sorted(severity.dropna().astype(int).unique()))
            ax.legend(fontsize=8, loc="upper left")
            plotted = True
    elif plotted:
        ax.legend(fontsize=8, loc="upper left")
    if not plotted:
        ax.text(0.5, 0.5, "No novelty or severity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Novelty")


def _plot_dedicated_risk_bound_panel(ax: Any, frame: pd.DataFrame) -> None:
    series = (
        ("ct_realized_risk", "CT risk"),
        ("ct_bound", "CT bound"),
        ("cw_realized_risk", "CW risk"),
        ("cw_bound", "CW bound"),
    )
    x = pd.to_numeric(frame["trace_index"], errors="coerce").to_numpy(dtype=float)
    plotted = False
    for column, label in series:
        if column not in frame.columns:
            continue
        y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).any():
            ax.plot(x, y, label=label)
            plotted = True
    ax.axhline(0.10, linestyle="--", linewidth=1.0, alpha=0.7, label="Risk budget")
    ax.set_ylabel("Risk / bound")
    if plotted:
        ax.legend(fontsize=8, ncol=3)
    else:
        ax.text(0.5, 0.5, "No risk or bound columns", ha="center", va="center", transform=ax.transAxes)


def _plot_dedicated_coverage_panel(ax: Any, frame: pd.DataFrame) -> None:
    x = pd.to_numeric(frame["trace_index"], errors="coerce").to_numpy(dtype=float)
    plotted = False
    for column, label in (("ct_coverage", "CT coverage"), ("cw_coverage", "CW coverage")):
        if column not in frame.columns:
            continue
        y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).any():
            ax.plot(x, y, label=label)
            plotted = True
    ax.set_ylabel("Coverage")
    ax.set_ylim(-0.02, 1.02)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No coverage columns", ha="center", va="center", transform=ax.transAxes)


def _add_severity_change_lines(axes: Sequence[Any], frame: pd.DataFrame) -> None:
    if "severity" not in frame.columns or len(frame) < 2:
        return
    severities = pd.to_numeric(frame["severity"], errors="coerce").tolist()
    x = pd.to_numeric(frame["trace_index"], errors="coerce").to_numpy(dtype=float)
    for index in range(1, len(severities)):
        if severities[index] == severities[index - 1]:
            continue
        position = (x[index] + x[index - 1]) / 2.0
        for ax in axes:
            ax.axvline(position, color="0.65", linewidth=0.8, alpha=0.45)
        axes[0].text(
            x[index],
            0.96,
            f"s={int(severities[index])}",
            transform=axes[0].get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="0.25",
        )


def _plot_novelty_panel(ax: Any, frame: pd.DataFrame) -> None:
    x = _window_positions(frame)
    novelty_col = _first_column(frame, ("novelty_distance", "distance", "nearest_distance"))
    if novelty_col is not None:
        y = pd.to_numeric(frame[novelty_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).any():
            ax.plot(x, y, label=_novelty_label(novelty_col))
            ax.set_ylabel("Novelty / distance")
            ax.legend(fontsize=8, loc="upper right")
            return
    if "context_id" in frame.columns and len(frame) > 0:
        contexts = frame["context_id"].astype(str).tolist()
        ordered_contexts = list(dict.fromkeys(contexts))
        mapping = {context: index for index, context in enumerate(ordered_contexts)}
        y = np.asarray([mapping[context] for context in contexts], dtype=float)
        ax.step(x, y, where="mid")
        ax.set_yticks(list(mapping.values()))
        ax.set_yticklabels([_compact_context_label(context) for context in mapping], fontsize=7)
        ax.set_ylabel("Context")
        return
    ax.text(0.5, 0.5, "No novelty or context data", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Novelty")


def _plot_risk_bound_panel(
    ax: Any,
    ct_windows: pd.DataFrame | None,
    cw_windows: pd.DataFrame | None,
) -> None:
    plotted = False
    plotted |= _plot_window_series(
        ax,
        ct_windows,
        candidates=("realized_risk", "mean_loss", "mean_risk"),
        label="CT risk",
    )
    plotted |= _plot_window_series(
        ax,
        ct_windows,
        candidates=("transferred_upper_bound", "risk_upper_bound", "mean_bound"),
        label="CT bound",
    )
    plotted |= _plot_window_series(
        ax,
        cw_windows,
        candidates=("realized_risk", "mean_loss", "mean_risk"),
        label="CW risk",
    )
    plotted |= _plot_window_series(
        ax,
        cw_windows,
        candidates=("weighted_risk_upper_bound", "risk_upper_bound", "mean_bound"),
        label="CW bound",
    )
    ax.axhline(0.10, linestyle="--", linewidth=1.0, alpha=0.7, label="Risk budget")
    ax.set_ylabel("Risk / bound")
    if plotted:
        ax.legend(fontsize=8, ncol=3)
    else:
        ax.text(0.5, 0.5, "No risk or bound columns", ha="center", va="center", transform=ax.transAxes)


def _plot_coverage_panel(
    ax: Any,
    ct_windows: pd.DataFrame | None,
    cw_windows: pd.DataFrame | None,
) -> None:
    plotted = False
    plotted |= _plot_window_series(
        ax,
        ct_windows,
        candidates=("coverage", "mean_coverage"),
        label="CT coverage",
    )
    plotted |= _plot_window_series(
        ax,
        cw_windows,
        candidates=("coverage", "mean_coverage"),
        label="CW coverage",
    )
    ax.set_ylabel("Coverage")
    ax.set_ylim(-0.02, 1.02)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No coverage columns", ha="center", va="center", transform=ax.transAxes)


def _plot_window_series(
    ax: Any,
    frame: pd.DataFrame | None,
    *,
    candidates: tuple[str, ...],
    label: str,
) -> bool:
    column = _first_column(frame, candidates)
    if frame is None or frame.empty or column is None:
        return False
    y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).any():
        return False
    ax.plot(_window_positions(frame), y, label=label)
    return True


def _window_positions(frame: pd.DataFrame) -> np.ndarray:
    return np.arange(len(frame), dtype=float)


def _novelty_label(column: str) -> str:
    if column == "novelty_distance":
        return "Novelty distance"
    if column == "nearest_distance":
        return "Nearest-context distance"
    return column.replace("_", " ")


def _compact_context_label(context: str) -> str:
    return context.replace("_", " ")


def _add_context_change_lines(axes: Sequence[Any], frame: pd.DataFrame) -> None:
    if "context_id" not in frame.columns or len(frame) < 2:
        return
    contexts = frame["context_id"].astype(str).tolist()
    change_positions = [
        index - 0.5
        for index in range(1, len(contexts))
        if contexts[index] != contexts[index - 1]
    ]
    for ax in axes:
        for position in change_positions:
            ax.axvline(position, color="0.7", linewidth=0.7, alpha=0.35)


def _plot_grouped_bound_metric(
    ax: Any,
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    show_xlabel: bool,
) -> None:
    datasets = _ordered_values(frame["dataset"], preferred=PREFERRED_DATASET_ORDER)
    methods = _ordered_values(frame["method"], preferred=("ct_crc", "cw_crc"))
    pivot = frame.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=datasets, columns=methods)

    x = np.arange(len(datasets), dtype=float)
    width = min(0.8 / max(len(methods), 1), 0.28)
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods, strict=True):
        values = pivot[method].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=_display_method(method),
        )
        _annotate_bars(ax, bars)

    ax.axhline(0.0, linestyle="--", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_dataset(dataset) for dataset in datasets])
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=min(2, max(1, len(methods))))


def _annotate_bars(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3 if height >= 0 else -9),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=7,
        )


def _context_smoothness_grid(n_panels: int) -> tuple[int, int, tuple[float, float]]:
    """Choose a readable grid for context-smoothness panels."""
    n_panels = max(int(n_panels), 1)
    if n_panels == 1:
        return 1, 1, (5.2, 4.2)
    if n_panels == 2:
        return 1, 2, (8.4, 4.0)
    if n_panels <= 4:
        return 2, 2, (8.8, 7.2)
    n_columns = 3
    n_rows = int(np.ceil(n_panels / n_columns))
    return n_rows, n_columns, (11.4, max(7.2, 3.45 * n_rows))


def _ordered_values(values: pd.Series, *, preferred: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset).replace("_", " "))


def _display_method(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method), str(method).replace("_", " "))


def _budget_axis_label(column: str) -> str:
    if column == "budget_windows_per_context":
        return "Budget windows per context"
    if column == "actual_labeled_examples":
        return "Labeled examples"
    return column.replace("_", " ")


def _label_efficiency_curve(frame: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[x_col, y_col])
    curve = frame[[x_col, y_col]].dropna()
    if curve.empty:
        return curve
    return curve.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)


def _plot_binned_smoothness_line(
    ax: Any,
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> None:
    numeric = frame[[x_col, y_col]].dropna()
    if len(numeric) < 8:
        return
    x = numeric[x_col].to_numpy(dtype=float)
    y = numeric[y_col].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 8 or len(np.unique(x)) < 4:
        return

    n_bins = min(10, max(5, int(np.sqrt(len(x)))))
    edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
    if len(np.unique(edges)) < 3:
        return
    bin_index = np.digitize(x, edges[1:-1], right=True)
    centers = []
    means = []
    for index in range(n_bins):
        mask = bin_index == index
        if not np.any(mask):
            continue
        centers.append(float(np.mean(x[mask])))
        means.append(float(np.mean(y[mask])))
    if len(centers) >= 2:
        ax.plot(centers, means, linewidth=1.8)


def _smoothness_annotation(frame: pd.DataFrame, *, x_col: str, y_col: str) -> str:
    n_plotted = len(frame)
    n_pairs = _summary_n_pairs(frame, fallback=n_plotted)
    parts = [f"n={n_pairs:g}"]
    summary_correlation = _summary_correlation(frame)
    if np.isfinite(summary_correlation):
        parts.append(f"corr={summary_correlation:.2f}")
    else:
        label, correlation = _smoothness_correlation(
            frame[x_col].to_numpy(dtype=float),
            frame[y_col].to_numpy(dtype=float),
        )
        if label is not None and np.isfinite(correlation):
            parts.append(f"{label}={correlation:.2f}")
    if n_plotted < 4:
        parts.append("few pairs" if n_pairs <= 4 else "summary row")
    return "\n".join(parts)


def _summary_n_pairs(frame: pd.DataFrame, *, fallback: int) -> float:
    if "n_pairs" not in frame.columns:
        return float(fallback)
    numeric = pd.to_numeric(frame["n_pairs"], errors="coerce").dropna()
    if numeric.empty:
        return float(fallback)
    return float(numeric.iloc[0]) if len(frame) <= 1 else float(np.nansum(numeric))


def _summary_correlation(frame: pd.DataFrame) -> float:
    if "distance_value_correlation" not in frame.columns or len(frame) > 1:
        return float("nan")
    numeric = pd.to_numeric(frame["distance_value_correlation"], errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.iloc[0])


def _smoothness_correlation(x: np.ndarray, y: np.ndarray) -> tuple[str | None, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None, float("nan")
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        statistic = float(result.statistic)
        if np.isfinite(statistic):
            return "rho", statistic
    except Exception:
        pass
    return "r", float(np.corrcoef(x, y)[0, 1])


def _plot_window_trace(
    ax: Any,
    frame: pd.DataFrame | None,
    *,
    risk_col: str | None,
    bound_col: str,
    risk_label: str,
    bound_label: str,
) -> None:
    if frame is None or frame.empty:
        return
    x = np.arange(len(frame))
    if risk_col is not None and risk_col in frame.columns:
        y = pd.to_numeric(frame[risk_col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, label=risk_label)
    if bound_col in frame.columns:
        y = pd.to_numeric(frame[bound_col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, label=bound_label)
