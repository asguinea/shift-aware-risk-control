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
from contextual_crc.evaluation.cw_sensitivity import SENSITIVITY_COLUMNS
from contextual_crc.utils.paths import ensure_dir


DEFAULT_SENSITIVITY_CSV = Path("outputs/runs/cw_sensitivity/cw_sensitivity.csv")
RISK_BUDGET = 0.10

DATASET_ORDER = (
    "cifar10c",
    "cifar100c",
    "synthetic_stronger_contextual_shift",
    "wilds_camelyon17",
    "officehome",
    "wilds_fmow",
)


def load_cw_sensitivity_table(path: str | Path = DEFAULT_SENSITIVITY_CSV) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.is_file():
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    try:
        frame = pd.read_csv(input_path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    for column in SENSITIVITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, list(SENSITIVITY_COLUMNS)]


def write_cw_sensitivity_artifacts(
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
    *,
    sensitivity_csv: str | Path = DEFAULT_SENSITIVITY_CSV,
    figure_format: str = "pdf",
    precision: int = 3,
) -> dict[str, str]:
    """Copy/render CW sensitivity artifacts when the dedicated sweep CSV exists."""
    table = load_cw_sensitivity_table(sensitivity_csv)
    if table.empty:
        return {}
    tables_dir = ensure_dir(output_tables_dir)
    figures_dir = ensure_dir(output_figures_dir)
    csv_path = tables_dir / "cw_sensitivity.csv"
    tex_path = tables_dir / "cw_sensitivity.tex"
    figure_path = figures_dir / f"cw_sensitivity_tradeoff.{figure_format.lstrip('.')}"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex_tabular(
            table,
            caption="CW-CRC Sensitivity",
            label="tab:cw-sensitivity",
            precision=precision,
        ),
        encoding="utf-8",
    )
    plot_cw_sensitivity_tradeoff(table, figure_path)
    return {
        "cw_sensitivity_csv": str(csv_path),
        "cw_sensitivity_tex": str(tex_path),
        "cw_sensitivity_tradeoff": str(figure_path),
    }


def plot_cw_sensitivity_tradeoff(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot coverage-risk tradeoffs, colored by bound underestimation."""
    output = Path(output_path)
    ensure_dir(output.parent)
    if table.empty:
        return _placeholder(output, "No CW sensitivity data")
    frame = table.copy()
    for column in ("risk", "coverage", "bound_underestimation", "mean_bound_gap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["risk", "coverage"])
    if frame.empty:
        return _placeholder(output, "No finite CW sensitivity metrics")

    datasets = _ordered_values(frame["dataset"], preferred=DATASET_ORDER)
    sweep_types = [value for value in ("target_effective_sample_size", "bound_margin") if value in set(frame["sweep_type"])]
    if not sweep_types:
        sweep_types = sorted(str(value) for value in frame["sweep_type"].dropna().unique())
    n_rows = max(1, len(sweep_types))
    n_cols = max(1, len(datasets))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 3.4 * n_cols), max(3.2, 3.1 * n_rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    color_values = pd.to_numeric(frame["bound_underestimation"], errors="coerce")
    finite_colors = color_values[np.isfinite(color_values)]
    color_min = float(finite_colors.min()) if not finite_colors.empty else 0.0
    color_max = float(finite_colors.max()) if not finite_colors.empty else 1.0
    if np.isclose(color_min, color_max):
        color_max = color_min + 1e-9
    mappable = None

    for row_index, sweep_type in enumerate(sweep_types):
        for col_index, dataset in enumerate(datasets):
            ax = axes[row_index, col_index]
            subset = frame[(frame["dataset"] == dataset) & (frame["sweep_type"] == sweep_type)].copy()
            if subset.empty:
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
            else:
                sizes = 44.0 + 220.0 * pd.to_numeric(subset["mean_bound_gap"], errors="coerce").abs().fillna(0.0)
                scatter = ax.scatter(
                    subset["coverage"],
                    subset["risk"],
                    c=subset["bound_underestimation"],
                    cmap="viridis",
                    vmin=color_min,
                    vmax=color_max,
                    s=sizes,
                    alpha=0.82,
                    edgecolor="0.25",
                    linewidth=0.4,
                )
                mappable = scatter
                current = subset[subset["is_current_setting"].astype(bool)]
                if not current.empty:
                    ax.scatter(
                        current["coverage"],
                        current["risk"],
                        marker="*",
                        s=130,
                        c="none",
                        edgecolor="black",
                        linewidth=1.0,
                        zorder=4,
                    )
                for _, point in subset.iterrows():
                    label = _point_label(point)
                    ax.annotate(label, (point["coverage"], point["risk"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
            ax.axhline(RISK_BUDGET, linestyle="--", linewidth=0.9, color="0.35", alpha=0.75)
            if row_index == 0:
                ax.set_title(_display_dataset(dataset), fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{_display_sweep_type(sweep_type)}\nRisk")
            if row_index == n_rows - 1:
                ax.set_xlabel("Coverage")
            ax.grid(alpha=0.3)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(bottom=-0.01)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
        cbar.set_label("Bound underestimation")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _point_label(row: pd.Series) -> str:
    if row.get("sweep_type") == "target_effective_sample_size":
        return f"ESS {float(row.get('target_effective_sample_size')):g}"
    if row.get("sweep_type") == "bound_margin":
        return f"m {float(row.get('bound_margin')):.2f}"
    return str(row.get("sweep_value", ""))


def _display_sweep_type(sweep_type: str) -> str:
    if sweep_type == "target_effective_sample_size":
        return "ESS sweep"
    if sweep_type == "bound_margin":
        return "Margin sweep"
    return str(sweep_type).replace("_", " ").title()


def _ordered_values(values: pd.Series, *, preferred: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _display_dataset(dataset: str) -> str:
    display = {
        "cifar10c": "CIFAR-10-C",
        "cifar100c": "CIFAR-100-C",
        "synthetic_contextual_shift": "Synthetic",
        "synthetic_stronger_contextual_shift": "Synthetic",
        "wilds_camelyon17": "Camelyon17-WILDS",
    }
    return display.get(str(dataset), str(dataset).replace("_", " "))


def _placeholder(output_path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
