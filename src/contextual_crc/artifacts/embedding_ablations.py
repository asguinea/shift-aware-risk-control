from __future__ import annotations

import hashlib
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
from contextual_crc.signatures.behavioral_signatures import get_signature_feature_columns
from contextual_crc.utils.paths import ensure_dir


ABLATION_COLUMNS = (
    "dataset",
    "dataset_display",
    "run_dir",
    "ablation",
    "ablation_display",
    "n_windows",
    "n_pairs",
    "spearman_rho",
    "pearson_r",
    "mean_pairwise_distance",
    "mean_abs_risk_diff",
    "available",
    "notes",
)

ABLATION_ORDER = (
    "full_embedding",
    "raw_signature_all",
    "confidence_uncertainty_only",
    "probability_logit_only",
    "feature_only",
    "metadata_only",
    "random_embedding",
)

ABLATION_DISPLAY_NAMES = {
    "full_embedding": "Full embedding",
    "raw_signature_all": "Raw signatures",
    "confidence_uncertainty_only": "Confidence/uncertainty",
    "probability_logit_only": "Probability/logit",
    "feature_only": "Feature-only",
    "metadata_only": "Metadata-only",
    "random_embedding": "Random embedding",
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

CONFIDENCE_PATTERNS = (
    "confidence",
    "entropy",
    "margin",
    "energy",
    "uncertainty",
    "selective",
    "acceptance",
    "accept_score",
    "score",
    "prob_max",
    "prob_entropy",
)

PROBABILITY_LOGIT_PATTERNS = (
    "prob_",
    "probability",
    "logit",
)

FEATURE_PATTERNS = (
    "feature",
    "feature_norm",
    "embedding",
    "embed",
    "representation",
    "distance",
)

METADATA_COLUMNS = (
    "context_id",
    "context_name",
    "corruption_type",
    "severity",
    "domain_id",
    "group_id",
    "hospital",
    "source",
)

RANDOM_SEED = 20260505


@dataclass(frozen=True)
class RepresentationResult:
    matrix: np.ndarray
    window_ids: list[str]
    available: bool
    notes: str


def select_signature_columns(
    signatures: pd.DataFrame,
    ablation: str,
    *,
    include_diagnostics: bool = False,
) -> list[str]:
    """Select signature feature columns for one ablation by flexible name matching."""
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=include_diagnostics)
    if ablation == "raw_signature_all":
        return feature_columns
    if ablation == "confidence_uncertainty_only":
        return _pattern_columns(feature_columns, CONFIDENCE_PATTERNS)
    if ablation == "probability_logit_only":
        return _pattern_columns(feature_columns, PROBABILITY_LOGIT_PATTERNS)
    if ablation == "feature_only":
        return _pattern_columns(feature_columns, FEATURE_PATTERNS)
    return []


def compute_pairwise_geometry_metrics(representation: np.ndarray, risks: Sequence[float]) -> dict[str, float]:
    """Compute distance/risk-difference correlations for window representations."""
    matrix = np.asarray(representation, dtype=float)
    risk_array = pd.to_numeric(pd.Series(risks), errors="coerce").to_numpy(dtype=float)
    if matrix.ndim != 2:
        raise ValueError("representation must be a 2D matrix")
    if matrix.shape[0] != risk_array.shape[0]:
        raise ValueError("representation and risks must have matching row counts")

    finite_rows = np.isfinite(risk_array) & np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_rows]
    risk_array = risk_array[finite_rows]
    n_windows = int(matrix.shape[0])
    if n_windows < 2:
        return _empty_metric_summary(n_windows)

    row_i, row_j = np.triu_indices(n_windows, k=1)
    distances = np.linalg.norm(matrix[row_i] - matrix[row_j], axis=1)
    risk_diffs = np.abs(risk_array[row_i] - risk_array[row_j])
    finite_pairs = np.isfinite(distances) & np.isfinite(risk_diffs)
    distances = distances[finite_pairs]
    risk_diffs = risk_diffs[finite_pairs]
    if distances.size == 0:
        return _empty_metric_summary(n_windows)

    return {
        "n_windows": float(n_windows),
        "n_pairs": float(distances.size),
        "spearman_rho": _spearman(distances, risk_diffs),
        "pearson_r": _pearson(distances, risk_diffs),
        "mean_pairwise_distance": float(np.mean(distances)),
        "mean_abs_risk_diff": float(np.mean(risk_diffs)),
    }


def build_representation(
    *,
    ablation: str,
    dataset: str,
    signatures: pd.DataFrame,
    embeddings: pd.DataFrame | None = None,
    random_seed: int = RANDOM_SEED,
) -> RepresentationResult:
    """Build one numeric representation matrix per window for an ablation."""
    if "window_id" not in signatures.columns:
        return RepresentationResult(np.empty((0, 0)), [], False, "signatures missing window_id")
    window_ids = [str(value) for value in signatures["window_id"].tolist()]
    if ablation == "full_embedding":
        if embeddings is None or embeddings.empty or "window_id" not in embeddings.columns:
            return RepresentationResult(np.empty((0, 0)), window_ids, False, "embeddings.parquet unavailable")
        z_columns = [column for column in embeddings.columns if str(column).startswith("z")]
        if not z_columns:
            return RepresentationResult(np.empty((0, 0)), window_ids, False, "no z* embedding columns")
        aligned = signatures[["window_id"]].merge(
            embeddings[["window_id", *z_columns]],
            on="window_id",
            how="left",
        )
        matrix, notes = _standardized_numeric_matrix(aligned[z_columns])
        return RepresentationResult(matrix, window_ids, matrix.shape[1] > 0, notes)

    if ablation in {
        "raw_signature_all",
        "confidence_uncertainty_only",
        "probability_logit_only",
        "feature_only",
    }:
        columns = select_signature_columns(signatures, ablation)
        if not columns:
            return RepresentationResult(np.empty((len(signatures), 0)), window_ids, False, "no matching signature columns")
        matrix, notes = _standardized_numeric_matrix(signatures[columns])
        if matrix.shape[1] == 0:
            return RepresentationResult(matrix, window_ids, False, "matching columns are all constant or non-finite")
        return RepresentationResult(matrix, window_ids, True, f"{len(columns)} selected columns; {notes}")

    if ablation == "metadata_only":
        metadata = build_metadata_frame(signatures)
        if metadata.empty or metadata.shape[1] == 0:
            return RepresentationResult(np.empty((len(signatures), 0)), window_ids, False, "no usable metadata columns")
        matrix, notes = _standardized_numeric_matrix(metadata)
        if matrix.shape[1] == 0:
            return RepresentationResult(matrix, window_ids, False, "metadata is constant across windows")
        return RepresentationResult(matrix, window_ids, True, f"one-hot metadata plus parsed numeric fields; {notes}")

    if ablation == "random_embedding":
        dimension = _embedding_dimension(embeddings)
        matrix = deterministic_random_embedding(dataset, window_ids, dimension=dimension, seed=random_seed)
        matrix, notes = _standardized_numeric_matrix(pd.DataFrame(matrix))
        return RepresentationResult(matrix, window_ids, matrix.shape[1] > 0, f"deterministic seed={random_seed}; {notes}")

    return RepresentationResult(np.empty((0, 0)), window_ids, False, f"unknown ablation: {ablation}")


def build_metadata_frame(signatures: pd.DataFrame) -> pd.DataFrame:
    """Encode deterministic per-window metadata for distance computation."""
    parts: list[pd.DataFrame] = []
    for column in METADATA_COLUMNS:
        if column not in signatures.columns:
            continue
        series = signatures[column]
        if pd.api.types.is_numeric_dtype(series):
            parts.append(pd.DataFrame({column: pd.to_numeric(series, errors="coerce")}))
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=column, dtype=float)
            if not dummies.empty:
                parts.append(dummies)

    context_text = _context_text(signatures)
    if context_text is not None:
        parsed = pd.DataFrame(
            {
                "parsed_severity": context_text.map(_parse_severity),
                "parsed_domain": context_text.map(_parse_domain),
            }
        )
        parsed = parsed.dropna(axis=1, how="all")
        if not parsed.empty:
            parts.append(parsed)

    if not parts:
        return pd.DataFrame(index=signatures.index)
    return pd.concat(parts, axis=1)


def deterministic_random_embedding(
    dataset: str,
    window_ids: Sequence[str],
    *,
    dimension: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate deterministic independent random vectors keyed by dataset and window id."""
    dimension = max(int(dimension), 1)
    rows: list[np.ndarray] = []
    for position, window_id in enumerate(window_ids):
        payload = f"{seed}|{dataset}|{window_id}|{position}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        row_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
        rng = np.random.default_rng(row_seed)
        rows.append(rng.normal(size=dimension))
    if not rows:
        return np.empty((0, dimension), dtype=float)
    return np.vstack(rows)


def make_embedding_ablation_table(run_outputs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Compute embedding geometry ablations for a set of experiment run outputs."""
    rows: list[dict[str, Any]] = []
    for output in run_outputs:
        rows.extend(_run_ablation_rows(output))
    if not rows:
        return pd.DataFrame(columns=ABLATION_COLUMNS)
    frame = pd.DataFrame(rows).loc[:, list(ABLATION_COLUMNS)]
    frame["_dataset_rank"] = frame["dataset"].map(_dataset_rank)
    frame["_ablation_rank"] = frame["ablation"].map(_ablation_rank)
    return frame.sort_values(
        ["_dataset_rank", "dataset", "_ablation_rank", "ablation"],
        kind="mergesort",
    ).drop(columns=["_dataset_rank", "_ablation_rank"]).reset_index(drop=True)


def write_embedding_ablation_artifacts(
    run_outputs: Sequence[Mapping[str, Any]],
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
    *,
    figure_format: str = "pdf",
    precision: int = 3,
) -> dict[str, str]:
    """Write embedding ablation CSV/LaTeX table and Spearman-rho figure."""
    tables_dir = ensure_dir(output_tables_dir)
    figures_dir = ensure_dir(output_figures_dir)
    table = make_embedding_ablation_table(run_outputs)

    csv_path = tables_dir / "embedding_ablations.csv"
    tex_path = tables_dir / "embedding_ablations.tex"
    figure_path = figures_dir / f"embedding_ablation_smoothness.{figure_format.lstrip('.')}"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex_tabular(
            table,
            caption="Behavioral Context Geometry Ablations",
            label="tab:embedding-ablations",
            precision=precision,
        ),
        encoding="utf-8",
    )
    plot_embedding_ablation_smoothness(table, figure_path)

    return {
        "embedding_ablations_csv": str(csv_path),
        "embedding_ablations_tex": str(tex_path),
        "embedding_ablation_smoothness": str(figure_path),
    }


def plot_embedding_ablation_smoothness(table: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot Spearman rho by dataset and ablation for available variants."""
    output = Path(output_path)
    ensure_dir(output.parent)
    if table.empty or "available" not in table.columns:
        return _placeholder(output, "No embedding ablation data")

    frame = table[table["available"].astype(bool)].copy()
    frame["spearman_rho"] = pd.to_numeric(frame["spearman_rho"], errors="coerce")
    frame = frame.dropna(subset=["spearman_rho"])
    if frame.empty:
        return _placeholder(output, "No available embedding ablation correlations")

    datasets = _ordered_values(frame["dataset"], preferred=DATASET_ORDER)
    n_rows, n_cols = _panel_grid(len(datasets))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 4.0 * n_cols), max(3.4, 3.2 * n_rows)),
        squeeze=False,
        sharey=True,
    )
    for panel_index, dataset in enumerate(datasets):
        row_index, col_index = divmod(panel_index, n_cols)
        ax = axes[row_index, col_index]
        subset = frame[frame["dataset"] == dataset].copy()
        subset["ablation_rank"] = subset["ablation"].map(_ablation_rank)
        subset = subset.sort_values("ablation_rank", kind="mergesort")
        x = np.arange(len(subset), dtype=float)
        ax.bar(x, subset["spearman_rho"].to_numpy(dtype=float), color="tab:blue", alpha=0.82)
        ax.axhline(0.0, linewidth=0.9, color="0.35", alpha=0.7)
        ax.set_title(_display_dataset(dataset), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_ablation_label(value) for value in subset["ablation"]], rotation=35, ha="right")
        if col_index == 0:
            ax.set_ylabel("Spearman rho")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-1.0, 1.0)

    for empty_index in range(len(datasets), n_rows * n_cols):
        row_index, col_index = divmod(empty_index, n_cols)
        axes[row_index, col_index].set_axis_off()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _run_ablation_rows(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(output.get("run_dir", ""))
    dataset = str(output.get("dataset") or output.get("run_name") or run_dir.name or "unknown")
    dataset_display = _display_dataset(dataset)
    loaded = _load_run_geometry_inputs(run_dir)
    rows: list[dict[str, Any]] = []

    if loaded is None:
        for ablation in ABLATION_ORDER:
            rows.append(
                _unavailable_row(
                    dataset=dataset,
                    dataset_display=dataset_display,
                    run_dir=run_dir,
                    ablation=ablation,
                    notes="missing signatures, embeddings, or realized window risks",
                )
            )
        return rows

    signatures, embeddings, risks = loaded
    aligned = signatures.merge(risks[["window_id", "realized_risk"]], on="window_id", how="inner")
    if aligned.empty:
        for ablation in ABLATION_ORDER:
            rows.append(
                _unavailable_row(
                    dataset=dataset,
                    dataset_display=dataset_display,
                    run_dir=run_dir,
                    ablation=ablation,
                    notes="no overlapping window_id values between signatures and risk outputs",
                )
            )
        return rows

    for ablation in ABLATION_ORDER:
        result = build_representation(
            ablation=ablation,
            dataset=dataset,
            signatures=aligned.drop(columns=["realized_risk"]),
            embeddings=embeddings,
        )
        if not result.available:
            rows.append(
                _unavailable_row(
                    dataset=dataset,
                    dataset_display=dataset_display,
                    run_dir=run_dir,
                    ablation=ablation,
                    n_windows=len(aligned),
                    notes=result.notes,
                )
            )
            continue
        metrics = compute_pairwise_geometry_metrics(result.matrix, aligned["realized_risk"])
        rows.append(
            {
                "dataset": dataset,
                "dataset_display": dataset_display,
                "run_dir": str(run_dir),
                "ablation": ablation,
                "ablation_display": ABLATION_DISPLAY_NAMES[ablation],
                "n_windows": metrics["n_windows"],
                "n_pairs": metrics["n_pairs"],
                "spearman_rho": metrics["spearman_rho"],
                "pearson_r": metrics["pearson_r"],
                "mean_pairwise_distance": metrics["mean_pairwise_distance"],
                "mean_abs_risk_diff": metrics["mean_abs_risk_diff"],
                "available": bool(np.isfinite(metrics["spearman_rho"])),
                "notes": result.notes if np.isfinite(metrics["spearman_rho"]) else f"{result.notes}; constant pairwise series",
            }
        )
    return rows


def _load_run_geometry_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    signatures_path = run_dir / "intermediate" / "signatures.parquet"
    embeddings_path = run_dir / "intermediate" / "embeddings.parquet"
    risk_path = run_dir / "results" / "ct_crc_windows.csv"
    if not signatures_path.is_file() or not embeddings_path.is_file() or not risk_path.is_file():
        return None
    try:
        signatures = pd.read_parquet(signatures_path, engine="pyarrow")
        embeddings = pd.read_parquet(embeddings_path, engine="pyarrow")
        risks = pd.read_csv(risk_path)
    except (OSError, ValueError, ImportError, pd.errors.EmptyDataError):
        return None
    if "window_id" not in signatures.columns or "window_id" not in risks.columns:
        return None
    risk_col = "realized_risk" if "realized_risk" in risks.columns else "mean_loss"
    if risk_col not in risks.columns:
        return None
    risks = risks[["window_id", risk_col]].rename(columns={risk_col: "realized_risk"})
    risks["realized_risk"] = pd.to_numeric(risks["realized_risk"], errors="coerce")
    risks = risks.dropna(subset=["realized_risk"])
    return signatures, embeddings, risks


def _standardized_numeric_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    if frame.empty:
        return np.empty((len(frame), 0), dtype=float), "no numeric columns"
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    retained: list[str] = []
    columns: list[np.ndarray] = []
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        if np.isnan(values).all():
            continue
        mean = float(np.nanmean(values))
        values = np.where(np.isfinite(values), values, mean)
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 0.0:
            continue
        columns.append((values - mean) / std)
        retained.append(str(column))
    if not columns:
        return np.empty((len(numeric), 0), dtype=float), "all numeric columns were constant or non-finite"
    matrix = np.column_stack(columns)
    return matrix, f"{len(retained)} standardized nonconstant columns"


def _pattern_columns(columns: Sequence[str], patterns: Sequence[str]) -> list[str]:
    selected = []
    for column in columns:
        lower = str(column).lower()
        if any(pattern in lower for pattern in patterns):
            selected.append(str(column))
    return selected


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(x_rank, y_rank)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _empty_metric_summary(n_windows: int) -> dict[str, float]:
    return {
        "n_windows": float(n_windows),
        "n_pairs": 0.0,
        "spearman_rho": float("nan"),
        "pearson_r": float("nan"),
        "mean_pairwise_distance": float("nan"),
        "mean_abs_risk_diff": float("nan"),
    }


def _embedding_dimension(embeddings: pd.DataFrame | None) -> int:
    if embeddings is None or embeddings.empty:
        return 4
    z_columns = [column for column in embeddings.columns if str(column).startswith("z")]
    return len(z_columns) if z_columns else 4


def _context_text(signatures: pd.DataFrame) -> pd.Series | None:
    columns = [column for column in ("context_id", "context_name") if column in signatures.columns]
    if not columns:
        return None
    text = signatures[columns[0]].astype(str)
    for column in columns[1:]:
        text = text + " " + signatures[column].astype(str)
    return text.str.lower()


def _parse_severity(text: str) -> float:
    import re

    match = re.search(r"(?:severity[_\s-]*|_s|\bs)(\d+)", str(text).lower())
    return float(match.group(1)) if match else float("nan")


def _parse_domain(text: str) -> float:
    import re

    match = re.search(r"(?:domain[_\s-]*)(\d+)", str(text).lower())
    return float(match.group(1)) if match else float("nan")


def _unavailable_row(
    *,
    dataset: str,
    dataset_display: str,
    run_dir: Path,
    ablation: str,
    notes: str,
    n_windows: int = 0,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_display": dataset_display,
        "run_dir": str(run_dir),
        "ablation": ablation,
        "ablation_display": ABLATION_DISPLAY_NAMES[ablation],
        "n_windows": float(n_windows),
        "n_pairs": 0.0,
        "spearman_rho": float("nan"),
        "pearson_r": float("nan"),
        "mean_pairwise_distance": float("nan"),
        "mean_abs_risk_diff": float("nan"),
        "available": False,
        "notes": notes,
    }


def _ordered_values(values: pd.Series, *, preferred: tuple[str, ...]) -> list[str]:
    observed = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


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


def _short_ablation_label(ablation: str) -> str:
    labels = {
        "full_embedding": "Full",
        "raw_signature_all": "Raw",
        "confidence_uncertainty_only": "Conf.",
        "probability_logit_only": "Prob./logit",
        "feature_only": "Feature",
        "metadata_only": "Metadata",
        "random_embedding": "Random",
    }
    return labels.get(ablation, ablation.replace("_", " "))


def _ablation_rank(ablation: str) -> int:
    try:
        return ABLATION_ORDER.index(str(ablation))
    except ValueError:
        return len(ABLATION_ORDER)


def _dataset_rank(dataset: str) -> int:
    try:
        return DATASET_ORDER.index(str(dataset))
    except ValueError:
        return len(DATASET_ORDER)


def _placeholder(output_path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
