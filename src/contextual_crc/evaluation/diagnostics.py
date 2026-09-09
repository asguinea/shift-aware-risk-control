from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.embeddings.distances import pairwise_distances
from contextual_crc.evaluation.metrics import (
    bound_underestimation_rate,
    mean_bound_gap,
    mean_excess_risk_over_bound,
    safe_mean,
    safe_std,
    summarize_window_results,
    temporal_variability,
)


def bound_calibration_diagnostics(
    results: pd.DataFrame,
    *,
    risk_col: str = "realized_risk",
    upper_bound_col: str = "transferred_upper_bound",
) -> dict[str, float]:
    """Diagnose whether reported bounds are conservative or optimistic."""
    _require_columns(results, [risk_col, upper_bound_col])
    gap = pd.to_numeric(results[upper_bound_col], errors="coerce") - pd.to_numeric(
        results[risk_col], errors="coerce"
    )
    return {
        "bound_underestimation_rate": bound_underestimation_rate(
            results[risk_col], results[upper_bound_col]
        ),
        "mean_bound_gap": mean_bound_gap(results[risk_col], results[upper_bound_col]),
        "std_bound_gap": safe_std(gap),
        "mean_excess_risk_over_bound": mean_excess_risk_over_bound(
            results[risk_col], results[upper_bound_col]
        ),
        "min_bound_gap": _safe_min(gap),
        "max_bound_gap": _safe_max(gap),
    }


def calibration_gap_by_group(
    results: pd.DataFrame,
    *,
    group_col: str = "context_id",
    risk_col: str = "realized_risk",
    upper_bound_col: str = "transferred_upper_bound",
) -> pd.DataFrame:
    """Compute bound-gap diagnostics separately by context or another group."""
    _require_columns(results, [group_col, risk_col, upper_bound_col])
    rows = []
    for group_value, group in results.groupby(group_col, sort=False, dropna=False):
        row = bound_calibration_diagnostics(
            group,
            risk_col=risk_col,
            upper_bound_col=upper_bound_col,
        )
        row[group_col] = group_value
        row["n_windows"] = float(len(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[[group_col, "n_windows"] + [key for key in rows[0] if key not in {group_col, "n_windows"}]]


def context_smoothness_diagnostics(
    embedding_frame: pd.DataFrame,
    *,
    value_col: str = "realized_risk",
    metric: str = "euclidean",
    eps: float = 1e-8,
) -> dict[str, float]:
    """Measure whether nearby context embeddings have similar diagnostic values."""
    pairs = context_smoothness_pairwise_frame(
        embedding_frame,
        value_col=value_col,
        metric=metric,
    )
    if pairs.empty:
        return {
            "n_pairs": 0.0,
            "mean_pairwise_distance": float("nan"),
            "mean_value_difference": float("nan"),
            "distance_value_correlation": float("nan"),
            "mean_local_slope": float("nan"),
            "max_local_slope": float("nan"),
        }

    pair_distances = pairs["distance"].to_numpy(dtype=float)
    pair_differences = pairs["risk_difference"].to_numpy(dtype=float)
    slopes = pair_differences / (pair_distances + eps)
    correlation = _safe_corr(pair_distances, pair_differences)
    return {
        "n_pairs": float(len(pair_distances)),
        "mean_pairwise_distance": safe_mean(pair_distances),
        "mean_value_difference": safe_mean(pair_differences),
        "distance_value_correlation": correlation,
        "mean_local_slope": safe_mean(slopes),
        "max_local_slope": _safe_max(slopes),
    }


def context_smoothness_pairwise_frame(
    embedding_frame: pd.DataFrame,
    *,
    value_col: str = "realized_risk",
    metric: str = "euclidean",
    dataset: str | None = None,
) -> pd.DataFrame:
    """Build pairwise context-distance/risk-difference rows for smoothness plots."""
    _require_columns(embedding_frame, [value_col])
    embedding_columns = get_embedding_columns(embedding_frame)
    if not embedding_columns:
        raise ValueError("embedding_frame must contain z* embedding columns")

    output_columns = _smoothness_pairwise_columns(include_dataset=dataset is not None)
    values = pd.to_numeric(embedding_frame[value_col], errors="coerce").to_numpy(dtype=float)
    embeddings = embedding_frame[embedding_columns].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(embeddings).all(axis=1)
    if int(np.sum(valid)) < 2:
        return pd.DataFrame(columns=output_columns)

    working = embedding_frame.loc[valid].reset_index(drop=True)
    values = values[valid]
    embeddings = embeddings[valid]
    distances = pairwise_distances(embeddings, embeddings, metric=metric)
    value_differences = np.abs(values[:, None] - values[None, :])
    upper = np.triu_indices(len(values), k=1)

    if "window_id" in working.columns:
        ids = working["window_id"].astype(str).to_numpy()
    else:
        ids = working.index.astype(str).to_numpy()

    rows: dict[str, Any] = {
        "window_id_i": ids[upper[0]],
        "window_id_j": ids[upper[1]],
        "distance": distances[upper],
        "risk_difference": value_differences[upper],
        "risk_i": values[upper[0]],
        "risk_j": values[upper[1]],
    }
    if "context_id" in working.columns:
        contexts = working["context_id"].astype(str).to_numpy()
        rows["context_id_i"] = contexts[upper[0]]
        rows["context_id_j"] = contexts[upper[1]]
    if dataset is not None:
        rows["dataset"] = str(dataset)
    return pd.DataFrame(rows).reindex(columns=output_columns)


def knn_risk_prediction_diagnostics(
    embedding_frame: pd.DataFrame,
    *,
    risk_col: str = "realized_risk",
    k: int = 3,
    metric: str = "euclidean",
) -> dict[str, float]:
    """Leave-one-out kNN diagnostic for risk predictability in context space."""
    _require_columns(embedding_frame, [risk_col])
    if k <= 0:
        raise ValueError("k must be positive")
    embedding_columns = get_embedding_columns(embedding_frame)
    if not embedding_columns:
        raise ValueError("embedding_frame must contain z* embedding columns")

    risks = pd.to_numeric(embedding_frame[risk_col], errors="coerce").to_numpy(dtype=float)
    embeddings = embedding_frame[embedding_columns].to_numpy(dtype=float)
    valid = np.isfinite(risks) & np.isfinite(embeddings).all(axis=1)
    risks = risks[valid]
    embeddings = embeddings[valid]
    if len(risks) < 2:
        return {
            "n_eval": float(len(risks)),
            "k": float(k),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "correlation": float("nan"),
        }

    k_eff = min(k, len(risks) - 1)
    distances = pairwise_distances(embeddings, embeddings, metric=metric)
    predictions = []
    observed = []
    for row_index in range(len(risks)):
        order = sorted(
            [index for index in range(len(risks)) if index != row_index],
            key=lambda index: (distances[row_index, index], index),
        )
        neighbor_indices = order[:k_eff]
        predictions.append(float(np.mean(risks[neighbor_indices])))
        observed.append(float(risks[row_index]))

    predictions_array = np.asarray(predictions, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    errors = predictions_array - observed_array
    return {
        "n_eval": float(len(observed_array)),
        "k": float(k_eff),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "correlation": _safe_corr(predictions_array, observed_array),
    }


def label_efficiency_summary(
    results: pd.DataFrame,
    *,
    label_count_col: str = "n_calibration",
    method_col: str | None = "method",
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
) -> pd.DataFrame:
    """Aggregate label-usage and performance columns for later label-efficiency plots."""
    _require_columns(results, [label_count_col, risk_col, coverage_col])
    group_keys = [method_col] if method_col is not None and method_col in results.columns else [None]
    rows: list[dict[str, Any]] = []
    groups = results.groupby(method_col, sort=False, dropna=False) if group_keys[0] else [(None, results)]
    for group_value, group in groups:
        row: dict[str, Any] = {
            "n_rows": float(len(group)),
            "mean_labels": safe_mean(group[label_count_col]),
            "total_labels": float(np.nansum(pd.to_numeric(group[label_count_col], errors="coerce"))),
            "mean_risk": safe_mean(group[risk_col]),
            "mean_coverage": safe_mean(group[coverage_col]),
        }
        if group_keys[0]:
            row[method_col] = group_value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    if group_keys[0]:
        return pd.DataFrame(rows)[[method_col] + [key for key in rows[0] if key != method_col]]
    return pd.DataFrame(rows)


def method_comparison_table(
    method_results: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    *,
    risk_budget: float,
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
    upper_bound_col: str | None = None,
    method_col: str = "method",
) -> pd.DataFrame:
    """Create a compact method-comparison table from one or more result frames."""
    rows: list[dict[str, Any]] = []
    if isinstance(method_results, Mapping):
        iterator = method_results.items()
    else:
        iterator = []
        for index, frame in enumerate(method_results):
            if method_col in frame.columns and len(frame) > 0:
                name = str(frame[method_col].iloc[0])
            else:
                name = f"method_{index}"
            iterator.append((name, frame))

    for method_name, frame in iterator:
        summary = summarize_window_results(
            frame,
            risk_budget=risk_budget,
            risk_col=risk_col,
            coverage_col=coverage_col,
            upper_bound_col=upper_bound_col if upper_bound_col in frame.columns else None,
            method_name=str(method_name),
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def window_diagnostics_summary(
    results: pd.DataFrame,
    *,
    risk_budget: float,
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
    upper_bound_col: str | None = None,
    time_col: str | None = "time_index",
    group_col: str | None = "context_id",
) -> dict[str, float]:
    """Combine risk, bound, and temporal diagnostics for one method output."""
    summary = summarize_window_results(
        results,
        risk_budget=risk_budget,
        risk_col=risk_col,
        coverage_col=coverage_col,
        upper_bound_col=upper_bound_col,
    )
    temporal = temporal_variability(
        results,
        value_col=risk_col,
        sort_by=time_col,
        group_col=group_col,
    )
    return {**summary, **{f"temporal_{key}": value for key, value in temporal.items()}}


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _smoothness_pairwise_columns(*, include_dataset: bool) -> list[str]:
    columns = [
        "window_id_i",
        "window_id_j",
        "distance",
        "risk_difference",
        "risk_i",
        "risk_j",
        "context_id_i",
        "context_id_j",
    ]
    if include_dataset:
        return ["dataset", *columns]
    return columns


def _safe_corr(first: np.ndarray | pd.Series, second: np.ndarray | pd.Series) -> float:
    first_array = pd.to_numeric(pd.Series(first).to_numpy(), errors="coerce").astype(float)
    second_array = pd.to_numeric(pd.Series(second).to_numpy(), errors="coerce").astype(float)
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("inputs must have matching lengths")
    mask = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[mask]
    second_array = second_array[mask]
    if len(first_array) < 2 or np.std(first_array) == 0.0 or np.std(second_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_array, second_array)[0, 1])


def _safe_min(values: np.ndarray | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values).to_numpy(), errors="coerce").astype(float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan")
    return float(np.min(numeric))


def _safe_max(values: np.ndarray | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values).to_numpy(), errors="coerce").astype(float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan")
    return float(np.max(numeric))
