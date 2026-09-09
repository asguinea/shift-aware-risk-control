from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_mean(values: np.ndarray | pd.Series) -> float:
    """NaN-safe mean that returns NaN for empty or all-NaN inputs."""
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_std(values: np.ndarray | pd.Series) -> float:
    """NaN-safe standard deviation that returns NaN for empty or all-NaN inputs."""
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def violation_rate(
    risks: np.ndarray | pd.Series,
    risk_budget: float,
) -> float:
    """Fraction of finite risks above the target budget."""
    _validate_risk_budget(risk_budget)
    finite = _finite_values(risks)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite > risk_budget))


def violation_gap(
    risks: np.ndarray | pd.Series,
    risk_budget: float,
) -> float:
    """Mean positive excess risk above the target budget."""
    _validate_risk_budget(risk_budget)
    finite = _finite_values(risks)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, finite - risk_budget)))


def max_violation_gap(
    risks: np.ndarray | pd.Series,
    risk_budget: float,
) -> float:
    """Maximum positive excess risk above the target budget."""
    _validate_risk_budget(risk_budget)
    finite = _finite_values(risks)
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.maximum(0.0, finite - risk_budget)))


def bound_violation_rate(
    upper_bounds: np.ndarray | pd.Series,
    risk_budget: float,
) -> float:
    """Fraction of finite upper bounds above the target budget."""
    return violation_rate(upper_bounds, risk_budget)


def bound_underestimation_rate(
    realized_risks: np.ndarray | pd.Series,
    upper_bounds: np.ndarray | pd.Series,
) -> float:
    """Fraction of finite pairs where realized risk exceeds the reported bound."""
    risks, bounds = _finite_pairs(realized_risks, upper_bounds)
    if risks.size == 0:
        return float("nan")
    return float(np.mean(risks > bounds))


def mean_bound_gap(
    realized_risks: np.ndarray | pd.Series,
    upper_bounds: np.ndarray | pd.Series,
) -> float:
    """Mean upper_bound - realized_risk; positive values indicate conservatism."""
    risks, bounds = _finite_pairs(realized_risks, upper_bounds)
    if risks.size == 0:
        return float("nan")
    return float(np.mean(bounds - risks))


def mean_excess_risk_over_bound(
    realized_risks: np.ndarray | pd.Series,
    upper_bounds: np.ndarray | pd.Series,
) -> float:
    """Mean positive realized-risk excess above the reported bound."""
    risks, bounds = _finite_pairs(realized_risks, upper_bounds)
    if risks.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, risks - bounds)))


def summarize_window_results(
    results: pd.DataFrame,
    *,
    risk_budget: float,
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
    upper_bound_col: str | None = None,
    method_name: str | None = None,
) -> dict[str, float | str]:
    """Summarize window-level method outputs into risk, coverage, and bound metrics."""
    _validate_risk_budget(risk_budget)
    _require_columns(results, [risk_col, coverage_col])

    risks = results[risk_col]
    coverage = results[coverage_col]
    summary: dict[str, float | str] = {}
    if method_name is not None:
        summary["method"] = method_name
    summary.update(
        {
            "n_windows": float(len(results)),
            "mean_risk": safe_mean(risks),
            "std_risk": safe_std(risks),
            "violation_rate": violation_rate(risks, risk_budget),
            "violation_gap": violation_gap(risks, risk_budget),
            "max_violation_gap": max_violation_gap(risks, risk_budget),
            "mean_coverage": safe_mean(coverage),
            "std_coverage": safe_std(coverage),
            "mean_deferral_rate": safe_mean(1.0 - pd.to_numeric(coverage, errors="coerce")),
        }
    )

    if upper_bound_col is not None:
        _require_columns(results, [upper_bound_col])
        bounds = results[upper_bound_col]
        summary.update(
            {
                "bound_violation_rate": bound_violation_rate(bounds, risk_budget),
                "bound_underestimation_rate": bound_underestimation_rate(risks, bounds),
                "mean_bound_gap": mean_bound_gap(risks, bounds),
                "mean_excess_risk_over_bound": mean_excess_risk_over_bound(risks, bounds),
                "mean_upper_bound": safe_mean(bounds),
            }
        )
    return summary


def temporal_variability(
    results: pd.DataFrame,
    *,
    value_col: str = "realized_risk",
    sort_by: str | None = "time_index",
    group_col: str | None = "context_id",
) -> dict[str, float]:
    """Summarize step-to-step variability of a window-level metric."""
    _require_columns(results, [value_col])
    if sort_by is not None and sort_by not in results.columns:
        sort_by = None
    if group_col is not None and group_col not in results.columns:
        group_col = None

    deltas: list[float] = []
    groups = results.groupby(group_col, sort=False, dropna=False) if group_col else [(None, results)]
    for _, group in groups:
        ordered = group.sort_values(sort_by, kind="mergesort") if sort_by else group
        values = pd.to_numeric(ordered[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size >= 2:
            deltas.extend(np.diff(values).tolist())

    deltas_array = np.asarray(deltas, dtype=float)
    abs_deltas = np.abs(deltas_array)
    return {
        "n_transitions": float(deltas_array.size),
        "mean_abs_delta": safe_mean(abs_deltas),
        "std_delta": safe_std(deltas_array),
        "max_abs_delta": float(np.max(abs_deltas)) if abs_deltas.size else float("nan"),
    }


def coverage_risk_correlation(
    results: pd.DataFrame,
    *,
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
) -> float:
    """Pearson correlation between coverage and realized risk."""
    _require_columns(results, [risk_col, coverage_col])
    risks, coverage = _finite_pairs(results[risk_col], results[coverage_col])
    if risks.size < 2:
        return float("nan")
    if np.std(risks) == 0.0 or np.std(coverage) == 0.0:
        return float("nan")
    return float(np.corrcoef(coverage, risks)[0, 1])


def aggregate_by_group(
    results: pd.DataFrame,
    *,
    group_col: str,
    risk_budget: float,
    risk_col: str = "realized_risk",
    coverage_col: str = "coverage",
    upper_bound_col: str | None = None,
) -> pd.DataFrame:
    """Compute window-result summaries independently for each metadata group."""
    _require_columns(results, [group_col])
    rows: list[dict[str, Any]] = []
    for group_value, group in results.groupby(group_col, sort=False, dropna=False):
        row = summarize_window_results(
            group,
            risk_budget=risk_budget,
            risk_col=risk_col,
            coverage_col=coverage_col,
            upper_bound_col=upper_bound_col,
        )
        row[group_col] = group_value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    leading = [group_col]
    trailing = [column for column in rows[0] if column not in leading]
    return pd.DataFrame(rows)[leading + trailing]


def _finite_values(values: np.ndarray | pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values).to_numpy(), errors="coerce").astype(float)
    return numeric[np.isfinite(numeric)]


def _finite_pairs(
    first: np.ndarray | pd.Series,
    second: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = pd.to_numeric(pd.Series(first).to_numpy(), errors="coerce").astype(float)
    second_array = pd.to_numeric(pd.Series(second).to_numpy(), errors="coerce").astype(float)
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("inputs must have matching lengths")
    mask = np.isfinite(first_array) & np.isfinite(second_array)
    return first_array[mask], second_array[mask]


def _validate_risk_budget(risk_budget: float) -> None:
    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("risk_budget must be in [0, 1]")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
