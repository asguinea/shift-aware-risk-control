from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.evaluation.metrics import (
    aggregate_by_group,
    bound_underestimation_rate,
    bound_violation_rate,
    coverage_risk_correlation,
    max_violation_gap,
    mean_bound_gap,
    mean_excess_risk_over_bound,
    safe_mean,
    safe_std,
    summarize_window_results,
    temporal_variability,
    violation_gap,
    violation_rate,
)


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "context_id": ["a", "a", "b", "b"],
            "time_index": [0, 1, 0, 1],
            "realized_risk": [0.05, 0.2, 0.1, np.nan],
            "coverage": [0.9, 0.8, 0.7, 0.6],
            "upper_bound": [0.1, 0.15, 0.2, np.nan],
        }
    )


def test_safe_mean_and_std() -> None:
    assert safe_mean(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)
    assert safe_std(np.array([1.0, np.nan, 3.0])) == pytest.approx(1.0)
    assert np.isnan(safe_mean(np.array([np.nan])))
    assert np.isnan(safe_std(np.array([])))


def test_violation_metrics_ignore_nan() -> None:
    risks = np.array([0.05, 0.2, np.nan, 0.3])

    assert violation_rate(risks, 0.1) == pytest.approx(2.0 / 3.0)
    assert violation_gap(risks, 0.1) == pytest.approx((0.0 + 0.1 + 0.2) / 3.0)
    assert max_violation_gap(risks, 0.1) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="risk_budget"):
        violation_rate(risks, 1.5)


def test_bound_diagnostics() -> None:
    risks = np.array([0.1, 0.2, 0.3, np.nan])
    bounds = np.array([0.2, 0.1, 0.4, 0.5])

    assert bound_violation_rate(bounds, 0.15) == pytest.approx(3.0 / 4.0)
    assert bound_underestimation_rate(risks, bounds) == pytest.approx(1.0 / 3.0)
    assert mean_bound_gap(risks, bounds) == pytest.approx((0.1 - 0.1 + 0.1) / 3.0)
    assert mean_excess_risk_over_bound(risks, bounds) == pytest.approx(0.1 / 3.0)


def test_summarize_window_results_with_bounds() -> None:
    summary = summarize_window_results(
        make_results(),
        risk_budget=0.1,
        upper_bound_col="upper_bound",
        method_name="method_a",
    )

    assert summary["method"] == "method_a"
    assert summary["n_windows"] == 4.0
    assert summary["mean_risk"] == pytest.approx((0.05 + 0.2 + 0.1) / 3.0)
    assert "bound_underestimation_rate" in summary
    assert "mean_upper_bound" in summary


def test_summarize_window_results_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        summarize_window_results(pd.DataFrame({"coverage": [0.5]}), risk_budget=0.1)


def test_temporal_variability() -> None:
    variability = temporal_variability(make_results())

    assert variability["n_transitions"] == 1.0
    assert variability["mean_abs_delta"] == pytest.approx(0.15)


def test_coverage_risk_correlation_and_group_aggregation() -> None:
    results = make_results()
    corr = coverage_risk_correlation(results)
    grouped = aggregate_by_group(results, group_col="context_id", risk_budget=0.1, upper_bound_col="upper_bound")

    assert np.isfinite(corr)
    assert set(grouped["context_id"]) == {"a", "b"}
    assert "mean_risk" in grouped.columns
