from __future__ import annotations

import pandas as pd
import pytest

from contextual_crc.evaluation.diagnostics import (
    bound_calibration_diagnostics,
    calibration_gap_by_group,
    context_smoothness_diagnostics,
    knn_risk_prediction_diagnostics,
    label_efficiency_summary,
    method_comparison_table,
    window_diagnostics_summary,
)


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "method": ["m"] * 4,
            "context_id": ["a", "a", "b", "b"],
            "time_index": [0, 1, 0, 1],
            "realized_risk": [0.05, 0.2, 0.12, 0.08],
            "coverage": [0.9, 0.8, 0.7, 0.6],
            "upper_bound": [0.1, 0.15, 0.2, 0.1],
            "n_calibration": [100, 100, 50, 50],
            "z0": [0.0, 0.2, 2.0, 2.2],
            "z1": [0.0, 0.1, 2.0, 2.1],
        }
    )


def test_bound_calibration_diagnostics() -> None:
    diagnostics = bound_calibration_diagnostics(make_results(), upper_bound_col="upper_bound")

    assert diagnostics["bound_underestimation_rate"] == pytest.approx(1.0 / 4.0)
    assert "mean_bound_gap" in diagnostics
    assert "mean_excess_risk_over_bound" in diagnostics


def test_calibration_gap_by_group() -> None:
    grouped = calibration_gap_by_group(make_results(), upper_bound_col="upper_bound")

    assert set(grouped["context_id"]) == {"a", "b"}
    assert "mean_bound_gap" in grouped.columns


def test_context_smoothness_diagnostics() -> None:
    diagnostics = context_smoothness_diagnostics(make_results())

    assert diagnostics["n_pairs"] == 6.0
    assert diagnostics["mean_pairwise_distance"] > 0.0
    assert diagnostics["mean_local_slope"] >= 0.0


def test_knn_risk_prediction_diagnostics() -> None:
    diagnostics = knn_risk_prediction_diagnostics(make_results(), k=2)

    assert diagnostics["n_eval"] == 4.0
    assert diagnostics["k"] == 2.0
    assert diagnostics["mae"] >= 0.0
    with pytest.raises(ValueError, match="k"):
        knn_risk_prediction_diagnostics(make_results(), k=0)


def test_label_efficiency_summary() -> None:
    summary = label_efficiency_summary(make_results())

    assert len(summary) == 1
    assert summary.iloc[0]["method"] == "m"
    assert summary.iloc[0]["mean_labels"] == pytest.approx(75.0)
    assert summary.iloc[0]["total_labels"] == pytest.approx(300.0)


def test_method_comparison_table() -> None:
    results = make_results()
    comparison = method_comparison_table(
        {"method_a": results, "method_b": results.assign(realized_risk=results["realized_risk"] * 0.5)},
        risk_budget=0.1,
        upper_bound_col="upper_bound",
    )

    assert comparison["method"].tolist() == ["method_a", "method_b"]
    assert "mean_risk" in comparison.columns
    assert "bound_underestimation_rate" in comparison.columns


def test_window_diagnostics_summary() -> None:
    summary = window_diagnostics_summary(
        make_results(),
        risk_budget=0.1,
        upper_bound_col="upper_bound",
    )

    assert "mean_risk" in summary
    assert "temporal_mean_abs_delta" in summary
