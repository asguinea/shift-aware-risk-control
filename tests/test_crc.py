from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.risk_control.crc import (
    RiskControlConfig,
    calibrate_threshold,
    evaluate_threshold,
    risk_upper_confidence_bound,
)
from contextual_crc.risk_control.losses import LossConfig
from contextual_crc.risk_control.policies import ThresholdPolicy


def make_calibration_examples(n: int = 200) -> pd.DataFrame:
    labels = np.arange(n) % 2
    preds = labels.copy()
    preds[:20] = 1 - preds[:20]
    confidence = np.linspace(0.95, 0.1, n)
    return pd.DataFrame(
        {
            "label": labels,
            "pred": preds,
            "confidence": confidence,
            "context_id": ["ctx"] * n,
        }
    )


def test_risk_upper_confidence_bound_returns_finite_unit_interval() -> None:
    upper = risk_upper_confidence_bound(np.array([0.0, 1.0, 0.0, 0.0]), delta=0.1)

    assert np.isfinite(upper)
    assert 0.0 <= upper <= 1.0


def test_invalid_delta_raises() -> None:
    with pytest.raises(ValueError, match="delta"):
        risk_upper_confidence_bound(np.array([0.0, 1.0]), delta=1.0)


def test_calibrate_threshold_returns_feasible_result_on_easy_data() -> None:
    config = RiskControlConfig(risk_budget=0.2, delta=0.2, min_coverage=0.2)

    result = calibrate_threshold(make_calibration_examples(), config)

    assert result.feasible
    assert result.risk_upper_bound <= config.risk_budget
    assert result.coverage >= config.min_coverage
    assert result.num_candidates > 0


def test_calibrate_threshold_returns_infeasible_when_budget_impossible() -> None:
    config = RiskControlConfig(risk_budget=0.0, delta=0.05)

    result = calibrate_threshold(make_calibration_examples(), config)

    assert not result.feasible
    assert result.risk_upper_bound > config.risk_budget


def test_accepted_only_error_rate_rejected_during_calibration() -> None:
    config = RiskControlConfig(loss=LossConfig("accepted_only_error_rate"))

    with pytest.raises(ValueError, match="accepted_only_error_rate"):
        calibrate_threshold(make_calibration_examples(), config)


def test_evaluate_threshold_returns_expected_keys() -> None:
    examples = make_calibration_examples()
    config = RiskControlConfig(risk_budget=0.2, delta=0.2)
    result = calibrate_threshold(examples, config)

    metrics_from_result = evaluate_threshold(examples, result)
    metrics_from_policy = evaluate_threshold(examples, ThresholdPolicy(threshold=result.threshold))
    metrics_from_float = evaluate_threshold(examples, result.threshold, config)

    for metrics in (metrics_from_result, metrics_from_policy, metrics_from_float):
        assert "coverage" in metrics
        assert "mean_loss" in metrics
        assert "threshold" in metrics
        assert "risk_upper_bound" in metrics
        assert "feasible" in metrics
