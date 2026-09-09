from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.cw_crc import (
    CWCRCConfig,
    WeightedRiskControlResult,
    assign_cwcrc_thresholds_to_examples,
    calibrate_cwcrc_for_query,
    calibrate_weighted_threshold,
    compute_window_weights,
    effective_sample_size,
    evaluate_cwcrc_window,
    evaluate_cwcrc_windows,
    kernel_values,
    normalize_weights,
    select_bandwidth_for_effective_sample_size,
    weighted_risk_upper_confidence_bound,
    window_weights_to_example_weights,
)
from contextual_crc.risk_control.losses import LossConfig


def make_examples(n: int = 12) -> pd.DataFrame:
    labels = np.arange(n) % 2
    preds = labels.copy()
    preds[[1, 5, 9]] = 1 - preds[[1, 5, 9]]
    return pd.DataFrame(
        {
            "example_id": [f"ex{i}" for i in range(n)],
            "dataset": ["fixture"] * n,
            "split": ["eval"] * n,
            "context_id": ["a"] * 6 + ["b"] * 6,
            "context_name": ["A"] * 6 + ["B"] * 6,
            "label": labels,
            "pred": preds,
            "confidence": np.linspace(0.95, 0.2, n),
        }
    )


def make_calibration_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "dataset": ["fixture"] * 3,
            "split": ["cal"] * 3,
            "context_id": ["a", "a", "b"],
            "context_name": ["A", "A", "B"],
            "start_pos": [0, 3, 6],
            "end_pos": [4, 8, 12],
            "n_examples": [4, 5, 6],
            "example_indices": [[0, 1, 2, 3], [3, 4, 5, 6, 7], [6, 7, 8, 9, 10, 11]],
        }
    )


def make_calibration_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "context_id": ["a", "a", "b"],
            "context_name": ["A", "A", "B"],
            "z0": [0.0, 0.5, 5.0],
            "z1": [0.0, 0.2, 5.0],
        }
    )


def make_eval_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["q0", "q1"],
            "dataset": ["fixture", "fixture"],
            "split": ["eval", "eval"],
            "context_id": ["a", "b"],
            "context_name": ["A", "B"],
            "start_pos": [0, 6],
            "end_pos": [4, 10],
            "n_examples": [4, 4],
            "example_indices": [[0, 1, 2, 3], [6, 7, 8, 9]],
        }
    )


def make_eval_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["q0", "q1"],
            "context_id": ["a", "b"],
            "context_name": ["A", "B"],
            "z0": [0.1, 4.8],
            "z1": [0.0, 5.1],
        }
    )


def make_config(**kwargs) -> CWCRCConfig:
    defaults = {
        "risk_control": RiskControlConfig(risk_budget=0.5, delta=0.2, min_coverage=0.0),
        "bandwidth": 1.0,
        "target_effective_sample_size": 2.0,
    }
    defaults.update(kwargs)
    return CWCRCConfig(**defaults)


def test_kernel_values() -> None:
    distances = np.array([0.0, 1.0, 2.0])

    gaussian = kernel_values(distances, 1.0, "gaussian")
    epanechnikov = kernel_values(distances, 1.0, "epanechnikov")
    uniform = kernel_values(distances, 1.0, "uniform")

    assert gaussian[0] > gaussian[1] > gaussian[2] > 0.0
    assert epanechnikov.tolist() == [1.0, 0.0, 0.0]
    assert uniform.tolist() == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="kernel"):
        kernel_values(distances, 1.0, "bad")


def test_normalize_weights_and_effective_sample_size() -> None:
    normalized = normalize_weights(np.array([1.0, 1.0, 2.0]))
    uniform = normalize_weights(np.zeros(4))

    assert normalized.sum() == pytest.approx(1.0)
    assert uniform.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert effective_sample_size(np.ones(4)) == pytest.approx(4.0)
    assert effective_sample_size(np.array([])) == 0.0


def test_compute_window_weights() -> None:
    weights = compute_window_weights(
        np.array([0.0, 0.0]),
        make_calibration_embedding_frame(),
        bandwidth=1.0,
    )

    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights.iloc[0]["weight"] > weights.iloc[1]["weight"] > weights.iloc[2]["weight"]


def test_select_bandwidth_for_effective_sample_size() -> None:
    bandwidth = select_bandwidth_for_effective_sample_size(
        np.array([0.0, 0.0]),
        make_calibration_embedding_frame(),
        target_effective_sample_size=2.0,
        grid_size=10,
    )
    all_zero = pd.DataFrame({"window_id": ["a", "b"], "z0": [0.0, 0.0], "z1": [0.0, 0.0]})
    zero_bandwidth = select_bandwidth_for_effective_sample_size(
        np.array([0.0, 0.0]),
        all_zero,
        target_effective_sample_size=2.0,
        grid_size=5,
    )

    assert bandwidth > 0.0
    assert zero_bandwidth > 0.0
    with pytest.raises(ValueError, match="grid_size"):
        select_bandwidth_for_effective_sample_size(
            np.array([0.0, 0.0]),
            make_calibration_embedding_frame(),
            grid_size=1,
        )


def test_window_weights_to_example_weights() -> None:
    windows = make_calibration_windows()
    window_weights = pd.DataFrame({"window_id": ["w0", "w1", "w2"], "weight": [0.6, 0.3, 0.1]})

    mean_weights = window_weights_to_example_weights(windows, window_weights, 12, aggregation="mean")
    sum_weights = window_weights_to_example_weights(windows, window_weights, 12, aggregation="sum")

    assert mean_weights.sum() == pytest.approx(1.0)
    assert sum_weights.sum() == pytest.approx(1.0)
    assert not np.allclose(mean_weights, sum_weights)

    bad_windows = windows.copy()
    bad_windows.at[0, "example_indices"] = [0, 99]
    with pytest.raises(ValueError, match="out of range"):
        window_weights_to_example_weights(bad_windows, window_weights, 12)


def test_weighted_risk_upper_confidence_bound() -> None:
    losses = np.array([0.0, 1.0, 0.0, 0.0])
    weights = np.ones(4)

    upper = weighted_risk_upper_confidence_bound(losses, weights, delta=0.2)

    assert np.isfinite(upper)
    assert 0.0 <= upper <= 1.0
    with pytest.raises(ValueError, match="matching"):
        weighted_risk_upper_confidence_bound(losses, np.ones(3), delta=0.2)
    with pytest.raises(ValueError, match="delta"):
        weighted_risk_upper_confidence_bound(losses, weights, delta=1.0)


def test_calibrate_weighted_threshold() -> None:
    examples = make_examples()
    weights = normalize_weights(np.ones(len(examples)))

    result = calibrate_weighted_threshold(examples, weights, make_config(), bandwidth=1.0)

    assert isinstance(result, WeightedRiskControlResult)
    assert result.num_candidates > 0
    assert result.effective_sample_size > 0.0

    bad_config = make_config(risk_control=RiskControlConfig(loss=LossConfig("accepted_only_error_rate")))
    with pytest.raises(ValueError, match="accepted_only_error_rate"):
        calibrate_weighted_threshold(examples, weights, bad_config, bandwidth=1.0)


def test_cwcrc_bound_margin_and_multiplier_adjust_upper_bound() -> None:
    examples = make_examples()
    weights = normalize_weights(np.ones(len(examples)))
    config = make_config(bound_margin=0.25, bound_multiplier=1.25)

    result = calibrate_weighted_threshold(examples, weights, config, bandwidth=1.0)

    assert result.raw_weighted_risk_upper_bound is not None
    assert result.weighted_risk_upper_bound == pytest.approx(
        1.25 * result.raw_weighted_risk_upper_bound + 0.25
    )
    assert not result.feasible

    with pytest.raises(ValueError, match="bound_margin"):
        calibrate_weighted_threshold(examples, weights, make_config(bound_margin=-0.01), bandwidth=1.0)
    with pytest.raises(ValueError, match="bound_multiplier"):
        calibrate_weighted_threshold(examples, weights, make_config(bound_multiplier=-0.01), bandwidth=1.0)


def test_calibrate_cwcrc_for_query() -> None:
    result, window_weights, example_weights = calibrate_cwcrc_for_query(
        make_examples(),
        make_calibration_windows(),
        make_calibration_embedding_frame(),
        np.array([0.0, 0.0]),
        make_config(bandwidth=None),
    )

    assert isinstance(result, WeightedRiskControlResult)
    assert len(window_weights) == 3
    assert example_weights.sum() == pytest.approx(1.0)
    assert result.effective_sample_size > 0.0


def test_evaluate_cwcrc_window() -> None:
    examples = make_examples()
    metrics = evaluate_cwcrc_window(
        examples,
        examples,
        make_calibration_windows(),
        make_calibration_embedding_frame(),
        make_eval_windows().iloc[0],
        make_eval_embedding_frame().iloc[0],
        make_config(),
    )

    for key in (
        "window_id",
        "threshold",
        "bandwidth",
        "effective_sample_size",
        "weighted_risk_estimate",
        "weighted_risk_upper_bound",
        "realized_risk",
        "coverage",
        "n_examples",
        "feasible",
        "violation",
        "bound_violation",
    ):
        assert key in metrics
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["effective_sample_size"] > 0.0


def test_evaluate_cwcrc_windows() -> None:
    examples = make_examples()
    windows = make_eval_windows()
    embeddings = make_eval_embedding_frame()

    evaluated = evaluate_cwcrc_windows(
        examples,
        examples,
        make_calibration_windows(),
        make_calibration_embedding_frame(),
        windows,
        embeddings,
        make_config(),
    )

    assert len(evaluated) == len(windows)
    with pytest.raises(ValueError, match="missing window_id"):
        evaluate_cwcrc_windows(
            examples,
            examples,
            make_calibration_windows(),
            make_calibration_embedding_frame(),
            windows,
            embeddings.iloc[:1],
            make_config(),
        )


def test_assign_cwcrc_thresholds_to_examples() -> None:
    examples = make_examples()
    windows = make_eval_windows()

    assignments = assign_cwcrc_thresholds_to_examples(
        examples,
        examples,
        make_calibration_windows(),
        make_calibration_embedding_frame(),
        windows,
        make_eval_embedding_frame(),
        make_config(),
    )

    assert len(assignments) == int(windows["n_examples"].sum())
    for column in ("window_id", "example_index", "threshold", "bandwidth", "effective_sample_size", "accepted", "loss"):
        assert column in assignments.columns
    assert assignments["accepted"].dtype == bool
