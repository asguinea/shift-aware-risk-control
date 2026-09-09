from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.source_certificate_audit import write_source_certificate_artifacts
from contextual_crc.evaluation.leakage_free import (
    _overlap_counts,
    make_leakage_free_window_split,
    subset_examples_and_remap_windows,
)
from contextual_crc.evaluation.source_certificate_audit import (
    decompose_transferred_bound,
    select_transfer_form_threshold,
)
from contextual_crc.risk_control.source_certificates import (
    accepted_error_risks,
    accepted_error_scores_finite,
    calibrate_crc_expected_risk,
    calibrate_dkw_source,
    calibrate_ltt_bonferroni_source,
    dkw_radius,
    dkw_upper_envelope,
    exact_binomial_ltt_p_value,
    fixed_sequence_rejections,
)
from contextual_crc.risk_control.ct_crc_dkw import dkw_uniform_epsilon


def _examples(n: int = 100) -> pd.DataFrame:
    scores = np.linspace(0.01, 0.99, n)
    labels = np.zeros(n, dtype=int)
    pred = np.zeros(n, dtype=int)
    pred[-10:] = 1
    return pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(n)],
            "context_id": "c0",
            "confidence": scores,
            "label": labels,
            "pred": pred,
        }
    )


def test_fixed_evaluation_source_and_audit_windows_are_disjoint():
    examples = _examples(12)
    windows = pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "context_id": ["c0"] * 3,
            "example_indices": [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))],
        }
    )
    split = make_leakage_free_window_split(windows, ["c0"], seed=123)
    cal_examples, cal_windows = subset_examples_and_remap_windows(
        examples, split.deployable_calibration_windows
    )
    eval_examples, _ = subset_examples_and_remap_windows(examples, split.evaluation_windows)
    assert _overlap_counts(
        cal_examples, cal_windows, eval_examples, split.evaluation_windows
    ) == {"example_overlap": 0, "window_overlap": 0}


def test_two_sided_and_one_sided_dkw_radii():
    n, delta = 400, 0.01
    assert dkw_radius(n, delta, side="two_sided") == pytest.approx(
        math.sqrt(math.log(2 / delta) / (2 * n))
    )
    assert dkw_radius(n, delta, side="one_sided") == pytest.approx(
        math.sqrt(math.log(1 / delta) / (2 * n))
    )
    assert dkw_radius(n, delta, side="one_sided") < dkw_radius(
        n, delta, side="two_sided"
    )
    assert dkw_radius(n, delta, side="two_sided") == dkw_uniform_epsilon(n, delta)


def test_one_sided_survival_direction_and_uniform_envelope():
    rng = np.random.default_rng(12)
    values = rng.uniform(0, 1, size=500)
    thresholds = np.linspace(0, 1, 101)
    envelope = dkw_upper_envelope(values, thresholds, delta=0.05, side="one_sided")
    true_survival = 1.0 - thresholds
    assert np.all(true_survival <= envelope + 1e-12)
    empirical = accepted_error_risks(values, thresholds)
    assert np.allclose(
        envelope,
        np.clip(empirical + dkw_radius(500, 0.05, side="one_sided"), 0, 1),
    )


def test_finite_error_score_sentinel_exactly_represents_accepted_error():
    examples = _examples(20)
    values = accepted_error_scores_finite(examples, correct_sentinel=-1.0)
    assert np.isfinite(values).all()
    assert (values[examples["pred"].to_numpy() == examples["label"].to_numpy()] == -1).all()
    for threshold in (0.0, 0.5, 1.0):
        represented = values >= threshold
        direct = (
            (examples["confidence"].to_numpy() >= threshold)
            & (examples["pred"].to_numpy() != examples["label"].to_numpy())
        )
        assert np.array_equal(represented, direct)


def test_budget_decomposition_sums_to_final_bound():
    result = decompose_transferred_bound(
        0.01,
        0.05,
        multiplier=1.25,
        margin=0.01,
        l_hat=0.2,
        distance=0.1,
        epsilon_k=0.0,
    )
    component_sum = (
        result["source_ucb"]
        + result["multiplier_increment"]
        + result["additive_margin"]
        + result["distance_penalty"]
        + result["embedding_penalty"]
        + result["clipping_adjustment"]
    )
    assert component_sum == pytest.approx(result["final_bound"])


def test_original_crc_monotone_rule_uses_exact_finite_sample_correction():
    examples = _examples(100)
    result = calibrate_crc_expected_risk(examples, risk_budget=0.1)
    corrected = (100 * result.empirical_risk + 1.0) / 101.0
    assert result.certificate == pytest.approx(corrected)
    assert corrected <= 0.1 + 1e-12
    # A lower threshold has no smaller risk, confirming least-conservative feasible selection.
    values = accepted_error_scores_finite(examples)
    lower = np.nextafter(result.threshold, -np.inf)
    lower_risk = accepted_error_risks(values, [lower])[0]
    lower_corrected = (100 * lower_risk + 1.0) / 101.0
    assert lower_corrected > 0.1 or lower_risk == result.empirical_risk


def test_exact_binomial_ltt_p_value_lower_tail_direction():
    assert exact_binomial_ltt_p_value(0, 10, 0.1) == pytest.approx(0.9**10)
    assert exact_binomial_ltt_p_value(10, 10, 0.1) == pytest.approx(1.0)
    assert exact_binomial_ltt_p_value(0, 100, 0.1) < exact_binomial_ltt_p_value(
        1, 100, 0.1
    )


def test_ltt_bonferroni_uses_grid_fwer_allocation():
    examples = _examples(400)
    grid = np.linspace(0, 1, 11)
    result = calibrate_ltt_bonferroni_source(
        examples, risk_target=0.1, delta=0.05, theta_grid=grid
    )
    if result.feasible:
        assert result.p_value <= 0.05 / len(grid)
        assert result.certified_set_size >= 1


def test_fixed_sequence_stops_at_first_non_rejection():
    rejected = fixed_sequence_rejections([0.001, 0.02, 0.2, 0.001], 0.05)
    assert rejected.tolist() == [True, True, False, False]


def test_target_selection_accepts_scores_not_deployment_labels():
    calibration = _examples(100)
    source_scores = calibration["confidence"].to_numpy()
    error_scores = accepted_error_scores_finite(calibration)
    deployment_scores = np.asarray([0.2, 0.4, 0.6, 0.8])
    kwargs = dict(
        source_epsilon=0.05,
        distance=0.1,
        l_hat=0.01,
        epsilon_k=0.0,
        multiplier=1.1,
        margin=0.01,
        risk_budget=0.1,
        form="T4_current_full",
    )
    first = select_transfer_form_threshold(
        deployment_scores, source_scores, error_scores, **kwargs
    )
    second = select_transfer_form_threshold(
        deployment_scores.copy(), source_scores, error_scores, **kwargs
    )
    assert first == second


def test_one_sided_dkw_is_no_more_conservative_on_same_data():
    examples = _examples(400)
    two = calibrate_dkw_source(
        examples, risk_budget=0.1, delta=0.05, side="two_sided"
    )
    one = calibrate_dkw_source(
        examples, risk_budget=0.1, delta=0.05, side="one_sided"
    )
    assert one.uncertainty < two.uncertainty
    assert one.coverage >= two.coverage


def test_source_certificate_artifact_presence_and_absence(tmp_path):
    assert write_source_certificate_artifacts(
        tmp_path,
        comparison_csv=tmp_path / "missing-comparison.csv",
        budget_csv=tmp_path / "missing-budget.csv",
    ) == {}
    comparison = pd.DataFrame(
        [
            {
                "row_type": "dataset_aggregate",
                "dataset": "fake",
                "method": "two_sided_dkw_source",
                "mean_empirical_source_risk": 0.01,
                "mean_source_certificate": 0.08,
                "mean_source_calibration_coverage": 0.5,
                "source_nontrivial_fraction": 1.0,
                "mean_heldout_source_risk": 0.02,
                "mean_heldout_source_coverage": 0.5,
                "target_risk": 0.01,
                "target_coverage": 0.4,
            }
        ]
    )
    budget = pd.DataFrame(
        [
            {
                "row_type": "dataset_mean",
                "dataset": "fake",
                "empirical_source_risk": 0.01,
                "source_uncertainty": 0.05,
                "multiplier_increment": 0.01,
                "additive_margin": 0.01,
                "distance_penalty": 0.0,
                "embedding_penalty": 0.0,
                "final_bound": 0.08,
            }
        ]
    )
    comparison_path = tmp_path / "comparison.csv"
    budget_path = tmp_path / "budget.csv"
    comparison.to_csv(comparison_path, index=False)
    budget.to_csv(budget_path, index=False)
    paths = write_source_certificate_artifacts(
        tmp_path, comparison_csv=comparison_path, budget_csv=budget_path
    )
    assert len(paths) == 4
    assert all((tmp_path / name).is_file() for name in (
        "source_certificate_comparison.csv",
        "source_certificate_comparison.tex",
        "ct_budget_decomposition.csv",
        "ct_budget_decomposition.tex",
    ))
