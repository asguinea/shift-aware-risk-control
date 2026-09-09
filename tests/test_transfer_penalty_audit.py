from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.transfer_penalty_audit import write_transfer_penalty_artifacts
from contextual_crc.evaluation.transfer_penalty_audit import (
    _build_certifiability,
    _run_loco,
    _select_primary_family,
)
from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.losses import LossConfig
from contextual_crc.risk_control.source_certificates import (
    accepted_error_scores_finite,
    calibrate_dkw_source,
    dkw_upper_envelope,
)
from contextual_crc.risk_control.transfer_penalty import (
    fit_transfer_penalty,
    observable_transfer_residual_upper,
    predeclared_penalty_grids,
    select_gamma_feasible_threshold,
    transfer_penalty_dominates_constraints,
)


def _pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_context": ["a", "a", "b", "b", "c", "c"],
            "target_context": ["b", "c", "a", "c", "a", "b"],
            "distance": [1.0, 2.0, 1.0, 1.5, 2.0, 1.5],
            "max_residual_upper": [0.03, 0.08, 0.02, 0.05, 0.07, 0.04],
        }
    )


def _examples(context: str, error_start: int = 16) -> pd.DataFrame:
    n = 20
    pred = np.zeros(n, dtype=int)
    pred[error_start:] = 1
    return pd.DataFrame(
        {
            "example_id": [f"{context}-{i}" for i in range(n)],
            "context_id": context,
            "confidence": np.linspace(0.05, 0.95, n),
            "pred": pred,
            "label": np.zeros(n, dtype=int),
        }
    )


def _context_data() -> dict[str, dict]:
    output = {}
    for index, context in enumerate(("a", "b", "c")):
        examples = _examples(context, 16 + index)
        calibration = calibrate_dkw_source(
            examples, risk_budget=0.5, delta=0.05 / 3, side="one_sided"
        )
        output[context] = {
            "examples": examples,
            "scores": examples["confidence"].to_numpy(),
            "error_scores": accepted_error_scores_finite(examples),
            "epsilon": 0.1,
            "centroid": np.asarray([float(index), float(index % 2)]),
            "n_windows": 2,
            "source_calibration": calibration,
        }
    return output


def test_uniform_envelopes_construct_observable_transfer_bound():
    source = np.asarray([-1.0, -1.0, 0.4, 0.8])
    target = np.asarray([-1.0, 0.3, 0.6, 0.9])
    theta = np.linspace(0, 1, 11)
    source_upper = dkw_upper_envelope(source, theta, delta=0.1, side="one_sided")
    target_upper = dkw_upper_envelope(target, theta, delta=0.1, side="one_sided")
    residual = observable_transfer_residual_upper(source_upper, target_upper)
    assert np.all(residual >= 0)
    assert np.all(residual + source_upper + 1e-12 >= target_upper)


def test_observable_residual_upper_bounds_population_residual_on_event():
    source_ucb = np.asarray([0.08, 0.09])
    target_ucb = np.asarray([0.10, 0.07])
    target_truth = np.asarray([0.09, 0.06])
    observed = observable_transfer_residual_upper(source_ucb, target_ucb)
    population_positive = np.maximum(0.0, target_truth - source_ucb)
    assert np.all(population_positive <= observed + 1e-12)


@pytest.mark.parametrize(
    "family",
    ["G0_no_penalty", "G1_global_residual", "G2_pure_lipschitz", "G3_lipschitz_intercept"],
)
def test_predeclared_penalty_families_evaluate(family: str):
    pairs = _pair_frame()
    slopes, intercepts = predeclared_penalty_grids(pairs["distance"], n_slopes=501)
    penalty = fit_transfer_penalty(
        pairs,
        family,
        slope_grid=slopes,
        intercept_grid=intercepts,
    )
    assert penalty.fit_status == "fitted"
    values = penalty.evaluate(np.asarray([0.5, 1.0, 2.0]))
    assert np.all(np.diff(values) >= -1e-12)
    if family != "G0_no_penalty":
        assert transfer_penalty_dominates_constraints(penalty, pairs)


def test_piecewise_penalty_is_monotone():
    pairs = _pair_frame()
    penalty = fit_transfer_penalty(
        pairs, "G4_piecewise_bins", bin_edges=[1.25, 1.75]
    )
    values = penalty.evaluate(np.linspace(0, 3, 20))
    assert np.all(np.diff(values) >= -1e-12)
    assert transfer_penalty_dominates_constraints(penalty, pairs)


def test_finite_grid_selection_dominates_all_constraints():
    pairs = _pair_frame()
    slopes, intercepts = predeclared_penalty_grids(pairs["distance"], n_slopes=1001)
    penalty = fit_transfer_penalty(
        pairs,
        "G3_lipschitz_intercept",
        slope_grid=slopes,
        intercept_grid=intercepts,
    )
    assert transfer_penalty_dominates_constraints(penalty, pairs)


def test_gamma_fitting_does_not_accept_target_labels():
    pairs = _pair_frame()
    first = fit_transfer_penalty(pairs, "G1_global_residual")
    mutated = pairs.copy()
    mutated["unused_final_target_label"] = [0, 1, 1, 0, 1, 0]
    second = fit_transfer_penalty(mutated, "G1_global_residual")
    assert first == second


def test_loco_excludes_heldout_context_from_gamma_fit():
    pairs = _pair_frame()
    context_data = _context_data()
    slopes, intercepts = predeclared_penalty_grids(pairs["distance"])
    risk = RiskControlConfig(
        loss=LossConfig(loss_name="accepted_error"), risk_budget=0.5, delta=0.05
    )
    result = _run_loco(
        "fake",
        ["a", "b", "c"],
        context_data,
        pairs,
        slopes,
        intercepts,
        None,
        risk,
    )
    folds = result[result["row_type"] == "context_fold"]
    assert folds["heldout_excluded_from_fit"].all()
    assert all(
        heldout not in fit_contexts.split("|")
        for heldout, fit_contexts in zip(folds["heldout_context"], folds["fit_contexts"], strict=True)
    )


def test_target_feasible_selection_and_accept_none():
    source = _examples("source", 18)
    source_scores = source["confidence"].to_numpy()
    errors = accepted_error_scores_finite(source)
    deployment = np.asarray([0.2, 0.4, 0.6, 0.8])
    feasible = select_gamma_feasible_threshold(
        deployment,
        source_scores,
        errors,
        source_epsilon=0.02,
        gamma=0.01,
        risk_budget=0.5,
    )
    assert feasible["nontrivial_feasible"]
    collapsed = select_gamma_feasible_threshold(
        deployment,
        source_scores,
        errors,
        source_epsilon=0.2,
        gamma=0.2,
        risk_budget=0.1,
    )
    assert collapsed["forced_accept_none"]
    assert collapsed["coverage_from_scores"] == 0.0
    assert collapsed["operational_bound"] == 0.0


def test_certifiability_flags_and_cifar_sample_sizes():
    context_data = _context_data()
    windows = pd.DataFrame(
        {
            "family": ["G1_global_residual"] * 2,
            "nontrivial_feasible": [True, False],
        }
    )
    result = _build_certifiability("cifar100c", context_data, windows, 0.05 / 6)
    assert result.iloc[0]["target_certifiable_fraction"] == 0.5
    assert result.iloc[0]["n_required_epsilon_0.05"] == 958
    assert result.iloc[0]["n_required_epsilon_0.02"] == 5985


def test_transfer_penalty_artifact_generation(tmp_path):
    assert write_transfer_penalty_artifacts(
        tmp_path,
        final_results_csv=tmp_path / "missing-final.csv",
        certifiability_csv=tmp_path / "missing-cert.csv",
    ) == {}
    final = pd.DataFrame(
        [
            {
                "dataset": "fake",
                "family": "G1_global_residual",
                "is_primary": True,
                "risk": 0.01,
                "coverage": 0.5,
                "bound_underestimation": 0.0,
                "mean_bound_gap": 0.05,
                "nontrivial_feasible_fraction": 1.0,
                "accept_none_fraction": 0.0,
                "mean_gamma": 0.01,
                "mean_source_ucb": 0.08,
                "mean_final_ucb": 0.09,
            }
        ]
    )
    cert = pd.DataFrame(
        [
            {
                "dataset": "fake",
                "family": "G1_global_residual",
                "source_certifiable_fraction": 1.0,
                "target_certifiable_fraction": 1.0,
                "n_validated_contexts": 3,
                "n_evaluation_windows": 2,
            }
        ]
    )
    final_path = tmp_path / "final.csv"
    cert_path = tmp_path / "cert.csv"
    final.to_csv(final_path, index=False)
    cert.to_csv(cert_path, index=False)
    outputs = write_transfer_penalty_artifacts(
        tmp_path, final_results_csv=final_path, certifiability_csv=cert_path
    )
    assert len(outputs) == 4
    assert all((tmp_path / name).is_file() for name in (
        "transfer_penalty_results.csv",
        "transfer_penalty_results.tex",
        "certifiability.csv",
        "certifiability.tex",
    ))


def test_primary_selection_prefers_safe_nontrivial_loco_candidate():
    loco = pd.DataFrame(
        {
            "row_type": ["dataset_aggregate"] * 3,
            "family": [
                "G1_global_residual",
                "G2_pure_lipschitz",
                "G3_lipschitz_intercept",
            ],
            "n_valid_folds": [4, 4, 4],
            "n_folds": [4, 4, 4],
            "bound_underestimation": [0.0, 0.0, 0.0],
            "nontrivial_feasible_fraction": [0.0, 0.5, 0.0],
            "coverage": [0.0, 0.4, 0.0],
        }
    )
    family, status = _select_primary_family(loco, m_contexts=4)
    assert family == "G2_pure_lipschitz"
    assert status == "selected_by_predeclared_loco_rule"
