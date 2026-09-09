from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.ct_selection_valid import (
    make_ct_selection_valid_table,
    write_ct_selection_valid_artifacts,
)
from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.ct_crc import CTCRCConfig
from contextual_crc.risk_control.ct_crc_dkw import (
    DKWCTResult,
    DKWValidatedContext,
    accepted_error_losses_from_scores,
    accepted_error_score_transform,
    build_dkw_validated_contexts,
    dkw_uniform_certificates,
    dkw_uniform_epsilon,
    empirical_accepted_error_risks,
    evaluate_dkw_target_feasible_windows,
    select_target_feasible_threshold,
)
from contextual_crc.risk_control.ct_crc_simultaneous import allocate_context_delta
from contextual_crc.risk_control.losses import compute_losses
from contextual_crc.risk_control.policies import apply_threshold_policy


def _nested_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context_id": ["a", "a", "a", "a"],
            "context_name": ["A", "A", "A", "A"],
            "confidence": [0.2, 0.5, 0.8, 0.9],
            "label": [0, 0, 1, 1],
            "pred": [1, 0, 0, 1],
        }
    )


def _manual_result(*, n: int = 100, risk_budget: float = 0.14, margin: float = 0.0) -> DKWCTResult:
    source_scores = np.linspace(0.01, 0.99, n)
    error_scores = np.full(n, -np.inf)
    error_scores[:3] = [0.7, 0.8, 0.9]
    delta = 0.1
    epsilon = dkw_uniform_epsilon(n, delta)
    context = DKWValidatedContext(
        context_id="a",
        context_name="A",
        centroid=np.asarray([0.0, 0.0]),
        calibration_scores=source_scores,
        error_scores=error_scores,
        source_threshold=0.9,
        source_certificate=epsilon + 0.01,
        risk_estimate=0.01,
        coverage=0.1,
        n_calibration=n,
        n_windows=1,
        feasible=True,
        epsilon=epsilon,
        delta=delta,
    )
    return DKWCTResult(
        config=CTCRCConfig(
            risk_control=RiskControlConfig(
                risk_budget=risk_budget,
                delta=0.05,
                bound="empirical_bernstein",
            ),
            l_hat=0.0,
            bound_margin=margin,
        ),
        contexts=[context],
        library_alpha=delta,
        per_context_delta=delta,
    )


def test_accepted_error_loss_is_nested_threshold_class() -> None:
    examples = _nested_examples()
    error_scores = accepted_error_score_transform(examples)
    previous = None
    for threshold in (-0.1, 0.2, 0.5, 0.8, 0.95):
        transformed = accepted_error_losses_from_scores(error_scores, threshold)
        accepted = apply_threshold_policy(examples, "confidence", threshold, "greater_equal")
        actual = compute_losses(examples, accepted, RiskControlConfig().loss)
        assert np.array_equal(transformed, actual)
        if previous is not None:
            assert np.all(transformed <= previous)
        previous = transformed


def test_error_score_transform_uses_negative_infinity_for_correct_examples() -> None:
    transformed = accepted_error_score_transform(_nested_examples())

    assert transformed.tolist()[0] == pytest.approx(0.2)
    assert np.isneginf(transformed[1])
    assert transformed.tolist()[2] == pytest.approx(0.8)
    assert np.isneginf(transformed[3])


def test_two_sided_dkw_epsilon_formula() -> None:
    expected = np.sqrt(np.log(2.0 / 0.025) / (2.0 * 500))
    assert dkw_uniform_epsilon(500, 0.025) == pytest.approx(expected)


def test_uniform_source_certificates_cover_empirical_risk_and_are_monotone() -> None:
    error_scores = accepted_error_score_transform(_nested_examples())
    thresholds = np.asarray([0.1, 0.2, 0.5, 0.8, 0.95])
    empirical = empirical_accepted_error_risks(error_scores, thresholds)
    certificates = dkw_uniform_certificates(error_scores, thresholds, delta=0.1)

    assert np.all(np.diff(empirical) <= 0.0)
    assert np.all(np.diff(certificates) <= 1e-12)
    assert np.all(certificates >= empirical)


def test_per_context_allocation_sums_to_library_alpha() -> None:
    per_context = allocate_context_delta(0.05, 6)
    assert 6 * per_context == pytest.approx(0.05)


def test_build_dkw_context_library_uses_uniform_delta() -> None:
    examples = pd.concat([_nested_examples(), _nested_examples().assign(context_id="b", context_name="B")])
    embeddings = pd.DataFrame(
        {
            "window_id": ["a0", "b0"],
            "context_id": ["a", "b"],
            "context_name": ["A", "B"],
            "z0": [0.0, 1.0],
        }
    )
    config = CTCRCConfig(risk_control=RiskControlConfig(risk_budget=0.9))
    result, audit = build_dkw_validated_contexts(
        examples,
        embeddings,
        config,
        library_alpha=0.05,
    )

    assert len(result.contexts) == 2
    assert result.per_context_delta == pytest.approx(0.025)
    assert np.allclose(audit["delta"].to_numpy(dtype=float), 0.025)


def test_target_feasible_selection_maximizes_unlabeled_coverage() -> None:
    result = _manual_result()
    scores = np.asarray([0.5, 0.8, 0.92, 0.95])
    selection = select_target_feasible_threshold(scores, np.asarray([0.0, 0.0]), result)

    assert selection.nontrivial_feasible is True
    assert selection.forced_accept_none is False
    assert selection.coverage == pytest.approx(0.5)
    assert selection.n_accepted == 2
    assert selection.transferred_upper_bound <= result.config.risk_control.risk_budget


def test_target_feasible_selection_uses_accept_none_fallback() -> None:
    result = _manual_result(n=10, risk_budget=0.05, margin=0.1)
    scores = np.asarray([0.5, 0.8, 0.92, 0.95])
    selection = select_target_feasible_threshold(scores, np.asarray([0.0, 0.0]), result)

    assert selection.nontrivial_feasible is False
    assert selection.forced_accept_none is True
    assert selection.coverage == 0.0
    assert selection.n_accepted == 0
    assert selection.transferred_upper_bound == 0.0


def test_target_selection_interface_has_no_deployment_labels() -> None:
    parameters = set(inspect.signature(select_target_feasible_threshold).parameters)
    assert parameters == {"deployment_scores", "query_embedding", "result"}


def test_target_feasible_window_evaluation_keeps_selected_bound_feasible() -> None:
    examples = pd.DataFrame(
        {
            "confidence": [0.5, 0.8, 0.92, 0.95],
            "label": [0, 1, 0, 1],
            "pred": [1, 1, 0, 0],
            "context_id": ["a"] * 4,
            "context_name": ["A"] * 4,
        }
    )
    windows = pd.DataFrame(
        {
            "window_id": ["w0"],
            "context_id": ["a"],
            "context_name": ["A"],
            "example_indices": [[0, 1, 2, 3]],
        }
    )
    embeddings = pd.DataFrame(
        {
            "window_id": ["w0"],
            "context_id": ["a"],
            "z0": [0.0],
            "z1": [0.0],
        }
    )
    result = _manual_result()
    evaluated = evaluate_dkw_target_feasible_windows(examples, windows, embeddings, result)

    assert len(evaluated) == 1
    assert evaluated.iloc[0]["transferred_upper_bound"] <= result.config.risk_control.risk_budget
    assert bool(evaluated.iloc[0]["bound_feasible"])
    assert evaluated.iloc[0]["num_accepted"] == 2


def test_target_threshold_is_unchanged_when_only_evaluation_labels_change() -> None:
    examples = pd.DataFrame(
        {
            "confidence": [0.5, 0.8, 0.92, 0.95],
            "label": [0, 1, 0, 1],
            "pred": [1, 1, 0, 0],
            "context_id": ["a"] * 4,
            "context_name": ["A"] * 4,
        }
    )
    changed_labels = examples.assign(label=[1, 0, 1, 0])
    windows = pd.DataFrame(
        {
            "window_id": ["w0"],
            "context_id": ["a"],
            "context_name": ["A"],
            "example_indices": [[0, 1, 2, 3]],
        }
    )
    embeddings = pd.DataFrame(
        {
            "window_id": ["w0"],
            "context_id": ["a"],
            "z0": [0.0],
            "z1": [0.0],
        }
    )
    result = _manual_result()

    original = evaluate_dkw_target_feasible_windows(examples, windows, embeddings, result)
    relabeled = evaluate_dkw_target_feasible_windows(changed_labels, windows, embeddings, result)

    assert relabeled.iloc[0]["threshold"] == pytest.approx(original.iloc[0]["threshold"])
    assert relabeled.iloc[0]["coverage"] == pytest.approx(original.iloc[0]["coverage"])
    assert relabeled.iloc[0]["transferred_upper_bound"] == pytest.approx(
        original.iloc[0]["transferred_upper_bound"]
    )


def _artifact_results() -> pd.DataFrame:
    variants = (
        "original_ct",
        "library_bonferroni_ct",
        "fully_threshold_bonferroni_ct",
        "dkw_target_feasible_ct",
    )
    return pd.DataFrame(
        {
            "dataset": ["cifar10c"] * 4,
            "variant": variants,
            "risk": [0.08, 0.07, 0.04, 0.05],
            "coverage": [0.8, 0.7, 0.3, 0.6],
            "bound_underestimation": [0.1, 0.0, 0.0, 0.0],
            "nontrivial_feasible_fraction": [0.2, 0.3, 1.0, 0.9],
        }
    )


def test_ct_selection_valid_artifact_generation_present_and_absent(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    assert write_ct_selection_valid_artifacts(
        tables_dir,
        results_csv=tmp_path / "missing.csv",
    ) == {}

    results_path = tmp_path / "results.csv"
    _artifact_results().to_csv(results_path, index=False)
    written = write_ct_selection_valid_artifacts(tables_dir, results_csv=results_path)

    assert Path(written["ct_selection_valid_csv"]).is_file()
    assert Path(written["ct_selection_valid_tex"]).is_file()
    table = make_ct_selection_valid_table(_artifact_results())
    assert table.iloc[0]["Dataset"] == "CIFAR-10-C"
    assert table.iloc[0]["DKW target-feasible CT"] == "0.050 / 0.600 / 0.000 / 0.900"
