from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.ct_simultaneous_validity import (
    make_ct_simultaneous_validity_table,
    write_ct_simultaneous_validity_artifacts,
)
from contextual_crc.risk_control.crc import RiskControlConfig, risk_upper_confidence_bound
from contextual_crc.risk_control.ct_crc import CTCRCConfig, evaluate_ctcrc_windows
from contextual_crc.risk_control.ct_crc_simultaneous import (
    allocate_context_delta,
    allocate_threshold_deltas,
    build_simultaneously_validated_contexts,
    count_candidate_thresholds,
    total_allocated_failure_probability,
)


def _examples() -> pd.DataFrame:
    rows = []
    for context_index, context_id in enumerate(("a", "b")):
        for index in range(40):
            label = index % 2
            is_error = index < 3 + context_index
            rows.append(
                {
                    "context_id": context_id,
                    "context_name": context_id.upper(),
                    "label": label,
                    "pred": 1 - label if is_error else label,
                    "confidence": 0.99 - 0.01 * index - 0.001 * context_index,
                }
            )
    return pd.DataFrame(rows)


def _calibration_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["a0", "a1", "b0", "b1"],
            "context_id": ["a", "a", "b", "b"],
            "context_name": ["A", "A", "B", "B"],
            "z0": [0.0, 0.1, 2.0, 2.1],
            "z1": [0.0, 0.0, 2.0, 2.0],
        }
    )


def _evaluation_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["eval_a", "eval_b"],
            "context_id": ["a", "b"],
            "context_name": ["A", "B"],
            "example_indices": [list(range(20)), list(range(40, 60))],
        }
    )


def _evaluation_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["eval_a", "eval_b"],
            "context_id": ["a", "b"],
            "context_name": ["A", "B"],
            "z0": [0.05, 2.05],
            "z1": [0.0, 2.0],
        }
    )


def _config() -> CTCRCConfig:
    return CTCRCConfig(
        risk_control=RiskControlConfig(
            risk_budget=0.5,
            delta=0.05,
            bound="empirical_bernstein",
        ),
        l_hat=0.01,
    )


def test_context_and_threshold_delta_allocations() -> None:
    counts = {"a": 10, "b": 20, "c": 5}
    context_delta = allocate_context_delta(0.05, len(counts))
    threshold_deltas = allocate_threshold_deltas(counts, 0.05)

    assert context_delta == pytest.approx(0.05 / 3)
    assert threshold_deltas["a"] == pytest.approx(context_delta / 10)
    assert threshold_deltas["b"] == pytest.approx(context_delta / 20)
    assert total_allocated_failure_probability(counts, threshold_deltas) == pytest.approx(0.05)
    assert total_allocated_failure_probability(counts, threshold_deltas) <= 0.05 + 1e-12


def test_candidate_threshold_counting_includes_both_extremes() -> None:
    examples = pd.DataFrame(
        {
            "context_id": ["a", "a", "a", "b", "b"],
            "confidence": [0.1, 0.1, 0.7, 0.2, 0.9],
        }
    )

    assert count_candidate_thresholds(examples) == {"a": 4, "b": 4}


def test_smaller_delta_is_at_least_as_conservative_for_identical_losses() -> None:
    losses = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0] * 20)
    original = risk_upper_confidence_bound(losses, 0.05, "empirical_bernstein")
    corrected = risk_upper_confidence_bound(losses, 0.05 / 30, "empirical_bernstein")

    assert corrected >= original


def test_fully_simultaneous_ct_evaluates_fake_cached_frames() -> None:
    examples = _examples()
    result, audit = build_simultaneously_validated_contexts(
        examples,
        _calibration_embeddings(),
        _config(),
        library_delta=0.05,
        variant="ct_crc_fully_simultaneous",
    )
    evaluated = evaluate_ctcrc_windows(
        examples,
        _evaluation_windows(),
        _evaluation_embeddings(),
        result,
    )

    assert len(result.contexts) == 2
    assert len(evaluated) == 2
    assert {"realized_risk", "coverage", "transferred_upper_bound"}.issubset(evaluated.columns)
    assert (audit["threshold_delta"] < audit["context_failure_budget"]).all()
    assert audit["allocated_failure_probability"].sum() == pytest.approx(0.05)


def test_single_context_allocation_and_build() -> None:
    examples = _examples().query("context_id == 'a'").reset_index(drop=True)
    embeddings = _calibration_embeddings().query("context_id == 'a'").reset_index(drop=True)
    result, audit = build_simultaneously_validated_contexts(
        examples,
        embeddings,
        _config(),
        library_delta=0.05,
        variant="ct_crc_library_bonferroni",
    )

    assert allocate_context_delta(0.05, 1) == pytest.approx(0.05)
    assert len(result.contexts) == 1
    assert audit.iloc[0]["context_failure_budget"] == pytest.approx(0.05)


def _artifact_results() -> pd.DataFrame:
    rows = []
    for variant, risk, coverage, under in (
        ("original_ct_crc", 0.08, 0.9, 0.1),
        ("ct_crc_library_bonferroni", 0.06, 0.7, 0.0),
        ("ct_crc_fully_simultaneous", 0.04, 0.4, 0.0),
    ):
        rows.append(
            {
                "dataset": "cifar10c",
                "variant": variant,
                "risk": risk,
                "coverage": coverage,
                "bound_underestimation": under,
            }
        )
    return pd.DataFrame(rows)


def test_artifact_generation_present_and_absent(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    assert write_ct_simultaneous_validity_artifacts(
        tables_dir,
        results_csv=tmp_path / "missing.csv",
    ) == {}

    results_path = tmp_path / "results.csv"
    _artifact_results().to_csv(results_path, index=False)
    written = write_ct_simultaneous_validity_artifacts(
        tables_dir,
        results_csv=results_path,
    )

    assert Path(written["ct_simultaneous_validity_csv"]).is_file()
    assert Path(written["ct_simultaneous_validity_tex"]).is_file()
    table = make_ct_simultaneous_validity_table(_artifact_results())
    assert table.iloc[0]["Dataset"] == "CIFAR-10-C"
    assert table.iloc[0]["Fully simultaneous CT"] == "0.040 / 0.400 / 0.000"


@pytest.mark.parametrize(
    ("library_delta", "n_contexts"),
    [(0.0, 2), (1.0, 2), (0.05, 0)],
)
def test_invalid_allocations_raise(library_delta: float, n_contexts: int) -> None:
    with pytest.raises(ValueError):
        allocate_context_delta(library_delta, n_contexts)
