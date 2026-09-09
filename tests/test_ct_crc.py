from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    CTCRCResult,
    ValidatedContext,
    assign_ctcrc_thresholds_to_examples,
    build_validated_contexts,
    estimate_lipschitz_from_contexts,
    evaluate_ctcrc_window,
    evaluate_ctcrc_windows,
    find_nearest_validated_context,
    transferred_upper_bound,
    validated_contexts_to_frame,
)


def make_calibration_examples(n_per_context: int = 80) -> pd.DataFrame:
    rows = []
    for context_id, context_name, base_confidence in (
        ("source", "Source", 0.95),
        ("shift", "Shift", 0.85),
    ):
        for index in range(n_per_context):
            label = index % 2
            is_error = index < 6 if context_id == "source" else index < 10
            pred = 1 - label if is_error else label
            rows.append(
                {
                    "example_id": f"{context_id}_{index}",
                    "dataset": "fixture",
                    "split": "calibration",
                    "context_id": context_id,
                    "context_name": context_name,
                    "label": label,
                    "pred": pred,
                    "confidence": base_confidence - 0.005 * index,
                }
            )
    return pd.DataFrame(rows)


def make_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["source_w0", "source_w1", "shift_w0", "shift_w1"],
            "context_id": ["source", "source", "shift", "shift"],
            "context_name": ["Source", "Source", "Shift", "Shift"],
            "z0": [0.0, 0.2, 4.0, 4.2],
            "z1": [0.0, 0.1, 4.0, 4.1],
        }
    )


def make_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["eval_w0", "eval_w1"],
            "dataset": ["fixture", "fixture"],
            "split": ["eval", "eval"],
            "context_id": ["source", "shift"],
            "context_name": ["Source", "Shift"],
            "start_pos": [0, 80],
            "end_pos": [20, 100],
            "n_examples": [20, 20],
            "example_indices": [list(range(0, 20)), list(range(80, 100))],
        }
    )


def make_eval_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["eval_w0", "eval_w1"],
            "context_id": ["source", "shift"],
            "context_name": ["Source", "Shift"],
            "z0": [0.1, 4.1],
            "z1": [0.05, 4.05],
        }
    )


def make_config() -> CTCRCConfig:
    return CTCRCConfig(
        risk_control=RiskControlConfig(risk_budget=0.35, delta=0.2, min_coverage=0.0),
        l_hat=0.05,
        epsilon_k=0.1,
    )


def test_transferred_upper_bound() -> None:
    assert transferred_upper_bound(0.1, 2.0, 0.05, 1.0) == pytest.approx(0.25)
    assert transferred_upper_bound(
        0.1,
        2.0,
        0.05,
        1.0,
        bound_margin=0.03,
        bound_multiplier=1.25,
    ) == pytest.approx(0.305)
    assert transferred_upper_bound(0.9, 10.0, 1.0, clip=True) == 1.0
    assert transferred_upper_bound(0.9, 10.0, 1.0, clip=False) > 1.0

    with pytest.raises(ValueError, match="distance"):
        transferred_upper_bound(0.1, -1.0, 0.1)
    with pytest.raises(ValueError, match="l_hat"):
        transferred_upper_bound(0.1, 1.0, -0.1)
    with pytest.raises(ValueError, match="epsilon"):
        transferred_upper_bound(0.1, 1.0, 0.1, -0.1)
    with pytest.raises(ValueError, match="bound_margin"):
        transferred_upper_bound(0.1, 1.0, 0.1, bound_margin=-0.1)
    with pytest.raises(ValueError, match="bound_multiplier"):
        transferred_upper_bound(0.1, 1.0, 0.1, bound_multiplier=-0.1)


def test_find_nearest_validated_context() -> None:
    contexts = [
        ValidatedContext("a", "A", np.array([0.0, 0.0]), 0.5, 0.1, 0.05, 0.8, 50, 2, True),
        ValidatedContext("b", "B", np.array([5.0, 5.0]), 0.6, 0.2, 0.1, 0.7, 50, 2, True),
    ]

    nearest, distance = find_nearest_validated_context(np.array([4.8, 5.1]), contexts)

    assert nearest.context_id == "b"
    assert distance < 0.3


def test_build_validated_contexts() -> None:
    result = build_validated_contexts(
        make_calibration_examples(),
        make_embedding_frame(),
        make_config(),
    )

    assert isinstance(result, CTCRCResult)
    assert len(result.contexts) == 2
    assert {context.context_id for context in result.contexts} == {"source", "shift"}
    assert result.global_result is not None
    assert all(context.n_windows == 2 for context in result.contexts)


def test_build_validated_contexts_missing_embedding_raises() -> None:
    embedding_frame = make_embedding_frame()
    embedding_frame = embedding_frame[embedding_frame["context_id"] != "shift"]

    with pytest.raises(ValueError, match="No calibration embedding"):
        build_validated_contexts(make_calibration_examples(), embedding_frame, make_config())


def test_evaluate_ctcrc_window() -> None:
    examples = make_calibration_examples()
    result = build_validated_contexts(examples, make_embedding_frame(), make_config())
    window = make_windows().iloc[0]
    embedding = make_eval_embedding_frame().iloc[0]

    metrics = evaluate_ctcrc_window(examples, window, embedding, result)

    for key in (
        "window_id",
        "nearest_context_id",
        "novelty_distance",
        "threshold",
        "certificate",
        "transferred_upper_bound",
        "realized_risk",
        "coverage",
        "n_examples",
        "violation",
        "bound_violation",
    ):
        assert key in metrics
    assert metrics["transferred_upper_bound"] >= metrics["certificate"]
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_evaluate_ctcrc_windows() -> None:
    examples = make_calibration_examples()
    result = build_validated_contexts(examples, make_embedding_frame(), make_config())
    windows = make_windows()
    embeddings = make_eval_embedding_frame()

    evaluated = evaluate_ctcrc_windows(examples, windows, embeddings, result)

    assert len(evaluated) == len(windows)
    assert set(evaluated["nearest_context_id"]) == {"source", "shift"}

    with pytest.raises(ValueError, match="missing window_id"):
        evaluate_ctcrc_windows(examples, windows, embeddings.iloc[:1], result)


def test_assign_ctcrc_thresholds_to_examples() -> None:
    examples = make_calibration_examples()
    result = build_validated_contexts(examples, make_embedding_frame(), make_config())
    windows = make_windows()

    assignments = assign_ctcrc_thresholds_to_examples(
        examples,
        windows,
        make_eval_embedding_frame(),
        result,
    )

    assert len(assignments) == int(windows["n_examples"].sum())
    for column in ("window_id", "example_index", "threshold", "nearest_context_id", "accepted", "loss"):
        assert column in assignments.columns
    assert assignments["accepted"].dtype == bool


def test_validated_contexts_to_frame_and_lipschitz() -> None:
    contexts = [
        ValidatedContext("a", "A", np.array([0.0, 0.0]), 0.5, 0.1, 0.05, 0.8, 50, 2, True),
        ValidatedContext("b", "B", np.array([2.0, 0.0]), 0.6, 0.2, 0.15, 0.7, 50, 2, True),
    ]
    frame = validated_contexts_to_frame(contexts)

    estimate = estimate_lipschitz_from_contexts(frame, quantile=None)
    single = estimate_lipschitz_from_contexts(frame.iloc[:1])

    assert {"context_id", "z0", "z1", "risk_estimate", "certificate"}.issubset(frame.columns)
    assert np.isfinite(estimate)
    assert estimate >= 0.0
    assert single == 0.0

    with pytest.raises(ValueError, match="risk_col"):
        estimate_lipschitz_from_contexts(frame.drop(columns=["risk_estimate"]))
    with pytest.raises(ValueError, match="quantile"):
        estimate_lipschitz_from_contexts(frame, quantile=1.5)
