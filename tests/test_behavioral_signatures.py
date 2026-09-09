from __future__ import annotations

import numpy as np
import pandas as pd

from contextual_crc.signatures.behavioral_signatures import (
    SignatureConfig,
    compute_quantile_features,
    compute_signatures_for_windows,
    compute_window_signature,
    get_signature_feature_columns,
)
from contextual_crc.signatures.windowing import WindowConfig, make_windows


def make_signature_fixture() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    examples = pd.DataFrame(
        {
            "example_id": [f"ex{i}" for i in range(6)],
            "dataset": ["fixture"] * 6,
            "split": ["eval"] * 6,
            "context_id": ["ctx"] * 6,
            "context_name": ["Context"] * 6,
            "label": [0, 1, 2, 0, 1, 2],
            "pred": [0, 2, 2, 0, 0, 2],
            "correct": [True, False, True, True, False, True],
            "confidence": [0.9, 0.6, 0.8, 0.7, 0.55, 0.95],
            "entropy": [0.2, 0.8, 0.4, 0.6, 0.9, 0.1],
            "margin": [0.7, 0.2, 0.5, 0.3, 0.1, 0.8],
            "energy": [-2.0, -1.2, -1.7, -1.4, -1.1, -2.2],
            "time_index": list(range(6)),
        }
    )
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.2, 0.6],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.55, 0.35, 0.1],
            [0.03, 0.02, 0.95],
        ]
    )
    arrays = {
        "logits": np.log(np.clip(probabilities, 1e-12, 1.0)),
        "probabilities": probabilities,
        "feature_embedding": np.array(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
        ),
        "ood_score": np.array([0.1, 0.4, 0.2, 0.5, 0.3, 0.6]),
    }
    return examples, arrays


def test_compute_quantile_features() -> None:
    features = compute_quantile_features(np.array([1.0, 2.0, 3.0]), "score", (0.1, 0.5, 0.9))

    assert "score_mean" in features
    assert "score_std" in features
    assert "score_q10" in features
    assert "score_q50" in features
    assert "score_q90" in features


def test_compute_window_signature_no_label_leakage_by_default() -> None:
    examples, arrays = make_signature_fixture()
    signature = compute_window_signature(examples, arrays, [0, 1, 2], SignatureConfig())

    assert not any(key.startswith("diag_") for key in signature)
    assert not any("label" in key for key in signature)
    assert not any("correct" in key for key in signature)
    assert "confidence_mean" in signature
    assert "prob_class_0_mean" in signature


def test_compute_window_signature_with_diagnostics() -> None:
    examples, arrays = make_signature_fixture()
    signature = compute_window_signature(
        examples,
        arrays,
        [0, 1, 2],
        SignatureConfig(include_label_diagnostics=True),
    )

    assert "diag_accuracy" in signature
    assert "diag_error_rate" in signature
    assert np.isclose(signature["diag_accuracy"], 2.0 / 3.0)


def test_compute_signatures_for_windows() -> None:
    examples, arrays = make_signature_fixture()
    windows = make_windows(
        examples,
        WindowConfig(window_size=3, stride=3, group_by_context=True),
    )

    signatures = compute_signatures_for_windows(examples, arrays, windows, SignatureConfig())

    assert len(signatures) == len(windows)
    for column in ("window_id", "dataset", "split", "context_id", "context_name", "n_examples"):
        assert column in signatures.columns
    feature_columns = get_signature_feature_columns(signatures)
    assert "confidence_mean" in feature_columns
    assert "logit_mean" in feature_columns


def test_get_signature_feature_columns_excludes_metadata_and_diag() -> None:
    signatures = pd.DataFrame(
        {
            "window_id": ["w0"],
            "dataset": ["fixture"],
            "split": ["eval"],
            "context_id": ["ctx"],
            "context_name": ["Context"],
            "start_pos": [0],
            "end_pos": [3],
            "n_examples": [3],
            "confidence_mean": [0.8],
            "diag_accuracy": [1.0],
            "non_numeric": ["x"],
        }
    )

    default_columns = get_signature_feature_columns(signatures)
    diagnostic_columns = get_signature_feature_columns(signatures, include_diagnostics=True)

    assert "confidence_mean" in default_columns
    assert "diag_accuracy" not in default_columns
    assert "diag_accuracy" in diagnostic_columns
    assert "window_id" not in diagnostic_columns
    assert "non_numeric" not in diagnostic_columns
