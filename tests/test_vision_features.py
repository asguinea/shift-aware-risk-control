from __future__ import annotations

import numpy as np

from contextual_crc.datasets.vision_features import (
    predict_proba_with_features,
    sanitize_probabilities,
    stable_softmax,
)


def test_stable_softmax_handles_extreme_logits() -> None:
    logits = np.array([[1000.0, -1000.0], [np.nan, np.inf], [-np.inf, -np.inf]])

    probabilities = stable_softmax(logits)

    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3), atol=1e-9)


def test_sanitize_probabilities_replaces_bad_rows() -> None:
    probabilities = np.array([[np.nan, np.nan], [0.0, 0.0], [0.25, 0.75]])

    sanitized = sanitize_probabilities(probabilities, num_classes=2)

    assert np.isfinite(sanitized).all()
    np.testing.assert_allclose(sanitized.sum(axis=1), np.ones(3), atol=1e-9)
    np.testing.assert_allclose(sanitized[0], [0.5, 0.5], atol=1e-9)
    np.testing.assert_allclose(sanitized[1], [0.5, 0.5], atol=1e-9)


def test_predict_proba_with_features_aligns_missing_classes() -> None:
    class FakeModel:
        classes_ = np.array([1])

        def predict_proba(self, features):
            return np.ones((len(features), 1))

    features = np.zeros((3, 2), dtype=np.float32)

    probabilities, logits = predict_proba_with_features(FakeModel(), features, 2)

    assert probabilities.shape == (3, 2)
    assert logits.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(logits).all()
    assert np.all(probabilities[:, 1] > probabilities[:, 0])
