from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.embeddings.context_encoder import (
    ContextEncoder,
    EncoderConfig,
    build_context_embedding_frame,
    get_embedding_columns,
    load_context_encoder,
    save_context_encoder,
)


def make_signatures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(6)],
            "dataset": ["fixture"] * 6,
            "split": ["eval"] * 6,
            "context_id": ["a", "a", "b", "b", "c", "c"],
            "context_name": ["A", "A", "B", "B", "C", "C"],
            "start_pos": [0, 10, 20, 30, 40, 50],
            "end_pos": [10, 20, 30, 40, 50, 60],
            "n_examples": [10] * 6,
            "confidence_mean": [0.9, 0.8, 0.6, 0.55, 0.4, 0.35],
            "entropy_mean": [0.1, 0.2, 0.4, 0.45, 0.7, 0.8],
            "margin_mean": [0.7, 0.6, 0.3, 0.25, 0.1, 0.05],
            "logit_mean": [2.0, 1.5, 0.5, 0.3, -0.2, -0.5],
            "diag_accuracy": [1.0, 0.9, 0.7, 0.6, 0.4, 0.3],
        }
    )


def feature_columns() -> list[str]:
    return ["confidence_mean", "entropy_mean", "margin_mean", "logit_mean"]


def test_identity_encoder_roundtrip(tmp_path) -> None:
    signatures = make_signatures()
    encoder = ContextEncoder(EncoderConfig(method="identity"))

    embeddings = encoder.fit_transform(signatures, feature_columns())

    assert embeddings.shape == (len(signatures), len(feature_columns()))
    assert encoder.fitted
    assert encoder.feature_columns == feature_columns()

    path = save_context_encoder(encoder, tmp_path / "encoder.pkl")
    loaded = load_context_encoder(path)

    np.testing.assert_allclose(loaded.transform(signatures), embeddings)


def test_standardized_encoder_has_zero_mean_on_train() -> None:
    signatures = make_signatures()
    encoder = ContextEncoder(EncoderConfig(method="standardized"))

    embeddings = encoder.fit_transform(signatures, feature_columns())

    assert embeddings.shape == (len(signatures), len(feature_columns()))
    np.testing.assert_allclose(embeddings.mean(axis=0), np.zeros(len(feature_columns())), atol=1e-12)


def test_pca_encoder_reduces_dimension_safely() -> None:
    signatures = make_signatures()
    encoder = ContextEncoder(EncoderConfig(method="pca", n_components=99, random_seed=123))

    embeddings = encoder.fit_transform(signatures, feature_columns())

    assert embeddings.shape == (len(signatures), min(len(signatures), len(feature_columns())))
    assert encoder.actual_n_components_ == min(len(signatures), len(feature_columns()))


def test_transform_before_fit_raises() -> None:
    encoder = ContextEncoder(EncoderConfig(method="identity"))

    with pytest.raises(RuntimeError):
        encoder.transform(make_signatures())


def test_missing_feature_column_raises() -> None:
    signatures = make_signatures()
    encoder = ContextEncoder(EncoderConfig(method="standardized"))
    encoder.fit(signatures, feature_columns())

    with pytest.raises(ValueError, match="entropy_mean"):
        encoder.transform(signatures.drop(columns=["entropy_mean"]))


def test_build_context_embedding_frame() -> None:
    signatures = make_signatures()
    embeddings = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ]
    )

    frame = build_context_embedding_frame(signatures, embeddings)

    assert "window_id" in frame.columns
    assert "context_id" in frame.columns
    assert ["z0", "z1", "z2"] == get_embedding_columns(frame)
    np.testing.assert_allclose(frame[["z0", "z1", "z2"]].to_numpy(), embeddings)


def test_get_embedding_columns_sorts_numerically() -> None:
    frame = pd.DataFrame({"z10": [1.0], "z2": [1.0], "z0": [1.0], "not_z": [1.0]})

    assert get_embedding_columns(frame) == ["z0", "z2", "z10"]


def test_nonfinite_values_are_handled() -> None:
    signatures = make_signatures()
    signatures.loc[0, "confidence_mean"] = np.nan
    signatures.loc[1, "entropy_mean"] = np.inf
    signatures.loc[2, "margin_mean"] = -np.inf
    encoder = ContextEncoder(EncoderConfig(method="standardized", fill_value=0.0))

    embeddings = encoder.fit_transform(signatures, feature_columns())

    assert np.isfinite(embeddings).all()
