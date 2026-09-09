from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.embeddings.distances import (
    context_centroids,
    distances_to_context_centroids,
    nearest_neighbors,
    pairwise_distances,
)


def test_pairwise_euclidean() -> None:
    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0], [6.0, 8.0]])

    distances = pairwise_distances(A, B, metric="euclidean")

    expected = np.array([[0.0, 10.0], [5.0, 5.0]])
    np.testing.assert_allclose(distances, expected)


def test_pairwise_cosine_handles_zero_vectors() -> None:
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    B = np.array([[0.0, 0.0], [1.0, 0.0]])

    distances = pairwise_distances(A, B, metric="cosine")

    assert distances.shape == (2, 2)
    assert np.isfinite(distances).all()
    assert distances[1, 1] == pytest.approx(0.0)


def test_pairwise_mahalanobis_shape_and_finite() -> None:
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[2.0, 2.0], [3.0, 3.0]])

    distances = pairwise_distances(A, B, metric="mahalanobis")

    assert distances.shape == (2, 2)
    assert np.isfinite(distances).all()


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        pairwise_distances(np.zeros((1, 2)), np.zeros((1, 2)), metric="bad")


def test_nearest_neighbors() -> None:
    queries = np.array([[0.1, 0.0], [2.9, 0.0]])
    references = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    reference_ids = ["a", "b", "c"]

    neighbors = nearest_neighbors(queries, references, reference_ids, k=2)

    assert len(neighbors) == 4
    first_query = neighbors[neighbors["query_index"] == 0].sort_values("neighbor_rank")
    assert first_query.iloc[0]["reference_id"] == "a"
    second_query = neighbors[neighbors["query_index"] == 1].sort_values("neighbor_rank")
    assert second_query.iloc[0]["reference_id"] == "b"

    with pytest.raises(ValueError, match="k"):
        nearest_neighbors(queries, references, reference_ids, k=0)
    with pytest.raises(ValueError, match="k"):
        nearest_neighbors(queries, references, reference_ids, k=4)


def make_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "context_id": ["a", "a", "b", "b"],
            "context_name": ["A", "A", "B", "B"],
            "z0": [0.0, 2.0, 10.0, 12.0],
            "z1": [0.0, 2.0, 10.0, 12.0],
        }
    )


def test_context_centroids() -> None:
    centroids = context_centroids(make_embedding_frame())

    assert len(centroids) == 2
    assert centroids.loc[centroids["context_id"] == "a", "n_windows"].iloc[0] == 2
    a = centroids[centroids["context_id"] == "a"].iloc[0]
    assert a["z0"] == pytest.approx(1.0)
    assert a["z1"] == pytest.approx(1.0)
    assert a["context_name"] == "A"


def test_distances_to_context_centroids() -> None:
    frame = make_embedding_frame()
    centroids = context_centroids(frame)

    distances = distances_to_context_centroids(frame, centroids)

    assert set(distances.columns) == {"window_id", "context_id", "distance"}
    assert len(distances) == len(frame) * len(centroids)
    assert np.isfinite(distances["distance"]).all()
