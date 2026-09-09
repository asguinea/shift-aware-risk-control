from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns


def pairwise_distances(
    A: np.ndarray,
    B: np.ndarray,
    metric: str = "euclidean",
    *,
    covariance: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute pairwise distances between two embedding matrices."""
    A, B = _validate_pairwise_inputs(A, B)

    if metric == "euclidean":
        diff = A[:, None, :] - B[None, :, :]
        return np.linalg.norm(diff, axis=2)

    if metric == "cosine":
        A_norm = np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = np.linalg.norm(B, axis=1, keepdims=True)
        A_safe = np.divide(A, A_norm, out=np.zeros_like(A, dtype=float), where=A_norm > 0)
        B_safe = np.divide(B, B_norm, out=np.zeros_like(B, dtype=float), where=B_norm > 0)
        similarity = A_safe @ B_safe.T
        return 1.0 - np.clip(similarity, -1.0, 1.0)

    if metric == "mahalanobis":
        if covariance is None:
            covariance = estimate_covariance(np.vstack([A, B]), eps=eps)
        else:
            covariance = np.asarray(covariance, dtype=float)
            if covariance.shape != (A.shape[1], A.shape[1]):
                raise ValueError(
                    "covariance shape must be [d, d], got "
                    f"{covariance.shape} for d={A.shape[1]}"
                )
            covariance = covariance + eps * np.eye(A.shape[1])
        inv_covariance = np.linalg.pinv(covariance)
        diff = A[:, None, :] - B[None, :, :]
        squared = np.einsum("nmd,de,nme->nm", diff, inv_covariance, diff)
        return np.sqrt(np.maximum(squared, 0.0))

    raise ValueError(f"Unknown distance metric: {metric}")


def estimate_covariance(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Estimate a regularized covariance matrix for embeddings."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n_samples, n_features = X.shape
    if n_features == 0:
        raise ValueError("X must have at least one feature")
    if n_samples < 2:
        covariance = np.zeros((n_features, n_features), dtype=float)
    else:
        covariance = np.cov(X, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])
    return np.asarray(covariance, dtype=float) + eps * np.eye(n_features)


def nearest_neighbors(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_ids: Sequence[str],
    metric: str = "euclidean",
    *,
    k: int = 1,
    covariance: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return k nearest reference embeddings for each query embedding."""
    references = list(reference_ids)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(references) != len(reference_embeddings):
        raise ValueError("reference_ids length must match reference_embeddings rows")
    if k > len(references):
        raise ValueError("k must be <= number of references")

    distances = pairwise_distances(
        query_embeddings,
        reference_embeddings,
        metric=metric,
        covariance=covariance,
    )
    rows: list[dict[str, int | float | str]] = []
    for query_index in range(distances.shape[0]):
        order = sorted(range(distances.shape[1]), key=lambda index: (distances[query_index, index], index))
        for rank, reference_index in enumerate(order[:k], start=1):
            rows.append(
                {
                    "query_index": query_index,
                    "neighbor_rank": rank,
                    "reference_id": references[reference_index],
                    "distance": float(distances[query_index, reference_index]),
                    "reference_index": reference_index,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["query_index", "neighbor_rank", "reference_id", "distance", "reference_index"],
    )


def context_centroids(
    embedding_frame: pd.DataFrame,
    context_col: str = "context_id",
) -> pd.DataFrame:
    """Compute one centroid per context from z* embedding columns."""
    if context_col not in embedding_frame.columns:
        raise ValueError(f"embedding_frame is missing context column: {context_col}")
    embedding_columns = get_embedding_columns(embedding_frame)
    if not embedding_columns:
        raise ValueError("embedding_frame must contain z* embedding columns")

    rows: list[dict[str, int | float | str]] = []
    for context_value, group in embedding_frame.groupby(context_col, sort=False, dropna=False):
        row: dict[str, int | float | str] = {
            context_col: context_value,
            "n_windows": int(len(group)),
        }
        if "context_name" in group.columns and group["context_name"].nunique(dropna=False) == 1:
            row["context_name"] = group["context_name"].iloc[0]
        centroid = group[embedding_columns].to_numpy(dtype=float).mean(axis=0)
        for column, value in zip(embedding_columns, centroid, strict=True):
            row[column] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def distances_to_context_centroids(
    query_embedding_frame: pd.DataFrame,
    centroid_frame: pd.DataFrame,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute long-form distances from query windows to context centroids."""
    if "window_id" not in query_embedding_frame.columns:
        raise ValueError("query_embedding_frame must include window_id")
    if "context_id" not in centroid_frame.columns:
        raise ValueError("centroid_frame must include context_id")

    query_columns = get_embedding_columns(query_embedding_frame)
    centroid_columns = get_embedding_columns(centroid_frame)
    if query_columns != centroid_columns:
        raise ValueError(
            "query and centroid embedding columns must match: "
            f"{query_columns} != {centroid_columns}"
        )

    distances = pairwise_distances(
        query_embedding_frame[query_columns].to_numpy(dtype=float),
        centroid_frame[centroid_columns].to_numpy(dtype=float),
        metric=metric,
    )
    rows: list[dict[str, float | str]] = []
    for query_index, window_id in enumerate(query_embedding_frame["window_id"].tolist()):
        for centroid_index, context_id in enumerate(centroid_frame["context_id"].tolist()):
            rows.append(
                {
                    "window_id": window_id,
                    "context_id": context_id,
                    "distance": float(distances[query_index, centroid_index]),
                }
            )
    return pd.DataFrame(rows, columns=["window_id", "context_id", "distance"])


def _validate_pairwise_inputs(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same embedding dimension: {A.shape[1]} != {B.shape[1]}")
    return A, B
