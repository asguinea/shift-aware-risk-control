from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from contextual_crc.utils.paths import ensure_dir


EMBEDDING_METADATA_COLUMNS = (
    "window_id",
    "dataset",
    "split",
    "context_id",
    "context_name",
    "start_pos",
    "end_pos",
    "n_examples",
)


@dataclass(frozen=True)
class EncoderConfig:
    method: str = "pca"
    n_components: int | None = 8
    whiten: bool = False
    random_seed: int = 123
    fill_value: float = 0.0


class ContextEncoder:
    """Fit-once context encoder for behavioral signature windows."""

    def __init__(self, config: EncoderConfig):
        if config.method not in {"identity", "standardized", "pca"}:
            raise ValueError("EncoderConfig.method must be one of: identity, standardized, pca")
        if config.n_components is not None and config.n_components <= 0:
            raise ValueError("EncoderConfig.n_components must be positive or None")

        self.config = config
        self.method = config.method
        self.actual_n_components_: int | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.feature_columns_: list[str] | None = None

    def fit(
        self,
        signatures: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> "ContextEncoder":
        columns = list(feature_columns)
        if not columns:
            raise ValueError("feature_columns must be non-empty")
        _require_columns(signatures, columns)

        X = self._select_matrix(signatures, columns)
        self.feature_columns_ = columns
        self.scaler_ = None
        self.pca_ = None

        if self.method == "identity":
            self.actual_n_components_ = X.shape[1]
            return self

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if self.method == "standardized":
            self.actual_n_components_ = X_scaled.shape[1]
            return self

        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        requested_components = self.config.n_components
        actual_components = max_components if requested_components is None else min(requested_components, max_components)
        if actual_components <= 0:
            raise ValueError("PCA requires at least one sample and one feature")

        self.pca_ = PCA(
            n_components=actual_components,
            whiten=self.config.whiten,
            random_state=self.config.random_seed,
        )
        self.pca_.fit(X_scaled)
        self.actual_n_components_ = actual_components
        return self

    def transform(self, signatures: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ContextEncoder.transform called before fit")
        assert self.feature_columns_ is not None
        _require_columns(signatures, self.feature_columns_)

        X = self._select_matrix(signatures, self.feature_columns_)
        if self.method == "identity":
            return X

        assert self.scaler_ is not None
        X_scaled = self.scaler_.transform(X)
        if self.method == "standardized":
            return X_scaled

        assert self.pca_ is not None
        return self.pca_.transform(X_scaled)

    def fit_transform(
        self,
        signatures: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> np.ndarray:
        return self.fit(signatures, feature_columns).transform(signatures)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.feature_columns_ or [])

    @property
    def fitted(self) -> bool:
        return self.feature_columns_ is not None and self.actual_n_components_ is not None

    def _select_matrix(self, signatures: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        matrix = signatures.loc[:, list(columns)].to_numpy(dtype=float)
        return np.nan_to_num(
            matrix,
            nan=self.config.fill_value,
            posinf=self.config.fill_value,
            neginf=self.config.fill_value,
        )


def save_context_encoder(encoder: ContextEncoder, path: str | Path) -> Path:
    """Persist a fitted or unfitted context encoder with pickle."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as handle:
        pickle.dump(encoder, handle)
    return output_path


def load_context_encoder(path: str | Path) -> ContextEncoder:
    """Load a context encoder saved by save_context_encoder."""
    input_path = Path(path)
    with input_path.open("rb") as handle:
        encoder = pickle.load(handle)
    if not isinstance(encoder, ContextEncoder):
        raise TypeError(f"Pickle did not contain a ContextEncoder: {type(encoder)!r}")
    return encoder


def build_context_embedding_frame(
    signatures: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Attach dense context embeddings to window metadata."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(signatures) != embeddings.shape[0]:
        raise ValueError(
            "embeddings row count must match signatures rows: "
            f"{embeddings.shape[0]} != {len(signatures)}"
        )
    if "window_id" not in signatures.columns:
        raise ValueError("signatures must include window_id")

    metadata_columns = [column for column in EMBEDDING_METADATA_COLUMNS if column in signatures.columns]
    frame = signatures.loc[:, metadata_columns].copy()
    for dimension in range(embeddings.shape[1]):
        frame[f"z{dimension}"] = embeddings[:, dimension]
    return frame


def get_embedding_columns(embedding_frame: pd.DataFrame) -> list[str]:
    """Return z* embedding columns sorted by numeric suffix."""
    columns: list[tuple[int, str]] = []
    for column in embedding_frame.columns:
        match = re.fullmatch(r"z(\d+)", column)
        if match:
            columns.append((int(match.group(1)), column))
    return [column for _, column in sorted(columns)]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"signatures is missing fitted feature columns: {missing}")
