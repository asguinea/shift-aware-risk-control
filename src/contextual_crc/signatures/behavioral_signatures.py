from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


SIGNATURE_METADATA_COLUMNS = (
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
class SignatureConfig:
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    include_confidence: bool = True
    include_entropy: bool = True
    include_margin: bool = True
    include_energy: bool = True
    include_probability_summary: bool = True
    include_logit_summary: bool = True
    include_ood_score: bool = True
    include_feature_norm: bool = True
    include_label_diagnostics: bool = False


def compute_quantile_features(
    values: np.ndarray,
    prefix: str,
    quantiles: Sequence[float],
) -> dict[str, float]:
    """Compute finite-only mean/std/quantile summary features."""
    numeric = np.asarray(values, dtype=float).reshape(-1)
    finite = numeric[np.isfinite(numeric)]

    features = {
        f"{prefix}_mean": float("nan"),
        f"{prefix}_std": float("nan"),
    }
    for quantile in quantiles:
        features[f"{prefix}_{_quantile_name(quantile)}"] = float("nan")

    if finite.size == 0:
        return features

    features[f"{prefix}_mean"] = float(np.mean(finite))
    features[f"{prefix}_std"] = float(np.std(finite))
    for quantile in quantiles:
        features[f"{prefix}_{_quantile_name(quantile)}"] = float(np.quantile(finite, quantile))
    return features


def compute_window_signature(
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    example_indices: Sequence[int],
    config: SignatureConfig,
) -> dict[str, float]:
    """Compute one label-free behavioral signature for a deployment window."""
    indices = np.asarray(list(example_indices), dtype=int)
    if indices.size == 0:
        raise ValueError("example_indices must be non-empty")

    window_examples = examples.iloc[indices]
    features: dict[str, float] = {}

    if config.include_confidence:
        features.update(_column_quantiles(window_examples, "confidence", config))
    if config.include_entropy:
        features.update(_column_quantiles(window_examples, "entropy", config))
    if config.include_margin:
        features.update(_column_quantiles(window_examples, "margin", config))
    if config.include_energy:
        features.update(_column_quantiles(window_examples, "energy", config))

    probabilities = arrays.get("probabilities")
    if config.include_probability_summary and probabilities is not None:
        features.update(_probability_features(probabilities[indices], config.quantiles))

    logits = arrays.get("logits")
    if config.include_logit_summary and logits is not None:
        features.update(_logit_features(logits[indices]))

    ood_score = arrays.get("ood_score")
    if config.include_ood_score and ood_score is not None:
        features.update(compute_quantile_features(ood_score[indices], "ood_score", config.quantiles))

    feature_embedding = arrays.get("feature_embedding")
    if config.include_feature_norm and feature_embedding is not None:
        feature_norm = np.linalg.norm(feature_embedding[indices], axis=1)
        features.update(compute_quantile_features(feature_norm, "feature_norm", config.quantiles))

    if config.include_label_diagnostics:
        features.update(_diagnostic_features(window_examples))

    return features


def compute_signatures_for_windows(
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    windows: pd.DataFrame,
    config: SignatureConfig,
) -> pd.DataFrame:
    """Compute compact behavioral signatures for a window table."""
    rows: list[dict[str, float | int | str]] = []
    for _, window in windows.iterrows():
        row: dict[str, float | int | str] = {
            column: window[column] for column in SIGNATURE_METADATA_COLUMNS if column in window
        }
        row.update(
            compute_window_signature(
                examples,
                arrays,
                window["example_indices"],
                config,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def get_signature_feature_columns(
    signatures: pd.DataFrame,
    include_diagnostics: bool = False,
) -> list[str]:
    """Return numeric, non-metadata columns usable as embedding inputs."""
    excluded = set(SIGNATURE_METADATA_COLUMNS)
    feature_columns: list[str] = []
    for column in signatures.columns:
        if column in excluded:
            continue
        if column.startswith("diag_") and not include_diagnostics:
            continue
        if pd.api.types.is_numeric_dtype(signatures[column]):
            feature_columns.append(column)
    return feature_columns


def _column_quantiles(
    examples: pd.DataFrame,
    column: str,
    config: SignatureConfig,
) -> dict[str, float]:
    if column not in examples.columns:
        return compute_quantile_features(np.array([], dtype=float), column, config.quantiles)
    return compute_quantile_features(examples[column].to_numpy(), column, config.quantiles)


def _probability_features(probabilities: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    row_entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_probability = np.max(probabilities, axis=1)

    features = {
        "prob_max_mean": float(np.mean(max_probability)),
        "prob_entropy_mean": float(np.mean(row_entropy)),
    }
    features.update(compute_quantile_features(max_probability, "prob_max", quantiles))
    features.update(compute_quantile_features(row_entropy, "prob_entropy", quantiles))
    class_means = np.mean(probabilities, axis=0)
    for class_index, value in enumerate(class_means):
        features[f"prob_class_{class_index}_mean"] = float(value)
    return features


def _logit_features(logits: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=float)
    sorted_logits = np.sort(logits, axis=1)
    top_gap = sorted_logits[:, -1] - sorted_logits[:, -2] if logits.shape[1] > 1 else np.zeros(len(logits))
    return {
        "logit_mean": float(np.mean(logits)),
        "logit_std": float(np.std(logits)),
        "logit_max_mean": float(np.mean(np.max(logits, axis=1))),
        "logit_top_gap_mean": float(np.mean(top_gap)),
    }


def _diagnostic_features(examples: pd.DataFrame) -> dict[str, float]:
    labels = examples["label"].astype(int).to_numpy()
    preds = examples["pred"].astype(int).to_numpy()
    correct = (preds == labels).astype(float)
    return {
        "diag_accuracy": float(np.mean(correct)),
        "diag_error_rate": float(1.0 - np.mean(correct)),
        "diag_label_entropy": _discrete_entropy(labels),
        "diag_pred_entropy": _discrete_entropy(preds),
        "diag_mean_label": float(np.mean(labels)),
    }


def _discrete_entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))


def _quantile_name(quantile: float) -> str:
    return f"q{int(round(100 * quantile)):02d}"
