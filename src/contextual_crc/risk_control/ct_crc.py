from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.embeddings.distances import pairwise_distances
from contextual_crc.risk_control.crc import (
    RiskControlConfig,
    RiskControlResult,
    calibrate_threshold,
)
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy


@dataclass(frozen=True)
class ValidatedContext:
    context_id: str
    context_name: str
    centroid: np.ndarray
    threshold: float
    certificate: float
    risk_estimate: float
    coverage: float
    n_calibration: int
    n_windows: int
    feasible: bool


@dataclass(frozen=True)
class CTCRCConfig:
    risk_control: RiskControlConfig
    metric: str = "euclidean"
    l_hat: float = 0.0
    epsilon_k: float = 0.0
    bound_margin: float = 0.0
    bound_multiplier: float = 1.0
    clip_bounds: bool = True
    fallback_to_global: bool = True


@dataclass(frozen=True)
class CTCRCResult:
    config: CTCRCConfig
    contexts: list[ValidatedContext]
    global_result: RiskControlResult | None = None


def validated_context_to_dict(context: ValidatedContext) -> dict[str, Any]:
    """Convert a validated context to a serializable dictionary."""
    payload = asdict(context)
    payload["centroid"] = np.asarray(context.centroid, dtype=float).tolist()
    return payload


def ctcrc_result_to_dict(result: CTCRCResult) -> dict[str, Any]:
    """Convert a CT-CRC-style result to a serializable dictionary."""
    payload = {
        "config": asdict(result.config),
        "contexts": [validated_context_to_dict(context) for context in result.contexts],
        "global_result": None,
    }
    if result.global_result is not None:
        payload["global_result"] = asdict(result.global_result)
    return payload


def build_validated_contexts(
    calibration_examples: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    config: CTCRCConfig,
    *,
    context_col: str = "context_id",
) -> CTCRCResult:
    """Build a validated context library with centroids and UCB-style certificates."""
    _validate_ctcrc_config(config)
    if context_col not in calibration_examples.columns:
        raise ValueError(f"calibration_examples is missing context column: {context_col}")
    if context_col not in calibration_embedding_frame.columns:
        raise ValueError(f"calibration_embedding_frame is missing context column: {context_col}")

    embedding_columns = get_embedding_columns(calibration_embedding_frame)
    if not embedding_columns:
        raise ValueError("calibration_embedding_frame must contain z* embedding columns")

    contexts: list[ValidatedContext] = []
    for context_value, context_examples in calibration_examples.groupby(
        context_col, sort=False, dropna=False
    ):
        context_windows = calibration_embedding_frame[
            calibration_embedding_frame[context_col] == context_value
        ]
        if len(context_windows) == 0:
            raise ValueError(f"No calibration embedding windows found for context: {context_value}")

        centroid = context_windows[embedding_columns].to_numpy(dtype=float).mean(axis=0)
        calibration_result = calibrate_threshold(context_examples, config.risk_control)
        contexts.append(
            ValidatedContext(
                context_id=str(context_value),
                context_name=_context_name(context_examples, context_value),
                centroid=centroid,
                threshold=calibration_result.threshold,
                certificate=calibration_result.risk_upper_bound,
                risk_estimate=calibration_result.risk_estimate,
                coverage=calibration_result.coverage,
                n_calibration=calibration_result.n_calibration,
                n_windows=int(len(context_windows)),
                feasible=calibration_result.feasible,
            )
        )

    if not contexts:
        raise ValueError("At least one validated context is required")

    global_result = (
        calibrate_threshold(calibration_examples, config.risk_control)
        if config.fallback_to_global
        else None
    )
    return CTCRCResult(config=config, contexts=contexts, global_result=global_result)


def find_nearest_validated_context(
    query_embedding: np.ndarray,
    contexts: Sequence[ValidatedContext],
    metric: str = "euclidean",
) -> tuple[ValidatedContext, float]:
    """Find the nearest validated context centroid for one query embedding."""
    if not contexts:
        raise ValueError("contexts must be non-empty")

    query = np.asarray(query_embedding, dtype=float)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError(f"query_embedding must have shape [d] or [1, d], got {query.shape}")

    centroids = np.vstack([np.asarray(context.centroid, dtype=float) for context in contexts])
    distances = pairwise_distances(query, centroids, metric=metric).reshape(-1)
    nearest_index = min(range(len(contexts)), key=lambda index: (distances[index], index))
    return contexts[nearest_index], float(distances[nearest_index])


def transferred_upper_bound(
    certificate: float,
    distance: float,
    l_hat: float,
    epsilon_k: float = 0.0,
    *,
    bound_margin: float = 0.0,
    bound_multiplier: float = 1.0,
    clip: bool = True,
) -> float:
    """Distance-degrade a source-context risk certificate for transfer."""
    if distance < 0.0:
        raise ValueError("distance must be nonnegative")
    if l_hat < 0.0:
        raise ValueError("l_hat must be nonnegative")
    if epsilon_k < 0.0:
        raise ValueError("epsilon_k must be nonnegative")
    if bound_margin < 0.0:
        raise ValueError("bound_margin must be nonnegative")
    if bound_multiplier < 0.0:
        raise ValueError("bound_multiplier must be nonnegative")

    upper = (
        float(bound_multiplier) * float(certificate)
        + float(l_hat) * float(distance)
        + float(l_hat) * float(epsilon_k)
        + float(bound_margin)
    )
    if clip:
        upper = float(np.clip(upper, 0.0, 1.0))
    return upper


def evaluate_ctcrc_window(
    examples: pd.DataFrame,
    window_row: pd.Series,
    embedding_row: pd.Series,
    result: CTCRCResult,
) -> dict[str, Any]:
    """Evaluate CT-CRC-style nearest-context transfer for one deployment window."""
    embedding_columns = _embedding_columns_from_series(embedding_row)
    query_embedding = embedding_row.loc[embedding_columns].to_numpy(dtype=float)
    nearest_context, novelty_distance = find_nearest_validated_context(
        query_embedding,
        result.contexts,
        metric=result.config.metric,
    )

    example_indices = [int(index) for index in window_row["example_indices"]]
    window_examples = examples.iloc[example_indices]
    accepted = apply_threshold_policy(
        window_examples,
        result.config.risk_control.score_column,
        nearest_context.threshold,
        result.config.risk_control.accept_if,
    )
    losses = compute_losses(window_examples, accepted, result.config.risk_control.loss)
    summary = summarize_policy_performance(window_examples, accepted, losses)
    transferred_bound = transferred_upper_bound(
        nearest_context.certificate,
        novelty_distance,
        result.config.l_hat,
        result.config.epsilon_k,
        bound_margin=result.config.bound_margin,
        bound_multiplier=result.config.bound_multiplier,
        clip=result.config.clip_bounds,
    )
    realized_risk = summary["mean_loss"]
    risk_budget = result.config.risk_control.risk_budget

    return {
        "window_id": window_row["window_id"],
        "context_id": window_row.get("context_id", "unknown"),
        "context_name": window_row.get("context_name", "unknown"),
        "nearest_context_id": nearest_context.context_id,
        "nearest_context_name": nearest_context.context_name,
        "novelty_distance": novelty_distance,
        "threshold": nearest_context.threshold,
        "certificate": nearest_context.certificate,
        "transferred_upper_bound": transferred_bound,
        "realized_risk": realized_risk,
        "mean_loss": realized_risk,
        "coverage": summary["coverage"],
        "n_examples": summary["n"],
        "feasible_nearest": bool(nearest_context.feasible),
        "violation": bool(np.isfinite(realized_risk) and realized_risk > risk_budget),
        "bound_violation": bool(transferred_bound > risk_budget),
        "num_accepted": summary["num_accepted"],
        "num_deferred": summary["num_deferred"],
        "accuracy_accepted": summary["accuracy_accepted"],
        "error_rate_accepted": summary["error_rate_accepted"],
        "accuracy_all": summary["accuracy_all"],
        "error_rate_all": summary["error_rate_all"],
    }


def evaluate_ctcrc_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    result: CTCRCResult,
) -> pd.DataFrame:
    """Evaluate CT-CRC-style nearest-context transfer for multiple windows."""
    if "window_id" not in windows.columns:
        raise ValueError("windows must include window_id")
    if "window_id" not in embedding_frame.columns:
        raise ValueError("embedding_frame must include window_id")

    embedding_by_window = embedding_frame.set_index("window_id", drop=False)
    missing = [window_id for window_id in windows["window_id"] if window_id not in embedding_by_window.index]
    if missing:
        raise ValueError(f"embedding_frame is missing window_id values: {missing}")

    rows = []
    for _, window_row in windows.iterrows():
        embedding_row = embedding_by_window.loc[window_row["window_id"]]
        if isinstance(embedding_row, pd.DataFrame):
            embedding_row = embedding_row.iloc[0]
        rows.append(evaluate_ctcrc_window(examples, window_row, embedding_row, result))
    return pd.DataFrame(rows)


def assign_ctcrc_thresholds_to_examples(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    result: CTCRCResult,
) -> pd.DataFrame:
    """Assign transferred thresholds to each window-example occurrence for debugging."""
    if "window_id" not in windows.columns or "window_id" not in embedding_frame.columns:
        raise ValueError("windows and embedding_frame must include window_id")

    embedding_by_window = embedding_frame.set_index("window_id", drop=False)
    rows: list[dict[str, Any]] = []
    for _, window_row in windows.iterrows():
        window_id = window_row["window_id"]
        if window_id not in embedding_by_window.index:
            raise ValueError(f"embedding_frame is missing window_id: {window_id}")
        embedding_row = embedding_by_window.loc[window_id]
        if isinstance(embedding_row, pd.DataFrame):
            embedding_row = embedding_row.iloc[0]

        embedding_columns = _embedding_columns_from_series(embedding_row)
        nearest_context, novelty_distance = find_nearest_validated_context(
            embedding_row.loc[embedding_columns].to_numpy(dtype=float),
            result.contexts,
            metric=result.config.metric,
        )
        example_indices = [int(index) for index in window_row["example_indices"]]
        window_examples = examples.iloc[example_indices]
        accepted = apply_threshold_policy(
            window_examples,
            result.config.risk_control.score_column,
            nearest_context.threshold,
            result.config.risk_control.accept_if,
        )
        losses = compute_losses(window_examples, accepted, result.config.risk_control.loss)
        for example_index, accepted_value, loss in zip(example_indices, accepted, losses, strict=True):
            rows.append(
                {
                    "window_id": window_id,
                    "example_index": int(example_index),
                    "threshold": nearest_context.threshold,
                    "nearest_context_id": nearest_context.context_id,
                    "novelty_distance": novelty_distance,
                    "accepted": bool(accepted_value),
                    "loss": float(loss),
                }
            )
    return pd.DataFrame(rows)


def estimate_lipschitz_from_contexts(
    context_summary: pd.DataFrame,
    *,
    risk_col: str = "risk_estimate",
    metric: str = "euclidean",
    eps: float = 1e-8,
    quantile: float | None = 0.9,
) -> float:
    """Estimate a local Lipschitz slope from context risks and centroids."""
    if risk_col not in context_summary.columns:
        raise ValueError(f"context_summary is missing risk_col: {risk_col}")
    if quantile is not None and not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1] or None")

    embedding_columns = get_embedding_columns(context_summary)
    if not embedding_columns:
        raise ValueError("context_summary must contain z* embedding columns")
    if len(context_summary) < 2:
        return 0.0

    embeddings = context_summary[embedding_columns].to_numpy(dtype=float)
    risks = context_summary[risk_col].to_numpy(dtype=float)
    if not np.isfinite(risks).all():
        raise ValueError(f"{risk_col} must be finite")

    distances = pairwise_distances(embeddings, embeddings, metric=metric)
    ratios: list[float] = []
    for i in range(len(context_summary)):
        for j in range(i + 1, len(context_summary)):
            ratios.append(float(abs(risks[i] - risks[j]) / (distances[i, j] + eps)))
    if not ratios:
        return 0.0
    if quantile is None:
        return float(np.max(ratios))
    return float(np.quantile(ratios, quantile))


def validated_contexts_to_frame(contexts: Sequence[ValidatedContext]) -> pd.DataFrame:
    """Convert validated contexts to a tabular centroid/risk summary."""
    rows: list[dict[str, Any]] = []
    for context in contexts:
        row = {
            "context_id": context.context_id,
            "context_name": context.context_name,
            "threshold": context.threshold,
            "certificate": context.certificate,
            "risk_estimate": context.risk_estimate,
            "coverage": context.coverage,
            "n_calibration": context.n_calibration,
            "n_windows": context.n_windows,
            "feasible": context.feasible,
        }
        for dimension, value in enumerate(np.asarray(context.centroid, dtype=float)):
            row[f"z{dimension}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_ctcrc_config(config: CTCRCConfig) -> None:
    if config.l_hat < 0.0:
        raise ValueError("l_hat must be nonnegative")
    if config.epsilon_k < 0.0:
        raise ValueError("epsilon_k must be nonnegative")
    if config.bound_margin < 0.0:
        raise ValueError("bound_margin must be nonnegative")
    if config.bound_multiplier < 0.0:
        raise ValueError("bound_multiplier must be nonnegative")


def _context_name(context_examples: pd.DataFrame, context_value: Any) -> str:
    if "context_name" not in context_examples.columns:
        return str(context_value)
    names = context_examples["context_name"].dropna().unique()
    if len(names) == 1:
        return str(names[0])
    return str(context_value)


def _embedding_columns_from_series(row: pd.Series) -> list[str]:
    columns = get_embedding_columns(pd.DataFrame(columns=row.index))
    if not columns:
        raise ValueError("embedding_row must contain z* embedding columns")
    return columns
