from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.embeddings.distances import pairwise_distances
from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy, candidate_thresholds


ALLOWED_KERNELS = ("gaussian", "epanechnikov", "uniform")
ALLOWED_AGGREGATIONS = ("mean", "sum")
ALLOWED_WEIGHTED_BOUNDS = ("hoeffding", "empirical_bernstein")


@dataclass(frozen=True)
class CWCRCConfig:
    risk_control: RiskControlConfig
    metric: str = "euclidean"
    kernel: str = "gaussian"
    bandwidth: float | None = None
    target_effective_sample_size: float | None = 200.0
    min_bandwidth: float = 1e-6
    max_bandwidth: float | None = None
    bandwidth_grid_size: int = 30
    example_weight_aggregation: str = "mean"
    finite_sample_correction: str = "hoeffding"
    normalize_weights: bool = True
    bound_margin: float = 0.0
    bound_multiplier: float = 1.0


@dataclass(frozen=True)
class WeightedRiskControlResult:
    threshold: float
    weighted_risk_estimate: float
    weighted_risk_upper_bound: float
    weighted_coverage: float
    effective_sample_size: float
    bandwidth: float
    feasible: bool
    num_candidates: int
    n_calibration_examples: int
    config: CWCRCConfig
    raw_weighted_risk_upper_bound: float | None = None


def weighted_result_to_dict(result: WeightedRiskControlResult) -> dict[str, Any]:
    """Convert a weighted risk-control result to a serializable dictionary."""
    payload = asdict(result)
    payload["config"] = asdict(result.config)
    payload["config"]["risk_control"] = asdict(result.config.risk_control)
    payload["config"]["risk_control"]["loss"] = asdict(result.config.risk_control.loss)
    return payload


def kernel_values(
    distances: np.ndarray,
    bandwidth: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Compute local context kernel values from nonnegative distances."""
    if kernel not in ALLOWED_KERNELS:
        raise ValueError(f"kernel must be one of {ALLOWED_KERNELS}, got {kernel!r}")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")

    distances = np.asarray(distances, dtype=float)
    if not np.isfinite(distances).all() or (distances < 0.0).any():
        raise ValueError("distances must be finite and nonnegative")

    scaled = distances / float(bandwidth)
    if kernel == "gaussian":
        return np.exp(-0.5 * scaled**2)
    if kernel == "epanechnikov":
        return np.maximum(0.0, 1.0 - scaled**2)
    return (distances <= bandwidth).astype(float)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights, falling back to uniform if all mass is zero."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        return weights
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("weights must be finite and nonnegative")

    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    """Return 1 / sum(w_i^2) after normalizing weights."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        return 0.0
    normalized = normalize_weights(weights)
    denominator = float(np.sum(normalized**2))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 / denominator)


def compute_window_weights(
    query_embedding: np.ndarray,
    calibration_embedding_frame: pd.DataFrame,
    *,
    metric: str = "euclidean",
    kernel: str = "gaussian",
    bandwidth: float,
) -> pd.DataFrame:
    """Compute normalized calibration-window weights for a query context embedding."""
    if "window_id" not in calibration_embedding_frame.columns:
        raise ValueError("calibration_embedding_frame must include window_id")
    embedding_columns = get_embedding_columns(calibration_embedding_frame)
    if not embedding_columns:
        raise ValueError("calibration_embedding_frame must contain z* embedding columns")

    query = _coerce_query_embedding(query_embedding)
    calibration_embeddings = calibration_embedding_frame[embedding_columns].to_numpy(dtype=float)
    distances = pairwise_distances(query, calibration_embeddings, metric=metric).reshape(-1)
    raw_weights = kernel_values(distances, bandwidth, kernel)
    weights = normalize_weights(raw_weights)
    return pd.DataFrame(
        {
            "window_id": calibration_embedding_frame["window_id"].to_numpy(),
            "distance": distances,
            "weight": weights,
        }
    )


def select_bandwidth_for_effective_sample_size(
    query_embedding: np.ndarray,
    calibration_embedding_frame: pd.DataFrame,
    *,
    metric: str = "euclidean",
    kernel: str = "gaussian",
    target_effective_sample_size: float = 200.0,
    min_bandwidth: float = 1e-6,
    max_bandwidth: float | None = None,
    grid_size: int = 30,
) -> float:
    """Select the smallest bandwidth reaching a target window-level ESS."""
    if target_effective_sample_size is None:
        raise ValueError("target_effective_sample_size must not be None")
    if target_effective_sample_size <= 0.0:
        raise ValueError("target_effective_sample_size must be positive")
    if min_bandwidth <= 0.0:
        raise ValueError("min_bandwidth must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    embedding_columns = get_embedding_columns(calibration_embedding_frame)
    if not embedding_columns:
        raise ValueError("calibration_embedding_frame must contain z* embedding columns")

    query = _coerce_query_embedding(query_embedding)
    calibration_embeddings = calibration_embedding_frame[embedding_columns].to_numpy(dtype=float)
    distances = pairwise_distances(query, calibration_embeddings, metric=metric).reshape(-1)
    if distances.size == 0:
        raise ValueError("calibration_embedding_frame must contain at least one row")

    if max_bandwidth is None:
        max_distance = float(np.max(distances)) if distances.size else 0.0
        max_bandwidth = max(max_distance + min_bandwidth, min_bandwidth * 10.0)
    if max_bandwidth < min_bandwidth:
        raise ValueError("max_bandwidth must be >= min_bandwidth")

    target = min(float(target_effective_sample_size), float(len(distances)))
    if max_bandwidth == min_bandwidth:
        grid = np.array([min_bandwidth], dtype=float)
    else:
        grid = np.geomspace(min_bandwidth, max_bandwidth, num=grid_size)

    for bandwidth in grid:
        ess = effective_sample_size(kernel_values(distances, float(bandwidth), kernel))
        if ess >= target:
            return float(bandwidth)
    return float(max_bandwidth)


def window_weights_to_example_weights(
    windows: pd.DataFrame,
    window_weights: pd.DataFrame,
    n_examples: int,
    *,
    aggregation: str = "mean",
) -> np.ndarray:
    """Map calibration-window weights to calibration-example weights."""
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {ALLOWED_AGGREGATIONS}, got {aggregation!r}")
    if n_examples < 0:
        raise ValueError("n_examples must be nonnegative")
    if "window_id" not in windows.columns or "example_indices" not in windows.columns:
        raise ValueError("windows must include window_id and example_indices")
    if "window_id" not in window_weights.columns or "weight" not in window_weights.columns:
        raise ValueError("window_weights must include window_id and weight")

    weights_by_window = dict(zip(window_weights["window_id"], window_weights["weight"], strict=True))
    inherited: list[list[float]] = [[] for _ in range(n_examples)]
    for _, window in windows.iterrows():
        window_id = window["window_id"]
        if window_id not in weights_by_window:
            continue
        weight = float(weights_by_window[window_id])
        for example_index in window["example_indices"]:
            index = int(example_index)
            if index < 0 or index >= n_examples:
                raise ValueError(f"example index out of range: {index}")
            inherited[index].append(weight)

    example_weights = np.zeros(n_examples, dtype=float)
    for index, values in enumerate(inherited):
        if not values:
            continue
        example_weights[index] = float(np.mean(values) if aggregation == "mean" else np.sum(values))
    return normalize_weights(example_weights)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a normalized weighted mean."""
    values, weights = _validate_weighted_values(values, weights)
    normalized = normalize_weights(weights)
    return float(np.sum(normalized * values))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a normalized weighted variance."""
    values, weights = _validate_weighted_values(values, weights)
    normalized = normalize_weights(weights)
    mean = float(np.sum(normalized * values))
    return float(np.sum(normalized * (values - mean) ** 2))


def weighted_risk_upper_confidence_bound(
    losses: np.ndarray,
    weights: np.ndarray,
    delta: float,
    bound: str = "hoeffding",
) -> float:
    """Compute a weighted UCB-style upper bound for losses in [0, 1]."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if bound not in ALLOWED_WEIGHTED_BOUNDS:
        raise ValueError(f"bound must be one of {ALLOWED_WEIGHTED_BOUNDS}, got {bound!r}")

    losses, weights = _validate_weighted_values(losses, weights)
    if losses.size == 0:
        raise ValueError("losses must be non-empty")
    if ((losses < 0.0) | (losses > 1.0)).any():
        raise ValueError("losses must be in [0, 1]")

    normalized = normalize_weights(weights)
    ess = effective_sample_size(normalized)
    if ess <= 0.0:
        raise ValueError("effective sample size must be positive")

    mean = float(np.sum(normalized * losses))
    if bound == "hoeffding":
        upper = mean + float(np.sqrt(np.log(1.0 / delta) / (2.0 * ess)))
    else:
        variance = weighted_variance(losses, normalized)
        log_term = float(np.log(3.0 / delta))
        upper = mean + float(np.sqrt(2.0 * variance * log_term / ess)) + 3.0 * log_term / ess
    return float(np.clip(upper, 0.0, 1.0))


def calibrate_weighted_threshold(
    calibration_examples: pd.DataFrame,
    example_weights: np.ndarray,
    config: CWCRCConfig,
    *,
    bandwidth: float,
) -> WeightedRiskControlResult:
    """Calibrate a scalar threshold with local context-weighted risk bounds."""
    _validate_cwcrc_config(config)
    if config.risk_control.loss.loss_name == "accepted_only_error_rate":
        raise ValueError("accepted_only_error_rate is diagnostic and cannot be used for weighted calibration")
    if len(calibration_examples) == 0:
        raise ValueError("calibration_examples must be non-empty")
    if config.risk_control.score_column not in calibration_examples.columns:
        raise ValueError(f"calibration_examples missing score column: {config.risk_control.score_column}")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")

    weights = np.asarray(example_weights, dtype=float).reshape(-1)
    if len(weights) != len(calibration_examples):
        raise ValueError("example_weights length must match calibration_examples")
    weights = normalize_weights(weights) if config.normalize_weights else weights
    if not np.isclose(weights.sum(), 1.0):
        weights = normalize_weights(weights)

    thresholds = candidate_thresholds(
        calibration_examples[config.risk_control.score_column].to_numpy(dtype=float),
        accept_if=config.risk_control.accept_if,
        include_extremes=True,
    )
    candidates: list[dict[str, float | bool]] = []
    for threshold in thresholds:
        accepted = apply_threshold_policy(
            calibration_examples,
            config.risk_control.score_column,
            float(threshold),
            config.risk_control.accept_if,
        )
        losses = compute_losses(calibration_examples, accepted, config.risk_control.loss)
        if not np.isfinite(losses).all():
            raise ValueError("weighted calibration losses must be finite")
        risk_estimate = weighted_mean(losses, weights)
        raw_risk_upper = weighted_risk_upper_confidence_bound(
            losses,
            weights,
            config.risk_control.delta,
            config.finite_sample_correction,
        )
        risk_upper = config.bound_multiplier * raw_risk_upper + config.bound_margin
        coverage = float(np.sum(weights * accepted.astype(float)))
        candidates.append(
            {
                "threshold": float(threshold),
                "weighted_risk_estimate": risk_estimate,
                "weighted_risk_upper_bound": risk_upper,
                "raw_weighted_risk_upper_bound": raw_risk_upper,
                "weighted_coverage": coverage,
                "feasible": bool(
                    risk_upper <= config.risk_control.risk_budget
                    and coverage >= config.risk_control.min_coverage
                ),
            }
        )

    feasible = [candidate for candidate in candidates if bool(candidate["feasible"])]
    if feasible:
        selected = sorted(
            feasible,
            key=lambda candidate: (
                -float(candidate["weighted_coverage"]),
                float(candidate["weighted_risk_upper_bound"]),
            ),
        )[0]
        is_feasible = True
    else:
        selected = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate["weighted_risk_upper_bound"]),
                -float(candidate["weighted_coverage"]),
            ),
        )[0]
        is_feasible = False

    return WeightedRiskControlResult(
        threshold=float(selected["threshold"]),
        weighted_risk_estimate=float(selected["weighted_risk_estimate"]),
        weighted_risk_upper_bound=float(selected["weighted_risk_upper_bound"]),
        weighted_coverage=float(selected["weighted_coverage"]),
        effective_sample_size=effective_sample_size(weights),
        bandwidth=float(bandwidth),
        feasible=is_feasible,
        num_candidates=len(candidates),
        n_calibration_examples=len(calibration_examples),
        config=config,
        raw_weighted_risk_upper_bound=float(selected["raw_weighted_risk_upper_bound"]),
    )


def calibrate_cwcrc_for_query(
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    query_embedding: np.ndarray,
    config: CWCRCConfig,
) -> tuple[WeightedRiskControlResult, pd.DataFrame, np.ndarray]:
    """Calibrate a CW-CRC-style threshold for one query context embedding."""
    _validate_cwcrc_config(config)
    bandwidth = config.bandwidth
    if bandwidth is None:
        bandwidth = select_bandwidth_for_effective_sample_size(
            query_embedding,
            calibration_embedding_frame,
            metric=config.metric,
            kernel=config.kernel,
            target_effective_sample_size=config.target_effective_sample_size,
            min_bandwidth=config.min_bandwidth,
            max_bandwidth=config.max_bandwidth,
            grid_size=config.bandwidth_grid_size,
        )
    elif bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")

    window_weights = compute_window_weights(
        query_embedding,
        calibration_embedding_frame,
        metric=config.metric,
        kernel=config.kernel,
        bandwidth=float(bandwidth),
    )
    example_weights = window_weights_to_example_weights(
        calibration_windows,
        window_weights,
        len(calibration_examples),
        aggregation=config.example_weight_aggregation,
    )
    result = calibrate_weighted_threshold(
        calibration_examples,
        example_weights,
        config,
        bandwidth=float(bandwidth),
    )
    return result, window_weights, example_weights


def evaluate_cwcrc_window(
    examples: pd.DataFrame,
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    window_row: pd.Series,
    embedding_row: pd.Series,
    config: CWCRCConfig,
) -> dict[str, Any]:
    """Evaluate one query window with CW-CRC-style local weighted calibration."""
    embedding_columns = _embedding_columns_from_series(embedding_row)
    query_embedding = embedding_row.loc[embedding_columns].to_numpy(dtype=float)
    weighted_result, _, _ = calibrate_cwcrc_for_query(
        calibration_examples,
        calibration_windows,
        calibration_embedding_frame,
        query_embedding,
        config,
    )

    example_indices = [int(index) for index in window_row["example_indices"]]
    window_examples = examples.iloc[example_indices]
    accepted = apply_threshold_policy(
        window_examples,
        config.risk_control.score_column,
        weighted_result.threshold,
        config.risk_control.accept_if,
    )
    losses = compute_losses(window_examples, accepted, config.risk_control.loss)
    summary = summarize_policy_performance(window_examples, accepted, losses)
    realized_risk = summary["mean_loss"]
    risk_budget = config.risk_control.risk_budget
    return {
        "window_id": window_row["window_id"],
        "context_id": window_row.get("context_id", "unknown"),
        "context_name": window_row.get("context_name", "unknown"),
        "threshold": weighted_result.threshold,
        "bandwidth": weighted_result.bandwidth,
        "effective_sample_size": weighted_result.effective_sample_size,
        "weighted_risk_estimate": weighted_result.weighted_risk_estimate,
        "weighted_risk_upper_bound": weighted_result.weighted_risk_upper_bound,
        "raw_weighted_risk_upper_bound": weighted_result.raw_weighted_risk_upper_bound,
        "realized_risk": realized_risk,
        "mean_loss": realized_risk,
        "coverage": summary["coverage"],
        "n_examples": summary["n"],
        "feasible": bool(weighted_result.feasible),
        "violation": bool(np.isfinite(realized_risk) and realized_risk > risk_budget),
        "bound_violation": bool(weighted_result.weighted_risk_upper_bound > risk_budget),
        "num_candidates": weighted_result.num_candidates,
        "num_accepted": summary["num_accepted"],
        "num_deferred": summary["num_deferred"],
        "accuracy_accepted": summary["accuracy_accepted"],
        "error_rate_accepted": summary["error_rate_accepted"],
        "accuracy_all": summary["accuracy_all"],
        "error_rate_all": summary["error_rate_all"],
    }


def evaluate_cwcrc_windows(
    examples: pd.DataFrame,
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    config: CWCRCConfig,
) -> pd.DataFrame:
    """Evaluate multiple query windows independently with CW-CRC-style calibration."""
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
        rows.append(
            evaluate_cwcrc_window(
                examples,
                calibration_examples,
                calibration_windows,
                calibration_embedding_frame,
                window_row,
                embedding_row,
                config,
            )
        )
    return pd.DataFrame(rows)


def assign_cwcrc_thresholds_to_examples(
    examples: pd.DataFrame,
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    config: CWCRCConfig,
) -> pd.DataFrame:
    """Assign CW-CRC-style thresholds to each window-example occurrence for debugging."""
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

        query_embedding = embedding_row.loc[_embedding_columns_from_series(embedding_row)].to_numpy(dtype=float)
        weighted_result, _, _ = calibrate_cwcrc_for_query(
            calibration_examples,
            calibration_windows,
            calibration_embedding_frame,
            query_embedding,
            config,
        )
        example_indices = [int(index) for index in window_row["example_indices"]]
        window_examples = examples.iloc[example_indices]
        accepted = apply_threshold_policy(
            window_examples,
            config.risk_control.score_column,
            weighted_result.threshold,
            config.risk_control.accept_if,
        )
        losses = compute_losses(window_examples, accepted, config.risk_control.loss)
        for example_index, accepted_value, loss in zip(example_indices, accepted, losses, strict=True):
            rows.append(
                {
                    "window_id": window_id,
                    "example_index": int(example_index),
                    "threshold": weighted_result.threshold,
                    "bandwidth": weighted_result.bandwidth,
                    "effective_sample_size": weighted_result.effective_sample_size,
                    "accepted": bool(accepted_value),
                    "loss": float(loss),
                }
            )
    return pd.DataFrame(rows)


def _validate_cwcrc_config(config: CWCRCConfig) -> None:
    if config.kernel not in ALLOWED_KERNELS:
        raise ValueError(f"kernel must be one of {ALLOWED_KERNELS}, got {config.kernel!r}")
    if config.example_weight_aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"example_weight_aggregation must be one of {ALLOWED_AGGREGATIONS}, "
            f"got {config.example_weight_aggregation!r}"
        )
    if config.finite_sample_correction not in ALLOWED_WEIGHTED_BOUNDS:
        raise ValueError(
            f"finite_sample_correction must be one of {ALLOWED_WEIGHTED_BOUNDS}, "
            f"got {config.finite_sample_correction!r}"
        )
    if config.bandwidth is not None and config.bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if config.min_bandwidth <= 0.0:
        raise ValueError("min_bandwidth must be positive")
    if config.max_bandwidth is not None and config.max_bandwidth < config.min_bandwidth:
        raise ValueError("max_bandwidth must be >= min_bandwidth")
    if config.bandwidth_grid_size < 2:
        raise ValueError("bandwidth_grid_size must be >= 2")
    if config.bound_margin < 0.0:
        raise ValueError("bound_margin must be nonnegative")
    if config.bound_multiplier < 0.0:
        raise ValueError("bound_multiplier must be nonnegative")


def _validate_weighted_values(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size != weights.size:
        raise ValueError("values and weights must have matching lengths")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("weights must be finite and nonnegative")
    return values, weights


def _coerce_query_embedding(query_embedding: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=float)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError(f"query_embedding must have shape [d] or [1, d], got {query.shape}")
    return query


def _embedding_columns_from_series(row: pd.Series) -> list[str]:
    columns = get_embedding_columns(pd.DataFrame(columns=row.index))
    if not columns:
        raise ValueError("embedding_row must contain z* embedding columns")
    return columns
