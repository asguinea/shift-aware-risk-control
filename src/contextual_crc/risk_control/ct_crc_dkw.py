from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    find_nearest_validated_context,
    transferred_upper_bound,
)
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy, candidate_thresholds
from contextual_crc.risk_control.ct_crc_simultaneous import allocate_context_delta


@dataclass(frozen=True)
class UniformThresholdCalibration:
    threshold: float
    risk_estimate: float
    certificate: float
    coverage: float
    n_calibration: int
    n_accepted: int
    feasible: bool
    num_candidates: int
    epsilon: float
    delta: float


@dataclass(frozen=True)
class DKWValidatedContext:
    context_id: str
    context_name: str
    centroid: np.ndarray
    calibration_scores: np.ndarray
    error_scores: np.ndarray
    source_threshold: float
    source_certificate: float
    risk_estimate: float
    coverage: float
    n_calibration: int
    n_windows: int
    feasible: bool
    epsilon: float
    delta: float


@dataclass(frozen=True)
class DKWCTResult:
    config: CTCRCConfig
    contexts: list[DKWValidatedContext]
    library_alpha: float
    per_context_delta: float


@dataclass(frozen=True)
class TargetFeasibleSelection:
    threshold: float
    source_risk_estimate: float
    source_certificate: float
    raw_source_certificate: float
    transferred_upper_bound: float
    raw_transferred_upper_bound: float
    coverage: float
    n_accepted: int
    num_candidates: int
    nontrivial_feasible: bool
    forced_accept_none: bool
    nearest_context_id: str
    nearest_context_name: str
    novelty_distance: float


def accepted_error_score_transform(
    examples: pd.DataFrame,
    *,
    score_column: str = "confidence",
) -> np.ndarray:
    """Map accepted-error loss to 1{V >= theta}, with V=-inf on correct examples."""
    missing = [column for column in (score_column, "label", "pred") if column not in examples.columns]
    if missing:
        raise ValueError(f"examples is missing required columns: {missing}")
    scores = pd.to_numeric(examples[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    incorrect = examples["pred"].astype(int).to_numpy() != examples["label"].astype(int).to_numpy()
    return np.where(incorrect, scores, -np.inf)


def accepted_error_losses_from_scores(
    error_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Evaluate the nested accepted-error indicators for one threshold."""
    values = np.asarray(error_scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("error_scores must be non-empty")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("error_scores may contain finite values and -inf only")
    return (values >= float(threshold)).astype(float)


def dkw_uniform_epsilon(n: int, delta: float) -> float:
    """Two-sided DKW/Massart radius sqrt(log(2/delta)/(2n))."""
    if int(n) < 1:
        raise ValueError("n must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return float(np.sqrt(np.log(2.0 / float(delta)) / (2.0 * int(n))))


def empirical_accepted_error_risks(
    error_scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute empirical survival probabilities P_n(V >= theta) efficiently."""
    values = np.asarray(error_scores, dtype=float).reshape(-1)
    candidates = np.asarray(thresholds, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("error_scores must be non-empty")
    if candidates.size == 0:
        raise ValueError("thresholds must be non-empty")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("error_scores may contain finite values and -inf only")
    if not np.isfinite(candidates).all():
        raise ValueError("thresholds must be finite")

    ordered = np.sort(values)
    first_accepted = np.searchsorted(ordered, candidates, side="left")
    return (values.size - first_accepted).astype(float) / values.size


def dkw_uniform_certificates(
    error_scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    """Return simultaneous upper envelopes for all supplied thresholds."""
    risks = empirical_accepted_error_risks(error_scores, thresholds)
    epsilon = dkw_uniform_epsilon(len(np.asarray(error_scores).reshape(-1)), delta)
    return np.clip(risks + epsilon, 0.0, 1.0)


def dkw_uniform_certificate(
    error_scores: np.ndarray,
    threshold: float,
    *,
    delta: float,
) -> float:
    """Return the DKW uniform source certificate at one threshold."""
    return float(
        dkw_uniform_certificates(
            error_scores,
            np.asarray([threshold], dtype=float),
            delta=delta,
        )[0]
    )


def calibrate_dkw_uniform_threshold(
    calibration_examples: pd.DataFrame,
    config: CTCRCConfig,
    *,
    delta: float,
) -> UniformThresholdCalibration:
    """Select the maximum-coverage source threshold under a uniform DKW envelope."""
    risk_config = config.risk_control
    _validate_supported_risk_config(config)
    if len(calibration_examples) == 0:
        raise ValueError("calibration_examples must be non-empty")

    scores = pd.to_numeric(
        calibration_examples[risk_config.score_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("calibration scores must be finite")
    thresholds = candidate_thresholds(scores, accept_if="greater_equal", include_extremes=True)
    error_scores = accepted_error_score_transform(
        calibration_examples,
        score_column=risk_config.score_column,
    )
    risks = empirical_accepted_error_risks(error_scores, thresholds)
    certificates = dkw_uniform_certificates(error_scores, thresholds, delta=delta)
    coverages = _coverage_at_thresholds(scores, thresholds)
    accepted_counts = np.rint(coverages * len(scores)).astype(int)
    feasible = (certificates <= risk_config.risk_budget) & (
        coverages >= risk_config.min_coverage
    )

    if feasible.any():
        indices = np.flatnonzero(feasible)
        selected_index = min(
            indices,
            key=lambda index: (
                -float(coverages[index]),
                float(certificates[index]),
                float(thresholds[index]),
            ),
        )
        selected_feasible = True
    else:
        selected_index = min(
            range(len(thresholds)),
            key=lambda index: (
                float(certificates[index]),
                -float(coverages[index]),
                float(thresholds[index]),
            ),
        )
        selected_feasible = False

    return UniformThresholdCalibration(
        threshold=float(thresholds[selected_index]),
        risk_estimate=float(risks[selected_index]),
        certificate=float(certificates[selected_index]),
        coverage=float(coverages[selected_index]),
        n_calibration=int(len(scores)),
        n_accepted=int(accepted_counts[selected_index]),
        feasible=selected_feasible,
        num_candidates=int(len(thresholds)),
        epsilon=dkw_uniform_epsilon(len(scores), delta),
        delta=float(delta),
    )


def build_dkw_validated_contexts(
    calibration_examples: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    config: CTCRCConfig,
    *,
    library_alpha: float,
    context_col: str = "context_id",
) -> tuple[DKWCTResult, pd.DataFrame]:
    """Build a library whose source envelopes are uniform over all thresholds."""
    _validate_supported_risk_config(config)
    if context_col not in calibration_examples.columns:
        raise ValueError(f"calibration_examples is missing context column: {context_col}")
    if context_col not in calibration_embedding_frame.columns:
        raise ValueError(f"calibration_embedding_frame is missing context column: {context_col}")
    embedding_columns = get_embedding_columns(calibration_embedding_frame)
    if not embedding_columns:
        raise ValueError("calibration_embedding_frame must contain z* embedding columns")

    grouped = list(calibration_examples.groupby(context_col, sort=False, dropna=False))
    per_context_delta = allocate_context_delta(library_alpha, len(grouped))
    contexts: list[DKWValidatedContext] = []
    audit_rows: list[dict[str, Any]] = []
    for context_value, context_examples in grouped:
        context_windows = calibration_embedding_frame[
            calibration_embedding_frame[context_col] == context_value
        ]
        if context_windows.empty:
            raise ValueError(f"No calibration embedding windows found for context: {context_value}")

        calibration = calibrate_dkw_uniform_threshold(
            context_examples,
            config,
            delta=per_context_delta,
        )
        scores = pd.to_numeric(
            context_examples[config.risk_control.score_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        error_scores = accepted_error_score_transform(
            context_examples,
            score_column=config.risk_control.score_column,
        )
        context_name = _context_name(context_examples, context_value)
        context = DKWValidatedContext(
            context_id=str(context_value),
            context_name=context_name,
            centroid=context_windows[embedding_columns].to_numpy(dtype=float).mean(axis=0),
            calibration_scores=scores,
            error_scores=error_scores,
            source_threshold=calibration.threshold,
            source_certificate=calibration.certificate,
            risk_estimate=calibration.risk_estimate,
            coverage=calibration.coverage,
            n_calibration=calibration.n_calibration,
            n_windows=int(len(context_windows)),
            feasible=calibration.feasible,
            epsilon=calibration.epsilon,
            delta=calibration.delta,
        )
        contexts.append(context)
        audit_rows.append(
            {
                "context_id": context.context_id,
                "context_name": context.context_name,
                "n_calibration": context.n_calibration,
                "n_windows": context.n_windows,
                "delta": context.delta,
                "dkw_epsilon": context.epsilon,
                "source_threshold": context.source_threshold,
                "source_risk_estimate": context.risk_estimate,
                "source_certificate": context.source_certificate,
                "source_coverage": context.coverage,
                "source_feasible": context.feasible,
                "source_candidate_count": calibration.num_candidates,
            }
        )

    return (
        DKWCTResult(
            config=config,
            contexts=contexts,
            library_alpha=float(library_alpha),
            per_context_delta=per_context_delta,
        ),
        pd.DataFrame(audit_rows),
    )


def dkw_contexts_to_frame(contexts: Sequence[DKWValidatedContext]) -> pd.DataFrame:
    """Convert DKW contexts to the frame consumed by the existing L-hat estimator."""
    rows: list[dict[str, Any]] = []
    for context in contexts:
        row: dict[str, Any] = {
            "context_id": context.context_id,
            "risk_estimate": context.risk_estimate,
            "threshold": context.source_threshold,
            "certificate": context.source_certificate,
        }
        for dimension, value in enumerate(np.asarray(context.centroid, dtype=float)):
            row[f"z{dimension}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def target_candidate_thresholds(
    deployment_scores: np.ndarray,
    source_scores: np.ndarray,
) -> np.ndarray:
    """Use all source/deployment score breakpoints plus finite accept-all/none extremes."""
    deployment = np.asarray(deployment_scores, dtype=float).reshape(-1)
    source = np.asarray(source_scores, dtype=float).reshape(-1)
    if deployment.size == 0:
        raise ValueError("deployment_scores must be non-empty")
    combined = np.concatenate([deployment, source])
    return candidate_thresholds(combined, accept_if="greater_equal", include_extremes=True)


def select_target_feasible_threshold(
    deployment_scores: np.ndarray,
    query_embedding: np.ndarray,
    result: DKWCTResult,
) -> TargetFeasibleSelection:
    """Select using source labels and deployment scores, never deployment labels."""
    scores = np.asarray(deployment_scores, dtype=float).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("deployment_scores must be non-empty and finite")

    nearest, distance = find_nearest_validated_context(
        query_embedding,
        result.contexts,
        metric=result.config.metric,
    )
    thresholds = target_candidate_thresholds(scores, nearest.calibration_scores)
    source_risks = empirical_accepted_error_risks(nearest.error_scores, thresholds)
    source_certificates = np.clip(source_risks + nearest.epsilon, 0.0, 1.0)
    raw_transferred = np.asarray(
        [
            transferred_upper_bound(
                certificate,
                distance,
                result.config.l_hat,
                result.config.epsilon_k,
                bound_margin=result.config.bound_margin,
                bound_multiplier=result.config.bound_multiplier,
                clip=result.config.clip_bounds,
            )
            for certificate in source_certificates
        ],
        dtype=float,
    )
    coverages = _coverage_at_thresholds(scores, thresholds)
    accepted_counts = np.rint(coverages * len(scores)).astype(int)
    feasible_nontrivial = (accepted_counts > 0) & (
        raw_transferred <= result.config.risk_control.risk_budget
    )

    if feasible_nontrivial.any():
        indices = np.flatnonzero(feasible_nontrivial)
        selected_index = min(
            indices,
            key=lambda index: (
                -int(accepted_counts[index]),
                float(raw_transferred[index]),
                float(thresholds[index]),
            ),
        )
        forced_accept_none = False
        selected_bound = float(raw_transferred[selected_index])
        selected_source_certificate = float(source_certificates[selected_index])
    else:
        accept_none_indices = np.flatnonzero(accepted_counts == 0)
        if accept_none_indices.size == 0:
            raise RuntimeError("candidate construction did not include accept-none")
        selected_index = int(accept_none_indices[0])
        forced_accept_none = True
        # Full defer has exactly zero operational loss on this deployment window.
        selected_bound = 0.0
        selected_source_certificate = 0.0

    return TargetFeasibleSelection(
        threshold=float(thresholds[selected_index]),
        source_risk_estimate=float(source_risks[selected_index]),
        source_certificate=selected_source_certificate,
        raw_source_certificate=float(source_certificates[selected_index]),
        transferred_upper_bound=selected_bound,
        raw_transferred_upper_bound=float(raw_transferred[selected_index]),
        coverage=float(coverages[selected_index]),
        n_accepted=int(accepted_counts[selected_index]),
        num_candidates=int(len(thresholds)),
        nontrivial_feasible=not forced_accept_none,
        forced_accept_none=forced_accept_none,
        nearest_context_id=nearest.context_id,
        nearest_context_name=nearest.context_name,
        novelty_distance=float(distance),
    )


def evaluate_dkw_source_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    result: DKWCTResult,
) -> pd.DataFrame:
    """Diagnostic old behavior: transfer each DKW source-selected threshold."""
    embedding_by_window = _validate_window_inputs(windows, embedding_frame)
    rows: list[dict[str, Any]] = []
    for _, window_row in windows.iterrows():
        embedding_row = _embedding_row(embedding_by_window, window_row["window_id"])
        query = embedding_row.loc[_embedding_columns_from_series(embedding_row)].to_numpy(dtype=float)
        nearest, distance = find_nearest_validated_context(query, result.contexts, result.config.metric)
        window_examples = examples.iloc[[int(index) for index in window_row["example_indices"]]]
        accepted = apply_threshold_policy(
            window_examples,
            result.config.risk_control.score_column,
            nearest.source_threshold,
            "greater_equal",
        )
        losses = compute_losses(window_examples, accepted, result.config.risk_control.loss)
        summary = summarize_policy_performance(window_examples, accepted, losses)
        bound = transferred_upper_bound(
            nearest.source_certificate,
            distance,
            result.config.l_hat,
            result.config.epsilon_k,
            bound_margin=result.config.bound_margin,
            bound_multiplier=result.config.bound_multiplier,
            clip=result.config.clip_bounds,
        )
        rows.append(
            _evaluation_row(
                window_row,
                nearest.context_id,
                nearest.context_name,
                distance,
                nearest.source_threshold,
                nearest.risk_estimate,
                nearest.source_certificate,
                bound,
                bound,
                summary,
                risk_budget=result.config.risk_control.risk_budget,
                nontrivial_feasible=bool(summary["num_accepted"] > 0 and bound <= result.config.risk_control.risk_budget),
                forced_accept_none=bool(summary["num_accepted"] == 0),
                num_candidates=np.nan,
            )
        )
    return pd.DataFrame(rows)


def evaluate_dkw_target_feasible_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    result: DKWCTResult,
) -> pd.DataFrame:
    """Evaluate per-window target-feasible selection; labels enter only after selection."""
    embedding_by_window = _validate_window_inputs(windows, embedding_frame)
    rows: list[dict[str, Any]] = []
    for _, window_row in windows.iterrows():
        embedding_row = _embedding_row(embedding_by_window, window_row["window_id"])
        query = embedding_row.loc[_embedding_columns_from_series(embedding_row)].to_numpy(dtype=float)
        indices = [int(index) for index in window_row["example_indices"]]
        window_examples = examples.iloc[indices]
        deployment_scores = pd.to_numeric(
            window_examples[result.config.risk_control.score_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        selection = select_target_feasible_threshold(deployment_scores, query, result)

        # Evaluation labels are deliberately accessed only after the score-only selection above.
        accepted = apply_threshold_policy(
            window_examples,
            result.config.risk_control.score_column,
            selection.threshold,
            "greater_equal",
        )
        losses = compute_losses(window_examples, accepted, result.config.risk_control.loss)
        summary = summarize_policy_performance(window_examples, accepted, losses)
        if int(summary["num_accepted"]) != selection.n_accepted:
            raise RuntimeError("score-only selection coverage does not match policy evaluation")
        if selection.transferred_upper_bound > result.config.risk_control.risk_budget + 1e-12:
            raise RuntimeError("selected target-feasible bound exceeds the risk budget")

        rows.append(
            _evaluation_row(
                window_row,
                selection.nearest_context_id,
                selection.nearest_context_name,
                selection.novelty_distance,
                selection.threshold,
                selection.source_risk_estimate,
                selection.source_certificate,
                selection.transferred_upper_bound,
                selection.raw_transferred_upper_bound,
                summary,
                risk_budget=result.config.risk_control.risk_budget,
                nontrivial_feasible=selection.nontrivial_feasible,
                forced_accept_none=selection.forced_accept_none,
                num_candidates=selection.num_candidates,
            )
        )
    return pd.DataFrame(rows)


def _coverage_at_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    candidates = np.asarray(thresholds, dtype=float).reshape(-1)
    first_accepted = np.searchsorted(ordered, candidates, side="left")
    return (ordered.size - first_accepted).astype(float) / ordered.size


def _validate_supported_risk_config(config: CTCRCConfig) -> None:
    risk_config = config.risk_control
    if risk_config.accept_if != "greater_equal":
        raise ValueError("DKW accepted-error calibration requires accept_if='greater_equal'")
    if risk_config.loss.loss_name != "accepted_error":
        raise ValueError("DKW accepted-error calibration requires loss_name='accepted_error'")


def _context_name(context_examples: pd.DataFrame, context_value: Any) -> str:
    if "context_name" not in context_examples.columns:
        return str(context_value)
    names = context_examples["context_name"].dropna().unique()
    return str(names[0]) if len(names) == 1 else str(context_value)


def _validate_window_inputs(
    windows: pd.DataFrame,
    embedding_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "window_id" not in windows.columns or "window_id" not in embedding_frame.columns:
        raise ValueError("windows and embedding_frame must include window_id")
    embedding_by_window = embedding_frame.set_index("window_id", drop=False)
    missing = [window_id for window_id in windows["window_id"] if window_id not in embedding_by_window.index]
    if missing:
        raise ValueError(f"embedding_frame is missing window_id values: {missing}")
    return embedding_by_window


def _embedding_row(embedding_by_window: pd.DataFrame, window_id: Any) -> pd.Series:
    row = embedding_by_window.loc[window_id]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _embedding_columns_from_series(row: pd.Series) -> list[str]:
    columns = get_embedding_columns(pd.DataFrame(columns=row.index))
    if not columns:
        raise ValueError("embedding_row must contain z* embedding columns")
    return columns


def _evaluation_row(
    window_row: pd.Series,
    nearest_context_id: str,
    nearest_context_name: str,
    novelty_distance: float,
    threshold: float,
    source_risk_estimate: float,
    source_certificate: float,
    transferred_bound: float,
    raw_transferred_bound: float,
    summary: dict[str, float],
    *,
    risk_budget: float,
    nontrivial_feasible: bool,
    forced_accept_none: bool,
    num_candidates: float | int,
) -> dict[str, Any]:
    return {
        "window_id": window_row["window_id"],
        "context_id": window_row.get("context_id", "unknown"),
        "context_name": window_row.get("context_name", "unknown"),
        "nearest_context_id": nearest_context_id,
        "nearest_context_name": nearest_context_name,
        "novelty_distance": novelty_distance,
        "threshold": threshold,
        "source_risk_estimate": source_risk_estimate,
        "source_certificate": source_certificate,
        "transferred_upper_bound": transferred_bound,
        "raw_transferred_upper_bound": raw_transferred_bound,
        "realized_risk": summary["mean_loss"],
        "mean_loss": summary["mean_loss"],
        "coverage": summary["coverage"],
        "n_examples": summary["n"],
        "num_accepted": summary["num_accepted"],
        "num_deferred": summary["num_deferred"],
        "accuracy_accepted": summary["accuracy_accepted"],
        "error_rate_accepted": summary["error_rate_accepted"],
        "accuracy_all": summary["accuracy_all"],
        "error_rate_all": summary["error_rate_all"],
        "nontrivial_feasible": nontrivial_feasible,
        "forced_accept_none": forced_accept_none,
        "bound_feasible": bool(transferred_bound <= risk_budget + 1e-12),
        "num_candidates": num_candidates,
    }
