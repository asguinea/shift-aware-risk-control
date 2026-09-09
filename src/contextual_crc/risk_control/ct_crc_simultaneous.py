from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pandas as pd

from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.risk_control.crc import calibrate_threshold
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    CTCRCResult,
    ValidatedContext,
)
from contextual_crc.risk_control.policies import candidate_thresholds


ALLOWED_SIMULTANEOUS_VARIANTS = (
    "ct_crc_library_bonferroni",
    "ct_crc_fully_simultaneous",
)


def allocate_context_delta(library_delta: float, n_contexts: int) -> float:
    """Allocate an overall failure budget equally across validated contexts."""
    _validate_library_delta(library_delta)
    if n_contexts < 1:
        raise ValueError("n_contexts must be positive")
    return float(library_delta / n_contexts)


def count_candidate_thresholds(
    calibration_examples: pd.DataFrame,
    *,
    score_column: str = "confidence",
    accept_if: str = "greater_equal",
    context_col: str = "context_id",
) -> dict[str, int]:
    """Count the exact finite threshold candidates searched in every context."""
    if context_col not in calibration_examples.columns:
        raise ValueError(f"calibration_examples is missing context column: {context_col}")
    if score_column not in calibration_examples.columns:
        raise ValueError(f"calibration_examples is missing score column: {score_column}")

    counts: dict[str, int] = {}
    for context_value, context_examples in calibration_examples.groupby(
        context_col,
        sort=False,
        dropna=False,
    ):
        thresholds = candidate_thresholds(
            context_examples[score_column].to_numpy(dtype=float),
            accept_if=accept_if,
            include_extremes=True,
        )
        counts[str(context_value)] = int(len(thresholds))
    if not counts:
        raise ValueError("At least one validated context is required")
    return counts


def allocate_threshold_deltas(
    candidate_counts: Mapping[str, int],
    library_delta: float,
) -> dict[str, float]:
    """Allocate context budget equally, then Bonferroni-correct within each search."""
    if not candidate_counts:
        raise ValueError("candidate_counts must be non-empty")
    context_delta = allocate_context_delta(library_delta, len(candidate_counts))
    allocations: dict[str, float] = {}
    for context_id, count in candidate_counts.items():
        if int(count) < 1:
            raise ValueError("candidate threshold counts must be positive")
        allocations[str(context_id)] = float(context_delta / int(count))
    return allocations


def total_allocated_failure_probability(
    candidate_counts: Mapping[str, int],
    threshold_deltas: Mapping[str, float],
) -> float:
    """Return the union-bound mass allocated over all context-threshold pairs."""
    if set(candidate_counts) != set(threshold_deltas):
        raise ValueError("candidate_counts and threshold_deltas must have identical keys")
    total = 0.0
    for context_id, count in candidate_counts.items():
        if int(count) < 1:
            raise ValueError("candidate threshold counts must be positive")
        delta = float(threshold_deltas[context_id])
        if not 0.0 < delta < 1.0:
            raise ValueError("threshold deltas must be in (0, 1)")
        total += int(count) * delta
    return float(total)


def build_simultaneously_validated_contexts(
    calibration_examples: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    config: CTCRCConfig,
    *,
    library_delta: float,
    variant: str,
    context_col: str = "context_id",
) -> tuple[CTCRCResult, pd.DataFrame]:
    """Build a CT library with explicit context and optional threshold-search correction."""
    if variant not in ALLOWED_SIMULTANEOUS_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_SIMULTANEOUS_VARIANTS}, got {variant!r}")
    _validate_library_delta(library_delta)
    if context_col not in calibration_examples.columns:
        raise ValueError(f"calibration_examples is missing context column: {context_col}")
    if context_col not in calibration_embedding_frame.columns:
        raise ValueError(f"calibration_embedding_frame is missing context column: {context_col}")

    embedding_columns = get_embedding_columns(calibration_embedding_frame)
    if not embedding_columns:
        raise ValueError("calibration_embedding_frame must contain z* embedding columns")

    candidate_counts = count_candidate_thresholds(
        calibration_examples,
        score_column=config.risk_control.score_column,
        accept_if=config.risk_control.accept_if,
        context_col=context_col,
    )
    context_delta = allocate_context_delta(library_delta, len(candidate_counts))
    threshold_deltas = (
        allocate_threshold_deltas(candidate_counts, library_delta)
        if variant == "ct_crc_fully_simultaneous"
        else {context_id: context_delta for context_id in candidate_counts}
    )

    contexts: list[ValidatedContext] = []
    audit_rows: list[dict[str, Any]] = []
    for context_value, context_examples in calibration_examples.groupby(
        context_col,
        sort=False,
        dropna=False,
    ):
        context_id = str(context_value)
        context_windows = calibration_embedding_frame[
            calibration_embedding_frame[context_col] == context_value
        ]
        if len(context_windows) == 0:
            raise ValueError(f"No calibration embedding windows found for context: {context_value}")

        candidate_count = candidate_counts[context_id]
        threshold_delta = threshold_deltas[context_id]
        context_risk_config = replace(config.risk_control, delta=threshold_delta)
        calibration_result = calibrate_threshold(context_examples, context_risk_config)
        if calibration_result.num_candidates != candidate_count:
            raise RuntimeError("candidate-threshold audit does not match calibration search")

        centroid = context_windows[embedding_columns].to_numpy(dtype=float).mean(axis=0)
        context_name = _context_name(context_examples, context_value)
        contexts.append(
            ValidatedContext(
                context_id=context_id,
                context_name=context_name,
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
        audit_rows.append(
            {
                "variant": variant,
                "context_id": context_id,
                "context_name": context_name,
                "candidate_threshold_count": candidate_count,
                "context_failure_budget": context_delta,
                "threshold_delta": threshold_delta,
                "allocated_failure_probability": (
                    candidate_count * threshold_delta
                    if variant == "ct_crc_fully_simultaneous"
                    else context_delta
                ),
                "threshold": calibration_result.threshold,
                "certificate": calibration_result.risk_upper_bound,
                "calibration_risk_estimate": calibration_result.risk_estimate,
                "calibration_coverage": calibration_result.coverage,
                "n_calibration": calibration_result.n_calibration,
                "n_windows": int(len(context_windows)),
                "feasible": calibration_result.feasible,
            }
        )

    global_result = (
        calibrate_threshold(calibration_examples, config.risk_control)
        if config.fallback_to_global
        else None
    )
    return CTCRCResult(config=config, contexts=contexts, global_result=global_result), pd.DataFrame(audit_rows)


def _validate_library_delta(library_delta: float) -> None:
    if not 0.0 < float(library_delta) < 1.0:
        raise ValueError("library_delta must be in (0, 1)")


def _context_name(context_examples: pd.DataFrame, context_value: Any) -> str:
    if "context_name" not in context_examples.columns:
        return str(context_value)
    names = context_examples["context_name"].dropna().unique()
    if len(names) == 1:
        return str(names[0])
    return str(context_value)
