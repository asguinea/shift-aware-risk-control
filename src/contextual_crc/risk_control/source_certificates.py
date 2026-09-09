from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

from contextual_crc.risk_control.policies import candidate_thresholds


DKWSide = Literal["one_sided", "two_sided"]


@dataclass(frozen=True)
class SourceCalibration:
    method: str
    threshold: float
    empirical_risk: float
    certificate: float
    coverage: float
    n_calibration: int
    n_accepted: int
    feasible: bool
    nontrivial: bool
    num_candidates: int
    guarantee_type: str
    uncertainty: float = float("nan")
    delta: float = float("nan")
    certified_set_size: int = 0
    p_value: float = float("nan")


def accepted_error_scores_finite(
    examples: pd.DataFrame,
    *,
    score_column: str = "confidence",
    correct_sentinel: float = -1.0,
) -> np.ndarray:
    """Represent accepted-error loss as 1{V >= theta} with finite V on correct rows.

    For confidence scores in [0, 1], any sentinel below zero makes this identity exact
    for every operational threshold theta in [0, 1].
    """
    required = {score_column, "pred", "label"}
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError(f"examples is missing required columns: {missing}")
    scores = pd.to_numeric(examples[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("confidence scores must be finite and in [0, 1]")
    if not np.isfinite(correct_sentinel) or correct_sentinel >= 0.0:
        raise ValueError("correct_sentinel must be finite and below the confidence range")
    incorrect = examples["pred"].astype(int).to_numpy() != examples["label"].astype(int).to_numpy()
    return np.where(incorrect, scores, float(correct_sentinel))


def accepted_error_risks(error_scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    values = np.asarray(error_scores, dtype=float).reshape(-1)
    candidates = np.asarray(thresholds, dtype=float).reshape(-1)
    if values.size == 0 or candidates.size == 0:
        raise ValueError("error_scores and thresholds must be non-empty")
    if not np.isfinite(values).all() or not np.isfinite(candidates).all():
        raise ValueError("error_scores and thresholds must be finite")
    ordered = np.sort(values)
    first = np.searchsorted(ordered, candidates, side="left")
    return (len(values) - first).astype(float) / len(values)


def dkw_radius(n: int, delta: float, *, side: DKWSide) -> float:
    """Return the one- or two-sided DKW/Massart uniform radius."""
    if int(n) < 1:
        raise ValueError("n must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if side == "one_sided":
        log_term = np.log(1.0 / float(delta))
    elif side == "two_sided":
        log_term = np.log(2.0 / float(delta))
    else:
        raise ValueError("side must be 'one_sided' or 'two_sided'")
    return float(np.sqrt(log_term / (2.0 * int(n))))


def dkw_upper_envelope(
    error_scores: np.ndarray,
    thresholds: Sequence[float],
    *,
    delta: float,
    side: DKWSide,
) -> np.ndarray:
    """Uniform upper envelope for survival risks P(V >= theta)."""
    risks = accepted_error_risks(error_scores, thresholds)
    return np.clip(risks + dkw_radius(len(np.asarray(error_scores).reshape(-1)), delta, side=side), 0, 1)


def calibrate_dkw_source(
    examples: pd.DataFrame,
    *,
    risk_budget: float,
    delta: float,
    side: DKWSide,
    score_column: str = "confidence",
) -> SourceCalibration:
    scores = _scores(examples, score_column)
    thresholds = candidate_thresholds(scores, accept_if="greater_equal", include_extremes=True)
    error_scores = accepted_error_scores_finite(examples, score_column=score_column)
    risks = accepted_error_risks(error_scores, thresholds)
    certificates = dkw_upper_envelope(error_scores, thresholds, delta=delta, side=side)
    coverages = coverage_at_thresholds(scores, thresholds)
    accepted = np.rint(coverages * len(scores)).astype(int)
    feasible = certificates <= float(risk_budget)
    selected = _select_max_coverage(feasible, coverages, certificates, thresholds)
    return SourceCalibration(
        method=f"{side}_dkw_source",
        threshold=float(thresholds[selected]),
        empirical_risk=float(risks[selected]),
        certificate=float(certificates[selected]),
        coverage=float(coverages[selected]),
        n_calibration=len(examples),
        n_accepted=int(accepted[selected]),
        feasible=bool(feasible[selected]),
        nontrivial=bool(feasible[selected] and accepted[selected] > 0),
        num_candidates=len(thresholds),
        guarantee_type="high_probability_uniform_threshold",
        uncertainty=float(certificates[selected] - risks[selected]),
        delta=float(delta),
    )


def calibrate_crc_expected_risk(
    examples: pd.DataFrame,
    *,
    risk_budget: float,
    score_column: str = "confidence",
    loss_bound: float = 1.0,
) -> SourceCalibration:
    """Original CRC rule for bounded monotone loss: (n Rhat + B)/(n+1) <= tau."""
    if loss_bound <= 0.0:
        raise ValueError("loss_bound must be positive")
    scores = _scores(examples, score_column)
    thresholds = candidate_thresholds(scores, accept_if="greater_equal", include_extremes=True)
    error_scores = accepted_error_scores_finite(examples, score_column=score_column)
    risks = accepted_error_risks(error_scores, thresholds)
    corrected = (len(examples) * risks + float(loss_bound)) / (len(examples) + 1.0)
    coverages = coverage_at_thresholds(scores, thresholds)
    accepted = np.rint(coverages * len(scores)).astype(int)
    feasible = corrected <= float(risk_budget)
    selected = _select_max_coverage(feasible, coverages, corrected, thresholds)
    return SourceCalibration(
        method="crc_expected_risk_source",
        threshold=float(thresholds[selected]),
        empirical_risk=float(risks[selected]),
        certificate=float(corrected[selected]),
        coverage=float(coverages[selected]),
        n_calibration=len(examples),
        n_accepted=int(accepted[selected]),
        feasible=bool(feasible[selected]),
        nontrivial=bool(feasible[selected] and accepted[selected] > 0),
        num_candidates=len(thresholds),
        guarantee_type="expected_risk",
        uncertainty=float(corrected[selected] - risks[selected]),
    )


def exact_binomial_ltt_p_value(k_errors: int, n: int, risk_target: float) -> float:
    """Lower-tail p-value for H0: p > risk_target against p <= risk_target."""
    if int(n) < 1 or not 0 <= int(k_errors) <= int(n):
        raise ValueError("require n >= 1 and 0 <= k_errors <= n")
    if not 0.0 < float(risk_target) < 1.0:
        raise ValueError("risk_target must be in (0, 1)")
    return float(binom.cdf(int(k_errors), int(n), float(risk_target)))


def calibrate_ltt_bonferroni_source(
    examples: pd.DataFrame,
    *,
    risk_target: float,
    delta: float,
    theta_grid: Sequence[float],
    score_column: str = "confidence",
) -> SourceCalibration:
    thresholds = _validate_grid(theta_grid)
    scores = _scores(examples, score_column)
    error_scores = accepted_error_scores_finite(examples, score_column=score_column)
    error_counts = np.rint(accepted_error_risks(error_scores, thresholds) * len(examples)).astype(int)
    p_values = np.asarray(
        [exact_binomial_ltt_p_value(k, len(examples), risk_target) for k in error_counts],
        dtype=float,
    )
    rejected = p_values <= float(delta) / len(thresholds)
    return _ltt_result(
        "ltt_bonferroni_source",
        examples,
        scores,
        error_counts,
        thresholds,
        p_values,
        rejected,
        risk_target,
        delta,
    )


def fixed_sequence_rejections(p_values: Sequence[float], alpha: float) -> np.ndarray:
    """Reject in supplied order until the first non-rejection, then stop."""
    values = np.asarray(p_values, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("p_values must be non-empty and finite")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rejected = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if value > float(alpha):
            break
        rejected[index] = True
    return rejected


def calibrate_ltt_fixed_sequence_source(
    examples: pd.DataFrame,
    *,
    risk_target: float,
    delta: float,
    theta_grid: Sequence[float],
    score_column: str = "confidence",
) -> SourceCalibration:
    thresholds = _validate_grid(theta_grid)
    scores = _scores(examples, score_column)
    error_scores = accepted_error_scores_finite(examples, score_column=score_column)
    error_counts = np.rint(accepted_error_risks(error_scores, thresholds) * len(examples)).astype(int)
    p_values = np.asarray(
        [exact_binomial_ltt_p_value(k, len(examples), risk_target) for k in error_counts],
        dtype=float,
    )
    # Test from most conservative (largest theta) toward highest coverage.
    reverse_rejected = fixed_sequence_rejections(p_values[::-1], delta)
    rejected = reverse_rejected[::-1]
    return _ltt_result(
        "ltt_fixed_sequence_source",
        examples,
        scores,
        error_counts,
        thresholds,
        p_values,
        rejected,
        risk_target,
        delta,
    )


def coverage_at_thresholds(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    candidates = np.asarray(thresholds, dtype=float).reshape(-1)
    first = np.searchsorted(ordered, candidates, side="left")
    return (len(ordered) - first).astype(float) / len(ordered)


def _ltt_result(
    method: str,
    examples: pd.DataFrame,
    scores: np.ndarray,
    error_counts: np.ndarray,
    thresholds: np.ndarray,
    p_values: np.ndarray,
    rejected: np.ndarray,
    risk_target: float,
    delta: float,
) -> SourceCalibration:
    risks = error_counts.astype(float) / len(examples)
    coverages = coverage_at_thresholds(scores, thresholds)
    accepted = np.rint(coverages * len(examples)).astype(int)
    if rejected.any():
        candidates = np.flatnonzero(rejected)
        selected = min(candidates, key=lambda i: (-coverages[i], thresholds[i]))
        feasible = True
    else:
        selected = len(thresholds) - 1
        feasible = False
    return SourceCalibration(
        method=method,
        threshold=float(thresholds[selected]),
        empirical_risk=float(risks[selected]),
        certificate=float(risk_target) if feasible else float("nan"),
        coverage=float(coverages[selected]),
        n_calibration=len(examples),
        n_accepted=int(accepted[selected]),
        feasible=feasible,
        nontrivial=bool(feasible and accepted[selected] > 0),
        num_candidates=len(thresholds),
        guarantee_type="high_probability_finite_family",
        delta=float(delta),
        certified_set_size=int(rejected.sum()),
        p_value=float(p_values[selected]),
    )


def _select_max_coverage(
    feasible: np.ndarray,
    coverages: np.ndarray,
    certificates: np.ndarray,
    thresholds: np.ndarray,
) -> int:
    if feasible.any():
        choices = np.flatnonzero(feasible)
        return int(min(choices, key=lambda i: (-coverages[i], certificates[i], thresholds[i])))
    return int(min(range(len(thresholds)), key=lambda i: (certificates[i], -coverages[i], thresholds[i])))


def _scores(examples: pd.DataFrame, score_column: str) -> np.ndarray:
    if len(examples) == 0 or score_column not in examples:
        raise ValueError("examples must be non-empty and contain the score column")
    scores = pd.to_numeric(examples[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("confidence scores must be finite and in [0, 1]")
    return scores


def _validate_grid(theta_grid: Sequence[float]) -> np.ndarray:
    values = np.asarray(theta_grid, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("theta_grid must be non-empty and finite")
    if ((values < 0.0) | (values > 1.0)).any() or np.any(np.diff(values) <= 0.0):
        raise ValueError("theta_grid must be strictly increasing in [0, 1]")
    return values
