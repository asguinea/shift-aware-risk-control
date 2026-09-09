from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from contextual_crc.risk_control.losses import (
    LossConfig,
    compute_losses,
    summarize_policy_performance,
)
from contextual_crc.risk_control.policies import (
    ThresholdPolicy,
    apply_threshold_policy,
    candidate_thresholds,
)


ALLOWED_BOUNDS = ("hoeffding", "empirical_bernstein")


@dataclass(frozen=True)
class RiskControlConfig:
    score_column: str = "confidence"
    accept_if: str = "greater_equal"
    loss: LossConfig = LossConfig()
    risk_budget: float = 0.1
    delta: float = 0.05
    bound: str = "hoeffding"
    min_coverage: float = 0.0


@dataclass(frozen=True)
class RiskControlResult:
    threshold: float
    risk_estimate: float
    risk_upper_bound: float
    coverage: float
    n_calibration: int
    n_accepted: int
    config: RiskControlConfig
    feasible: bool
    num_candidates: int


def risk_upper_confidence_bound(
    losses: np.ndarray,
    delta: float,
    bound: str = "hoeffding",
) -> float:
    """Compute a bounded risk upper confidence bound for finite losses in [0, 1]."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if bound not in ALLOWED_BOUNDS:
        raise ValueError(f"bound must be one of {ALLOWED_BOUNDS}, got {bound!r}")

    losses = np.asarray(losses, dtype=float).reshape(-1)
    if losses.size == 0:
        raise ValueError("losses must be non-empty")
    if not np.isfinite(losses).all():
        raise ValueError("losses must be finite")
    if ((losses < 0.0) | (losses > 1.0)).any():
        raise ValueError("losses must be in [0, 1]")

    n = losses.size
    mean = float(np.mean(losses))
    if bound == "hoeffding":
        upper = mean + float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))
    else:
        variance = float(np.var(losses, ddof=1)) if n > 1 else 0.0
        log_term = float(np.log(3.0 / delta))
        upper = mean + float(np.sqrt(2.0 * variance * log_term / n)) + 3.0 * log_term / n
    return float(np.clip(upper, 0.0, 1.0))


def calibrate_threshold(
    calibration_examples: pd.DataFrame,
    config: RiskControlConfig,
) -> RiskControlResult:
    """Calibrate a scalar threshold using a practical risk-UCB baseline."""
    _validate_risk_config(config)
    if config.loss.loss_name == "accepted_only_error_rate":
        raise ValueError("accepted_only_error_rate is diagnostic and cannot be used for UCB calibration")

    if len(calibration_examples) == 0:
        raise ValueError("calibration_examples must be non-empty")
    if config.score_column not in calibration_examples.columns:
        raise ValueError(f"calibration_examples missing score column: {config.score_column}")

    scores = calibration_examples[config.score_column].to_numpy(dtype=float)
    thresholds = candidate_thresholds(scores, accept_if=config.accept_if, include_extremes=True)

    candidates: list[dict[str, float | int | bool]] = []
    for threshold in thresholds:
        accepted = apply_threshold_policy(
            calibration_examples,
            config.score_column,
            float(threshold),
            config.accept_if,
        )
        losses = compute_losses(calibration_examples, accepted, config.loss)
        if not np.isfinite(losses).all():
            raise ValueError("calibration losses must be finite")
        risk_estimate = float(np.mean(losses))
        risk_upper_bound = risk_upper_confidence_bound(losses, config.delta, config.bound)
        coverage = float(np.mean(accepted))
        candidates.append(
            {
                "threshold": float(threshold),
                "risk_estimate": risk_estimate,
                "risk_upper_bound": risk_upper_bound,
                "coverage": coverage,
                "n_accepted": int(accepted.sum()),
                "feasible": bool(
                    risk_upper_bound <= config.risk_budget and coverage >= config.min_coverage
                ),
            }
        )

    feasible_candidates = [candidate for candidate in candidates if bool(candidate["feasible"])]
    if feasible_candidates:
        selected = sorted(
            feasible_candidates,
            key=lambda candidate: (-float(candidate["coverage"]), float(candidate["risk_upper_bound"])),
        )[0]
        feasible = True
    else:
        selected = sorted(
            candidates,
            key=lambda candidate: (float(candidate["risk_upper_bound"]), -float(candidate["coverage"])),
        )[0]
        feasible = False

    return RiskControlResult(
        threshold=float(selected["threshold"]),
        risk_estimate=float(selected["risk_estimate"]),
        risk_upper_bound=float(selected["risk_upper_bound"]),
        coverage=float(selected["coverage"]),
        n_calibration=int(len(calibration_examples)),
        n_accepted=int(selected["n_accepted"]),
        config=config,
        feasible=feasible,
        num_candidates=len(candidates),
    )


def evaluate_threshold(
    examples: pd.DataFrame,
    result: RiskControlResult | ThresholdPolicy | float,
    config: RiskControlConfig | None = None,
) -> dict[str, float]:
    """Evaluate a calibrated or raw threshold policy on labeled examples."""
    risk_upper_bound = float("nan")
    feasible = float("nan")

    if isinstance(result, RiskControlResult):
        effective_config = config or result.config
        threshold = result.threshold
        risk_upper_bound = result.risk_upper_bound
        feasible = float(result.feasible)
        accepted = apply_threshold_policy(
            examples,
            effective_config.score_column,
            threshold,
            effective_config.accept_if,
        )
    elif isinstance(result, ThresholdPolicy):
        if config is None:
            effective_config = RiskControlConfig(
                score_column=result.score_column,
                accept_if=result.accept_if,
            )
        else:
            effective_config = config
        threshold = result.threshold
        accepted = result.decide(examples)
    else:
        if config is None:
            raise ValueError("config is required when evaluating a raw threshold")
        effective_config = config
        threshold = float(result)
        accepted = apply_threshold_policy(
            examples,
            effective_config.score_column,
            threshold,
            effective_config.accept_if,
        )

    losses = compute_losses(examples, accepted, effective_config.loss)
    metrics = summarize_policy_performance(examples, accepted, losses)
    metrics.update(
        {
            "threshold": float(threshold),
            "risk_upper_bound": float(risk_upper_bound),
            "feasible": float(feasible),
        }
    )
    return metrics


def _validate_risk_config(config: RiskControlConfig) -> None:
    if not 0.0 <= config.risk_budget <= 1.0:
        raise ValueError("risk_budget must be in [0, 1]")
    if not 0.0 < config.delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if config.bound not in ALLOWED_BOUNDS:
        raise ValueError(f"bound must be one of {ALLOWED_BOUNDS}, got {config.bound!r}")
    if not 0.0 <= config.min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
