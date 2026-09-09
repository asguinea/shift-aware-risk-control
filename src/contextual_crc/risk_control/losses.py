from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ALLOWED_LOSSES = (
    "accepted_error",
    "accepted_error_with_deferral_cost",
    "accepted_only_error_rate",
)


@dataclass(frozen=True)
class LossConfig:
    loss_name: str = "accepted_error"
    deferral_cost: float = 0.0


def compute_losses(
    examples: pd.DataFrame,
    accepted: np.ndarray,
    config: LossConfig,
) -> np.ndarray:
    """Compute per-example operational losses for a selective policy."""
    _validate_loss_inputs(examples, accepted, config)

    accepted = np.asarray(accepted, dtype=bool)
    labels = examples["label"].astype(int).to_numpy()
    preds = examples["pred"].astype(int).to_numpy()
    incorrect = preds != labels

    if config.loss_name == "accepted_error":
        return (accepted & incorrect).astype(float)

    if config.loss_name == "accepted_error_with_deferral_cost":
        losses = np.zeros(len(examples), dtype=float)
        losses[accepted & incorrect] = 1.0
        losses[~accepted] = float(config.deferral_cost)
        return losses

    if config.loss_name == "accepted_only_error_rate":
        losses = np.full(len(examples), np.nan, dtype=float)
        losses[accepted] = incorrect[accepted].astype(float)
        return losses

    raise ValueError(f"Unknown loss_name: {config.loss_name}")


def summarize_policy_performance(
    examples: pd.DataFrame,
    accepted: np.ndarray,
    losses: np.ndarray,
) -> dict[str, float]:
    """Summarize coverage, losses, and diagnostic accuracy/error metrics."""
    if "label" not in examples.columns or "pred" not in examples.columns:
        raise ValueError("examples must include label and pred columns")

    accepted = np.asarray(accepted, dtype=bool)
    losses = np.asarray(losses, dtype=float)
    if len(accepted) != len(examples):
        raise ValueError("accepted length must match examples length")
    if len(losses) != len(examples):
        raise ValueError("losses length must match examples length")

    n = len(examples)
    labels = examples["label"].astype(int).to_numpy()
    preds = examples["pred"].astype(int).to_numpy()
    correct = preds == labels
    num_accepted = int(accepted.sum())
    num_deferred = int(n - num_accepted)

    if num_accepted > 0:
        accepted_accuracy = float(np.mean(correct[accepted]))
        accepted_error = float(1.0 - accepted_accuracy)
    else:
        accepted_accuracy = float("nan")
        accepted_error = float("nan")

    if np.isfinite(losses).any():
        mean_loss = float(np.nanmean(losses))
    else:
        mean_loss = float("nan")

    accuracy_all = float(np.mean(correct)) if n > 0 else float("nan")
    return {
        "n": float(n),
        "coverage": float(num_accepted / n) if n > 0 else float("nan"),
        "deferral_rate": float(num_deferred / n) if n > 0 else float("nan"),
        "mean_loss": mean_loss,
        "num_accepted": float(num_accepted),
        "num_deferred": float(num_deferred),
        "accuracy_accepted": accepted_accuracy,
        "error_rate_accepted": accepted_error,
        "accuracy_all": accuracy_all,
        "error_rate_all": float(1.0 - accuracy_all) if np.isfinite(accuracy_all) else float("nan"),
    }


def _validate_loss_inputs(
    examples: pd.DataFrame,
    accepted: np.ndarray,
    config: LossConfig,
) -> None:
    missing = [column for column in ("label", "pred") if column not in examples.columns]
    if missing:
        raise ValueError(f"examples is missing required loss columns: {missing}")
    if config.loss_name not in ALLOWED_LOSSES:
        raise ValueError(f"Unknown loss_name: {config.loss_name}")
    if not 0.0 <= config.deferral_cost <= 1.0:
        raise ValueError("deferral_cost must be in [0, 1]")
    if len(accepted) != len(examples):
        raise ValueError("accepted length must match examples length")
