from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from contextual_crc.risk_control.crc import (
    RiskControlConfig,
    RiskControlResult,
    calibrate_threshold,
    evaluate_threshold,
)
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy


def risk_control_result_to_dict(result: RiskControlResult) -> dict[str, Any]:
    """Convert a risk-control calibration result to a serializable dictionary."""
    payload = asdict(result)
    payload["config"] = asdict(result.config)
    payload["config"]["loss"] = asdict(result.config.loss)
    return payload


def run_global_crc_baseline(
    calibration_examples: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    config: RiskControlConfig,
) -> dict[str, Any]:
    """Run one global CRC-style threshold baseline and evaluate by context."""
    result = calibrate_threshold(calibration_examples, config)
    overall = evaluate_threshold(evaluation_examples, result)

    rows: list[dict[str, Any]] = []
    if "context_id" in evaluation_examples.columns:
        for context_id, group in evaluation_examples.groupby("context_id", sort=False, dropna=False):
            metrics = evaluate_threshold(group, result)
            metrics["context_id"] = context_id
            rows.append(metrics)

    return {
        "method": "global_crc",
        "result": risk_control_result_to_dict(result),
        "overall": overall,
        "by_context": pd.DataFrame(rows),
    }


def run_oracle_context_crc_baseline(
    calibration_examples: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    config: RiskControlConfig,
    context_col: str = "context_id",
) -> dict[str, Any]:
    """Run a same-context oracle/local CRC-style threshold baseline."""
    _require_column(calibration_examples, context_col, "calibration_examples")
    _require_column(evaluation_examples, context_col, "evaluation_examples")

    rows: list[dict[str, Any]] = []
    accepted_parts: list[np.ndarray] = []
    evaluated_parts: list[pd.DataFrame] = []

    for context_value, eval_group in evaluation_examples.groupby(context_col, sort=False, dropna=False):
        cal_group = calibration_examples[calibration_examples[context_col] == context_value]
        if len(cal_group) == 0:
            rows.append(_missing_group_metrics(context_col, context_value, len(eval_group)))
            continue

        result = calibrate_threshold(cal_group, config)
        accepted = apply_threshold_policy(
            eval_group,
            config.score_column,
            result.threshold,
            config.accept_if,
        )
        losses = compute_losses(eval_group, accepted, config.loss)
        metrics = summarize_policy_performance(eval_group, accepted, losses)
        metrics.update(
            {
                context_col: context_value,
                "threshold": result.threshold,
                "risk_upper_bound": result.risk_upper_bound,
                "feasible": float(result.feasible),
                "n_calibration": float(result.n_calibration),
            }
        )
        rows.append(metrics)
        accepted_parts.append(accepted)
        evaluated_parts.append(eval_group)

    overall = _aggregate_context_specific(evaluated_parts, accepted_parts, config)
    return {
        "method": "oracle_context_crc",
        "by_context": pd.DataFrame(rows),
        "overall": overall,
    }


def run_metadata_stratified_crc_baseline(
    calibration_examples: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    config: RiskControlConfig,
    stratify_col: str = "context_id",
    fallback_to_global: bool = True,
) -> dict[str, Any]:
    """Calibrate CRC-style thresholds per metadata stratum."""
    _require_column(calibration_examples, stratify_col, "calibration_examples")
    _require_column(evaluation_examples, stratify_col, "evaluation_examples")

    global_result = calibrate_threshold(calibration_examples, config) if fallback_to_global else None
    stratum_results: dict[Any, RiskControlResult] = {}
    for stratum_value, cal_group in calibration_examples.groupby(stratify_col, sort=False, dropna=False):
        if len(cal_group) > 0:
            stratum_results[stratum_value] = calibrate_threshold(cal_group, config)

    rows: list[dict[str, Any]] = []
    accepted_parts: list[np.ndarray] = []
    evaluated_parts: list[pd.DataFrame] = []

    for stratum_value, eval_group in evaluation_examples.groupby(stratify_col, sort=False, dropna=False):
        result = stratum_results.get(stratum_value)
        used_fallback = False
        if result is None and global_result is not None:
            result = global_result
            used_fallback = True

        if result is None:
            rows.append(_missing_group_metrics(stratify_col, stratum_value, len(eval_group)))
            continue

        accepted = apply_threshold_policy(
            eval_group,
            config.score_column,
            result.threshold,
            config.accept_if,
        )
        losses = compute_losses(eval_group, accepted, config.loss)
        metrics = summarize_policy_performance(eval_group, accepted, losses)
        metrics.update(
            {
                stratify_col: stratum_value,
                "threshold": result.threshold,
                "risk_upper_bound": result.risk_upper_bound,
                "feasible": float(result.feasible),
                "n_calibration": float(result.n_calibration),
                "used_fallback": bool(used_fallback),
            }
        )
        rows.append(metrics)
        accepted_parts.append(accepted)
        evaluated_parts.append(eval_group)

    overall = _aggregate_context_specific(evaluated_parts, accepted_parts, config)
    return {
        "method": "metadata_stratified_crc",
        "stratify_col": stratify_col,
        "by_stratum": pd.DataFrame(rows),
        "overall": overall,
    }


def _aggregate_context_specific(
    evaluated_parts: list[pd.DataFrame],
    accepted_parts: list[np.ndarray],
    config: RiskControlConfig,
) -> dict[str, float]:
    if not evaluated_parts:
        return {
            "n": 0.0,
            "coverage": float("nan"),
            "deferral_rate": float("nan"),
            "mean_loss": float("nan"),
            "num_accepted": 0.0,
            "num_deferred": 0.0,
            "accuracy_accepted": float("nan"),
            "error_rate_accepted": float("nan"),
            "accuracy_all": float("nan"),
            "error_rate_all": float("nan"),
        }

    examples = pd.concat(evaluated_parts, ignore_index=True)
    accepted = np.concatenate(accepted_parts).astype(bool)
    losses = compute_losses(examples, accepted, config.loss)
    return summarize_policy_performance(examples, accepted, losses)


def _missing_group_metrics(column: str, value: Any, n: int) -> dict[str, Any]:
    return {
        column: value,
        "n": float(n),
        "coverage": float("nan"),
        "deferral_rate": float("nan"),
        "mean_loss": float("nan"),
        "num_accepted": float("nan"),
        "num_deferred": float("nan"),
        "accuracy_accepted": float("nan"),
        "error_rate_accepted": float("nan"),
        "accuracy_all": float("nan"),
        "error_rate_all": float("nan"),
        "threshold": float("nan"),
        "risk_upper_bound": float("nan"),
        "feasible": 0.0,
        "n_calibration": 0.0,
    }


def _require_column(frame: pd.DataFrame, column: str, frame_name: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column: {column}")
