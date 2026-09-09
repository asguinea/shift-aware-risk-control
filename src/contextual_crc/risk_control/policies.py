from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ALLOWED_ACCEPT_IF = ("greater_equal", "less_equal")


@dataclass(frozen=True)
class ThresholdPolicy:
    score_column: str = "confidence"
    threshold: float = 0.5
    accept_if: str = "greater_equal"

    def decide(self, examples: pd.DataFrame) -> np.ndarray:
        return apply_threshold_policy(
            examples,
            score_column=self.score_column,
            threshold=self.threshold,
            accept_if=self.accept_if,
        )


def apply_threshold_policy(
    examples: pd.DataFrame,
    score_column: str,
    threshold: float,
    accept_if: str = "greater_equal",
) -> np.ndarray:
    """Apply a scalar threshold selective-prediction policy."""
    if score_column not in examples.columns:
        raise ValueError(f"examples is missing score column: {score_column}")
    if accept_if not in ALLOWED_ACCEPT_IF:
        raise ValueError(f"accept_if must be one of {ALLOWED_ACCEPT_IF}, got {accept_if!r}")

    scores = pd.to_numeric(examples[score_column], errors="coerce").to_numpy(dtype=float)
    if accept_if == "greater_equal":
        return scores >= threshold
    return scores <= threshold


def candidate_thresholds(
    scores: np.ndarray,
    accept_if: str = "greater_equal",
    include_extremes: bool = True,
) -> np.ndarray:
    """Return finite candidate thresholds plus optional accept-all/none extremes."""
    if accept_if not in ALLOWED_ACCEPT_IF:
        raise ValueError(f"accept_if must be one of {ALLOWED_ACCEPT_IF}, got {accept_if!r}")

    finite_scores = np.asarray(scores, dtype=float).reshape(-1)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        raise ValueError("scores must contain at least one finite value")

    unique_scores = np.unique(finite_scores)
    if not include_extremes:
        return unique_scores if accept_if == "greater_equal" else unique_scores[::-1]

    min_score = float(unique_scores[0])
    max_score = float(unique_scores[-1])
    span = max(max_score - min_score, 1.0)
    eps = max(span * 1e-9, 1e-12)

    if accept_if == "greater_equal":
        candidates = np.concatenate(([min_score - eps], unique_scores, [max_score + eps]))
        return np.unique(candidates)

    candidates = np.concatenate(([max_score + eps], unique_scores[::-1], [min_score - eps]))
    _, first_indices = np.unique(candidates, return_index=True)
    return candidates[np.sort(first_indices)]
