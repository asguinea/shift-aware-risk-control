from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.risk_control.policies import (
    ThresholdPolicy,
    apply_threshold_policy,
    candidate_thresholds,
)


def test_apply_greater_equal_threshold() -> None:
    examples = pd.DataFrame({"confidence": [0.2, 0.5, 0.8]})

    accepted = apply_threshold_policy(examples, "confidence", 0.5, "greater_equal")

    assert accepted.tolist() == [False, True, True]
    assert ThresholdPolicy(threshold=0.5).decide(examples).tolist() == [False, True, True]


def test_apply_less_equal_threshold() -> None:
    examples = pd.DataFrame({"energy": [-3.0, -1.0, 0.5]})

    accepted = apply_threshold_policy(examples, "energy", -1.0, "less_equal")

    assert accepted.tolist() == [True, True, False]


def test_missing_score_column_raises() -> None:
    with pytest.raises(ValueError, match="score"):
        apply_threshold_policy(pd.DataFrame({"x": [1.0]}), "confidence", 0.5)


def test_candidate_thresholds_include_accept_all_and_none_extremes() -> None:
    scores = np.array([0.2, 0.5, 0.8, np.nan])

    greater = candidate_thresholds(scores, accept_if="greater_equal")
    less = candidate_thresholds(scores, accept_if="less_equal")

    assert greater[0] < 0.2
    assert greater[-1] > 0.8
    assert less[0] > 0.8
    assert less[-1] < 0.2


def test_invalid_accept_if_raises() -> None:
    with pytest.raises(ValueError, match="accept_if"):
        candidate_thresholds(np.array([1.0]), accept_if="bad")
    with pytest.raises(ValueError, match="accept_if"):
        apply_threshold_policy(pd.DataFrame({"x": [1.0]}), "x", 0.0, accept_if="bad")
