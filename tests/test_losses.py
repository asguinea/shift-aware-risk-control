from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.risk_control.losses import (
    LossConfig,
    compute_losses,
    summarize_policy_performance,
)


def make_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 1, 1, 0],
            "pred": [0, 0, 1, 1],
        }
    )


def test_accepted_error_loss_correct() -> None:
    losses = compute_losses(
        make_examples(),
        np.array([True, True, False, True]),
        LossConfig("accepted_error"),
    )

    assert losses.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_accepted_error_with_deferral_cost_correct() -> None:
    losses = compute_losses(
        make_examples(),
        np.array([True, True, False, True]),
        LossConfig("accepted_error_with_deferral_cost", deferral_cost=0.25),
    )

    assert losses.tolist() == [0.0, 1.0, 0.25, 1.0]


def test_accepted_only_error_rate_returns_nan_for_deferred() -> None:
    losses = compute_losses(
        make_examples(),
        np.array([True, True, False, True]),
        LossConfig("accepted_only_error_rate"),
    )

    assert losses[0] == 0.0
    assert losses[1] == 1.0
    assert np.isnan(losses[2])
    assert losses[3] == 1.0


def test_summarize_policy_performance_handles_no_accepted_examples() -> None:
    examples = make_examples()
    accepted = np.array([False, False, False, False])
    losses = compute_losses(examples, accepted, LossConfig("accepted_error"))

    summary = summarize_policy_performance(examples, accepted, losses)

    assert summary["coverage"] == 0.0
    assert summary["deferral_rate"] == 1.0
    assert np.isnan(summary["accuracy_accepted"])
    assert np.isnan(summary["error_rate_accepted"])


def test_invalid_loss_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        compute_losses(make_examples(), np.array([True] * 4), LossConfig("bad"))


def test_invalid_deferral_cost_raises() -> None:
    with pytest.raises(ValueError, match="deferral_cost"):
        compute_losses(
            make_examples(),
            np.array([True] * 4),
            LossConfig("accepted_error_with_deferral_cost", deferral_cost=1.5),
        )
