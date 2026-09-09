from __future__ import annotations

import numpy as np
import pandas as pd

from contextual_crc.risk_control.baselines import (
    run_global_crc_baseline,
    run_metadata_stratified_crc_baseline,
    run_oracle_context_crc_baseline,
)
from contextual_crc.risk_control.crc import RiskControlConfig


def make_examples(n_per_context: int = 80) -> pd.DataFrame:
    rows = []
    for context_id, base_confidence in (("source", 0.9), ("shift", 0.7)):
        for index in range(n_per_context):
            label = index % 2
            is_error = index < 8 if context_id == "source" else index < 16
            pred = 1 - label if is_error else label
            confidence = base_confidence - 0.006 * index
            rows.append(
                {
                    "example_id": f"{context_id}_{index}",
                    "context_id": context_id,
                    "context_name": context_id.title(),
                    "label": label,
                    "pred": pred,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(rows)


def split_examples(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_parts = []
    evaluation_parts = []
    for _, group in examples.groupby("context_id", sort=False):
        calibration_parts.append(group.iloc[:40])
        evaluation_parts.append(group.iloc[40:])
    return pd.concat(calibration_parts, ignore_index=True), pd.concat(evaluation_parts, ignore_index=True)


def make_config() -> RiskControlConfig:
    return RiskControlConfig(risk_budget=0.25, delta=0.2, min_coverage=0.0)


def test_global_baseline_returns_method_overall_by_context() -> None:
    calibration, evaluation = split_examples(make_examples())

    result = run_global_crc_baseline(calibration, evaluation, make_config())

    assert result["method"] == "global_crc"
    assert "overall" in result
    assert "by_context" in result
    assert len(result["by_context"]) == 2
    assert result["result"]["threshold"] == result["overall"]["threshold"]


def test_oracle_context_baseline_returns_per_context_results() -> None:
    calibration, evaluation = split_examples(make_examples())

    result = run_oracle_context_crc_baseline(calibration, evaluation, make_config())

    assert result["method"] == "oracle_context_crc"
    assert len(result["by_context"]) == 2
    assert set(result["by_context"]["context_id"]) == {"source", "shift"}
    assert "coverage" in result["overall"]


def test_metadata_stratified_baseline_works_with_fallback() -> None:
    calibration, evaluation = split_examples(make_examples())

    result = run_metadata_stratified_crc_baseline(calibration, evaluation, make_config())

    assert result["method"] == "metadata_stratified_crc"
    assert result["stratify_col"] == "context_id"
    assert len(result["by_stratum"]) == 2
    assert not result["by_stratum"]["used_fallback"].any()


def test_missing_stratum_with_fallback_to_global_works() -> None:
    calibration, evaluation = split_examples(make_examples())
    calibration = calibration[calibration["context_id"] != "shift"].reset_index(drop=True)

    result = run_metadata_stratified_crc_baseline(
        calibration,
        evaluation,
        make_config(),
        fallback_to_global=True,
    )

    by_stratum = result["by_stratum"]
    shift_row = by_stratum[by_stratum["context_id"] == "shift"].iloc[0]
    assert bool(shift_row["used_fallback"])
    assert np.isfinite(shift_row["threshold"])
    assert "coverage" in result["overall"]
