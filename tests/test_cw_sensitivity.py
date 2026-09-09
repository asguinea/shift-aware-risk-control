from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.cw_sensitivity import (
    load_cw_sensitivity_table,
    write_cw_sensitivity_artifacts,
)
from contextual_crc.evaluation.cw_sensitivity import (
    SENSITIVITY_COLUMNS,
    build_compact_cw_sensitivity_grid,
    summarize_cw_sensitivity_windows,
)


def test_build_compact_ess_and_margin_sweep_grid() -> None:
    settings = build_compact_cw_sensitivity_grid(
        {"target_effective_sample_size": 6, "bound_margin": 0.03, "bound_multiplier": 1.25},
        ess_grid=[3, 6],
        margin_grid=[0.0, 0.03],
    )

    assert [setting.sweep_type for setting in settings] == [
        "target_effective_sample_size",
        "target_effective_sample_size",
        "bound_margin",
        "bound_margin",
    ]
    assert [setting.target_effective_sample_size for setting in settings] == [3.0, 6.0, 6.0, 6.0]
    assert [setting.bound_margin for setting in settings] == [0.03, 0.03, 0.0, 0.03]
    assert [setting.is_current_setting for setting in settings] == [False, True, False, True]


def test_summarize_cw_sensitivity_results_records_ess() -> None:
    windows = pd.DataFrame(
        {
            "realized_risk": [0.05, 0.12, 0.02],
            "coverage": [0.8, 0.6, 1.0],
            "weighted_risk_upper_bound": [0.08, 0.10, 0.04],
            "effective_sample_size": [3.0, 6.0, 9.0],
            "threshold": [0.7, 0.8, 0.9],
        }
    )

    summary = summarize_cw_sensitivity_windows(windows, risk_budget=0.10)

    assert summary["risk"] == pytest.approx(0.06333333333333334)
    assert summary["coverage"] == pytest.approx(0.8)
    assert summary["bound_underestimation"] == pytest.approx(1 / 3)
    assert summary["mean_bound_gap"] == pytest.approx(0.01)
    assert summary["violation_rate"] == pytest.approx(1 / 3)
    assert summary["mean_effective_sample_size"] == pytest.approx(6.0)
    assert summary["median_effective_sample_size"] == pytest.approx(6.0)
    assert summary["mean_selected_threshold"] == pytest.approx(0.8)
    assert summary["n_windows"] == pytest.approx(3.0)


def test_summarize_cw_sensitivity_handles_missing_ess() -> None:
    windows = pd.DataFrame(
        {
            "realized_risk": [0.05, 0.12],
            "coverage": [0.8, 0.6],
            "weighted_risk_upper_bound": [0.08, 0.10],
        }
    )

    summary = summarize_cw_sensitivity_windows(windows, risk_budget=0.10)

    assert np.isnan(summary["mean_effective_sample_size"])
    assert np.isnan(summary["median_effective_sample_size"])
    assert np.isnan(summary["mean_selected_threshold"])
    assert summary["n_windows"] == pytest.approx(2.0)


def test_load_missing_cw_sensitivity_table_is_empty(tmp_path: Path) -> None:
    table = load_cw_sensitivity_table(tmp_path / "missing.csv")
    written = write_cw_sensitivity_artifacts(
        tmp_path / "artifacts" / "tables",
        tmp_path / "artifacts" / "figures",
        sensitivity_csv=tmp_path / "missing.csv",
    )

    assert table.empty
    assert list(table.columns) == list(SENSITIVITY_COLUMNS)
    assert written == {}


def test_cw_sensitivity_artifact_generation_when_present(tmp_path: Path) -> None:
    sensitivity_csv = tmp_path / "cw_sensitivity.csv"
    _fake_sensitivity_table().to_csv(sensitivity_csv, index=False)
    tables_dir = tmp_path / "artifacts" / "tables"
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_cw_sensitivity_artifacts(
        tables_dir,
        figures_dir,
        sensitivity_csv=sensitivity_csv,
    )

    assert (tables_dir / "cw_sensitivity.csv").is_file()
    assert (tables_dir / "cw_sensitivity.tex").is_file()
    assert (figures_dir / "cw_sensitivity_tradeoff.pdf").is_file()
    assert "cw_sensitivity_csv" in written


def _fake_sensitivity_table() -> pd.DataFrame:
    rows = []
    for dataset in ("fixture_a", "fixture_b"):
        for sweep_type, values in (
            ("target_effective_sample_size", [3.0, 6.0]),
            ("bound_margin", [0.0, 0.03]),
        ):
            for value in values:
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_display": dataset,
                        "run_dir": f"runs/{dataset}",
                        "sweep_type": sweep_type,
                        "sweep_value": value,
                        "target_effective_sample_size": 6.0 if sweep_type == "bound_margin" else value,
                        "bound_margin": value if sweep_type == "bound_margin" else 0.03,
                        "bound_multiplier": 1.0,
                        "is_current_setting": value == 6.0 or value == 0.03,
                        "risk": 0.08 + 0.01 * value,
                        "coverage": 0.5 + 0.01 * value,
                        "bound_underestimation": 0.1,
                        "mean_bound_gap": 0.02,
                        "violation_rate": 0.0,
                        "mean_effective_sample_size": value if sweep_type == "target_effective_sample_size" else 6.0,
                        "median_effective_sample_size": value if sweep_type == "target_effective_sample_size" else 6.0,
                        "mean_selected_threshold": 0.7,
                        "n_windows": 3,
                        "notes": "fake",
                    }
                )
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
