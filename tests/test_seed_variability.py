from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from contextual_crc.artifacts.seed_variability import write_seed_variability_artifacts
from contextual_crc.evaluation.seed_variability import (
    METRIC_COLUMNS,
    SMOOTHNESS_COLUMNS,
    SeedVariabilityRunInputs,
    compute_seed_variability_from_inputs,
    normalize_seed_list,
    resample_window_ids,
    summarize_seed_metrics,
    summarize_seed_smoothness,
)


def test_seed_list_defaults_and_deduplicates() -> None:
    assert normalize_seed_list(None) == [0, 1, 2]
    assert normalize_seed_list([]) == [0, 1, 2]
    assert normalize_seed_list(["2", "2", "3"]) == [2, 3]


def test_window_resampling_is_reproducible() -> None:
    window_ids = ["w0", "w1", "w2", "w3"]
    first = resample_window_ids(window_ids, seed=7)
    second = resample_window_ids(window_ids, seed=7)
    different = resample_window_ids(window_ids, seed=8)
    assert first == second
    assert len(first) == len(window_ids)
    assert first != different


def test_seed_variability_computation_and_aggregation(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    metrics, smoothness = compute_seed_variability_from_inputs(inputs, seeds=[0, 1])

    assert len(metrics) == 10
    assert len(smoothness) == 2
    assert set(metrics["method"]) == {
        "global_crc",
        "metadata_stratified_crc",
        "ct_crc",
        "cw_crc",
        "oracle_context_crc",
    }
    metric_summary = summarize_seed_metrics(metrics)
    smoothness_summary = summarize_seed_smoothness(smoothness)
    assert len(metric_summary) == 5
    assert set(["mean_risk", "std_risk", "mean_coverage", "std_coverage", "n_seeds"]).issubset(metric_summary.columns)
    assert float(metric_summary["n_seeds"].min()) == 2.0
    assert len(smoothness_summary) == 1
    assert float(smoothness_summary["n_seeds"].iloc[0]) == 2.0
    assert np.isfinite(smoothness_summary["mean_smoothness_rho"].iloc[0])


def test_missing_method_and_smoothness_are_graceful(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    reduced = SeedVariabilityRunInputs(
        dataset=inputs.dataset,
        run_dir=inputs.run_dir,
        risk_budget=inputs.risk_budget,
        base_windows=inputs.base_windows,
        method_windows={"global_crc": inputs.method_windows["global_crc"]},
        smoothness_pairs=None,
    )
    metrics, smoothness = compute_seed_variability_from_inputs(reduced, seeds=[0])
    missing_ct = metrics[metrics["method"] == "ct_crc"].iloc[0]
    assert np.isnan(float(missing_ct["risk"]))
    assert float(missing_ct["n_windows"]) == 0.0
    assert np.isnan(float(smoothness["spearman_rho"].iloc[0]))


def test_seed_variability_artifacts_skip_when_absent(tmp_path: Path) -> None:
    written = write_seed_variability_artifacts(
        tmp_path / "tables",
        tmp_path / "figures",
        metrics_csv=tmp_path / "missing_metrics.csv",
        smoothness_csv=tmp_path / "missing_smoothness.csv",
    )
    assert written == {}


def test_seed_variability_artifacts_write_when_present(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    metrics, smoothness = compute_seed_variability_from_inputs(inputs, seeds=[0, 1])
    metrics_csv = tmp_path / "seed_variability_metrics.csv"
    smoothness_csv = tmp_path / "seed_variability_smoothness.csv"
    metrics.loc[:, list(METRIC_COLUMNS)].to_csv(metrics_csv, index=False)
    smoothness.loc[:, list(SMOOTHNESS_COLUMNS)].to_csv(smoothness_csv, index=False)

    written = write_seed_variability_artifacts(
        tmp_path / "tables",
        tmp_path / "figures",
        metrics_csv=metrics_csv,
        smoothness_csv=smoothness_csv,
    )
    assert Path(written["seed_variability_metrics_csv"]).is_file()
    assert Path(written["seed_variability_metrics_tex"]).is_file()
    assert Path(written["seed_variability_smoothness_csv"]).is_file()
    assert Path(written["seed_variability_smoothness_tex"]).is_file()
    assert Path(written["seed_variability_figure"]).is_file()


def _fake_inputs(tmp_path: Path) -> SeedVariabilityRunInputs:
    base_windows = pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "context_id": ["a", "a", "b", "b"],
            "n_examples": [10, 10, 10, 10],
        }
    )
    ct = pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "context_id": ["a", "a", "b", "b"],
            "risk": [0.02, 0.08, 0.12, 0.18],
            "coverage": [0.9, 0.8, 0.7, 0.6],
            "upper_bound": [0.06, 0.11, 0.14, 0.20],
            "n_examples": [10, 10, 10, 10],
        }
    )
    cw = ct.assign(risk=[0.01, 0.05, 0.09, 0.12], coverage=[0.75, 0.7, 0.65, 0.6])
    global_rows = ct.assign(risk=[0.08, 0.08, 0.08, 0.08], coverage=[0.8, 0.8, 0.8, 0.8])
    metadata = ct.assign(risk=[0.04, 0.04, 0.1, 0.1], coverage=[0.85, 0.85, 0.75, 0.75])
    oracle = ct.assign(risk=[0.03, 0.03, 0.07, 0.07], coverage=[0.9, 0.9, 0.8, 0.8])
    smoothness_pairs = pd.DataFrame(
        {
            "window_id_i": ["w0", "w0", "w0", "w1", "w1", "w2"],
            "window_id_j": ["w1", "w2", "w3", "w2", "w3", "w3"],
            "distance": [1.0, 2.0, 3.0, 1.5, 2.5, 1.0],
            "risk_difference": [0.01, 0.08, 0.13, 0.07, 0.12, 0.05],
        }
    )
    return SeedVariabilityRunInputs(
        dataset="fake_dataset",
        run_dir=tmp_path,
        risk_budget=0.10,
        base_windows=base_windows,
        method_windows={
            "global_crc": global_rows,
            "metadata_stratified_crc": metadata,
            "ct_crc": ct,
            "cw_crc": cw,
            "oracle_context_crc": oracle,
        },
        smoothness_pairs=smoothness_pairs,
    )
