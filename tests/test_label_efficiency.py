from __future__ import annotations

from pathlib import Path

import pandas as pd

from contextual_crc.datasets.synthetic import SyntheticConfig, build_synthetic_prediction_cache
from contextual_crc.evaluation.label_efficiency import (
    examples_from_windows,
    run_label_efficiency_from_cache,
    subsample_calibration_windows,
)
from contextual_crc.signatures.windowing import WindowConfig, make_windows


def make_label_efficiency_config(tmp_path: Path, cache_dir: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "synthetic_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "synthetic_label_test",
            "cache_dir": str(cache_dir),
        },
        "windowing": {
            "window_size": 20,
            "stride": 20,
            "group_by_context": True,
            "sort_by": "time_index",
            "drop_incomplete": True,
        },
        "signatures": {"include_label_diagnostics": True},
        "encoder": {"method": "pca", "n_components": 2, "whiten": False, "random_seed": 123},
        "risk_control": {
            "score_column": "confidence",
            "accept_if": "greater_equal",
            "risk_budget": 0.25,
            "delta": 0.2,
            "bound": "hoeffding",
            "min_coverage": 0.0,
            "loss": {"loss_name": "accepted_error", "deferral_cost": 0.0},
        },
        "splits": {
            "calibration_contexts": ["source", "covariate_s1", "label_s1", "mixture_s1"],
            "evaluation_contexts": "all",
        },
        "ct_crc": {
            "metric": "euclidean",
            "l_hat": "auto",
            "l_hat_quantile": 0.9,
            "epsilon_k": 0.0,
            "fallback_to_global": True,
        },
        "cw_crc": {
            "metric": "euclidean",
            "kernel": "gaussian",
            "bandwidth": None,
            "target_effective_sample_size": 2,
            "min_bandwidth": 1.0e-6,
            "max_bandwidth": None,
            "bandwidth_grid_size": 8,
            "example_weight_aggregation": "mean",
            "finite_sample_correction": "hoeffding",
            "normalize_weights": True,
        },
    }


def test_subsample_calibration_windows_and_examples_from_windows() -> None:
    examples = pd.DataFrame({"context_id": ["a"] * 6 + ["b"] * 6, "value": range(12)})
    windows = make_windows(
        examples.assign(
            example_id=[f"ex{i}" for i in range(12)],
            dataset="fixture",
            split="eval",
            context_name=examples["context_id"],
            label=0,
            pred=0,
            confidence=0.9,
            time_index=0,
        ),
        WindowConfig(window_size=2, stride=2, group_by_context=True, drop_incomplete=True),
    )

    selected = subsample_calibration_windows(windows, windows_per_context=1, random_seed=123)
    subset = examples_from_windows(examples, selected)

    assert len(selected) == 2
    assert len(subset) == 4
    assert "original_index" in subset.columns


def test_run_label_efficiency_from_cache_writes_outputs(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "synthetic"
    build_synthetic_prediction_cache(
        cache_dir,
        SyntheticConfig(n_train=300, n_per_context=60, random_seed=123),
        overwrite=True,
    )
    output_dir = tmp_path / "label_efficiency"

    saved = run_label_efficiency_from_cache(
        experiment_config=make_label_efficiency_config(tmp_path, cache_dir),
        budgets=[1, 2],
        output_dir=output_dir,
        dataset_kind="synthetic",
        overwrite=True,
    )

    frame = pd.read_csv(saved / "label_efficiency.csv")
    assert (saved / "label_efficiency_manifest.json").is_file()
    assert set(frame["budget_windows_per_context"]) == {1, 2}
    assert {"global_crc", "ct_crc"}.issubset(set(frame["method"]))
    assert frame["actual_labeled_examples"].min() > 0
