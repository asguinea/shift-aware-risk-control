from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from contextual_crc.evaluation.synthetic_experiment import run_synthetic_experiment


def make_small_config(tmp_path: Path) -> dict:
    cache_dir = tmp_path / "cache" / "synthetic_small"
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "synthetic_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "synthetic_test",
            "cache_dir": str(cache_dir),
        },
        "synthetic_cache": {
            "build_if_missing": True,
            "overwrite_cache": False,
            "config_path": "configs/synthetic.yaml",
        },
        "synthetic": {
            "dataset_name": "synthetic_contextual_shift",
            "num_classes": 3,
            "input_dim": 2,
            "n_train": 300,
            "n_per_context": 60,
            "random_seed": 123,
            "source_noise_scale": 0.8,
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
        "outputs": {"save_intermediate": True},
    }


def test_run_synthetic_experiment_smoke(tmp_path) -> None:
    config = make_small_config(tmp_path)

    run_dir = run_synthetic_experiment(config, overwrite=True)

    assert run_dir.is_dir()
    assert (run_dir / "results" / "method_comparison.csv").is_file()
    assert (run_dir / "results" / "ct_crc_windows.csv").is_file()
    assert (run_dir / "results" / "cw_crc_windows.csv").is_file()
    assert (run_dir / "results" / "run_manifest.json").is_file()

    comparison = pd.read_csv(run_dir / "results" / "method_comparison.csv")
    assert set(comparison["method"]) == {
        "global_crc",
        "oracle_context_crc",
        "metadata_stratified_crc",
        "ct_crc",
        "cw_crc",
    }


def test_encoder_fit_uses_calibration_only_and_excludes_diag(tmp_path) -> None:
    config = make_small_config(tmp_path)

    run_dir = run_synthetic_experiment(config, overwrite=True)

    with (run_dir / "results" / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["calibration_contexts"] == ["source", "covariate_s1", "label_s1", "mixture_s1"]
    assert manifest["num_signature_features"] > 0
    assert not any(column.startswith("diag_") for column in manifest["signature_feature_columns"])


def test_script_runs(tmp_path) -> None:
    config = make_small_config(tmp_path)
    config_path = tmp_path / "experiment.yaml"
    import yaml

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_experiment.py",
            "--config",
            str(config_path),
            "--overwrite",
            "--run-name",
            "synthetic_script_test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs" / "synthetic_script_test" / "results" / "method_comparison.csv").is_file()
