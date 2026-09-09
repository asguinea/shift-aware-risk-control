from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from contextual_crc.datasets import cifar_c
from contextual_crc.datasets.cifar_c import CIFARCConfig, build_cifar_c_prediction_cache
from contextual_crc.evaluation.cifar_c_experiment import run_cifar_c_experiment
from contextual_crc.evaluation.synthetic_experiment import (
    parse_ctcrc_config,
    parse_cwcrc_config,
    parse_risk_control_config,
)


def create_fake_cifar_c(root: Path, n_per_severity: int = 30) -> None:
    rng = np.random.default_rng(123)
    base = root / "cifar10c"
    base.mkdir(parents=True, exist_ok=True)
    labels = np.arange(5 * n_per_severity) % 10
    np.save(base / "labels.npy", labels.astype(np.int64))
    for offset, corruption in enumerate(("gaussian_noise", "brightness", "contrast")):
        images = rng.integers(0, 256, size=(5 * n_per_severity, 32, 32, 3), dtype=np.uint8)
        images = np.clip(images.astype(np.int16) + offset * 3, 0, 255).astype(np.uint8)
        np.save(base / f"{corruption}.npy", images)


def build_fake_cache(tmp_path: Path, monkeypatch) -> Path:
    data_root = tmp_path / "data"
    create_fake_cifar_c(data_root)
    rng = np.random.default_rng(321)
    train_images = rng.integers(0, 256, size=(80, 32, 32, 3), dtype=np.uint8)
    train_labels = np.arange(80) % 10

    def fake_load_clean_cifar_train(dataset, data_root, *, train_subset_size=None, random_seed=123):
        return train_images, train_labels

    monkeypatch.setattr(cifar_c, "load_clean_cifar_train", fake_load_clean_cifar_train)
    config = CIFARCConfig(
        dataset="cifar10c",
        data_root=data_root,
        corruptions=("gaussian_noise", "brightness", "contrast"),
        severities=(1, 3, 5),
        max_examples_per_context=30,
        train_subset_size=80,
        random_seed=123,
        download=False,
    )
    cache_dir = tmp_path / "cache" / "cifar10c_fake"
    return build_cifar_c_prediction_cache(cache_dir, config, overwrite=True)


def make_experiment_config(tmp_path: Path, cache_dir: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "cifar_c_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "cifar_test",
            "cache_dir": str(cache_dir),
        },
        "cifar_cache": {
            "build_if_missing": False,
            "overwrite_cache": False,
            "config_path": "configs/cifar10c.yaml",
        },
        "windowing": {
            "window_size": 10,
            "stride": 10,
            "group_by_context": True,
            "sort_by": "time_index",
            "drop_incomplete": True,
        },
        "signatures": {"include_label_diagnostics": True},
        "encoder": {"method": "pca", "n_components": 2, "whiten": False, "random_seed": 123},
        "risk_control": {
            "score_column": "confidence",
            "accept_if": "greater_equal",
            "risk_budget": 0.35,
            "delta": 0.2,
            "bound": "hoeffding",
            "min_coverage": 0.0,
            "loss": {"loss_name": "accepted_error", "deferral_cost": 0.0},
        },
        "splits": {
            "calibration_contexts": ["gaussian_noise_s1", "brightness_s1", "contrast_s1"],
            "evaluation_contexts": "all",
        },
        "ct_crc": {
            "metric": "euclidean",
            "l_hat": "auto",
            "l_hat_quantile": 0.9,
            "epsilon_k": 0.0,
            "bound_margin": 0.03,
            "bound_multiplier": 1.25,
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
            "bound_margin": 0.03,
            "bound_multiplier": 1.25,
        },
        "outputs": {"save_intermediate": True},
    }


def test_run_cifar_c_experiment_smoke(tmp_path, monkeypatch) -> None:
    cache_dir = build_fake_cache(tmp_path, monkeypatch)
    config = make_experiment_config(tmp_path, cache_dir)

    run_dir = run_cifar_c_experiment(config, overwrite=True)

    assert run_dir.is_dir()
    for relative in (
        "results/method_comparison.csv",
        "results/ct_crc_windows.csv",
        "results/cw_crc_windows.csv",
        "results/run_manifest.json",
        "intermediate/windows.parquet",
        "intermediate/signatures.parquet",
        "intermediate/embeddings.parquet",
    ):
        assert (run_dir / relative).is_file()

    comparison = pd.read_csv(run_dir / "results" / "method_comparison.csv")
    assert {
        "global_crc",
        "oracle_context_crc",
        "metadata_stratified_context",
        "metadata_stratified_corruption",
        "metadata_stratified_severity",
        "ct_crc",
        "cw_crc",
    }.issubset(set(comparison["method"]))

    with (run_dir / "results" / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["dataset"] == "cifar10c"
    assert manifest["num_signature_features"] > 0
    assert not any(column.startswith("diag_") for column in manifest["signature_feature_columns"])


def test_missing_calibration_context_raises(tmp_path, monkeypatch) -> None:
    cache_dir = build_fake_cache(tmp_path, monkeypatch)
    config = make_experiment_config(tmp_path, cache_dir)
    config["splits"]["calibration_contexts"] = ["does_not_exist"]

    with pytest.raises(ValueError, match="Available contexts"):
        run_cifar_c_experiment(config, overwrite=True)


def test_config_parser_accepts_ct_cw_bound_safety_fields() -> None:
    risk_config = parse_risk_control_config(
        {
            "risk_budget": 0.1,
            "delta": 0.05,
            "loss": {"loss_name": "accepted_error", "deferral_cost": 0.0},
        }
    )

    ct_config = parse_ctcrc_config(
        {
            "l_hat": "auto",
            "epsilon_k": 0.0,
            "bound_margin": 0.03,
            "bound_multiplier": 1.25,
            "clip_bounds": True,
        },
        risk_config,
    )
    cw_config = parse_cwcrc_config(
        {
            "target_effective_sample_size": 12,
            "finite_sample_correction": "empirical_bernstein",
            "bound_margin": 0.04,
            "bound_multiplier": 1.5,
        },
        risk_config,
    )

    assert ct_config.bound_margin == pytest.approx(0.03)
    assert ct_config.bound_multiplier == pytest.approx(1.25)
    assert ct_config.clip_bounds is True
    assert cw_config.bound_margin == pytest.approx(0.04)
    assert cw_config.bound_multiplier == pytest.approx(1.5)


def test_script_runs_with_existing_fake_cache(tmp_path, monkeypatch) -> None:
    cache_dir = build_fake_cache(tmp_path, monkeypatch)
    config = make_experiment_config(tmp_path, cache_dir)
    config_path = tmp_path / "cifar_experiment.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cifar_c_experiment.py",
            "--config",
            str(config_path),
            "--overwrite",
            "--run-name",
            "cifar_script_test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs" / "cifar_script_test" / "results" / "method_comparison.csv").is_file()
