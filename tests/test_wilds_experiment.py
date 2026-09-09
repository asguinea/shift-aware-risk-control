from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.cache.schema import REQUIRED_EXAMPLE_COLUMNS
from contextual_crc.datasets.wilds_dataset import (
    compute_camelyon17_prediction_frame_and_arrays,
    compute_fmow_prediction_frame_and_arrays,
)
from contextual_crc.evaluation.wilds_experiment import run_wilds_experiment


def create_fake_civilcomments_csv(root: Path, *, n_per_group: int = 30) -> None:
    base = root / "civilcomments"
    base.mkdir(parents=True, exist_ok=True)
    train_rows = []
    for index in range(90):
        label = index % 2
        train_rows.append(
            {
                "text": (
                    f"kind useful civil training note {index}"
                    if label == 0
                    else f"awful toxic insult training note {index}"
                ),
                "label": label,
            }
        )
    eval_rows = []
    for group_index, group in enumerate(("group_a", "group_b", "group_c")):
        domain = f"domain_{group_index % 2}"
        for index in range(n_per_group):
            label = (index + group_index) % 2
            eval_rows.append(
                {
                    "text": (
                        f"calm constructive eval comment {group} {index}"
                        if label == 0
                        else f"toxic hostile eval comment {group} {index}"
                    ),
                    "label": label,
                    "group_id": group,
                    "domain_id": domain,
                }
            )
    pd.DataFrame(train_rows).to_csv(base / "train.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(base / "eval.csv", index=False)


def make_cache_config(tmp_path: Path, data_root: Path) -> Path:
    cache_config = {
        "project": {"name": "shift-aware-risk-control", "mode": "wilds_civilcomments_cache", "seed": 123},
        "paths": {
            "data_root": str(data_root),
            "cache_dir": str(tmp_path / "cache" / "wilds_civilcomments"),
        },
        "wilds": {
            "dataset": "civilcomments",
            "source": "csv",
            "download": False,
            "model_backend": "sklearn_tfidf",
            "train_subset_size": None,
            "eval_subset_size": None,
            "random_seed": 123,
            "text_column": "text",
            "label_column": "label",
            "context_column": "group_id",
            "max_features": 500,
            "ngram_range": [1, 2],
        },
    }
    config_path = tmp_path / "wilds_cache.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cache_config, handle, sort_keys=False)
    return config_path


def make_experiment_config(tmp_path: Path, cache_config_path: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "wilds_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "wilds_test",
            "cache_dir": str(tmp_path / "cache" / "wilds_civilcomments"),
        },
        "wilds_cache": {
            "build_if_missing": True,
            "overwrite_cache": False,
            "config_path": str(cache_config_path),
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
            "calibration_contexts": "auto",
            "num_auto_calibration_contexts": 2,
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


def make_ready_config(tmp_path: Path) -> dict:
    data_root = tmp_path / "data" / "wilds"
    create_fake_civilcomments_csv(data_root)
    cache_config_path = make_cache_config(tmp_path, data_root)
    return make_experiment_config(tmp_path, cache_config_path)


def test_run_wilds_experiment_smoke(tmp_path) -> None:
    config = make_ready_config(tmp_path)

    run_dir = run_wilds_experiment(config, overwrite=True)

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
        "metadata_stratified_domain",
        "metadata_stratified_group",
        "ct_crc",
        "cw_crc",
    }.issubset(set(comparison["method"]))


def test_auto_calibration_contexts_recorded(tmp_path) -> None:
    config = make_ready_config(tmp_path)

    run_dir = run_wilds_experiment(config, overwrite=True)

    with (run_dir / "results" / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["available_contexts"] == ["group_a", "group_b", "group_c"]
    assert manifest["calibration_contexts"] == ["group_a", "group_b"]
    assert manifest["num_signature_features"] > 0
    assert not any(column.startswith("diag_") for column in manifest["signature_feature_columns"])


def test_missing_calibration_context_raises(tmp_path) -> None:
    config = make_ready_config(tmp_path)
    config["splits"]["calibration_contexts"] = ["does_not_exist"]

    with pytest.raises(ValueError, match="Available contexts"):
        run_wilds_experiment(config, overwrite=True)


def test_script_runs_with_fake_csv(tmp_path) -> None:
    config = make_ready_config(tmp_path)
    config_path = tmp_path / "wilds_experiment.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_wilds_experiment.py",
            "--config",
            str(config_path),
            "--overwrite",
            "--run-name",
            "wilds_script_test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs" / "wilds_script_test" / "results" / "method_comparison.csv").is_file()


def test_run_wilds_experiment_with_fake_camelyon_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "wilds_camelyon17"
    rows = []
    labels = []
    probabilities = []
    features = []
    for domain in range(3):
        for index in range(20):
            label = (index + domain) % 2
            confidence = 0.75 if label == 1 else 0.7
            rows.append(
                {
                    "original_index": domain * 20 + index,
                    "split": "eval",
                    "domain_id": domain,
                    "group_id": domain,
                    "source": "wilds-camelyon17",
                    "time_index": index,
                }
            )
            labels.append(label)
            probabilities.append([1.0 - confidence, confidence] if label == 1 else [confidence, 1.0 - confidence])
            features.append([float(domain), float(index) / 20.0, float(label)])
    labels_array = pd.Series(labels).to_numpy(dtype=int)
    probabilities_array = pd.DataFrame(probabilities).to_numpy(dtype=float)
    logits = np.log(np.clip(probabilities_array, 1e-12, 1.0))
    examples, arrays = compute_camelyon17_prediction_frame_and_arrays(
        eval_df=pd.DataFrame(rows),
        labels=labels_array,
        probabilities=probabilities_array,
        logits=logits,
        feature_embedding=pd.DataFrame(features).to_numpy(dtype=float),
    )
    metadata = {
        "dataset": "wilds_camelyon17",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "context_fields": ["context_id", "domain_id", "group_id"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "schema_version": "1.0",
    }
    save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=True)

    config = make_experiment_config(tmp_path, tmp_path / "unused.yaml")
    config["paths"]["run_name"] = "wilds_camelyon_test"
    config["paths"]["cache_dir"] = str(cache_dir)
    config["wilds_cache"]["build_if_missing"] = False
    config["wilds_cache"]["config_path"] = "configs/wilds_camelyon17.yaml"

    run_dir = run_wilds_experiment(config, overwrite=True)

    assert (run_dir / "results" / "method_comparison.csv").is_file()
    comparison = pd.read_csv(run_dir / "results" / "method_comparison.csv")
    assert {"ct_crc", "cw_crc", "metadata_stratified_domain"}.issubset(set(comparison["method"]))


def test_run_wilds_experiment_with_fake_fmow_cache(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "wilds_fmow"
    rows = []
    labels = []
    probabilities = []
    features = []
    regions = ("region_americas", "region_europe", "region_asia")
    for region_index, region in enumerate(regions):
        for index in range(20):
            label = (index + region_index) % 5
            confidence = 0.62 + 0.02 * (label % 3)
            probs = np.full(5, (1.0 - confidence) / 4.0, dtype=float)
            probs[label] = confidence
            rows.append(
                {
                    "original_index": region_index * 20 + index,
                    "split": "eval",
                    "domain_id": region_index,
                    "group_id": region_index,
                    "region": region,
                    "year": 2014 + index % 5,
                    "source": "wilds-fmow",
                    "time_index": index,
                }
            )
            labels.append(label)
            probabilities.append(probs)
            features.append([float(region_index), float(index) / 20.0, float(label), confidence])
    labels_array = pd.Series(labels).to_numpy(dtype=int)
    probabilities_array = pd.DataFrame(probabilities).to_numpy(dtype=float)
    logits = np.log(np.clip(probabilities_array, 1e-12, 1.0))
    examples, arrays = compute_fmow_prediction_frame_and_arrays(
        eval_df=pd.DataFrame(rows),
        labels=labels_array,
        probabilities=probabilities_array,
        logits=logits,
        feature_embedding=pd.DataFrame(features).to_numpy(dtype=float),
        context_column="region",
    )
    metadata = {
        "dataset": "wilds_fmow",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 5,
        "context_fields": ["context_id", "domain_id", "group_id", "region", "year"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "schema_version": "1.0",
    }
    save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=True)

    config = make_experiment_config(tmp_path, tmp_path / "unused.yaml")
    config["paths"]["run_name"] = "wilds_fmow_test"
    config["paths"]["cache_dir"] = str(cache_dir)
    config["wilds_cache"]["build_if_missing"] = False
    config["wilds_cache"]["config_path"] = "configs/wilds_fmow.yaml"
    config["splits"]["num_auto_calibration_contexts"] = 2

    run_dir = run_wilds_experiment(config, overwrite=True)

    assert (run_dir / "results" / "method_comparison.csv").is_file()
    comparison = pd.read_csv(run_dir / "results" / "method_comparison.csv")
    assert {"ct_crc", "cw_crc", "metadata_stratified_domain"}.issubset(set(comparison["method"]))
    with (run_dir / "results" / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["dataset"] == "wilds_fmow"
    assert len(manifest["available_contexts"]) == 3
