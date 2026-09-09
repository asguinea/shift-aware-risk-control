from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.datasets.officehome import compute_officehome_prediction_frame_and_arrays
from contextual_crc.evaluation.officehome_experiment import run_officehome_experiment


def test_run_officehome_experiment_with_fake_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "officehome"
    _write_fake_officehome_cache(cache_dir, n_per_domain=16)
    config = _make_officehome_experiment_config(tmp_path, cache_dir)

    run_dir = run_officehome_experiment(config, overwrite=True)

    comparison_path = run_dir / "results" / "method_comparison.csv"
    assert comparison_path.is_file()
    comparison = pd.read_csv(comparison_path)
    assert {
        "global_crc",
        "metadata_stratified_context",
        "ct_crc",
        "cw_crc",
    }.issubset(set(comparison["method"]))

    with (run_dir / "results" / "run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["dataset"] == "officehome"
    assert manifest["calibration_contexts"] == ["real_world", "product"]


def _write_fake_officehome_cache(cache_dir: Path, *, n_per_domain: int) -> None:
    rows = []
    probabilities = []
    features = []
    domains = ("art", "clipart", "product", "real_world")
    class_names = ("backpack", "bike", "mug")
    for domain_index, domain in enumerate(domains):
        for index in range(n_per_domain):
            label = (index + domain_index) % len(class_names)
            confidence = 0.62 + 0.04 * (domain_index % 2)
            probs = np.full(len(class_names), (1.0 - confidence) / (len(class_names) - 1), dtype=float)
            probs[label] = confidence
            rows.append(
                {
                    "image_path": f"{domain}/{class_names[label]}/{index}.jpg",
                    "domain": domain,
                    "class_name": class_names[label],
                    "label": label,
                }
            )
            probabilities.append(probs)
            features.append([float(domain_index), float(index) / n_per_domain, float(label), confidence])
    probabilities_array = np.asarray(probabilities, dtype=float)
    logits = np.log(np.clip(probabilities_array, 1e-12, 1.0))
    examples, arrays = compute_officehome_prediction_frame_and_arrays(
        eval_index=pd.DataFrame(rows),
        probabilities=probabilities_array,
        logits=logits,
        feature_embedding=np.asarray(features, dtype=float),
    )
    metadata = {
        "dataset": "officehome",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": len(class_names),
        "context_fields": ["context_id", "domain_id", "group_id", "domain"],
    }
    save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=True)


def _make_officehome_experiment_config(tmp_path: Path, cache_dir: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "officehome_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "officehome_test",
            "cache_dir": str(cache_dir),
        },
        "officehome_cache": {
            "build_if_missing": False,
            "overwrite_cache": False,
            "config_path": "configs/officehome_resnet18.yaml",
        },
        "windowing": {
            "window_size": 4,
            "stride": 4,
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
            "calibration_contexts": ["real_world", "product"],
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
