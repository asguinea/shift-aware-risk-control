from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.evaluation.online_trace import run_cifar_online_trace


def test_run_cifar_online_trace_with_fake_cache(tmp_path: Path) -> None:
    cache_dir = _write_fake_cifar_trace_cache(tmp_path / "cache", n_per_severity=20)
    config = _trace_config(tmp_path, cache_dir)

    run_dir = run_cifar_online_trace(config, overwrite=True)

    trace_path = run_dir / "results" / "online_trace.csv"
    manifest_path = run_dir / "results" / "run_manifest.json"
    assert trace_path.is_file()
    assert manifest_path.is_file()
    assert (run_dir / "intermediate" / "windows.parquet").is_file()
    assert (run_dir / "intermediate" / "signatures.parquet").is_file()
    assert (run_dir / "intermediate" / "embeddings.parquet").is_file()

    trace = pd.read_csv(trace_path)
    required_columns = {
        "trace_index",
        "window_id",
        "context_id",
        "corruption_type",
        "severity",
        "novelty_distance",
        "ct_realized_risk",
        "ct_bound",
        "ct_coverage",
        "ct_violation",
        "ct_bound_underestimated",
        "cw_realized_risk",
        "cw_bound",
        "cw_coverage",
        "cw_violation",
        "cw_bound_underestimated",
    }
    assert required_columns.issubset(trace.columns)
    assert trace["trace_index"].is_monotonic_increasing
    assert trace["severity"].tolist() == sorted(trace["severity"].tolist())
    assert set(trace["severity"]) == {1, 3, 5}

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["dataset"] == "cifar10c"
    assert manifest["corruption_type"] == "gaussian_noise"
    assert manifest["evaluation_severities"] == [1, 3, 5]


def _write_fake_cifar_trace_cache(cache_dir: Path, *, n_per_severity: int) -> Path:
    rows = []
    probabilities = []
    logits = []
    features = []
    for severity in (1, 3, 5):
        for index in range(n_per_severity):
            label = (index + severity) % 3
            pred = label if index % (severity + 1) else (label + 1) % 3
            confidence = max(0.4, 0.82 - 0.07 * severity)
            probs = np.full(3, (1.0 - confidence) / 2.0, dtype=float)
            probs[pred] = confidence
            rows.append(
                {
                    "example_id": f"gaussian_noise/s{severity}/{index}",
                    "dataset": "cifar10c",
                    "split": "eval",
                    "context_id": f"gaussian_noise_s{severity}",
                    "context_name": f"Gaussian noise severity {severity}",
                    "label": label,
                    "pred": pred,
                    "correct": pred == label,
                    "confidence": float(probs.max()),
                    "entropy": float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)))),
                    "margin": float(np.sort(probs)[-1] - np.sort(probs)[-2]),
                    "energy": float(-np.log(np.sum(np.exp(np.log(np.clip(probs, 1e-12, 1.0)))))),
                    "severity": severity,
                    "corruption_type": "gaussian_noise",
                    "time_index": index,
                }
            )
            probabilities.append(probs)
            logits.append(np.log(np.clip(probs, 1e-12, 1.0)))
            features.append([float(severity), float(index) / n_per_severity, float(label), float(pred)])
    examples = pd.DataFrame(rows)
    arrays = {
        "logits": np.asarray(logits, dtype=float),
        "probabilities": np.asarray(probabilities, dtype=float),
        "feature_embedding": np.asarray(features, dtype=float),
    }
    metadata = {
        "dataset": "cifar10c",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 3,
        "context_fields": ["context_id", "context_name", "corruption_type", "severity"],
    }
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=True)


def _trace_config(tmp_path: Path, cache_dir: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "cifar_online_trace", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "cifar10c_trace_test",
            "cache_dir": str(cache_dir),
        },
        "trace": {
            "dataset": "cifar10c",
            "corruption_type": "gaussian_noise",
            "severities": [1, 3, 5],
            "calibration_severities": [1],
            "evaluation_severities": [1, 3, 5],
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
        "ct_crc": {
            "metric": "euclidean",
            "l_hat": "auto",
            "l_hat_quantile": 0.9,
            "epsilon_k": 0.0,
            "fallback_to_global": True,
            "bound_margin": 0.03,
            "bound_multiplier": 1.25,
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
            "bound_multiplier": 1.0,
        },
        "outputs": {"save_intermediate": True},
    }
