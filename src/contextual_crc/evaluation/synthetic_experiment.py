from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from contextual_crc.cache.io import cache_exists, load_prediction_cache
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.config.loader import load_config
from contextual_crc.datasets.synthetic import SyntheticConfig, build_synthetic_prediction_cache
from contextual_crc.embeddings.context_encoder import (
    ContextEncoder,
    EncoderConfig,
    build_context_embedding_frame,
)
from contextual_crc.evaluation.diagnostics import (
    bound_calibration_diagnostics,
    context_smoothness_diagnostics,
    context_smoothness_pairwise_frame,
    knn_risk_prediction_diagnostics,
)
from contextual_crc.evaluation.metrics import summarize_window_results
from contextual_crc.risk_control.baselines import (
    run_global_crc_baseline,
    run_metadata_stratified_crc_baseline,
    run_oracle_context_crc_baseline,
)
from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    build_validated_contexts,
    estimate_lipschitz_from_contexts,
    evaluate_ctcrc_windows,
    validated_contexts_to_frame,
)
from contextual_crc.risk_control.cw_crc import CWCRCConfig, evaluate_cwcrc_windows
from contextual_crc.risk_control.losses import LossConfig
from contextual_crc.signatures.behavioral_signatures import (
    SignatureConfig,
    compute_signatures_for_windows,
    get_signature_feature_columns,
)
from contextual_crc.signatures.io import save_signatures, save_windows
from contextual_crc.signatures.windowing import WindowConfig, make_windows
from contextual_crc.utils.paths import ensure_dir, make_run_dir
from contextual_crc.utils.seed import set_seed


def run_synthetic_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    run_name_override: str | None = None,
) -> Path:
    """Run the complete synthetic evaluation pipeline and write structured outputs."""
    seed = int(config.get("project", {}).get("seed", 123))
    set_seed(seed)

    paths = config.get("paths", {})
    output_root = Path(paths.get("output_root", "outputs/runs"))
    run_name = run_name_override or paths.get("run_name", "synthetic_main")
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir = make_run_dir(output_root, run_name)
    results_dir = ensure_dir(run_dir / "results")
    intermediate_dir = ensure_dir(run_dir / "intermediate")

    resolved_config = dict(config)
    resolved_config.setdefault("paths", {})["run_name"] = run_name
    _write_yaml(resolved_config, run_dir / "configs" / "resolved_config.yaml")

    cache_dir = _ensure_synthetic_cache(resolved_config)
    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)

    window_config = parse_window_config(resolved_config.get("windowing", {}))
    signature_config = parse_signature_config(resolved_config.get("signatures", {}))
    encoder_config = parse_encoder_config(resolved_config.get("encoder", {}))
    risk_config = parse_risk_control_config(resolved_config.get("risk_control", {}))

    windows = make_windows(examples, window_config)
    signatures = compute_signatures_for_windows(examples, arrays, windows, signature_config)
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=False)
    if not feature_columns:
        raise ValueError("No non-diagnostic signature feature columns available for encoder fitting")

    calibration_contexts = list(resolved_config.get("splits", {}).get("calibration_contexts", []))
    if not calibration_contexts:
        raise ValueError("splits.calibration_contexts must be non-empty")
    evaluation_contexts = _resolve_evaluation_contexts(
        resolved_config.get("splits", {}).get("evaluation_contexts", "all"),
        examples,
    )

    calibration_examples, calibration_windows = _subset_examples_and_remap_windows(
        examples,
        windows,
        calibration_contexts,
    )
    evaluation_examples = examples[examples["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)

    calibration_window_ids = set(calibration_windows["window_id"])
    evaluation_window_ids = set(evaluation_windows["window_id"])
    calibration_signatures = signatures[signatures["window_id"].isin(calibration_window_ids)].reset_index(drop=True)

    encoder = ContextEncoder(encoder_config)
    calibration_embeddings = encoder.fit_transform(calibration_signatures, feature_columns)
    all_embeddings = encoder.transform(signatures)
    embedding_frame = build_context_embedding_frame(signatures, all_embeddings)
    calibration_embedding_frame = build_context_embedding_frame(
        calibration_signatures,
        calibration_embeddings,
    )
    evaluation_embedding_frame = embedding_frame[
        embedding_frame["window_id"].isin(evaluation_window_ids)
    ].reset_index(drop=True)

    if resolved_config.get("outputs", {}).get("save_intermediate", True):
        save_windows(windows, intermediate_dir / "windows.parquet")
        save_signatures(signatures, intermediate_dir / "signatures.parquet")
        embedding_frame.to_parquet(intermediate_dir / "embeddings.parquet", engine="pyarrow", index=False)

    global_result = run_global_crc_baseline(calibration_examples, evaluation_examples, risk_config)
    oracle_result = run_oracle_context_crc_baseline(calibration_examples, evaluation_examples, risk_config)
    stratified_result = run_metadata_stratified_crc_baseline(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="context_id",
    )

    global_result["by_context"].to_csv(results_dir / "global_by_context.csv", index=False)
    oracle_result["by_context"].to_csv(results_dir / "oracle_by_context.csv", index=False)
    stratified_result["by_stratum"].to_csv(results_dir / "stratified_by_context.csv", index=False)

    ct_config = parse_ctcrc_config(resolved_config.get("ct_crc", {}), risk_config)
    ct_result = build_validated_contexts(calibration_examples, calibration_embedding_frame, ct_config)
    if resolved_config.get("ct_crc", {}).get("l_hat") == "auto":
        l_hat = _estimate_ct_l_hat(ct_result, resolved_config.get("ct_crc", {}))
        ct_config = CTCRCConfig(
            risk_control=risk_config,
            metric=ct_config.metric,
            l_hat=l_hat,
            epsilon_k=ct_config.epsilon_k,
            bound_margin=ct_config.bound_margin,
            bound_multiplier=ct_config.bound_multiplier,
            clip_bounds=ct_config.clip_bounds,
            fallback_to_global=ct_config.fallback_to_global,
        )
        ct_result = build_validated_contexts(calibration_examples, calibration_embedding_frame, ct_config)

    ct_windows = evaluate_ctcrc_windows(examples, evaluation_windows, evaluation_embedding_frame, ct_result)
    cw_config = parse_cwcrc_config(resolved_config.get("cw_crc", {}), risk_config)
    cw_windows = evaluate_cwcrc_windows(
        examples,
        calibration_examples,
        calibration_windows,
        calibration_embedding_frame,
        evaluation_windows,
        evaluation_embedding_frame,
        cw_config,
    )

    ct_windows.to_csv(results_dir / "ct_crc_windows.csv", index=False)
    cw_windows.to_csv(results_dir / "cw_crc_windows.csv", index=False)

    ct_summary = summarize_window_results(
        ct_windows,
        risk_budget=risk_config.risk_budget,
        upper_bound_col="transferred_upper_bound",
        method_name="ct_crc",
    )
    cw_summary = summarize_window_results(
        cw_windows,
        risk_budget=risk_config.risk_budget,
        upper_bound_col="weighted_risk_upper_bound",
        method_name="cw_crc",
    )
    _write_json(ct_summary, results_dir / "ct_crc_summary.json")
    _write_json(cw_summary, results_dir / "cw_crc_summary.json")

    bound_diagnostics = {
        "ct_crc": bound_calibration_diagnostics(
            ct_windows,
            upper_bound_col="transferred_upper_bound",
        ),
        "cw_crc": bound_calibration_diagnostics(
            cw_windows,
            upper_bound_col="weighted_risk_upper_bound",
        ),
    }
    _write_json(bound_diagnostics, results_dir / "bound_diagnostics.json")

    smoothness_frame = signatures.merge(
        embedding_frame[["window_id"] + [column for column in embedding_frame.columns if column.startswith("z")]],
        on="window_id",
        how="left",
    )
    if "diag_error_rate" in smoothness_frame.columns:
        value_col = "diag_error_rate"
    elif "diag_accuracy" in smoothness_frame.columns:
        smoothness_frame["diag_error_rate"] = 1.0 - smoothness_frame["diag_accuracy"]
        value_col = "diag_error_rate"
    else:
        ct_risk = ct_windows[["window_id", "realized_risk"]]
        smoothness_frame = smoothness_frame.merge(ct_risk, on="window_id", how="left")
        value_col = "realized_risk"
    context_smoothness_summary = context_smoothness_diagnostics(smoothness_frame, value_col=value_col)
    context_smoothness_pairs = context_smoothness_pairwise_frame(
        smoothness_frame,
        value_col=value_col,
        dataset=str(metadata.get("dataset") or run_name),
    )
    context_smoothness_pairs.to_csv(results_dir / "context_smoothness.csv", index=False)
    _write_json(context_smoothness_summary, results_dir / "context_smoothness_summary.json")
    knn_diagnostics = knn_risk_prediction_diagnostics(smoothness_frame, risk_col=value_col, k=3)
    _write_json(knn_diagnostics, results_dir / "knn_diagnostics.json")

    method_comparison = _build_method_comparison(
        global_result,
        oracle_result,
        stratified_result,
        ct_summary,
        cw_summary,
    )
    method_comparison.to_csv(results_dir / "method_comparison.csv", index=False)

    manifest = {
        "run_name": run_name,
        "seed": seed,
        "cache_dir": str(cache_dir),
        "num_examples": int(len(examples)),
        "num_windows": int(len(windows)),
        "num_signature_features": int(len(feature_columns)),
        "signature_feature_columns": feature_columns,
        "calibration_contexts": calibration_contexts,
        "evaluation_contexts": evaluation_contexts,
        "methods_run": [
            "global_crc",
            "oracle_context_crc",
            "metadata_stratified_crc",
            "ct_crc",
            "cw_crc",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_dataset": metadata.get("dataset"),
    }
    _write_json(manifest, results_dir / "run_manifest.json")
    return run_dir


def parse_window_config(raw: dict[str, Any]) -> WindowConfig:
    return WindowConfig(
        window_size=int(raw.get("window_size", 100)),
        stride=int(raw.get("stride", 50)),
        group_by_context=bool(raw.get("group_by_context", True)),
        sort_by=raw.get("sort_by", "time_index"),
        drop_incomplete=bool(raw.get("drop_incomplete", True)),
    )


def parse_signature_config(raw: dict[str, Any]) -> SignatureConfig:
    return SignatureConfig(
        include_label_diagnostics=bool(raw.get("include_label_diagnostics", False)),
    )


def parse_encoder_config(raw: dict[str, Any]) -> EncoderConfig:
    return EncoderConfig(
        method=raw.get("method", "pca"),
        n_components=raw.get("n_components", 8),
        whiten=bool(raw.get("whiten", False)),
        random_seed=int(raw.get("random_seed", 123)),
    )


def parse_risk_control_config(raw: dict[str, Any]) -> RiskControlConfig:
    loss_raw = raw.get("loss", {})
    return RiskControlConfig(
        score_column=raw.get("score_column", "confidence"),
        accept_if=raw.get("accept_if", "greater_equal"),
        loss=LossConfig(
            loss_name=loss_raw.get("loss_name", "accepted_error"),
            deferral_cost=float(loss_raw.get("deferral_cost", 0.0)),
        ),
        risk_budget=float(raw.get("risk_budget", 0.1)),
        delta=float(raw.get("delta", 0.05)),
        bound=raw.get("bound", "hoeffding"),
        min_coverage=float(raw.get("min_coverage", 0.0)),
    )


def parse_ctcrc_config(raw: dict[str, Any], risk_config: RiskControlConfig) -> CTCRCConfig:
    l_hat_raw = raw.get("l_hat", 0.0)
    l_hat = 0.0 if l_hat_raw == "auto" else float(l_hat_raw)
    return CTCRCConfig(
        risk_control=risk_config,
        metric=raw.get("metric", "euclidean"),
        l_hat=l_hat,
        epsilon_k=float(raw.get("epsilon_k", 0.0)),
        bound_margin=float(raw.get("bound_margin", 0.0)),
        bound_multiplier=float(raw.get("bound_multiplier", 1.0)),
        clip_bounds=bool(raw.get("clip_bounds", True)),
        fallback_to_global=bool(raw.get("fallback_to_global", True)),
    )


def parse_cwcrc_config(raw: dict[str, Any], risk_config: RiskControlConfig) -> CWCRCConfig:
    return CWCRCConfig(
        risk_control=risk_config,
        metric=raw.get("metric", "euclidean"),
        kernel=raw.get("kernel", "gaussian"),
        bandwidth=raw.get("bandwidth"),
        target_effective_sample_size=raw.get("target_effective_sample_size", 200.0),
        min_bandwidth=float(raw.get("min_bandwidth", 1e-6)),
        max_bandwidth=raw.get("max_bandwidth"),
        bandwidth_grid_size=int(raw.get("bandwidth_grid_size", 30)),
        example_weight_aggregation=raw.get("example_weight_aggregation", "mean"),
        finite_sample_correction=raw.get("finite_sample_correction", "hoeffding"),
        normalize_weights=bool(raw.get("normalize_weights", True)),
        bound_margin=float(raw.get("bound_margin", 0.0)),
        bound_multiplier=float(raw.get("bound_multiplier", 1.0)),
    )


def _ensure_synthetic_cache(config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "outputs/caches/synthetic_default"))
    cache_config = config.get("synthetic_cache", {})
    if cache_exists(cache_dir) and not bool(cache_config.get("overwrite_cache", False)):
        return cache_dir
    if not cache_exists(cache_dir) and not bool(cache_config.get("build_if_missing", True)):
        raise FileNotFoundError(f"Synthetic cache not found: {cache_dir}")

    if "synthetic" in config:
        synthetic_config = SyntheticConfig(**config.get("synthetic", {}))
    else:
        synthetic_config_path = cache_config.get("config_path", "configs/synthetic.yaml")
        synthetic_raw = load_config(synthetic_config_path)
        synthetic_config = SyntheticConfig(**synthetic_raw.get("synthetic", {}))
    build_synthetic_prediction_cache(
        cache_dir,
        synthetic_config,
        overwrite=bool(cache_config.get("overwrite_cache", False)) or not cache_exists(cache_dir),
    )
    return cache_dir


def _resolve_evaluation_contexts(raw: Any, examples: pd.DataFrame) -> list[str]:
    if raw == "all" or raw == ["all"]:
        return [str(value) for value in examples["context_id"].drop_duplicates().tolist()]
    return [str(value) for value in raw]


def _subset_examples_and_remap_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    contexts: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = examples["context_id"].isin(contexts)
    original_positions = np.flatnonzero(mask.to_numpy())
    original_to_local = {int(original): local for local, original in enumerate(original_positions)}
    subset_examples = examples.iloc[original_positions].reset_index(drop=True)
    subset_windows = windows[windows["context_id"].isin(contexts)].copy().reset_index(drop=True)

    remapped_indices = []
    for indices in subset_windows["example_indices"]:
        local_indices = []
        for index in indices:
            index = int(index)
            if index not in original_to_local:
                raise ValueError(f"Window contains example outside requested contexts: {index}")
            local_indices.append(original_to_local[index])
        remapped_indices.append(local_indices)
    subset_windows["example_indices"] = remapped_indices
    return subset_examples, subset_windows


def _estimate_ct_l_hat(ct_result: Any, raw_config: dict[str, Any]) -> float:
    context_frame = validated_contexts_to_frame(ct_result.contexts)
    if len(context_frame) < 2:
        return 0.0
    try:
        return estimate_lipschitz_from_contexts(
            context_frame,
            risk_col="risk_estimate",
            metric=raw_config.get("metric", "euclidean"),
            quantile=float(raw_config.get("l_hat_quantile", 0.9)),
        )
    except ValueError:
        return 0.0


def _build_method_comparison(
    global_result: dict[str, Any],
    oracle_result: dict[str, Any],
    stratified_result: dict[str, Any],
    ct_summary: dict[str, Any],
    cw_summary: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        _baseline_row("global_crc", global_result["overall"], "example-level overall"),
        _baseline_row("oracle_context_crc", oracle_result["overall"], "example-level context oracle"),
        _baseline_row(
            "metadata_stratified_crc",
            stratified_result["overall"],
            "example-level metadata strata",
        ),
        _window_row(ct_summary, "transferred_upper_bound"),
        _window_row(cw_summary, "weighted_risk_upper_bound"),
    ]
    return pd.DataFrame(rows)


def _baseline_row(method: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "method": method,
        "mean_risk": metrics.get("mean_loss"),
        "violation_rate": np.nan,
        "mean_coverage": metrics.get("coverage"),
        "mean_bound": np.nan,
        "bound_underestimation_rate": np.nan,
        "n_windows": np.nan,
        "n_examples": metrics.get("n"),
        "notes": notes,
    }


def _window_row(summary: dict[str, Any], bound_name: str) -> dict[str, Any]:
    return {
        "method": summary.get("method"),
        "mean_risk": summary.get("mean_risk"),
        "violation_rate": summary.get("violation_rate"),
        "mean_coverage": summary.get("mean_coverage"),
        "mean_bound": summary.get("mean_upper_bound"),
        "bound_underestimation_rate": summary.get("bound_underestimation_rate"),
        "n_windows": summary.get("n_windows"),
        "n_examples": np.nan,
        "notes": f"window-level {bound_name}",
    }


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
