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
from contextual_crc.datasets.cifar_c import CIFARCConfig, build_cifar_c_prediction_cache
from contextual_crc.embeddings.context_encoder import ContextEncoder, build_context_embedding_frame
from contextual_crc.evaluation.diagnostics import (
    bound_calibration_diagnostics,
    context_smoothness_diagnostics,
    context_smoothness_pairwise_frame,
    knn_risk_prediction_diagnostics,
)
from contextual_crc.evaluation.metrics import summarize_window_results
from contextual_crc.evaluation.synthetic_experiment import (
    parse_ctcrc_config,
    parse_cwcrc_config,
    parse_encoder_config,
    parse_risk_control_config,
    parse_signature_config,
    parse_window_config,
)
from contextual_crc.risk_control.baselines import (
    run_global_crc_baseline,
    run_metadata_stratified_crc_baseline,
    run_oracle_context_crc_baseline,
)
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    build_validated_contexts,
    estimate_lipschitz_from_contexts,
    evaluate_ctcrc_windows,
    validated_contexts_to_frame,
)
from contextual_crc.risk_control.cw_crc import evaluate_cwcrc_windows
from contextual_crc.signatures.behavioral_signatures import (
    compute_signatures_for_windows,
    get_signature_feature_columns,
)
from contextual_crc.signatures.io import save_signatures, save_windows
from contextual_crc.signatures.windowing import make_windows
from contextual_crc.utils.paths import ensure_dir, make_run_dir
from contextual_crc.utils.seed import set_seed


def run_cifar_c_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    run_name_override: str | None = None,
    build_cache_override: bool | None = None,
    download_override: bool | None = None,
    cache_dir_override: str | None = None,
) -> Path:
    """Run the CIFAR-C evaluation pipeline and write structured outputs."""
    seed = int(config.get("project", {}).get("seed", 123))
    set_seed(seed)

    resolved_config = dict(config)
    paths = resolved_config.setdefault("paths", {})
    if cache_dir_override is not None:
        paths["cache_dir"] = cache_dir_override
    output_root = Path(paths.get("output_root", "outputs/runs"))
    run_name = run_name_override or paths.get("run_name", "cifar_c_main")
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir = make_run_dir(output_root, run_name)
    results_dir = ensure_dir(run_dir / "results")
    intermediate_dir = ensure_dir(run_dir / "intermediate")

    paths["run_name"] = run_name
    _write_yaml(resolved_config, run_dir / "configs" / "resolved_config.yaml")

    cache_dir = _ensure_cifar_cache(
        resolved_config,
        build_cache_override=build_cache_override,
        download_override=download_override,
    )
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

    calibration_contexts = [str(value) for value in resolved_config.get("splits", {}).get("calibration_contexts", [])]
    if not calibration_contexts:
        raise ValueError("splits.calibration_contexts must be non-empty")
    _validate_contexts_available(calibration_contexts, examples)
    evaluation_contexts = _resolve_evaluation_contexts(
        resolved_config.get("splits", {}).get("evaluation_contexts", "all"),
        examples,
    )
    _validate_contexts_available(evaluation_contexts, examples)

    calibration_examples, calibration_windows = _subset_examples_and_remap_windows(
        examples,
        windows,
        calibration_contexts,
    )
    evaluation_examples = examples[examples["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    if calibration_windows.empty:
        raise ValueError("No calibration windows available for requested calibration_contexts")
    if evaluation_windows.empty:
        raise ValueError("No evaluation windows available for requested evaluation_contexts")

    calibration_window_ids = set(calibration_windows["window_id"])
    evaluation_window_ids = set(evaluation_windows["window_id"])
    calibration_signatures = signatures[signatures["window_id"].isin(calibration_window_ids)].reset_index(drop=True)

    encoder = ContextEncoder(encoder_config)
    calibration_embeddings = encoder.fit_transform(calibration_signatures, feature_columns)
    all_embeddings = encoder.transform(signatures)
    embedding_frame = build_context_embedding_frame(signatures, all_embeddings)
    calibration_embedding_frame = build_context_embedding_frame(calibration_signatures, calibration_embeddings)
    evaluation_embedding_frame = embedding_frame[
        embedding_frame["window_id"].isin(evaluation_window_ids)
    ].reset_index(drop=True)

    if resolved_config.get("outputs", {}).get("save_intermediate", True):
        save_windows(windows, intermediate_dir / "windows.parquet")
        save_signatures(signatures, intermediate_dir / "signatures.parquet")
        embedding_frame.to_parquet(intermediate_dir / "embeddings.parquet", engine="pyarrow", index=False)

    global_result = run_global_crc_baseline(calibration_examples, evaluation_examples, risk_config)
    oracle_result = run_oracle_context_crc_baseline(calibration_examples, evaluation_examples, risk_config)
    stratified_context = run_metadata_stratified_crc_baseline(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="context_id",
    )
    stratified_corruption = _maybe_run_stratified(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="corruption_type",
    )
    stratified_severity = _maybe_run_stratified(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="severity",
    )

    global_result["by_context"].to_csv(results_dir / "global_by_context.csv", index=False)
    oracle_result["by_context"].to_csv(results_dir / "oracle_by_context.csv", index=False)
    stratified_context["by_stratum"].to_csv(results_dir / "stratified_by_context.csv", index=False)
    _write_optional_stratified(stratified_corruption, results_dir / "stratified_by_corruption.csv")
    _write_optional_stratified(stratified_severity, results_dir / "stratified_by_severity.csv")

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
        "ct_crc": bound_calibration_diagnostics(ct_windows, upper_bound_col="transferred_upper_bound"),
        "cw_crc": bound_calibration_diagnostics(cw_windows, upper_bound_col="weighted_risk_upper_bound"),
    }
    _write_json(bound_diagnostics, results_dir / "bound_diagnostics.json")

    smoothness_frame, smoothness_value_col = _build_smoothness_frame(signatures, embedding_frame, ct_windows)
    context_smoothness_summary = context_smoothness_diagnostics(
        smoothness_frame,
        value_col=smoothness_value_col,
    )
    context_smoothness_pairs = context_smoothness_pairwise_frame(
        smoothness_frame,
        value_col=smoothness_value_col,
        dataset=str(metadata.get("dataset") or run_name),
    )
    context_smoothness_pairs.to_csv(results_dir / "context_smoothness.csv", index=False)
    _write_json(context_smoothness_summary, results_dir / "context_smoothness_summary.json")
    _write_json(
        knn_risk_prediction_diagnostics(smoothness_frame, risk_col=smoothness_value_col, k=3),
        results_dir / "knn_diagnostics.json",
    )

    skipped = _skipped_stratified_notes(stratified_corruption, stratified_severity)
    method_comparison = _build_method_comparison(
        global_result,
        oracle_result,
        stratified_context,
        stratified_corruption,
        stratified_severity,
        ct_summary,
        cw_summary,
        skipped,
    )
    method_comparison.to_csv(results_dir / "method_comparison.csv", index=False)

    methods_run = [
        "global_crc",
        "oracle_context_crc",
        "metadata_stratified_context",
        "ct_crc",
        "cw_crc",
    ]
    if stratified_corruption is not None:
        methods_run.append("metadata_stratified_corruption")
    if stratified_severity is not None:
        methods_run.append("metadata_stratified_severity")

    manifest = {
        "run_name": run_name,
        "seed": seed,
        "cache_dir": str(cache_dir),
        "dataset": metadata.get("dataset"),
        "num_examples": int(len(examples)),
        "num_windows": int(len(windows)),
        "num_signature_features": int(len(feature_columns)),
        "signature_feature_columns": feature_columns,
        "calibration_contexts": calibration_contexts,
        "evaluation_contexts": evaluation_contexts,
        "methods_run": methods_run,
        "skipped_methods": skipped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest, results_dir / "run_manifest.json")
    return run_dir


def _ensure_cifar_cache(
    config: dict[str, Any],
    *,
    build_cache_override: bool | None,
    download_override: bool | None,
) -> Path:
    paths = config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "outputs/caches/cifar_c_subset"))
    cache_config = config.get("cifar_cache", {})
    build_if_missing = bool(cache_config.get("build_if_missing", False))
    if build_cache_override is not None:
        build_if_missing = bool(build_cache_override)
    overwrite_cache = bool(cache_config.get("overwrite_cache", False))

    if cache_exists(cache_dir) and not overwrite_cache and not build_cache_override:
        return cache_dir
    if not cache_exists(cache_dir) and not build_if_missing:
        raise FileNotFoundError(
            f"CIFAR-C cache not found: {cache_dir}. Build it with scripts/build_cifar_c_cache.py "
            "or set cifar_cache.build_if_missing=true / pass --build-cache."
        )

    raw_cache_config = load_config(cache_config.get("config_path", "configs/cifar10c.yaml"))
    cifar_raw = dict(raw_cache_config.get("cifar_c", {}))
    cifar_paths = raw_cache_config.get("paths", {})
    cifar_raw["data_root"] = cifar_paths.get("data_root", cifar_raw.get("data_root", "data"))
    if download_override is not None:
        cifar_raw["download"] = bool(download_override)
    cifar_config = CIFARCConfig(**cifar_raw)
    build_cifar_c_prediction_cache(
        cache_dir,
        cifar_config,
        overwrite=overwrite_cache or bool(build_cache_override) or not cache_exists(cache_dir),
    )
    return cache_dir


def _resolve_evaluation_contexts(raw: Any, examples: pd.DataFrame) -> list[str]:
    if raw == "all" or raw == ["all"]:
        return [str(value) for value in examples["context_id"].drop_duplicates().tolist()]
    return [str(value) for value in raw]


def _validate_contexts_available(contexts: list[str], examples: pd.DataFrame) -> None:
    available = [str(value) for value in examples["context_id"].drop_duplicates().tolist()]
    missing = [context for context in contexts if context not in set(available)]
    if missing:
        raise ValueError(f"Requested contexts not found: {missing}. Available contexts: {available}")


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


def _maybe_run_stratified(
    calibration_examples: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    risk_config: Any,
    *,
    stratify_col: str,
) -> dict[str, Any] | None:
    if stratify_col not in calibration_examples.columns or stratify_col not in evaluation_examples.columns:
        return None
    return run_metadata_stratified_crc_baseline(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col=stratify_col,
    )


def _write_optional_stratified(result: dict[str, Any] | None, path: Path) -> None:
    if result is None:
        pd.DataFrame().to_csv(path, index=False)
    else:
        result["by_stratum"].to_csv(path, index=False)


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


def _build_smoothness_frame(
    signatures: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    ct_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    z_columns = [column for column in embedding_frame.columns if column.startswith("z")]
    smoothness_frame = signatures.merge(
        embedding_frame[["window_id"] + z_columns],
        on="window_id",
        how="left",
    )
    if "diag_error_rate" in smoothness_frame.columns:
        return smoothness_frame, "diag_error_rate"
    if "diag_accuracy" in smoothness_frame.columns:
        smoothness_frame["diag_error_rate"] = 1.0 - smoothness_frame["diag_accuracy"]
        return smoothness_frame, "diag_error_rate"
    smoothness_frame = smoothness_frame.merge(ct_windows[["window_id", "realized_risk"]], on="window_id", how="left")
    return smoothness_frame, "realized_risk"


def _skipped_stratified_notes(
    stratified_corruption: dict[str, Any] | None,
    stratified_severity: dict[str, Any] | None,
) -> list[str]:
    skipped = []
    if stratified_corruption is None:
        skipped.append("metadata_stratified_corruption skipped: corruption_type column unavailable")
    if stratified_severity is None:
        skipped.append("metadata_stratified_severity skipped: severity column unavailable")
    return skipped


def _build_method_comparison(
    global_result: dict[str, Any],
    oracle_result: dict[str, Any],
    stratified_context: dict[str, Any],
    stratified_corruption: dict[str, Any] | None,
    stratified_severity: dict[str, Any] | None,
    ct_summary: dict[str, Any],
    cw_summary: dict[str, Any],
    skipped: list[str],
) -> pd.DataFrame:
    rows = [
        _baseline_row("global_crc", global_result["overall"], "example-level overall"),
        _baseline_row("oracle_context_crc", oracle_result["overall"], "example-level context oracle"),
        _baseline_row(
            "metadata_stratified_context",
            stratified_context["overall"],
            "example-level context strata",
        ),
    ]
    if stratified_corruption is not None:
        rows.append(
            _baseline_row(
                "metadata_stratified_corruption",
                stratified_corruption["overall"],
                "example-level corruption strata",
            )
        )
    if stratified_severity is not None:
        rows.append(
            _baseline_row(
                "metadata_stratified_severity",
                stratified_severity["overall"],
                "example-level severity strata",
            )
        )
    rows.extend(
        [
            _window_row(ct_summary, "transferred_upper_bound"),
            _window_row(cw_summary, "weighted_risk_upper_bound"),
        ]
    )
    if skipped:
        for note in skipped:
            rows.append({"method": note.split()[0], "notes": note})
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
