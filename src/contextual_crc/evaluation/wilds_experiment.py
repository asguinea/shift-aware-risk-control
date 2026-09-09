from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from contextual_crc.cache.io import cache_exists, load_prediction_cache
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.config.loader import load_config
from contextual_crc.datasets.wilds_dataset import WILDSConfig, build_wilds_prediction_cache
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


LOGGER = logging.getLogger(__name__)


def run_wilds_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    run_name_override: str | None = None,
    build_cache_override: bool | None = None,
    source_override: str | None = None,
    download_override: bool | None = None,
    cache_dir_override: str | None = None,
) -> Path:
    """Run the WILDS/CivilComments evaluation pipeline and write structured outputs."""
    seed = int(config.get("project", {}).get("seed", 123))
    set_seed(seed)

    resolved_config = dict(config)
    paths = resolved_config.setdefault("paths", {})
    if cache_dir_override is not None:
        paths["cache_dir"] = cache_dir_override
    wilds_cache = resolved_config.setdefault("wilds_cache", {})
    if build_cache_override is not None:
        wilds_cache["build_if_missing"] = bool(build_cache_override)
    if source_override is not None:
        wilds_cache["source_override"] = source_override
    if download_override is not None:
        wilds_cache["download_override"] = bool(download_override)
    output_root = Path(paths.get("output_root", "outputs/runs"))
    run_name = run_name_override or paths.get("run_name", "wilds_civilcomments_main")
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir = make_run_dir(output_root, run_name)
    results_dir = ensure_dir(run_dir / "results")
    intermediate_dir = ensure_dir(run_dir / "intermediate")

    paths["run_name"] = run_name
    _write_yaml(resolved_config, run_dir / "configs" / "resolved_config.yaml")

    cache_dir = _ensure_wilds_cache(
        resolved_config,
        build_cache_override=build_cache_override,
        source_override=source_override,
        download_override=download_override,
    )
    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)

    window_config = parse_window_config(resolved_config.get("windowing", {}))
    signature_config = parse_signature_config(resolved_config.get("signatures", {}))
    encoder_config = parse_encoder_config(resolved_config.get("encoder", {}))
    risk_config = parse_risk_control_config(resolved_config.get("risk_control", {}))

    windows = make_windows(examples, window_config)
    if windows.empty:
        raise ValueError("Windowing produced no windows; reduce window_size or set drop_incomplete=false")
    signatures = compute_signatures_for_windows(examples, arrays, windows, signature_config)
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=False)
    if not feature_columns:
        raise ValueError("No non-diagnostic signature feature columns available for encoder fitting")

    available_contexts = _available_contexts(examples)
    calibration_contexts = _resolve_calibration_contexts(
        resolved_config.get("splits", {}),
        available_contexts,
    )
    _validate_contexts_available(calibration_contexts, available_contexts)
    evaluation_contexts = _resolve_evaluation_contexts(
        resolved_config.get("splits", {}).get("evaluation_contexts", "all"),
        available_contexts,
    )
    _validate_contexts_available(evaluation_contexts, available_contexts)

    calibration_examples, calibration_windows = _subset_examples_and_remap_windows(
        examples,
        windows,
        calibration_contexts,
    )
    evaluation_examples = examples[examples["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    if calibration_examples.empty or calibration_windows.empty:
        raise ValueError("No calibration examples/windows available for requested calibration_contexts")
    if evaluation_examples.empty or evaluation_windows.empty:
        raise ValueError("No evaluation examples/windows available for requested evaluation_contexts")

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
    stratified_domain = _maybe_run_stratified(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="domain_id",
    )
    stratified_group = _maybe_run_stratified(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="group_id",
    )

    global_result["by_context"].to_csv(results_dir / "global_by_context.csv", index=False)
    oracle_result["by_context"].to_csv(results_dir / "oracle_by_context.csv", index=False)
    stratified_context["by_stratum"].to_csv(results_dir / "stratified_by_context.csv", index=False)
    _write_optional_stratified(stratified_domain, results_dir / "stratified_by_domain.csv")
    _write_optional_stratified(stratified_group, results_dir / "stratified_by_group.csv")

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

    skipped = _skipped_stratified_notes(stratified_domain, stratified_group)
    method_comparison = _build_method_comparison(
        global_result,
        oracle_result,
        stratified_context,
        stratified_domain,
        stratified_group,
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
    if stratified_domain is not None:
        methods_run.append("metadata_stratified_domain")
    if stratified_group is not None:
        methods_run.append("metadata_stratified_group")

    manifest = {
        "run_name": run_name,
        "seed": seed,
        "cache_dir": str(cache_dir),
        "dataset": metadata.get("dataset"),
        "num_examples": int(len(examples)),
        "num_windows": int(len(windows)),
        "num_signature_features": int(len(feature_columns)),
        "signature_feature_columns": feature_columns,
        "available_contexts": available_contexts,
        "calibration_contexts": calibration_contexts,
        "evaluation_contexts": evaluation_contexts,
        "methods_run": methods_run,
        "skipped_methods": skipped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest, results_dir / "run_manifest.json")
    return run_dir


def _ensure_wilds_cache(
    config: dict[str, Any],
    *,
    build_cache_override: bool | None,
    source_override: str | None,
    download_override: bool | None,
) -> Path:
    paths = config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "outputs/caches/wilds_civilcomments_subset"))
    cache_config = config.get("wilds_cache", {})
    build_if_missing = bool(cache_config.get("build_if_missing", False))
    if build_cache_override is not None:
        build_if_missing = bool(build_cache_override)
    overwrite_cache = bool(cache_config.get("overwrite_cache", False))

    if cache_exists(cache_dir) and not overwrite_cache and not build_cache_override:
        return cache_dir
    if not cache_exists(cache_dir) and not build_if_missing:
        raise FileNotFoundError(
            f"WILDS cache not found: {cache_dir}. Build it with scripts/build_wilds_cache.py "
            "or set wilds_cache.build_if_missing=true / pass --build-cache."
        )

    raw_cache_config = load_config(cache_config.get("config_path", "configs/wilds_civilcomments.yaml"))
    wilds_raw = dict(raw_cache_config.get("wilds", {}))
    wilds_paths = raw_cache_config.get("paths", {})
    wilds_raw["data_root"] = wilds_paths.get("data_root", wilds_raw.get("data_root", "data/wilds"))
    if source_override is not None:
        wilds_raw["source"] = source_override
    if download_override is not None:
        wilds_raw["download"] = bool(download_override)
    wilds_config = WILDSConfig(**wilds_raw)
    build_wilds_prediction_cache(
        cache_dir,
        wilds_config,
        overwrite=overwrite_cache or bool(build_cache_override) or not cache_exists(cache_dir),
    )
    return cache_dir


def _available_contexts(examples: pd.DataFrame) -> list[str]:
    return sorted(str(value) for value in examples["context_id"].drop_duplicates().tolist())


def _resolve_calibration_contexts(splits_config: dict[str, Any], available_contexts: list[str]) -> list[str]:
    raw = splits_config.get("calibration_contexts", [])
    if raw == "auto":
        requested = int(splits_config.get("num_auto_calibration_contexts", 2))
        if requested <= 0:
            raise ValueError("splits.num_auto_calibration_contexts must be positive")
        if len(available_contexts) < requested:
            LOGGER.warning(
                "Requested %s auto calibration contexts but only %s are available; using all contexts.",
                requested,
                len(available_contexts),
            )
        return available_contexts[: min(requested, len(available_contexts))]
    contexts = [str(value) for value in raw]
    if not contexts:
        raise ValueError("splits.calibration_contexts must be a non-empty list or 'auto'")
    return contexts


def _resolve_evaluation_contexts(raw: Any, available_contexts: list[str]) -> list[str]:
    if raw == "all" or raw == ["all"]:
        return list(available_contexts)
    contexts = [str(value) for value in raw]
    if not contexts:
        raise ValueError("splits.evaluation_contexts must be non-empty or 'all'")
    return contexts


def _validate_contexts_available(contexts: list[str], available_contexts: list[str]) -> None:
    available_set = set(available_contexts)
    missing = [context for context in contexts if context not in available_set]
    if missing:
        raise ValueError(f"Requested contexts not found: {missing}. Available contexts: {available_contexts}")


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
        LOGGER.warning("CT-CRC l_hat auto requested with fewer than two calibration contexts; using 0.0")
        return 0.0
    try:
        return estimate_lipschitz_from_contexts(
            context_frame,
            risk_col="risk_estimate",
            metric=raw_config.get("metric", "euclidean"),
            quantile=float(raw_config.get("l_hat_quantile", 0.9)),
        )
    except ValueError as exc:
        LOGGER.warning("CT-CRC l_hat estimation failed; using 0.0. Error: %s", exc)
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
    stratified_domain: dict[str, Any] | None,
    stratified_group: dict[str, Any] | None,
) -> list[str]:
    skipped = []
    if stratified_domain is None:
        skipped.append("metadata_stratified_domain skipped: domain_id column unavailable")
    if stratified_group is None:
        skipped.append("metadata_stratified_group skipped: group_id column unavailable")
    return skipped


def _build_method_comparison(
    global_result: dict[str, Any],
    oracle_result: dict[str, Any],
    stratified_context: dict[str, Any],
    stratified_domain: dict[str, Any] | None,
    stratified_group: dict[str, Any] | None,
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
            "example-level context_id strata",
        ),
    ]
    if stratified_domain is not None:
        rows.append(
            _baseline_row(
                "metadata_stratified_domain",
                stratified_domain["overall"],
                "example-level domain_id strata",
            )
        )
    if stratified_group is not None:
        rows.append(
            _baseline_row(
                "metadata_stratified_group",
                stratified_group["overall"],
                "example-level group_id strata",
            )
        )
    rows.extend(
        [
            _window_row(ct_summary, "transferred_upper_bound"),
            _window_row(cw_summary, "weighted_risk_upper_bound"),
        ]
    )
    for note in skipped:
        rows.append(
            {
                "method": note.split()[0],
                "mean_risk": np.nan,
                "violation_rate": np.nan,
                "mean_coverage": np.nan,
                "mean_bound": np.nan,
                "bound_underestimation_rate": np.nan,
                "n_windows": np.nan,
                "n_examples": np.nan,
                "notes": note,
            }
        )
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
