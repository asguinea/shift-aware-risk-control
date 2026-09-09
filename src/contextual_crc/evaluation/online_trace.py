from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.embeddings.context_encoder import ContextEncoder, build_context_embedding_frame
from contextual_crc.evaluation.synthetic_experiment import (
    parse_ctcrc_config,
    parse_cwcrc_config,
    parse_encoder_config,
    parse_risk_control_config,
    parse_signature_config,
    parse_window_config,
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
TRACE_REQUIRED_COLUMNS = ("corruption_type", "severity", "context_id")
TRACE_OUTPUT_COLUMNS = (
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
)


def run_cifar_online_trace(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    run_name_override: str | None = None,
    cache_dir_override: str | None = None,
) -> Path:
    """Run a dedicated severity-ordered CIFAR-C online trace from cached predictions."""
    seed = int(config.get("project", {}).get("seed", 123))
    set_seed(seed)

    resolved_config = dict(config)
    paths = resolved_config.setdefault("paths", {})
    if cache_dir_override is not None:
        paths["cache_dir"] = cache_dir_override
    output_root = Path(paths.get("output_root", "outputs/runs"))
    run_name = run_name_override or paths.get("run_name", "cifar10c_resnet18_online_trace")
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir = make_run_dir(output_root, run_name)
    results_dir = ensure_dir(run_dir / "results")
    intermediate_dir = ensure_dir(run_dir / "intermediate")

    paths["run_name"] = run_name
    _write_yaml(resolved_config, run_dir / "configs" / "resolved_config.yaml")

    cache_dir = Path(paths.get("cache_dir", "outputs/caches/cifar10c_resnet18_subset"))
    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)
    _validate_trace_columns(examples)

    trace_config = dict(resolved_config.get("trace", {}))
    dataset = str(trace_config.get("dataset", metadata.get("dataset", "cifar10c")))
    corruption_type = str(trace_config.get("corruption_type", "gaussian_noise"))
    severities = _as_int_list(trace_config.get("severities", [1, 3, 5]))
    calibration_severities = _as_int_list(trace_config.get("calibration_severities", [severities[0]]))
    evaluation_severities = _as_int_list(trace_config.get("evaluation_severities", severities))

    filtered_examples, filtered_arrays = _filter_trace_examples(
        examples,
        arrays,
        corruption_type=corruption_type,
        severities=severities,
    )
    window_config = parse_window_config(resolved_config.get("windowing", {}))
    signature_config = parse_signature_config(resolved_config.get("signatures", {}))
    encoder_config = parse_encoder_config(resolved_config.get("encoder", {}))
    risk_config = parse_risk_control_config(resolved_config.get("risk_control", {}))

    windows = make_windows(filtered_examples, window_config)
    if windows.empty:
        raise ValueError("Windowing produced no trace windows; reduce window_size or set drop_incomplete=false")
    windows = _attach_window_trace_metadata(windows, filtered_examples)
    windows = _order_trace_windows(windows, severity_order=severities)
    signatures = compute_signatures_for_windows(filtered_examples, filtered_arrays, windows, signature_config)
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=False)
    if not feature_columns:
        raise ValueError("No non-diagnostic signature feature columns available for encoder fitting")

    calibration_examples, calibration_windows = _subset_examples_and_remap_windows_by_severity(
        filtered_examples,
        windows,
        calibration_severities,
    )
    evaluation_windows = windows[windows["severity"].isin(evaluation_severities)].copy().reset_index(drop=True)
    if calibration_examples.empty or calibration_windows.empty:
        raise ValueError("No calibration examples/windows available for requested calibration_severities")
    if evaluation_windows.empty:
        raise ValueError("No evaluation windows available for requested evaluation_severities")

    calibration_window_ids = set(calibration_windows["window_id"])
    evaluation_window_ids = set(evaluation_windows["window_id"])
    calibration_signatures = signatures[signatures["window_id"].isin(calibration_window_ids)].reset_index(drop=True)
    encoder = ContextEncoder(encoder_config)
    calibration_embeddings = encoder.fit_transform(calibration_signatures, feature_columns)
    all_embeddings = encoder.transform(signatures)
    embedding_frame = build_context_embedding_frame(signatures, all_embeddings)
    embedding_frame = _attach_window_trace_metadata(embedding_frame, filtered_examples, windows=windows)
    calibration_embedding_frame = build_context_embedding_frame(calibration_signatures, calibration_embeddings)
    calibration_embedding_frame = _attach_window_trace_metadata(
        calibration_embedding_frame,
        calibration_examples,
        windows=calibration_windows,
    )
    evaluation_embedding_frame = embedding_frame[
        embedding_frame["window_id"].isin(evaluation_window_ids)
    ].reset_index(drop=True)

    if resolved_config.get("outputs", {}).get("save_intermediate", True):
        save_windows(windows, intermediate_dir / "windows.parquet")
        save_signatures(signatures, intermediate_dir / "signatures.parquet")
        embedding_frame.to_parquet(intermediate_dir / "embeddings.parquet", engine="pyarrow", index=False)

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

    ct_windows = evaluate_ctcrc_windows(
        filtered_examples,
        evaluation_windows,
        evaluation_embedding_frame,
        ct_result,
    )
    cw_config = parse_cwcrc_config(resolved_config.get("cw_crc", {}), risk_config)
    cw_windows = evaluate_cwcrc_windows(
        filtered_examples,
        calibration_examples,
        calibration_windows,
        calibration_embedding_frame,
        evaluation_windows,
        evaluation_embedding_frame,
        cw_config,
    )
    online_trace = _build_online_trace_frame(
        evaluation_windows,
        ct_windows,
        cw_windows,
        risk_budget=risk_config.risk_budget,
        severity_order=evaluation_severities,
    )
    online_trace.to_csv(results_dir / "online_trace.csv", index=False)

    manifest = {
        "run_name": run_name,
        "seed": seed,
        "cache_dir": str(cache_dir),
        "dataset": dataset,
        "corruption_type": corruption_type,
        "severities": severities,
        "calibration_severities": calibration_severities,
        "evaluation_severities": evaluation_severities,
        "num_examples": int(len(filtered_examples)),
        "num_windows": int(len(windows)),
        "num_trace_windows": int(len(online_trace)),
        "num_signature_features": int(len(feature_columns)),
        "signature_feature_columns": feature_columns,
        "methods_run": ["ct_crc", "cw_crc"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest, results_dir / "run_manifest.json")
    return run_dir


def _validate_trace_columns(examples: pd.DataFrame) -> None:
    missing = [column for column in TRACE_REQUIRED_COLUMNS if column not in examples.columns]
    if missing:
        raise ValueError(f"CIFAR online trace requires cache columns: {missing}")


def _filter_trace_examples(
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    corruption_type: str,
    severities: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    working = examples.copy()
    working["severity"] = pd.to_numeric(working["severity"], errors="coerce").astype("Int64")
    available_corruptions = sorted(str(value) for value in working["corruption_type"].dropna().unique())
    if corruption_type not in set(available_corruptions):
        raise ValueError(
            f"Requested corruption_type {corruption_type!r} not found. "
            f"Available corruptions: {available_corruptions}"
        )
    corruption_examples = working[working["corruption_type"] == corruption_type]
    available_severities = sorted(int(value) for value in corruption_examples["severity"].dropna().unique())
    missing_severities = [int(value) for value in severities if int(value) not in set(available_severities)]
    if missing_severities:
        raise ValueError(
            f"Requested severities {missing_severities} not found for corruption {corruption_type!r}. "
            f"Available severities: {available_severities}"
        )

    mask = (working["corruption_type"] == corruption_type) & working["severity"].isin([int(value) for value in severities])
    positions = np.flatnonzero(mask.to_numpy())
    filtered = working.iloc[positions].copy()
    filtered["_source_position"] = positions
    filtered["severity"] = filtered["severity"].astype(int)
    filtered = filtered.sort_values(["severity", "time_index", "example_id"], kind="mergesort").reset_index(drop=True)
    sorted_positions = filtered.pop("_source_position").to_numpy(dtype=int)
    return filtered, _subset_arrays(arrays, sorted_positions)


def _subset_arrays(arrays: dict[str, np.ndarray], positions: np.ndarray) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {}
    max_position = int(np.max(positions)) if len(positions) else -1
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] == len(positions):
            subset[key] = array
        elif max_position < array.shape[0]:
            subset[key] = array[positions]
        else:
            raise ValueError(f"Array {key!r} length {array.shape[0]} cannot be aligned to filtered examples")
    return subset


def _attach_window_trace_metadata(
    frame: pd.DataFrame,
    examples: pd.DataFrame,
    *,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    source = windows if windows is not None else frame
    if {"corruption_type", "severity"}.issubset(frame.columns):
        return frame
    metadata_rows = []
    for _, row in source.iterrows():
        indices = [int(index) for index in row["example_indices"]] if "example_indices" in row else []
        if indices:
            window_examples = examples.iloc[indices]
            severity = int(window_examples["severity"].mode().iloc[0])
            corruption = str(window_examples["corruption_type"].mode().iloc[0])
            if "time_index" in window_examples.columns:
                first_time = int(pd.to_numeric(window_examples["time_index"], errors="coerce").min())
            else:
                first_time = int(row.get("start_pos", 0))
        else:
            severity = int(row.get("severity", 0))
            corruption = str(row.get("corruption_type", "unknown"))
            first_time = int(row.get("start_pos", 0))
        metadata_rows.append(
            {
                "window_id": row["window_id"],
                "corruption_type": corruption,
                "severity": severity,
                "trace_sort_time": first_time,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    return frame.merge(metadata, on="window_id", how="left")


def _order_trace_windows(windows: pd.DataFrame, *, severity_order: Sequence[int]) -> pd.DataFrame:
    order = {int(severity): index for index, severity in enumerate(severity_order)}
    working = windows.copy()
    working["_severity_order"] = working["severity"].map(order).fillna(len(order)).astype(int)
    working = working.sort_values(
        ["_severity_order", "trace_sort_time", "window_id"],
        kind="mergesort",
    ).drop(columns=["_severity_order"])
    return working.reset_index(drop=True)


def _subset_examples_and_remap_windows_by_severity(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    severities: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    severity_set = {int(value) for value in severities}
    mask = examples["severity"].astype(int).isin(severity_set)
    original_positions = np.flatnonzero(mask.to_numpy())
    original_to_local = {int(original): local for local, original in enumerate(original_positions)}
    subset_examples = examples.iloc[original_positions].reset_index(drop=True)
    subset_windows = windows[windows["severity"].astype(int).isin(severity_set)].copy().reset_index(drop=True)
    remapped_indices = []
    for indices in subset_windows["example_indices"]:
        remapped = []
        for index in indices:
            index = int(index)
            if index not in original_to_local:
                raise ValueError(f"Window contains example outside requested severities: {index}")
            remapped.append(original_to_local[index])
        remapped_indices.append(remapped)
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
            quantile=float(raw_config.get("l_hat_quantile", 0.95)),
        )
    except ValueError as exc:
        LOGGER.warning("Falling back to l_hat=0.0 for CIFAR online trace: %s", exc)
        return 0.0


def _build_online_trace_frame(
    evaluation_windows: pd.DataFrame,
    ct_windows: pd.DataFrame,
    cw_windows: pd.DataFrame,
    *,
    risk_budget: float,
    severity_order: Sequence[int],
) -> pd.DataFrame:
    base = evaluation_windows[["window_id", "context_id", "corruption_type", "severity", "trace_sort_time"]].copy()
    base = _order_trace_windows(base, severity_order=severity_order)
    ct = ct_windows[
        ["window_id", "novelty_distance", "realized_risk", "transferred_upper_bound", "coverage", "violation"]
    ].rename(
        columns={
            "realized_risk": "ct_realized_risk",
            "transferred_upper_bound": "ct_bound",
            "coverage": "ct_coverage",
            "violation": "ct_violation",
        }
    )
    cw = cw_windows[
        ["window_id", "realized_risk", "weighted_risk_upper_bound", "coverage", "violation"]
    ].rename(
        columns={
            "realized_risk": "cw_realized_risk",
            "weighted_risk_upper_bound": "cw_bound",
            "coverage": "cw_coverage",
            "violation": "cw_violation",
        }
    )
    trace = base.merge(ct, on="window_id", how="left").merge(cw, on="window_id", how="left")
    trace["trace_index"] = np.arange(len(trace), dtype=int)
    trace["ct_bound_underestimated"] = trace["ct_realized_risk"] > trace["ct_bound"]
    trace["cw_bound_underestimated"] = trace["cw_realized_risk"] > trace["cw_bound"]
    trace["ct_violation"] = trace["ct_violation"].fillna(trace["ct_realized_risk"] > risk_budget).astype(bool)
    trace["cw_violation"] = trace["cw_violation"].fillna(trace["cw_realized_risk"] > risk_budget).astype(bool)
    return trace.loc[:, list(TRACE_OUTPUT_COLUMNS)]


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    return [int(value) for value in values]


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
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
