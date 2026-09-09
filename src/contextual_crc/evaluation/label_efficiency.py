from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.embeddings.context_encoder import ContextEncoder, build_context_embedding_frame
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
from contextual_crc.signatures.windowing import make_windows
from contextual_crc.utils.paths import ensure_dir
from contextual_crc.utils.seed import set_seed


LABEL_EFFICIENCY_COLUMNS = (
    "dataset",
    "budget_windows_per_context",
    "actual_labeled_examples",
    "method",
    "mean_risk",
    "mean_coverage",
    "violation_rate",
    "mean_bound",
    "bound_underestimation_rate",
    "seed",
)


def subsample_calibration_examples(
    examples: pd.DataFrame,
    *,
    context_col: str = "context_id",
    n_per_context: int,
    random_seed: int = 123,
) -> pd.DataFrame:
    """Deterministically sample up to n examples per context."""
    if n_per_context <= 0:
        raise ValueError("n_per_context must be positive")
    if context_col not in examples.columns:
        raise ValueError(f"examples is missing context column: {context_col}")
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    for _, group in examples.groupby(context_col, sort=True, dropna=False):
        indices = group.index.to_numpy()
        if len(indices) > n_per_context:
            indices = rng.choice(indices, size=n_per_context, replace=False)
        selected.extend(int(index) for index in np.sort(indices))
    return examples.loc[sorted(selected)].copy()


def subsample_calibration_windows(
    windows: pd.DataFrame,
    *,
    context_col: str = "context_id",
    windows_per_context: int,
    random_seed: int = 123,
) -> pd.DataFrame:
    """Deterministically sample up to a window budget per calibration context."""
    if windows_per_context <= 0:
        raise ValueError("windows_per_context must be positive")
    if context_col not in windows.columns:
        raise ValueError(f"windows is missing context column: {context_col}")
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    for _, group in windows.groupby(context_col, sort=True, dropna=False):
        indices = group.index.to_numpy()
        if len(indices) > windows_per_context:
            indices = rng.choice(indices, size=windows_per_context, replace=False)
        selected.extend(int(index) for index in np.sort(indices))
    return windows.loc[sorted(selected)].copy().reset_index(drop=True)


def examples_from_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Return unique examples touched by selected windows with original row positions."""
    indices = _unique_window_example_indices(windows)
    subset = examples.iloc[indices].copy()
    subset.insert(0, "original_index", indices)
    return subset.reset_index(drop=True)


def run_label_efficiency_from_cache(
    *,
    experiment_config: dict,
    budgets: Sequence[int],
    output_dir: str | Path,
    dataset_kind: str,
    overwrite: bool = False,
) -> Path:
    """Run window-budget label-efficiency evaluation from an existing prediction cache."""
    if dataset_kind not in {"synthetic", "cifar"}:
        raise ValueError("dataset_kind must be 'synthetic' or 'cifar'")
    budget_values = [int(value) for value in budgets]
    if not budget_values or any(value <= 0 for value in budget_values):
        raise ValueError("budgets must contain positive integers")

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Label-efficiency output directory exists and is not empty: {output_path}")
    ensure_dir(output_path)

    seed = int(experiment_config.get("project", {}).get("seed", 123))
    set_seed(seed)
    paths = experiment_config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "outputs/caches/synthetic_default"))
    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)
    dataset = str(metadata.get("dataset") or (examples["dataset"].iloc[0] if len(examples) else dataset_kind))

    window_config = parse_window_config(experiment_config.get("windowing", {}))
    signature_config = parse_signature_config(experiment_config.get("signatures", {}))
    encoder_config = parse_encoder_config(experiment_config.get("encoder", {}))
    risk_config = parse_risk_control_config(experiment_config.get("risk_control", {}))

    windows = make_windows(examples, window_config)
    signatures = compute_signatures_for_windows(examples, arrays, windows, signature_config)
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=False)
    if not feature_columns:
        raise ValueError("No non-diagnostic signature feature columns available for encoder fitting")

    calibration_contexts = [str(value) for value in experiment_config.get("splits", {}).get("calibration_contexts", [])]
    if not calibration_contexts:
        raise ValueError("splits.calibration_contexts must be non-empty")
    evaluation_contexts = _resolve_evaluation_contexts(
        experiment_config.get("splits", {}).get("evaluation_contexts", "all"),
        examples,
    )
    calibration_windows_all = windows[windows["context_id"].isin(calibration_contexts)].reset_index(drop=True)
    evaluation_examples = examples[examples["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    if calibration_windows_all.empty:
        raise ValueError("No calibration windows available for requested calibration_contexts")
    if evaluation_windows.empty:
        raise ValueError("No evaluation windows available for requested evaluation_contexts")

    rows: list[dict[str, Any]] = []
    for budget in budget_values:
        selected_windows = subsample_calibration_windows(
            calibration_windows_all,
            windows_per_context=budget,
            random_seed=seed + budget,
        )
        calibration_examples, calibration_windows = _subset_examples_and_remap_windows(examples, selected_windows)
        selected_window_ids = set(selected_windows["window_id"])
        evaluation_window_ids = set(evaluation_windows["window_id"])
        calibration_signatures = signatures[signatures["window_id"].isin(selected_window_ids)].reset_index(drop=True)
        evaluation_signatures = signatures[signatures["window_id"].isin(evaluation_window_ids)].reset_index(drop=True)

        encoder = ContextEncoder(encoder_config)
        calibration_embeddings = encoder.fit_transform(calibration_signatures, feature_columns)
        evaluation_embeddings = encoder.transform(evaluation_signatures)
        calibration_embedding_frame = build_context_embedding_frame(calibration_signatures, calibration_embeddings)
        evaluation_embedding_frame = build_context_embedding_frame(evaluation_signatures, evaluation_embeddings)

        rows.extend(
            _evaluate_budget(
                examples=examples,
                calibration_examples=calibration_examples,
                evaluation_examples=evaluation_examples,
                calibration_windows=calibration_windows,
                calibration_embedding_frame=calibration_embedding_frame,
                evaluation_windows=evaluation_windows,
                evaluation_embedding_frame=evaluation_embedding_frame,
                experiment_config=experiment_config,
                risk_config=risk_config,
                dataset=dataset,
                budget=budget,
                seed=seed,
            )
        )

    frame = pd.DataFrame(rows)
    for column in LABEL_EFFICIENCY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.loc[:, list(LABEL_EFFICIENCY_COLUMNS)]
    frame.to_csv(output_path / "label_efficiency.csv", index=False)
    _write_json(
        {
            "dataset": dataset,
            "dataset_kind": dataset_kind,
            "budgets": budget_values,
            "seed": seed,
            "cache_dir": str(cache_dir),
            "definition": "window-level label budget; selected calibration windows per context define labeled examples",
        },
        output_path / "label_efficiency_manifest.json",
    )
    return output_path


def _evaluate_budget(
    *,
    examples: pd.DataFrame,
    calibration_examples: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    calibration_embedding_frame: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
    evaluation_embedding_frame: pd.DataFrame,
    experiment_config: dict,
    risk_config: Any,
    dataset: str,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    actual_labeled = len(calibration_examples)
    rows: list[dict[str, Any]] = []

    global_result = run_global_crc_baseline(calibration_examples, evaluation_examples, risk_config)
    rows.append(_baseline_row(dataset, budget, actual_labeled, "global_crc", global_result["overall"], seed))

    stratified = run_metadata_stratified_crc_baseline(
        calibration_examples,
        evaluation_examples,
        risk_config,
        stratify_col="context_id",
    )
    rows.append(
        _baseline_row(
            dataset,
            budget,
            actual_labeled,
            "metadata_stratified_context",
            stratified["overall"],
            seed,
        )
    )

    ct_config = parse_ctcrc_config(experiment_config.get("ct_crc", {}), risk_config)
    ct_result = build_validated_contexts(calibration_examples, calibration_embedding_frame, ct_config)
    if experiment_config.get("ct_crc", {}).get("l_hat") == "auto":
        l_hat = _estimate_ct_l_hat(ct_result, experiment_config.get("ct_crc", {}))
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
    ct_summary = summarize_window_results(
        ct_windows,
        risk_budget=risk_config.risk_budget,
        upper_bound_col="transferred_upper_bound",
        method_name="ct_crc",
    )
    rows.append(_window_row(dataset, budget, actual_labeled, ct_summary, "mean_upper_bound", seed))

    cw_config = parse_cwcrc_config(experiment_config.get("cw_crc", {}), risk_config)
    cw_windows = evaluate_cwcrc_windows(
        examples,
        calibration_examples,
        calibration_windows,
        calibration_embedding_frame,
        evaluation_windows,
        evaluation_embedding_frame,
        cw_config,
    )
    cw_summary = summarize_window_results(
        cw_windows,
        risk_budget=risk_config.risk_budget,
        upper_bound_col="weighted_risk_upper_bound",
        method_name="cw_crc",
    )
    rows.append(_window_row(dataset, budget, actual_labeled, cw_summary, "mean_upper_bound", seed))
    return rows


def _baseline_row(
    dataset: str,
    budget: int,
    actual_labeled: int,
    method: str,
    metrics: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "budget_windows_per_context": int(budget),
        "actual_labeled_examples": int(actual_labeled),
        "method": method,
        "mean_risk": metrics.get("mean_loss"),
        "mean_coverage": metrics.get("coverage"),
        "violation_rate": np.nan,
        "mean_bound": np.nan,
        "bound_underestimation_rate": np.nan,
        "seed": int(seed),
    }


def _window_row(
    dataset: str,
    budget: int,
    actual_labeled: int,
    summary: dict[str, Any],
    bound_key: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "budget_windows_per_context": int(budget),
        "actual_labeled_examples": int(actual_labeled),
        "method": summary.get("method"),
        "mean_risk": summary.get("mean_risk"),
        "mean_coverage": summary.get("mean_coverage"),
        "violation_rate": summary.get("violation_rate"),
        "mean_bound": summary.get(bound_key),
        "bound_underestimation_rate": summary.get("bound_underestimation_rate"),
        "seed": int(seed),
    }


def _subset_examples_and_remap_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_indices = _unique_window_example_indices(windows)
    original_to_local = {int(original): local for local, original in enumerate(original_indices)}
    subset_examples = examples.iloc[original_indices].copy().reset_index(drop=True)
    remapped_windows = windows.copy().reset_index(drop=True)
    remapped = []
    for indices in remapped_windows["example_indices"]:
        remapped.append([original_to_local[int(index)] for index in indices if int(index) in original_to_local])
    remapped_windows["example_indices"] = remapped
    remapped_windows["n_examples"] = [len(indices) for indices in remapped]
    return subset_examples, remapped_windows


def _unique_window_example_indices(windows: pd.DataFrame) -> list[int]:
    if "example_indices" not in windows.columns:
        raise ValueError("windows must include example_indices")
    values: set[int] = set()
    for indices in windows["example_indices"]:
        values.update(int(index) for index in indices)
    return sorted(values)


def _resolve_evaluation_contexts(raw: Any, examples: pd.DataFrame) -> list[str]:
    if raw == "all" or raw == ["all"]:
        return [str(value) for value in examples["context_id"].drop_duplicates().tolist()]
    return [str(value) for value in raw]


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


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
