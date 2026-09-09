from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from contextual_crc.artifacts.risk_coverage import (
    compute_risk_coverage_curve,
    matched_coverage_point,
    matched_risk_point,
)
from contextual_crc.cache.io import cache_exists, load_prediction_cache
from contextual_crc.config.loader import load_config
from contextual_crc.embeddings.context_encoder import (
    ContextEncoder,
    build_context_embedding_frame,
    save_context_encoder,
)
from contextual_crc.evaluation.ct_selection_valid import _build_dkw_with_auto_l_hat
from contextual_crc.evaluation.diagnostics import context_smoothness_pairwise_frame
from contextual_crc.evaluation.synthetic_experiment import (
    _estimate_ct_l_hat,
    parse_ctcrc_config,
    parse_cwcrc_config,
    parse_encoder_config,
    parse_risk_control_config,
    parse_signature_config,
)
from contextual_crc.risk_control.baselines import (
    run_global_crc_baseline,
    run_metadata_stratified_crc_baseline,
    run_oracle_context_crc_baseline,
)
from contextual_crc.risk_control.ct_crc import build_validated_contexts, evaluate_ctcrc_windows
from contextual_crc.risk_control.ct_crc_dkw import evaluate_dkw_target_feasible_windows
from contextual_crc.risk_control.cw_crc import evaluate_cwcrc_windows
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy
from contextual_crc.signatures.behavioral_signatures import (
    compute_signatures_for_windows,
    get_signature_feature_columns,
)
from contextual_crc.signatures.io import load_windows, save_signatures, save_windows
from contextual_crc.utils.paths import ensure_dir
from contextual_crc.utils.seed import set_seed


METHODS = (
    "global_crc",
    "metadata_stratified_crc",
    "original_ct_crc",
    "dkw_target_feasible_ct",
    "cw_crc",
    "oracle_context_crc",
)
DEPLOYABLE_METHODS = METHODS[:-1]


@dataclass(frozen=True)
class LeakageFreeSplit:
    assignments: pd.DataFrame
    deployable_calibration_windows: pd.DataFrame
    oracle_calibration_windows: pd.DataFrame
    evaluation_windows: pd.DataFrame


def make_leakage_free_window_split(
    windows: pd.DataFrame,
    calibration_contexts: Iterable[str],
    *,
    seed: int = 123,
) -> LeakageFreeSplit:
    """Split complete windows within every context, independently and deterministically."""
    required = {"window_id", "context_id", "example_indices"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"windows is missing required columns: {missing}")
    if windows["window_id"].duplicated().any():
        raise ValueError("window_id values must be unique")

    source_contexts = {str(value) for value in calibration_contexts}
    available_contexts = {str(value) for value in windows["context_id"].unique()}
    missing_sources = sorted(source_contexts - available_contexts)
    if missing_sources:
        raise ValueError(f"validated contexts have no windows: {missing_sources}")

    assignment_parts: list[pd.DataFrame] = []
    for context_value, context_windows in windows.groupby("context_id", sort=False, dropna=False):
        n_windows = len(context_windows)
        if n_windows < 2:
            raise ValueError(
                f"Context {context_value!r} has {n_windows} window(s); at least two are required "
                "for a meaningful leakage-free calibration/evaluation split."
            )
        n_calibration = max(1, math.floor(2 * n_windows / 3))
        n_calibration = min(n_calibration, n_windows - 1)
        context_seed = _stable_context_seed(seed, str(context_value))
        order = np.random.default_rng(context_seed).permutation(n_windows)
        calibration_positions = set(order[:n_calibration].tolist())
        assigned = context_windows.copy()
        assigned["oracle_partition"] = [
            "calibration" if position in calibration_positions else "evaluation"
            for position in range(n_windows)
        ]
        is_source = str(context_value) in source_contexts
        assigned["context_role"] = "validated_source" if is_source else "target_held_out"
        assigned["deployable_partition"] = [
            "calibration" if is_source and role == "calibration" else "not_calibration"
            for role in assigned["oracle_partition"]
        ]
        assignment_parts.append(assigned)

    assignments = pd.concat(assignment_parts, ignore_index=True)
    deployable = assignments[
        assignments["deployable_partition"] == "calibration"
    ].reset_index(drop=True)
    oracle = assignments[assignments["oracle_partition"] == "calibration"].reset_index(drop=True)
    evaluation = assignments[assignments["oracle_partition"] == "evaluation"].reset_index(drop=True)
    _assert_window_disjoint(deployable, evaluation, "deployable")
    _assert_window_disjoint(oracle, evaluation, "oracle")
    return LeakageFreeSplit(assignments, deployable, oracle, evaluation)


def subset_examples_and_remap_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract exactly the examples in complete windows and remap positional indices."""
    positions = sorted({int(index) for indices in windows["example_indices"] for index in indices})
    if not positions:
        raise ValueError("windows must contain at least one example")
    if positions[0] < 0 or positions[-1] >= len(examples):
        raise ValueError("window example index is outside the examples table")
    old_to_new = {old: new for new, old in enumerate(positions)}
    subset = examples.iloc[positions].reset_index(drop=True)
    remapped = windows.copy().reset_index(drop=True)
    remapped["example_indices"] = [
        [old_to_new[int(index)] for index in indices] for indices in remapped["example_indices"]
    ]
    return subset, remapped


def fit_calibration_only_encoder(
    signatures: pd.DataFrame,
    calibration_window_ids: Iterable[str],
    encoder_config: Any,
) -> tuple[ContextEncoder, pd.DataFrame, dict[str, Any]]:
    """Fit StandardScaler/PCA on named calibration signatures and transform all rows."""
    fit_ids = {str(value) for value in calibration_window_ids}
    signature_ids = signatures["window_id"].astype(str)
    fit_signatures = signatures[signature_ids.isin(fit_ids)].reset_index(drop=True)
    observed_ids = set(fit_signatures["window_id"].astype(str))
    if observed_ids != fit_ids:
        raise ValueError(f"Missing calibration signatures: {sorted(fit_ids - observed_ids)}")
    feature_columns = get_signature_feature_columns(signatures, include_diagnostics=False)
    if not feature_columns or any(column.startswith("diag_") for column in feature_columns):
        raise ValueError("encoder inputs must be non-empty and exclude diag_* columns")

    encoder = ContextEncoder(encoder_config)
    encoder.fit(fit_signatures, feature_columns)
    embeddings = encoder.transform(signatures)
    embedding_frame = build_context_embedding_frame(signatures, embeddings)
    audit = {
        "fit_window_ids": sorted(fit_ids),
        "fit_row_count": len(fit_signatures),
        "transform_row_count": len(signatures),
        "feature_columns": feature_columns,
        "diagnostic_columns_excluded": True,
        "scaler_fit_scope": "deployable calibration-window signatures only",
        "pca_fit_scope": "deployable calibration-window signatures only",
    }
    return encoder, embedding_frame, audit


def run_leakage_free_evaluation(
    config: dict[str, Any],
    *,
    seed: int = 123,
    library_alpha: float = 0.05,
    overwrite: bool = False,
    output_dir: str | Path = "outputs/runs/leakage_free_main",
    artifact_tables_dir: str | Path = "outputs/artifacts/tables",
) -> Path:
    """Run a leakage-free evaluation using existing prediction caches."""
    set_seed(seed)
    output_path = Path(output_dir)
    main_path = output_path / "main_results.csv"
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_path}")
    ensure_dir(output_path)

    run_dirs = [Path(value) for value in config.get("paths", {}).get("run_dirs", [])]
    if not run_dirs:
        raise ValueError("artifact config must define paths.run_dirs")
    started = datetime.now(timezone.utc)
    audit_frames: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    frontier_rows: list[pd.DataFrame] = []
    smoothness_rows: list[dict[str, Any]] = []
    dataset_manifests: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        evaluated = evaluate_leakage_free_run(
            run_dir,
            seed=seed,
            library_alpha=library_alpha,
            output_root=output_path,
        )
        audit_frames.append(evaluated["audit"])
        result_rows.extend(evaluated["results"])
        frontier_rows.extend(evaluated["frontiers"])
        smoothness_rows.append(evaluated["smoothness"])
        dataset_manifests.append(evaluated["manifest"])

    audit = pd.concat(audit_frames, ignore_index=True)
    results = pd.DataFrame(result_rows)
    frontiers = pd.concat(frontier_rows, ignore_index=True)
    matched = _build_matched_frontier(results, frontiers)
    smoothness = pd.DataFrame(smoothness_rows)
    audit.to_csv(output_path / "leakage_audit.csv", index=False)
    results.to_csv(main_path, index=False)
    matched.to_csv(output_path / "matched_frontier.csv", index=False)
    smoothness.to_csv(output_path / "smoothness.csv", index=False)
    frontiers.to_csv(output_path / "risk_coverage_frontiers.csv", index=False)

    from contextual_crc.artifacts.leakage_free import write_leakage_free_artifacts

    artifact_paths = write_leakage_free_artifacts(
        artifact_tables_dir,
        main_results_csv=main_path,
        matched_frontier_csv=output_path / "matched_frontier.csv",
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "workflow": "leakage_free_evaluation",
        "workflow_version": 1,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": (finished - started).total_seconds(),
        "seed": seed,
        "split_rule": (
            "Within each context, seeded shuffle; n_cal=max(1,floor(2*n_windows/3)) "
            "capped at n_windows-1. Target calibration partitions are oracle-only."
        ),
        "library_alpha": library_alpha,
        "uses_cached_predictions": True,
        "reruns_model_inference": False,
        "rebuilds_feature_cache": False,
        "retunes_hyperparameters": False,
        "common_evaluation_windows_across_all_methods": True,
        "evaluation_labels_used_for": ["final metrics", "post-hoc frontiers", "smoothness audit"],
        "evaluation_labels_excluded_from": [
            "scaler fit",
            "PCA fit",
            "validated prototypes",
            "source calibration",
            "L_hat estimation",
            "deployment policy selection",
        ],
        "datasets": dataset_manifests,
        "artifact_paths": artifact_paths,
        "outputs": {
            "leakage_audit": str(output_path / "leakage_audit.csv"),
            "main_results": str(main_path),
            "matched_frontier": str(output_path / "matched_frontier.csv"),
            "smoothness": str(output_path / "smoothness.csv"),
            "frontiers": str(output_path / "risk_coverage_frontiers.csv"),
        },
    }
    _write_json(manifest, output_path / "run_manifest.json")
    return main_path


def evaluate_leakage_free_run(
    run_dir: str | Path,
    *,
    seed: int,
    library_alpha: float,
    output_root: Path,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    old_manifest = _read_json(run_path / "results" / "run_manifest.json")
    resolved = load_config(run_path / "configs" / "resolved_config.yaml")
    cache_dir = Path(old_manifest["cache_dir"])
    if not cache_exists(cache_dir):
        raise FileNotFoundError(f"Prediction cache is missing: {cache_dir}")
    examples, arrays, cache_metadata = load_prediction_cache(cache_dir)
    if "example_id" not in examples.columns or examples["example_id"].duplicated().any():
        raise ValueError("Leakage-free evaluation requires unique stable example_id values in every cache")
    old_windows = load_windows(run_path / "intermediate" / "windows.parquet")
    dataset = str(
        old_manifest.get("dataset")
        or cache_metadata.get("dataset")
        or old_manifest.get("schema_dataset")
        or run_path.name
    )
    safe_dataset = _safe_name(dataset)
    dataset_dir = ensure_dir(output_root / safe_dataset)
    intermediate_dir = ensure_dir(dataset_dir / "intermediate")
    results_dir = ensure_dir(dataset_dir / "results")
    calibration_contexts = [str(value) for value in old_manifest.get("calibration_contexts", [])]
    if not calibration_contexts:
        raise ValueError(f"No resolved calibration contexts in {run_path}")

    split = make_leakage_free_window_split(old_windows, calibration_contexts, seed=seed)
    deploy_examples, deploy_windows = subset_examples_and_remap_windows(
        examples, split.deployable_calibration_windows
    )
    oracle_examples, oracle_windows = subset_examples_and_remap_windows(
        examples, split.oracle_calibration_windows
    )
    evaluation_examples, _ = subset_examples_and_remap_windows(examples, split.evaluation_windows)
    deploy_overlap = _overlap_counts(
        deploy_examples, deploy_windows, evaluation_examples, split.evaluation_windows
    )
    oracle_overlap = _overlap_counts(
        oracle_examples, oracle_windows, evaluation_examples, split.evaluation_windows
    )
    if any(deploy_overlap.values()) or any(oracle_overlap.values()):
        raise RuntimeError("leakage-free split contains overlapping IDs")

    signature_config = parse_signature_config(resolved.get("signatures", {}))
    signatures = compute_signatures_for_windows(examples, arrays, old_windows, signature_config)
    encoder_config = parse_encoder_config(resolved.get("encoder", {}))
    encoder, embeddings, encoder_audit = fit_calibration_only_encoder(
        signatures,
        split.deployable_calibration_windows["window_id"],
        encoder_config,
    )
    deploy_ids = set(split.deployable_calibration_windows["window_id"].astype(str))
    eval_ids = set(split.evaluation_windows["window_id"].astype(str))
    deploy_embeddings = embeddings[embeddings["window_id"].astype(str).isin(deploy_ids)].reset_index(drop=True)
    eval_embeddings = embeddings[embeddings["window_id"].astype(str).isin(eval_ids)].reset_index(drop=True)
    if set(encoder_audit["fit_window_ids"]) & eval_ids:
        raise RuntimeError("encoder fit rows include evaluation windows")

    assignment_windows = split.assignments.copy()
    assignment_windows["common_evaluation"] = assignment_windows["oracle_partition"].eq("evaluation")
    save_windows(assignment_windows, intermediate_dir / "windows.parquet")
    save_signatures(signatures, intermediate_dir / "signatures.parquet")
    embeddings.to_parquet(intermediate_dir / "embeddings.parquet", engine="pyarrow", index=False)
    save_context_encoder(encoder, intermediate_dir / "context_encoder.pkl")
    split.assignments.drop(columns=["example_indices"], errors="ignore").to_csv(
        intermediate_dir / "window_assignments.csv", index=False
    )

    risk_config = parse_risk_control_config(resolved.get("risk_control", {}))
    global_result = run_global_crc_baseline(deploy_examples, evaluation_examples, risk_config)
    metadata_result = run_metadata_stratified_crc_baseline(
        deploy_examples, evaluation_examples, risk_config, stratify_col="context_id"
    )
    oracle_result = run_oracle_context_crc_baseline(
        oracle_examples, evaluation_examples, risk_config, context_col="context_id"
    )
    baseline_windows = {
        "global_crc": _evaluate_baseline_windows(
            examples, split.evaluation_windows, global_result, "global_crc", risk_config
        ),
        "metadata_stratified_crc": _evaluate_baseline_windows(
            examples, split.evaluation_windows, metadata_result, "metadata_stratified_crc", risk_config
        ),
        "oracle_context_crc": _evaluate_baseline_windows(
            examples, split.evaluation_windows, oracle_result, "oracle_context_crc", risk_config
        ),
    }

    raw_ct = resolved.get("ct_crc", {})
    base_ct_config = parse_ctcrc_config(raw_ct, risk_config)
    original_ct = build_validated_contexts(deploy_examples, deploy_embeddings, base_ct_config)
    if raw_ct.get("l_hat") == "auto":
        original_l_hat = _estimate_ct_l_hat(original_ct, raw_ct)
        original_ct = build_validated_contexts(
            deploy_examples,
            deploy_embeddings,
            replace(base_ct_config, l_hat=original_l_hat),
        )
    else:
        original_l_hat = float(base_ct_config.l_hat)
    original_windows = evaluate_ctcrc_windows(
        examples, split.evaluation_windows, eval_embeddings, original_ct
    )

    dkw_result, dkw_context_audit = _build_dkw_with_auto_l_hat(
        deploy_examples,
        deploy_embeddings,
        base_ct_config,
        raw_ct_config=raw_ct,
        library_alpha=library_alpha,
    )
    dkw_windows = evaluate_dkw_target_feasible_windows(
        examples, split.evaluation_windows, eval_embeddings, dkw_result
    )
    if not bool((dkw_windows["transferred_upper_bound"] <= risk_config.risk_budget + 1e-12).all()):
        raise RuntimeError("corrected CT selected an infeasible deployment operating point")
    forced = dkw_windows["forced_accept_none"].astype(bool)
    if bool((forced & (pd.to_numeric(dkw_windows["coverage"]) > 0.0)).any()):
        raise RuntimeError("accept-none fallback has positive coverage")

    cw_config = parse_cwcrc_config(resolved.get("cw_crc", {}), risk_config)
    cw_windows = evaluate_cwcrc_windows(
        examples,
        deploy_examples,
        deploy_windows,
        deploy_embeddings,
        split.evaluation_windows,
        eval_embeddings,
        cw_config,
    )
    method_windows = {
        **baseline_windows,
        "original_ct_crc": original_windows,
        "dkw_target_feasible_ct": dkw_windows,
        "cw_crc": cw_windows,
    }
    bounds = {
        "global_crc": "upper_bound",
        "metadata_stratified_crc": "upper_bound",
        "original_ct_crc": "transferred_upper_bound",
        "dkw_target_feasible_ct": "transferred_upper_bound",
        "cw_crc": "weighted_risk_upper_bound",
        "oracle_context_crc": "upper_bound",
    }
    for method, frame in method_windows.items():
        frame.to_csv(results_dir / f"{method}_windows.csv", index=False)
    dkw_context_audit.to_csv(results_dir / "dkw_source_contexts.csv", index=False)

    results: list[dict[str, Any]] = []
    common_eval_window_ids = set(split.evaluation_windows["window_id"].astype(str))
    for method in METHODS:
        frame = method_windows[method]
        if set(frame["window_id"].astype(str)) != common_eval_window_ids:
            raise RuntimeError(f"{method} did not use the common evaluation windows")
        method_cal_examples = oracle_examples if method == "oracle_context_crc" else deploy_examples
        method_cal_windows = oracle_windows if method == "oracle_context_crc" else deploy_windows
        overlap = _overlap_counts(
            method_cal_examples,
            method_cal_windows,
            evaluation_examples,
            split.evaluation_windows,
        )
        row = _summarize_method(
            dataset,
            run_path.name,
            method,
            frame,
            upper_bound_col=bounds[method],
            risk_budget=risk_config.risk_budget,
            n_calibration_examples=len(method_cal_examples),
            n_calibration_windows=len(method_cal_windows),
            n_evaluation_examples=len(evaluation_examples),
            overlap=overlap,
        )
        if method == "dkw_target_feasible_ct":
            row.update(
                {
                    "resolved_l_hat": dkw_result.config.l_hat,
                    "source_certificate_type": "two_sided_dkw_uniform",
                    "per_context_delta": dkw_result.per_context_delta,
                    "library_alpha": dkw_result.library_alpha,
                }
            )
        elif method == "original_ct_crc":
            row["resolved_l_hat"] = original_l_hat
            row["source_certificate_type"] = risk_config.bound
        elif method == "cw_crc":
            row.update(
                {
                    "cw_target_ess": cw_config.target_effective_sample_size,
                    "cw_bound_margin": cw_config.bound_margin,
                    "cw_bound_multiplier": cw_config.bound_multiplier,
                }
            )
        results.append(row)

    frontiers = [
        _method_frontier(
            dataset,
            method,
            examples,
            split.evaluation_windows,
            method_windows[method],
            risk_config.score_column,
        )
        for method in (
            "global_crc",
            "metadata_stratified_crc",
            "dkw_target_feasible_ct",
            "cw_crc",
            "oracle_context_crc",
        )
    ]
    smoothness = _smoothness_summary(dataset, signatures, eval_embeddings)
    audit = _build_leakage_audit(
        dataset,
        examples,
        old_windows,
        calibration_contexts,
        deploy_examples,
        deploy_windows,
        oracle_examples,
        oracle_windows,
        evaluation_examples,
        split.evaluation_windows,
    )
    dataset_manifest = {
        "dataset": dataset,
        "source_run_dir": str(run_path),
        "prediction_cache": str(cache_dir),
        "prediction_cache_examples_sha256": _sha256(cache_dir / "examples.parquet"),
        "calibration_contexts": calibration_contexts,
        "target_contexts": sorted(
            set(examples["context_id"].astype(str).unique()) - set(calibration_contexts)
        ),
        "deployable_calibration_examples": len(deploy_examples),
        "deployable_calibration_windows": len(deploy_windows),
        "oracle_calibration_examples": len(oracle_examples),
        "oracle_calibration_windows": len(oracle_windows),
        "common_evaluation_examples": len(evaluation_examples),
        "common_evaluation_windows": len(split.evaluation_windows),
        "deployable_overlap": deploy_overlap,
        "oracle_overlap": oracle_overlap,
        "encoder": encoder_audit,
        "original_ct_resolved_l_hat": original_l_hat,
        "dkw_resolved_l_hat": dkw_result.config.l_hat,
        "l_hat_fit_scope": "validated deployable calibration-side contexts only",
        "common_evaluation_window_ids_sha256": _hash_values(common_eval_window_ids),
        "dataset_output_dir": str(dataset_dir),
    }
    _write_json(dataset_manifest, dataset_dir / "run_manifest.json")
    return {
        "audit": audit,
        "results": results,
        "frontiers": frontiers,
        "smoothness": smoothness,
        "manifest": dataset_manifest,
    }


def _evaluate_baseline_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    baseline_result: dict[str, Any],
    method: str,
    risk_config: Any,
) -> pd.DataFrame:
    if method == "global_crc":
        threshold_by_context: dict[str, float] = {}
        bound_by_context: dict[str, float] = {}
        fallback_threshold = float(baseline_result["result"]["threshold"])
        fallback_bound = float(baseline_result["result"]["risk_upper_bound"])
    else:
        table_key = "by_stratum" if method == "metadata_stratified_crc" else "by_context"
        table = baseline_result[table_key]
        threshold_by_context = dict(
            zip(table["context_id"].astype(str), pd.to_numeric(table["threshold"]), strict=True)
        )
        bound_by_context = dict(
            zip(table["context_id"].astype(str), pd.to_numeric(table["risk_upper_bound"]), strict=True)
        )
        fallback_threshold = float("nan")
        fallback_bound = float("nan")

    rows: list[dict[str, Any]] = []
    for _, window in windows.iterrows():
        context_id = str(window["context_id"])
        threshold = float(threshold_by_context.get(context_id, fallback_threshold))
        upper_bound = float(bound_by_context.get(context_id, fallback_bound))
        window_examples = examples.iloc[[int(value) for value in window["example_indices"]]]
        accepted = apply_threshold_policy(
            window_examples, risk_config.score_column, threshold, risk_config.accept_if
        )
        losses = compute_losses(window_examples, accepted, risk_config.loss)
        summary = summarize_policy_performance(window_examples, accepted, losses)
        rows.append(
            {
                "window_id": window["window_id"],
                "context_id": context_id,
                "threshold": threshold,
                "upper_bound": upper_bound,
                "realized_risk": summary["mean_loss"],
                "mean_loss": summary["mean_loss"],
                "coverage": summary["coverage"],
                "n_examples": summary["n"],
                "num_accepted": summary["num_accepted"],
                "num_deferred": summary["num_deferred"],
                "accuracy_accepted": summary["accuracy_accepted"],
                "error_rate_accepted": summary["error_rate_accepted"],
            }
        )
    return pd.DataFrame(rows)


def _summarize_method(
    dataset: str,
    run_name: str,
    method: str,
    windows: pd.DataFrame,
    *,
    upper_bound_col: str,
    risk_budget: float,
    n_calibration_examples: int,
    n_calibration_windows: int,
    n_evaluation_examples: int,
    overlap: dict[str, int],
) -> dict[str, Any]:
    risks = pd.to_numeric(windows["realized_risk"], errors="coerce")
    coverage = pd.to_numeric(windows["coverage"], errors="coerce")
    bounds = pd.to_numeric(windows[upper_bound_col], errors="coerce")
    thresholds = pd.to_numeric(windows["threshold"], errors="coerce")
    row: dict[str, Any] = {
        "dataset": dataset,
        "run_name": run_name,
        "method": method,
        "risk": float(risks.mean()),
        "coverage": float(coverage.mean()),
        "bound_underestimation": float((bounds + 1e-12 < risks).mean()),
        "mean_bound_gap": float((bounds - risks).mean()),
        "mean_upper_bound": float(bounds.mean()),
        "n_calibration_examples": n_calibration_examples,
        "n_calibration_windows": n_calibration_windows,
        "n_evaluation_examples": n_evaluation_examples,
        "n_evaluation_windows": len(windows),
        "calibration_evaluation_example_overlap": overlap["example_overlap"],
        "calibration_evaluation_window_overlap": overlap["window_overlap"],
        "mean_selected_threshold": float(thresholds.mean()),
        "median_selected_threshold": float(thresholds.median()),
        "risk_budget": risk_budget,
    }
    if "nontrivial_feasible" in windows:
        row["nontrivial_feasible_fraction"] = float(windows["nontrivial_feasible"].astype(bool).mean())
    if "forced_accept_none" in windows:
        row["accept_none_fraction"] = float(windows["forced_accept_none"].astype(bool).mean())
    if method == "dkw_target_feasible_ct":
        row["mean_transferred_bound"] = float(bounds.mean())
        row["maximum_transferred_bound"] = float(bounds.max())
    if "effective_sample_size" in windows:
        ess = pd.to_numeric(windows["effective_sample_size"], errors="coerce")
        row["mean_effective_sample_size"] = float(ess.mean())
        row["median_effective_sample_size"] = float(ess.median())
    return row


def _method_frontier(
    dataset: str,
    method: str,
    examples: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
    method_windows: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    thresholds = method_windows.set_index("window_id")["threshold"]
    margins: list[float] = []
    incorrect: list[bool] = []
    for _, window in evaluation_windows.iterrows():
        threshold = float(thresholds.loc[window["window_id"]])
        selected = examples.iloc[[int(value) for value in window["example_indices"]]]
        margins.extend(pd.to_numeric(selected[score_column]).to_numpy(dtype=float) - threshold)
        incorrect.extend((selected["pred"].astype(int) != selected["label"].astype(int)).tolist())
    curve = compute_risk_coverage_curve(margins, incorrect, selected_threshold=0.0)
    curve.insert(0, "method", method)
    curve.insert(0, "dataset", dataset)
    curve["post_hoc_evaluation"] = True
    curve["frontier_definition"] = "additive offset around deployment-selected per-window threshold"
    return curve


def _build_matched_frontier(results: pd.DataFrame, frontiers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in results.groupby("dataset", sort=False):
        global_rows = group[group["method"] == "global_crc"]
        if global_rows.empty:
            continue
        global_coverage = float(global_rows.iloc[0]["coverage"])
        for method in (
            "global_crc",
            "metadata_stratified_crc",
            "dkw_target_feasible_ct",
            "cw_crc",
            "oracle_context_crc",
        ):
            curve = frontiers[(frontiers["dataset"] == dataset) & (frontiers["method"] == method)]
            coverage_match = matched_coverage_point(curve, global_coverage)
            risk_match = matched_risk_point(curve, 0.10)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "global_coverage_target": global_coverage,
                    "risk_at_global_coverage": _match(coverage_match, "risk_at_coverage"),
                    "achieved_coverage_for_match": _match(coverage_match, "achieved_coverage"),
                    "coverage_at_tau_0_10": _match(risk_match, "coverage_at_risk"),
                    "achieved_risk_for_match": _match(risk_match, "achieved_risk"),
                    "post_hoc_evaluation": True,
                    "uses_evaluation_labels": True,
                }
            )
    return pd.DataFrame(rows)


def _smoothness_summary(
    dataset: str,
    signatures: pd.DataFrame,
    evaluation_embeddings: pd.DataFrame,
) -> dict[str, Any]:
    diagnostic = signatures[[
        column
        for column in ("window_id", "diag_error_rate", "diag_accuracy")
        if column in signatures.columns
    ]].copy()
    if "diag_error_rate" not in diagnostic and "diag_accuracy" in diagnostic:
        diagnostic["diag_error_rate"] = 1.0 - pd.to_numeric(diagnostic["diag_accuracy"])
    if "diag_error_rate" not in diagnostic:
        raise ValueError("smoothness audit requires diagnostic realized error rates")
    clean = evaluation_embeddings.merge(
        diagnostic[["window_id", "diag_error_rate"]], on="window_id", how="inner"
    ).rename(columns={"diag_error_rate": "realized_risk"})
    pairs = context_smoothness_pairwise_frame(
        clean, value_col="realized_risk", dataset=dataset
    )
    return {
        "dataset": dataset,
        "spearman_rho": float(pairs["distance"].corr(pairs["risk_difference"], method="spearman")),
        "pearson_r": float(pairs["distance"].corr(pairs["risk_difference"], method="pearson")),
        "n_pairs": len(pairs),
        "realized_risk_definition": "evaluation-window base-classifier error rate (accept-all)",
        "windows_scope": "common evaluation windows only",
        "labels_used_for_fit": False,
    }


def _build_leakage_audit(
    dataset: str,
    examples: pd.DataFrame,
    old_windows: pd.DataFrame,
    calibration_contexts: list[str],
    deploy_examples: pd.DataFrame,
    deploy_windows: pd.DataFrame,
    oracle_examples: pd.DataFrame,
    oracle_windows: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
) -> pd.DataFrame:
    source_old_windows = old_windows[
        old_windows["context_id"].astype(str).isin(calibration_contexts)
    ].reset_index(drop=True)
    source_old_examples, _ = subset_examples_and_remap_windows(examples, source_old_windows)
    all_old_examples, _ = subset_examples_and_remap_windows(examples, old_windows)
    rows: list[dict[str, Any]] = []
    for method in DEPLOYABLE_METHODS:
        old_overlap = _overlap_counts(
            source_old_examples, source_old_windows, all_old_examples, old_windows
        )
        rows.append(
            _audit_row(
                dataset,
                "old",
                method,
                len(source_old_examples),
                len(source_old_windows),
                len(all_old_examples),
                len(old_windows),
                old_overlap,
                "Source-context calibration used every source window; reported evaluation reused all windows.",
            )
        )
    old_oracle_overlap = _overlap_counts(
        source_old_examples, source_old_windows, source_old_examples, source_old_windows
    )
    rows.append(
        _audit_row(
            dataset,
            "old",
            "oracle_context_crc",
            len(source_old_examples),
            len(source_old_windows),
            len(source_old_examples),
            len(source_old_windows),
            old_oracle_overlap,
            "Oracle calibrated and evaluated the same labeled source-context examples; target contexts were omitted.",
        )
    )
    deploy_overlap = _overlap_counts(
        deploy_examples, deploy_windows, evaluation_examples, evaluation_windows
    )
    oracle_overlap = _overlap_counts(
        oracle_examples, oracle_windows, evaluation_examples, evaluation_windows
    )
    for method in DEPLOYABLE_METHODS:
        rows.append(
            _audit_row(
                dataset,
                "leakage_free",
                method,
                len(deploy_examples),
                len(deploy_windows),
                len(evaluation_examples),
                len(evaluation_windows),
                deploy_overlap,
                "Whole-window source calibration; common disjoint evaluation windows.",
            )
        )
    rows.append(
        _audit_row(
            dataset,
            "leakage_free",
            "oracle_context_crc",
            len(oracle_examples),
            len(oracle_windows),
            len(evaluation_examples),
            len(evaluation_windows),
            oracle_overlap,
            "Oracle-only within-context calibration partitions; common disjoint evaluation windows.",
        )
    )
    return pd.DataFrame(rows)


def _audit_row(
    dataset: str,
    protocol: str,
    method: str,
    n_cal_examples: int,
    n_cal_windows: int,
    n_eval_examples: int,
    n_eval_windows: int,
    overlap: dict[str, int],
    cause: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "protocol": protocol,
        "method": method,
        "n_calibration_examples": n_cal_examples,
        "n_calibration_windows": n_cal_windows,
        "n_evaluation_examples": n_eval_examples,
        "n_evaluation_windows": n_eval_windows,
        "shared_example_ids": overlap["example_overlap"],
        "shared_window_ids": overlap["window_overlap"],
        "context_overlap_allowed": True,
        "root_cause_or_resolution": cause,
    }


def _overlap_counts(
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    evaluation_examples: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
) -> dict[str, int]:
    cal_examples = set(calibration_examples["example_id"].astype(str))
    eval_examples = set(evaluation_examples["example_id"].astype(str))
    cal_windows = set(calibration_windows["window_id"].astype(str))
    eval_windows = set(evaluation_windows["window_id"].astype(str))
    return {
        "example_overlap": len(cal_examples & eval_examples),
        "window_overlap": len(cal_windows & eval_windows),
    }


def _assert_window_disjoint(calibration: pd.DataFrame, evaluation: pd.DataFrame, name: str) -> None:
    overlap = set(calibration["window_id"].astype(str)) & set(evaluation["window_id"].astype(str))
    if overlap:
        raise RuntimeError(f"{name} calibration/evaluation window overlap: {sorted(overlap)}")
    calibration_examples = {
        int(index) for values in calibration["example_indices"] for index in values
    }
    evaluation_examples = {int(index) for values in evaluation["example_indices"] for index in values}
    if calibration_examples & evaluation_examples:
        raise RuntimeError(f"{name} calibration/evaluation positional example overlap")


def _stable_context_seed(seed: int, context_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{context_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _hash_values(values: Iterable[str]) -> str:
    joined = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _match(value: dict[str, Any] | None, key: str) -> float:
    return float("nan") if value is None or key not in value else float(value[key])


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
