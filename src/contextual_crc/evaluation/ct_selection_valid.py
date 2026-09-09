from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contextual_crc.cache.io import cache_exists
from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.metrics import safe_mean, summarize_window_results
from contextual_crc.artifacts.risk_coverage import matched_coverage_point, matched_risk_point
from contextual_crc.evaluation.synthetic_experiment import parse_ctcrc_config, parse_risk_control_config
from contextual_crc.risk_control.ct_crc import estimate_lipschitz_from_contexts
from contextual_crc.risk_control.ct_crc_dkw import (
    DKWCTResult,
    build_dkw_validated_contexts,
    dkw_contexts_to_frame,
    empirical_accepted_error_risks,
    dkw_uniform_certificates,
    evaluate_dkw_source_windows,
    evaluate_dkw_target_feasible_windows,
)
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


RESULTS_FILENAME = "ct_selection_valid_results.csv"
CONTEXTS_FILENAME = "ct_selection_valid_contexts.csv"


def run_ct_selection_valid(
    config: dict[str, Any],
    *,
    library_alpha: float = 0.05,
    overwrite: bool = False,
    output_dir: str | Path = "outputs/runs/ct_selection_valid",
    simultaneous_results: str | Path = (
        "outputs/runs/ct_simultaneous_validity/ct_simultaneous_validity.csv"
    ),
    frontier_results: str | Path = "outputs/artifacts/tables/risk_coverage_frontiers.csv",
) -> Path:
    """Audit selection validity using existing prediction caches and saved embeddings."""
    output_path = Path(output_dir)
    results_path = output_path / RESULTS_FILENAME
    if results_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {results_path}")

    run_dirs = [Path(path) for path in config.get("paths", {}).get("run_dirs", [])]
    if not run_dirs:
        raise ValueError("artifact config must define paths.run_dirs")
    simultaneous_path = Path(simultaneous_results)
    simultaneous_summary = pd.read_csv(simultaneous_path) if simultaneous_path.is_file() else pd.DataFrame()

    ensure_dir(output_path)
    windows_dir = ensure_dir(output_path / "windows")
    result_rows: list[dict[str, Any]] = []
    context_frames: list[pd.DataFrame] = []
    cache_inputs: list[dict[str, str]] = []
    started_at = datetime.now(timezone.utc)

    for run_dir in run_dirs:
        rows, context_audit, dkw_windows, inputs = evaluate_selection_valid_run(
            run_dir,
            library_alpha=library_alpha,
            simultaneous_summary=simultaneous_summary,
            simultaneous_root=simultaneous_path.parent,
        )
        result_rows.extend(rows)
        context_frames.append(context_audit)
        cache_inputs.append(inputs)
        dataset = str(rows[0]["dataset"])
        safe_dataset = dataset.replace("/", "_").replace("\\", "_")
        for variant, frame in dkw_windows.items():
            frame.to_csv(windows_dir / f"{safe_dataset}__{variant}.csv", index=False)

    results = pd.DataFrame(result_rows)
    frontier_path = Path(frontier_results)
    if frontier_path.is_file():
        results = _add_matched_baseline_comparisons(results, pd.read_csv(frontier_path))
    results.to_csv(results_path, index=False)
    pd.concat(context_frames, ignore_index=True).to_csv(
        output_path / CONTEXTS_FILENAME,
        index=False,
    )

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime_seconds": (finished_at - started_at).total_seconds(),
        "library_alpha": library_alpha,
        "run_dirs": [str(path) for path in run_dirs],
        "simultaneous_results": str(simultaneous_path),
        "frontier_results": str(frontier_path) if frontier_path.is_file() else None,
        "variants": results["variant"].drop_duplicates().tolist(),
        "uses_cached_predictions": True,
        "uses_saved_embeddings": True,
        "reruns_model_inference": False,
        "retunes_transfer_parameters": False,
        "candidate_construction": (
            "finite unique current-window and nearest-source calibration confidence scores, "
            "plus finite accept-all and accept-none extremes"
        ),
        "selection_inputs": "deployment confidence scores and saved embedding only; no deployment labels",
        "accept_none_bound": (
            "zero for the explicit full-defer current-window fallback; raw DKW transfer is retained "
            "separately in window outputs"
        ),
        "validity_scope": {
            "source_envelope": "two-sided DKW/Massart, uniform over all real thresholds",
            "context_simultaneity": "union bound with delta_j=alpha/m",
            "transfer": "conditional on the existing local smoothness assumption and L_hat",
            "adaptive_theta_requires_uniform_A3": True,
        },
        "cache_inputs": cache_inputs,
        "results_csv": str(results_path),
        "contexts_csv": str(output_path / CONTEXTS_FILENAME),
    }
    _write_json(manifest, output_path / "run_manifest.json")
    return results_path


def evaluate_selection_valid_run(
    run_dir: str | Path,
    *,
    library_alpha: float,
    simultaneous_summary: pd.DataFrame,
    simultaneous_root: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """Evaluate one dataset from the exact saved prediction and embedding files."""
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "results" / "run_manifest.json")
    resolved_config = load_config(run_path / "configs" / "resolved_config.yaml")
    cache_dir = Path(manifest["cache_dir"])
    if not cache_exists(cache_dir):
        raise FileNotFoundError(f"Prediction cache is missing or incomplete: {cache_dir}")

    examples_path = cache_dir / "examples.parquet"
    embeddings_path = run_path / "intermediate" / "embeddings.parquet"
    windows_path = run_path / "intermediate" / "windows.parquet"
    examples = pd.read_parquet(examples_path, engine="pyarrow")
    embeddings = pd.read_parquet(embeddings_path, engine="pyarrow")
    windows = load_windows(windows_path)

    calibration_contexts = [str(value) for value in manifest.get("calibration_contexts", [])]
    evaluation_contexts = [str(value) for value in manifest.get("evaluation_contexts", [])]
    calibration_examples = examples[examples["context_id"].isin(calibration_contexts)].reset_index(drop=True)
    calibration_embeddings = embeddings[
        embeddings["context_id"].isin(calibration_contexts)
    ].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_ids = set(evaluation_windows["window_id"])
    evaluation_embeddings = embeddings[embeddings["window_id"].isin(evaluation_ids)].reset_index(drop=True)

    risk_config = parse_risk_control_config(resolved_config.get("risk_control", {}))
    base_ct_config = parse_ctcrc_config(resolved_config.get("ct_crc", {}), risk_config)
    dkw_result, context_audit = _build_dkw_with_auto_l_hat(
        calibration_examples,
        calibration_embeddings,
        base_ct_config,
        raw_ct_config=resolved_config.get("ct_crc", {}),
        library_alpha=library_alpha,
    )
    dataset = _dataset_name(manifest, run_path.name)
    context_audit.insert(0, "run_name", run_path.name)
    context_audit.insert(0, "dataset", dataset)
    context_audit["l_hat"] = dkw_result.config.l_hat
    _assert_uniform_context_sanity(dkw_result)

    rows: list[dict[str, Any]] = []
    original_windows = pd.read_csv(run_path / "results" / "ct_crc_windows.csv")
    rows.append(
        _summarize_variant(
            dataset,
            run_path.name,
            "original_ct",
            original_windows,
            risk_budget=risk_config.risk_budget,
            library_alpha=np.nan,
            per_context_delta=risk_config.delta,
            certificate_type="pointwise_empirical_bernstein",
            notes="Saved original CT operating point; transferred feasibility was not enforced.",
        )
    )

    simultaneous_variants = (
        (
            "ct_crc_library_bonferroni",
            "library_bonferroni_ct",
            "context_bonferroni_empirical_bernstein_pointwise_threshold",
        ),
        (
            "ct_crc_fully_simultaneous",
            "fully_threshold_bonferroni_ct",
            "context_and_threshold_bonferroni_empirical_bernstein",
        ),
    )
    for old_name, new_name, certificate_type in simultaneous_variants:
        summary_match = simultaneous_summary[
            (simultaneous_summary.get("run_name") == run_path.name)
            & (simultaneous_summary.get("variant") == old_name)
        ] if not simultaneous_summary.empty else pd.DataFrame()
        window_path = simultaneous_root / "windows" / f"{dataset}__{old_name}.csv"
        if summary_match.empty or not window_path.is_file():
            continue
        source_row = summary_match.iloc[0]
        prior_windows = pd.read_csv(window_path)
        rows.append(
            _summarize_variant(
                dataset,
                run_path.name,
                new_name,
                prior_windows,
                risk_budget=risk_config.risk_budget,
                library_alpha=float(source_row["overall_library_delta"]),
                per_context_delta=float(source_row["per_context_delta"]),
                certificate_type=certificate_type,
                notes="Read from the simultaneous-validity run; transferred feasibility was not enforced.",
            )
        )

    dkw_source = evaluate_dkw_source_windows(
        examples,
        evaluation_windows,
        evaluation_embeddings,
        dkw_result,
    )
    dkw_target = evaluate_dkw_target_feasible_windows(
        examples,
        evaluation_windows,
        evaluation_embeddings,
        dkw_result,
    )
    if not bool((dkw_target["transferred_upper_bound"] <= risk_config.risk_budget + 1e-12).all()):
        raise RuntimeError("target-feasible evaluation produced an infeasible selected bound")

    dkw_windows = {
        "dkw_uniform_source_ct": _label_windows(dkw_source, dataset, "dkw_uniform_source_ct"),
        "dkw_target_feasible_ct": _label_windows(dkw_target, dataset, "dkw_target_feasible_ct"),
    }
    rows.extend(
        [
            _summarize_variant(
                dataset,
                run_path.name,
                "dkw_uniform_source_ct",
                dkw_source,
                risk_budget=risk_config.risk_budget,
                library_alpha=library_alpha,
                per_context_delta=dkw_result.per_context_delta,
                certificate_type="two_sided_dkw_uniform",
                notes=(
                    "Diagnostic: uniform source envelope, but old source-selected threshold is "
                    "transferred without enforcing target feasibility."
                ),
            ),
            _summarize_variant(
                dataset,
                run_path.name,
                "dkw_target_feasible_ct",
                dkw_target,
                risk_budget=risk_config.risk_budget,
                library_alpha=library_alpha,
                per_context_delta=dkw_result.per_context_delta,
                certificate_type="two_sided_dkw_uniform",
                notes=(
                    "Primary: score-only per-window maximum-coverage selection subject to the "
                    "transferred bound; explicit full-defer fallback."
                ),
            ),
        ]
    )

    inputs = {
        "run_name": run_path.name,
        "examples_path": str(examples_path),
        "examples_sha256": _sha256(examples_path),
        "windows_path": str(windows_path),
        "windows_sha256": _sha256(windows_path),
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": _sha256(embeddings_path),
    }
    return rows, context_audit, dkw_windows, inputs


def _build_dkw_with_auto_l_hat(
    calibration_examples: pd.DataFrame,
    calibration_embeddings: pd.DataFrame,
    base_config: Any,
    *,
    raw_ct_config: dict[str, Any],
    library_alpha: float,
) -> tuple[DKWCTResult, pd.DataFrame]:
    result, audit = build_dkw_validated_contexts(
        calibration_examples,
        calibration_embeddings,
        base_config,
        library_alpha=library_alpha,
    )
    if raw_ct_config.get("l_hat") != "auto":
        return result, audit

    context_frame = dkw_contexts_to_frame(result.contexts)
    if len(context_frame) < 2:
        l_hat = 0.0
    else:
        try:
            l_hat = estimate_lipschitz_from_contexts(
                context_frame,
                risk_col="risk_estimate",
                metric=raw_ct_config.get("metric", "euclidean"),
                quantile=float(raw_ct_config.get("l_hat_quantile", 0.9)),
            )
        except ValueError:
            l_hat = 0.0
    return replace(result, config=replace(base_config, l_hat=l_hat)), audit


def _assert_uniform_context_sanity(result: DKWCTResult) -> None:
    if len(result.contexts) * result.per_context_delta > result.library_alpha + 1e-12:
        raise RuntimeError("per-context failure allocations exceed library alpha")
    for context in result.contexts:
        thresholds = np.unique(context.calibration_scores)
        certificates = dkw_uniform_certificates(
            context.error_scores,
            thresholds,
            delta=context.delta,
        )
        if np.any(np.diff(certificates) > 1e-12):
            raise RuntimeError("DKW source envelope increased as threshold increased")
        empirical = empirical_accepted_error_risks(context.error_scores, thresholds)
        if np.any(certificates + 1e-12 < empirical):
            raise RuntimeError("DKW source envelope fell below empirical risk")


def _summarize_variant(
    dataset: str,
    run_name: str,
    variant: str,
    windows: pd.DataFrame,
    *,
    risk_budget: float,
    library_alpha: float,
    per_context_delta: float,
    certificate_type: str,
    notes: str,
) -> dict[str, Any]:
    summary = summarize_window_results(
        windows,
        risk_budget=risk_budget,
        upper_bound_col="transferred_upper_bound",
        method_name=variant,
    )
    bounds = pd.to_numeric(windows["transferred_upper_bound"], errors="coerce")
    coverage = pd.to_numeric(windows["coverage"], errors="coerce")
    thresholds = pd.to_numeric(windows["threshold"], errors="coerce")
    num_accepted = pd.to_numeric(windows.get("num_accepted"), errors="coerce")
    n_examples = pd.to_numeric(windows.get("n_examples"), errors="coerce")
    risks = pd.to_numeric(windows["realized_risk"], errors="coerce")
    accepted_total = float(np.nansum(num_accepted))
    accepted_errors = float(np.nansum(risks * n_examples))

    if "nontrivial_feasible" in windows.columns:
        nontrivial = windows["nontrivial_feasible"].astype(bool)
    else:
        nontrivial = (coverage > 0.0) & (bounds <= risk_budget)
    if "forced_accept_none" in windows.columns:
        forced_none = windows["forced_accept_none"].astype(bool)
    else:
        forced_none = coverage <= 0.0

    return {
        "dataset": dataset,
        "run_name": run_name,
        "variant": variant,
        "risk": summary["mean_risk"],
        "coverage": summary["mean_coverage"],
        "bound_underestimation": summary["bound_underestimation_rate"],
        "mean_bound_gap": summary["mean_bound_gap"],
        "nontrivial_feasible_fraction": safe_mean(nontrivial),
        "n_nontrivial_feasible_windows": int(np.sum(nontrivial)),
        "forced_accept_none_fraction": safe_mean(forced_none),
        "n_forced_accept_none_windows": int(np.sum(forced_none)),
        "bound_feasible_fraction": safe_mean(bounds <= risk_budget),
        "n_bound_feasible_windows": int(np.sum(bounds <= risk_budget)),
        "mean_selected_threshold": safe_mean(thresholds),
        "median_selected_threshold": float(np.nanmedian(thresholds)),
        "mean_transferred_bound": safe_mean(bounds),
        "max_transferred_bound": float(np.nanmax(bounds)),
        "n_evaluation_windows": int(len(windows)),
        "library_alpha": library_alpha,
        "per_context_delta": per_context_delta,
        "certificate_type": certificate_type,
        "conditional_accepted_error_rate": (
            accepted_errors / accepted_total if accepted_total > 0.0 else np.nan
        ),
        "mean_accepted_examples_per_window": safe_mean(num_accepted),
        "notes": notes,
    }


def _label_windows(frame: pd.DataFrame, dataset: str, variant: str) -> pd.DataFrame:
    labeled = frame.copy()
    labeled.insert(0, "variant", variant)
    labeled.insert(0, "dataset", dataset)
    return labeled


def _add_matched_baseline_comparisons(
    results: pd.DataFrame,
    frontiers: pd.DataFrame,
) -> pd.DataFrame:
    """Add descriptive Global/Metadata matches for the primary DKW operating point."""
    enriched = results.copy()
    columns = (
        "global_risk_at_matched_coverage",
        "metadata_risk_at_matched_coverage",
        "delta_risk_vs_global_at_matched_coverage",
        "delta_risk_vs_metadata_at_matched_coverage",
        "global_coverage_at_matched_risk",
        "metadata_coverage_at_matched_risk",
        "delta_coverage_vs_global_at_matched_risk",
        "delta_coverage_vs_metadata_at_matched_risk",
    )
    for column in columns:
        enriched[column] = np.nan

    targets = enriched[enriched["variant"] == "dkw_target_feasible_ct"]
    for index, target in targets.iterrows():
        dataset_frame = frontiers[frontiers["dataset"] == target["dataset"]]
        global_curve = dataset_frame[dataset_frame["method"] == "global_crc"]
        metadata_curve = dataset_frame[dataset_frame["method"] == "metadata_stratified"]
        global_coverage_match = matched_coverage_point(global_curve, float(target["coverage"]))
        metadata_coverage_match = matched_coverage_point(metadata_curve, float(target["coverage"]))
        global_risk_match = matched_risk_point(global_curve, float(target["risk"]))
        metadata_risk_match = matched_risk_point(metadata_curve, float(target["risk"]))

        global_risk = _match_value(global_coverage_match, "risk_at_coverage")
        metadata_risk = _match_value(metadata_coverage_match, "risk_at_coverage")
        global_coverage = _match_value(global_risk_match, "coverage_at_risk")
        metadata_coverage = _match_value(metadata_risk_match, "coverage_at_risk")
        enriched.loc[index, "global_risk_at_matched_coverage"] = global_risk
        enriched.loc[index, "metadata_risk_at_matched_coverage"] = metadata_risk
        enriched.loc[index, "delta_risk_vs_global_at_matched_coverage"] = (
            float(target["risk"]) - global_risk
        )
        enriched.loc[index, "delta_risk_vs_metadata_at_matched_coverage"] = (
            float(target["risk"]) - metadata_risk
        )
        enriched.loc[index, "global_coverage_at_matched_risk"] = global_coverage
        enriched.loc[index, "metadata_coverage_at_matched_risk"] = metadata_coverage
        enriched.loc[index, "delta_coverage_vs_global_at_matched_risk"] = (
            float(target["coverage"]) - global_coverage
        )
        enriched.loc[index, "delta_coverage_vs_metadata_at_matched_risk"] = (
            float(target["coverage"]) - metadata_coverage
        )
    return enriched


def _match_value(match: dict[str, Any] | None, key: str) -> float:
    if match is None or key not in match:
        return float("nan")
    return float(match[key])


def _dataset_name(manifest: dict[str, Any], fallback: str) -> str:
    return str(manifest.get("dataset") or manifest.get("schema_dataset") or fallback)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
