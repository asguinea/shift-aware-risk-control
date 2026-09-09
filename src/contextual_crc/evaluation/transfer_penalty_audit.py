from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.config.loader import load_config
from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.evaluation.leakage_free import subset_examples_and_remap_windows
from contextual_crc.evaluation.synthetic_experiment import parse_risk_control_config
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy, candidate_thresholds
from contextual_crc.risk_control.source_certificates import (
    accepted_error_scores_finite,
    calibrate_dkw_source,
    dkw_radius,
    dkw_upper_envelope,
)
from contextual_crc.risk_control.transfer_penalty import (
    TransferPenalty,
    fit_transfer_penalty,
    observable_transfer_residual_upper,
    predeclared_penalty_grids,
    select_gamma_feasible_threshold,
    transfer_penalty_dominates_constraints,
)
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


BASE_FAMILIES = (
    "G0_no_penalty",
    "G1_global_residual",
    "G2_pure_lipschitz",
    "G3_lipschitz_intercept",
)
OLD_FAMILY = "old_auto_l_hat"


def run_transfer_penalty_audit(
    *,
    evaluation_dir: str | Path = "outputs/runs/leakage_free_main",
    output_dir: str | Path = "outputs/runs/transfer_penalty_audit",
    artifact_tables_dir: str | Path = "outputs/artifacts/tables",
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_dir)
    final_path = output_path / "final_transfer_results.csv"
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_path}")
    ensure_dir(output_path)
    evaluation_path = Path(evaluation_dir)
    evaluation_manifest_path = evaluation_path / "run_manifest.json"
    evaluation_manifest = _read_json(evaluation_manifest_path)
    started = datetime.now(timezone.utc)
    pair_frames: list[pd.DataFrame] = []
    loco_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    cert_frames: list[pd.DataFrame] = []
    dataset_manifests: list[dict[str, Any]] = []

    for dataset_manifest in evaluation_manifest["datasets"]:
        evaluated = evaluate_transfer_penalty_dataset(dataset_manifest)
        pair_frames.append(evaluated["pairs"])
        loco_frames.append(evaluated["loco"])
        candidate_frames.append(evaluated["candidates"])
        final_frames.append(evaluated["final"])
        cert_frames.append(evaluated["certifiability"])
        dataset_manifests.append(evaluated["manifest"])

    pairs = pd.concat(pair_frames, ignore_index=True)
    loco = pd.concat(loco_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    final = pd.concat(final_frames, ignore_index=True)
    certifiability = pd.concat(cert_frames, ignore_index=True)
    pairs.to_csv(output_path / "transfer_pair_audit.csv", index=False)
    loco.to_csv(output_path / "loco_transfer_results.csv", index=False)
    candidates.to_csv(output_path / "transfer_penalty_candidates.csv", index=False)
    final.to_csv(final_path, index=False)
    certifiability.to_csv(output_path / "certifiability.csv", index=False)

    from contextual_crc.artifacts.transfer_penalty_audit import write_transfer_penalty_artifacts

    artifact_paths = write_transfer_penalty_artifacts(
        artifact_tables_dir,
        final_results_csv=final_path,
        certifiability_csv=output_path / "certifiability.csv",
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "workflow": "transfer_penalty_audit",
        "workflow_version": 1,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": (finished - started).total_seconds(),
        "evaluation_dir": str(evaluation_path),
        "evaluation_manifest_sha256": _sha256(evaluation_manifest_path),
        "uses_fixed_split_encoder_embeddings": True,
        "uses_source_certificate_one_sided_dkw": True,
        "alpha": 0.05,
        "tau": 0.10,
        "legacy_multiplier_used": False,
        "legacy_margin_used": False,
        "final_evaluation_labels_used_for_gamma": False,
        "observable_residual_inequality": (
            "On the simultaneous event R_b(theta)<=U_b(theta), "
            "R_b(theta)-U_a(theta)<=U_b(theta)-U_a(theta); fit Gamma to the positive part."
        ),
        "gamma_grid_selection_validity": (
            "The context-and-threshold-uniform DKW event simultaneously covers every observed "
            "pair constraint, so selecting a dominating Gamma from the predeclared finite grid "
            "requires no additional Gamma-family union bound for observed-pair domination."
        ),
        "unseen_context_scope": (
            "Conditional on the structural assumption that the validated-pair envelope extends "
            "to deployment contexts; LOCO is empirical validation, not an unconditional context-level guarantee."
        ),
        "primary_selection_rule": (
            "Using calibration-side LOCO only: minimize bound-underestimation, then maximize "
            "the nontrivial-feasible fraction and coverage, then prefer family complexity "
            "G1<G2<G3<G4. Camelyon defaults to G1 with an "
            "insufficient-context-diversity warning because two contexts cannot support LOCO fitting."
        ),
        "datasets": dataset_manifests,
        "artifact_paths": artifact_paths,
    }
    _write_json(manifest, output_path / "run_manifest.json")
    return final_path


def evaluate_transfer_penalty_dataset(dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(dataset_manifest["dataset"])
    dataset_dir = Path(str(dataset_manifest["dataset_output_dir"]))
    source_run = Path(str(dataset_manifest["source_run_dir"]))
    resolved = load_config(source_run / "configs" / "resolved_config.yaml")
    examples, _, _ = load_prediction_cache(Path(str(dataset_manifest["prediction_cache"])))
    windows = load_windows(dataset_dir / "intermediate" / "windows.parquet")
    embeddings = pd.read_parquet(dataset_dir / "intermediate" / "embeddings.parquet")
    calibration_windows_original = windows[
        windows["deployable_partition"] == "calibration"
    ].reset_index(drop=True)
    evaluation_windows = windows[windows["common_evaluation"].astype(bool)].reset_index(drop=True)
    calibration_examples, calibration_windows = subset_examples_and_remap_windows(
        examples, calibration_windows_original
    )
    if _example_ids(calibration_examples) & _window_example_ids(examples, evaluation_windows):
        raise RuntimeError("Fixed calibration/evaluation overlap detected")

    risk_config = parse_risk_control_config(resolved.get("risk_control", {}))
    contexts = [str(value) for value in calibration_examples["context_id"].drop_duplicates()]
    m_contexts = len(contexts)
    delta_context = 0.05 / m_contexts
    context_data = _build_context_data(
        calibration_examples,
        calibration_windows,
        embeddings,
        contexts,
        delta_context=delta_context,
        risk_budget=risk_config.risk_budget,
        score_column=risk_config.score_column,
    )
    pair_rows = _build_pair_constraints(dataset, context_data, delta_context)
    pair_constraints = pair_rows[pair_rows["row_type"] == "ordered_pair"].reset_index(drop=True)
    geometry_distances = pair_constraints["distance"].to_numpy(dtype=float)
    slope_grid, intercept_grid = predeclared_penalty_grids(geometry_distances)
    bin_edges = _predeclared_bin_edges(geometry_distances) if m_contexts >= 6 else None
    old_saved_l_hat = float(dataset_manifest["dkw_resolved_l_hat"])

    loco = _run_loco(
        dataset,
        contexts,
        context_data,
        pair_constraints,
        slope_grid,
        intercept_grid,
        bin_edges,
        risk_config,
    )
    primary_family, primary_status = _select_primary_family(loco, m_contexts)
    final_penalties = _fit_final_penalties(
        pair_constraints,
        slope_grid,
        intercept_grid,
        bin_edges,
        old_saved_l_hat,
    )
    candidates = _candidate_table(
        dataset,
        final_penalties,
        pair_constraints,
        loco,
        primary_family,
        primary_status,
        context_data,
    )
    final_rows, window_results = _evaluate_final_penalties(
        dataset,
        examples,
        evaluation_windows,
        embeddings,
        context_data,
        final_penalties,
        risk_config,
        primary_family,
    )
    certifiability = _build_certifiability(
        dataset,
        context_data,
        window_results,
        delta_context,
    )

    data_audit = _dataset_pair_audit_row(
        dataset,
        context_data,
        pair_constraints,
        calibration_windows_original,
        evaluation_windows,
    )
    pairs = pd.concat([pd.DataFrame([data_audit]), pair_rows], ignore_index=True)
    manifest = {
        "dataset": dataset,
        "n_validated_contexts": m_contexts,
        "delta_context": delta_context,
        "calibration_windows_per_context": {
            context: int(context_data[context]["n_windows"]) for context in contexts
        },
        "audit_windows_per_validated_context": {
            context: int(
                len(
                    evaluation_windows[
                        evaluation_windows["context_id"].astype(str) == context
                    ]
                )
            )
            for context in contexts
        },
        "n_unordered_context_pairs": math.comb(m_contexts, 2),
        "n_ordered_context_pairs": m_contexts * (m_contexts - 1),
        "distance_min": float(pair_constraints["distance"].min()),
        "distance_median": float(pair_constraints["distance"].median()),
        "distance_max": float(pair_constraints["distance"].max()),
        "context_diversity_assessment": _context_diversity_assessment(m_contexts),
        "old_saved_l_hat": old_saved_l_hat,
        "primary_family": primary_family,
        "primary_selection_status": primary_status,
        "geometry_predeclared_slope_grid": {
            "size": len(slope_grid),
            "min": float(slope_grid.min()),
            "max": float(slope_grid.max()),
        },
        "geometry_predeclared_intercept_grid": {
            "size": len(intercept_grid),
            "min": float(intercept_grid.min()),
            "max": float(intercept_grid.max()),
        },
        "piecewise_bin_edges": [] if bin_edges is None else [float(v) for v in bin_edges],
    }
    return {
        "pairs": pairs,
        "loco": loco,
        "candidates": candidates,
        "final": final_rows,
        "certifiability": certifiability,
        "manifest": manifest,
    }


def _build_context_data(
    calibration_examples: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    embeddings: pd.DataFrame,
    contexts: Iterable[str],
    *,
    delta_context: float,
    risk_budget: float,
    score_column: str,
) -> dict[str, dict[str, Any]]:
    embedding_cols = get_embedding_columns(embeddings)
    cal_ids = set(calibration_windows["window_id"].astype(str))
    cal_embeddings = embeddings[embeddings["window_id"].astype(str).isin(cal_ids)]
    output: dict[str, dict[str, Any]] = {}
    for context in contexts:
        examples = calibration_examples[
            calibration_examples["context_id"].astype(str) == context
        ].reset_index(drop=True)
        context_embeddings = cal_embeddings[cal_embeddings["context_id"].astype(str) == context]
        scores = pd.to_numeric(examples[score_column], errors="coerce").to_numpy(dtype=float)
        errors = accepted_error_scores_finite(examples, score_column=score_column)
        calibration = calibrate_dkw_source(
            examples,
            risk_budget=risk_budget,
            delta=delta_context,
            side="one_sided",
            score_column=score_column,
        )
        output[context] = {
            "examples": examples,
            "scores": scores,
            "error_scores": errors,
            "epsilon": dkw_radius(len(examples), delta_context, side="one_sided"),
            "centroid": context_embeddings[embedding_cols].to_numpy(dtype=float).mean(axis=0),
            "n_windows": len(context_embeddings),
            "source_calibration": calibration,
        }
    return output


def _build_pair_constraints(
    dataset: str,
    context_data: Mapping[str, Mapping[str, Any]],
    delta_context: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contexts = list(context_data)
    for source in contexts:
        for target in contexts:
            if source == target:
                continue
            source_data = context_data[source]
            target_data = context_data[target]
            thresholds = candidate_thresholds(
                np.concatenate([source_data["scores"], target_data["scores"]]),
                accept_if="greater_equal",
                include_extremes=True,
            )
            source_upper = dkw_upper_envelope(
                source_data["error_scores"],
                thresholds,
                delta=delta_context,
                side="one_sided",
            )
            target_upper = dkw_upper_envelope(
                target_data["error_scores"],
                thresholds,
                delta=delta_context,
                side="one_sided",
            )
            residual_upper = observable_transfer_residual_upper(source_upper, target_upper)
            maximum = int(np.argmax(residual_upper))
            distance = float(
                np.linalg.norm(source_data["centroid"] - target_data["centroid"])
            )
            rows.append(
                {
                    "row_type": "ordered_pair",
                    "dataset": dataset,
                    "source_context": source,
                    "target_context": target,
                    "distance": distance,
                    "max_residual_upper": float(residual_upper[maximum]),
                    "median_residual_upper": float(np.median(residual_upper)),
                    "mean_residual_upper": float(np.mean(residual_upper)),
                    "threshold_at_max": float(thresholds[maximum]),
                    "source_ucb_at_max": float(source_upper[maximum]),
                    "target_ucb_at_max": float(target_upper[maximum]),
                    "n_threshold_constraints": len(thresholds),
                    "source_n": len(source_data["scores"]),
                    "target_n": len(target_data["scores"]),
                    "delta_context": delta_context,
                }
            )
    return pd.DataFrame(rows)


def _run_loco(
    dataset: str,
    contexts: list[str],
    context_data: Mapping[str, Mapping[str, Any]],
    pair_constraints: pd.DataFrame,
    slope_grid: np.ndarray,
    intercept_grid: np.ndarray,
    bin_edges: np.ndarray | None,
    risk_config: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = list(BASE_FAMILIES) + [OLD_FAMILY]
    if bin_edges is not None:
        families.insert(-1, "G4_piecewise_bins")
    for heldout in contexts:
        training_contexts = [context for context in contexts if context != heldout]
        training_pairs = pair_constraints[
            pair_constraints["source_context"].isin(training_contexts)
            & pair_constraints["target_context"].isin(training_contexts)
        ].reset_index(drop=True)
        penalties = _fit_fold_penalties(
            training_pairs,
            slope_grid,
            intercept_grid,
            bin_edges,
            training_contexts,
            context_data,
        )
        nearest, distance = _nearest_context(
            context_data[heldout]["centroid"], training_contexts, context_data
        )
        pseudo = context_data[heldout]["examples"]
        deployment_scores = context_data[heldout]["scores"]
        for family in families:
            penalty = penalties[family]
            base_row = {
                "row_type": "context_fold",
                "dataset": dataset,
                "family": family,
                "heldout_context": heldout,
                "fit_contexts": "|".join(training_contexts),
                "heldout_excluded_from_fit": True,
                "nearest_context": nearest,
                "distance": distance,
                "fit_status": penalty.fit_status,
                "intercept": penalty.intercept,
                "slope": penalty.slope,
            }
            if penalty.fit_status != "fitted":
                rows.append(base_row)
                continue
            gamma = float(penalty.evaluate(distance))
            selection = select_gamma_feasible_threshold(
                deployment_scores,
                context_data[nearest]["scores"],
                context_data[nearest]["error_scores"],
                source_epsilon=context_data[nearest]["epsilon"],
                gamma=gamma,
                risk_budget=risk_config.risk_budget,
            )
            metrics = _evaluate_after_selection(pseudo, selection, risk_config)
            rows.append(
                {
                    **base_row,
                    "gamma": gamma,
                    "threshold": selection["threshold"],
                    "source_ucb": selection["source_ucb"],
                    "final_ucb": selection["operational_bound"],
                    "realized_risk": metrics["risk"],
                    "coverage": metrics["coverage"],
                    "bound_underestimation": float(
                        selection["operational_bound"] + 1e-12 < metrics["risk"]
                    ),
                    "bound_gap": selection["operational_bound"] - metrics["risk"],
                    "nontrivial_feasible": selection["nontrivial_feasible"],
                    "forced_accept_none": selection["forced_accept_none"],
                }
            )
    context_rows = pd.DataFrame(rows)
    summaries = []
    for family, group in context_rows.groupby("family", sort=False):
        valid = group[group["fit_status"] == "fitted"]
        summaries.append(
            {
                "row_type": "dataset_aggregate",
                "dataset": dataset,
                "family": family,
                "n_folds": len(group),
                "n_valid_folds": len(valid),
                "risk": valid["realized_risk"].mean() if len(valid) else np.nan,
                "coverage": valid["coverage"].mean() if len(valid) else np.nan,
                "bound_underestimation": (
                    valid["bound_underestimation"].mean() if len(valid) else np.nan
                ),
                "mean_bound_gap": valid["bound_gap"].mean() if len(valid) else np.nan,
                "nontrivial_feasible_fraction": (
                    valid["nontrivial_feasible"].astype(float).mean() if len(valid) else np.nan
                ),
                "accept_none_fraction": (
                    valid["forced_accept_none"].astype(float).mean() if len(valid) else np.nan
                ),
                "mean_gamma": valid["gamma"].mean() if len(valid) else np.nan,
            }
        )
    return pd.concat([context_rows, pd.DataFrame(summaries)], ignore_index=True)


def _fit_fold_penalties(
    training_pairs: pd.DataFrame,
    slope_grid: np.ndarray,
    intercept_grid: np.ndarray,
    bin_edges: np.ndarray | None,
    training_contexts: list[str],
    context_data: Mapping[str, Mapping[str, Any]],
) -> dict[str, TransferPenalty]:
    penalties = {
        family: fit_transfer_penalty(
            training_pairs,
            family,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            bin_edges=bin_edges,
        )
        for family in BASE_FAMILIES
    }
    if bin_edges is not None:
        penalties["G4_piecewise_bins"] = fit_transfer_penalty(
            training_pairs, "G4_piecewise_bins", bin_edges=bin_edges
        )
    old_l = _old_auto_l_hat(training_contexts, context_data)
    penalties[OLD_FAMILY] = TransferPenalty(
        family=OLD_FAMILY,
        slope=old_l,
        interpretation="old 0.95 point-risk-difference/distance diagnostic",
    )
    return penalties


def _fit_final_penalties(
    pair_constraints: pd.DataFrame,
    slope_grid: np.ndarray,
    intercept_grid: np.ndarray,
    bin_edges: np.ndarray | None,
    old_saved_l_hat: float,
) -> dict[str, TransferPenalty]:
    penalties = {
        family: fit_transfer_penalty(
            pair_constraints,
            family,
            slope_grid=slope_grid,
            intercept_grid=intercept_grid,
            bin_edges=bin_edges,
        )
        for family in BASE_FAMILIES
    }
    if bin_edges is not None:
        penalties["G4_piecewise_bins"] = fit_transfer_penalty(
            pair_constraints, "G4_piecewise_bins", bin_edges=bin_edges
        )
    penalties[OLD_FAMILY] = TransferPenalty(
        family=OLD_FAMILY,
        slope=old_saved_l_hat,
        interpretation="saved legacy auto-L diagnostic; no legacy multiplier/margin",
    )
    return penalties


def _old_auto_l_hat(
    contexts: list[str], context_data: Mapping[str, Mapping[str, Any]]
) -> float:
    if len(contexts) < 2:
        return 0.0
    point_risks = []
    centroids = []
    for context in contexts:
        calibration = context_data[context]["source_calibration"]
        point_risks.append(float(calibration.empirical_risk))
        centroids.append(np.asarray(context_data[context]["centroid"], dtype=float))
    slopes = []
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            distance = float(np.linalg.norm(centroids[i] - centroids[j]))
            if distance > 0:
                slopes.append(abs(point_risks[i] - point_risks[j]) / (distance + 1e-8))
    return float(np.quantile(slopes, 0.95)) if slopes else 0.0


def _select_primary_family(loco: pd.DataFrame, m_contexts: int) -> tuple[str, str]:
    if m_contexts < 3:
        return "G1_global_residual", "insufficient_context_diversity_default_simplest"
    aggregate = loco[
        (loco["row_type"] == "dataset_aggregate")
        & loco["family"].isin(
            ["G1_global_residual", "G2_pure_lipschitz", "G3_lipschitz_intercept", "G4_piecewise_bins"]
        )
    ].copy()
    aggregate = aggregate[aggregate["n_valid_folds"] == aggregate["n_folds"]]
    if aggregate.empty:
        return "G1_global_residual", "no_complete_loco_family_default_simplest"
    complexity = {
        "G1_global_residual": 1,
        "G2_pure_lipschitz": 2,
        "G3_lipschitz_intercept": 3,
        "G4_piecewise_bins": 4,
    }
    aggregate["complexity"] = aggregate["family"].map(complexity)
    selected = aggregate.sort_values(
        ["bound_underestimation", "nontrivial_feasible_fraction", "coverage", "complexity"],
        ascending=[True, False, False, True],
    ).iloc[0]
    return str(selected["family"]), "selected_by_predeclared_loco_rule"


def _candidate_table(
    dataset: str,
    penalties: Mapping[str, TransferPenalty],
    pair_constraints: pd.DataFrame,
    loco: pd.DataFrame,
    primary_family: str,
    primary_status: str,
    context_data: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    summaries = loco[loco["row_type"] == "dataset_aggregate"].set_index("family")
    point_risks = [
        float(context_data[context]["source_calibration"].empirical_risk)
        for context in context_data
    ]
    rows = []
    for family, penalty in penalties.items():
        loco_row = summaries.loc[family] if family in summaries.index else pd.Series(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "fit_status": penalty.fit_status,
                "intercept": penalty.intercept,
                "slope": penalty.slope,
                "bin_edges": json.dumps(penalty.bin_edges),
                "bin_values": json.dumps(penalty.bin_values),
                "dominates_all_observed_pair_constraints": (
                    transfer_penalty_dominates_constraints(penalty, pair_constraints)
                    if family not in ("G0_no_penalty", OLD_FAMILY)
                    else False
                ),
                "interpretation": penalty.interpretation,
                "loco_risk": loco_row.get("risk", np.nan),
                "loco_coverage": loco_row.get("coverage", np.nan),
                "loco_bound_underestimation": loco_row.get("bound_underestimation", np.nan),
                "loco_nontrivial_feasible_fraction": loco_row.get(
                    "nontrivial_feasible_fraction", np.nan
                ),
                "loco_valid_folds": loco_row.get("n_valid_folds", 0),
                "is_primary": family == primary_family,
                "primary_selection_status": primary_status if family == primary_family else "",
                "old_point_risk_min": min(point_risks),
                "old_point_risk_max": max(point_risks),
                "old_point_risk_range": max(point_risks) - min(point_risks),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_final_penalties(
    dataset: str,
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embeddings: pd.DataFrame,
    context_data: Mapping[str, Mapping[str, Any]],
    penalties: Mapping[str, TransferPenalty],
    risk_config: Any,
    primary_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embedding_cols = get_embedding_columns(embeddings)
    by_window = embeddings.set_index("window_id")
    window_rows: list[dict[str, Any]] = []
    contexts = list(context_data)
    for family, penalty in penalties.items():
        if penalty.fit_status != "fitted":
            continue
        for _, window in windows.iterrows():
            query = by_window.loc[window["window_id"], embedding_cols].to_numpy(dtype=float)
            nearest, distance = _nearest_context(query, contexts, context_data)
            gamma = float(penalty.evaluate(distance))
            window_examples = examples.iloc[
                [int(value) for value in window["example_indices"]]
            ]
            deployment_scores = pd.to_numeric(
                window_examples[risk_config.score_column], errors="coerce"
            ).to_numpy(dtype=float)
            selection = select_gamma_feasible_threshold(
                deployment_scores,
                context_data[nearest]["scores"],
                context_data[nearest]["error_scores"],
                source_epsilon=context_data[nearest]["epsilon"],
                gamma=gamma,
                risk_budget=risk_config.risk_budget,
            )
            # Final labels are first used after Gamma and theta have both been frozen/selected.
            metrics = _evaluate_after_selection(window_examples, selection, risk_config)
            window_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "is_primary": family == primary_family,
                    "window_id": window["window_id"],
                    "context_id": window["context_id"],
                    "nearest_context": nearest,
                    "distance": distance,
                    "gamma": gamma,
                    "source_ucb": selection["source_ucb"],
                    "final_ucb": selection["operational_bound"],
                    "raw_final_ucb": selection["raw_final_ucb"],
                    "threshold": selection["threshold"],
                    "realized_risk": metrics["risk"],
                    "coverage": metrics["coverage"],
                    "bound_underestimation": float(
                        selection["operational_bound"] + 1e-12 < metrics["risk"]
                    ),
                    "bound_gap": selection["operational_bound"] - metrics["risk"],
                    "nontrivial_feasible": selection["nontrivial_feasible"],
                    "forced_accept_none": selection["forced_accept_none"],
                }
            )
    window_frame = pd.DataFrame(window_rows)
    summaries = []
    for family, group in window_frame.groupby("family", sort=False):
        summaries.append(
            {
                "dataset": dataset,
                "family": family,
                "is_primary": family == primary_family,
                "risk": group["realized_risk"].mean(),
                "coverage": group["coverage"].mean(),
                "bound_underestimation": group["bound_underestimation"].mean(),
                "mean_bound_gap": group["bound_gap"].mean(),
                "nontrivial_feasible_fraction": group["nontrivial_feasible"].astype(float).mean(),
                "accept_none_fraction": group["forced_accept_none"].astype(float).mean(),
                "mean_gamma": group["gamma"].mean(),
                "median_gamma": group["gamma"].median(),
                "max_gamma": group["gamma"].max(),
                "mean_source_ucb": group["source_ucb"].mean(),
                "mean_final_ucb": group["final_ucb"].mean(),
                "n_evaluation_windows": len(group),
            }
        )
    return pd.DataFrame(summaries), window_frame


def _build_certifiability(
    dataset: str,
    context_data: Mapping[str, Mapping[str, Any]],
    window_results: pd.DataFrame,
    delta_context: float,
) -> pd.DataFrame:
    source_fraction = float(
        np.mean(
            [
                bool(data["source_calibration"].nontrivial)
                for data in context_data.values()
            ]
        )
    )
    rows = []
    for family, group in window_results.groupby("family", sort=False):
        row = {
            "row_type": "dataset_family",
            "dataset": dataset,
            "family": family,
            "source_certifiable_fraction": source_fraction,
            "target_certifiable_fraction": group["nontrivial_feasible"].astype(float).mean(),
            "n_validated_contexts": len(context_data),
            "n_evaluation_windows": len(group),
            "tau": 0.10,
            "alpha": 0.05,
        }
        if dataset == "cifar100c":
            for epsilon in (0.05, 0.04, 0.03, 0.02):
                row[f"n_required_epsilon_{epsilon:.2f}"] = int(
                    math.ceil(math.log(1.0 / delta_context) / (2.0 * epsilon**2))
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_pair_audit_row(
    dataset: str,
    context_data: Mapping[str, Mapping[str, Any]],
    pairs: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
) -> dict[str, Any]:
    m = len(context_data)
    audit_counts = [
        len(evaluation_windows[evaluation_windows["context_id"].astype(str) == context])
        for context in context_data
    ]
    calibration_counts = [int(data["n_windows"]) for data in context_data.values()]
    return {
        "row_type": "dataset_audit",
        "dataset": dataset,
        "n_validated_contexts": m,
        "calibration_windows_min": min(calibration_counts),
        "calibration_windows_median": float(np.median(calibration_counts)),
        "calibration_windows_max": max(calibration_counts),
        "audit_windows_min": min(audit_counts),
        "audit_windows_median": float(np.median(audit_counts)),
        "audit_windows_max": max(audit_counts),
        "n_distinct_unordered_pairs": math.comb(m, 2),
        "n_usable_ordered_pairs": len(pairs),
        "distance_min": pairs["distance"].min(),
        "distance_median": pairs["distance"].median(),
        "distance_max": pairs["distance"].max(),
        "context_diversity_assessment": _context_diversity_assessment(m),
        "n_total_calibration_windows": len(calibration_windows),
        "n_common_evaluation_windows": len(evaluation_windows),
    }


def _predeclared_bin_edges(distances: np.ndarray) -> np.ndarray | None:
    values = np.asarray(distances, dtype=float)
    edges = np.unique(np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
    return edges if len(edges) == 2 and edges[0] < edges[1] else None


def _context_diversity_assessment(m_contexts: int) -> str:
    if m_contexts <= 2:
        return "insufficient: two contexts provide one distinct distance and no meaningful LOCO fit"
    if m_contexts < 6:
        return "limited: LOCO is possible but distance-family identification is weak"
    return "moderate: 15 distinct unordered pairs support simple-family LOCO, not rich functions"


def _nearest_context(
    query: np.ndarray,
    contexts: Iterable[str],
    context_data: Mapping[str, Mapping[str, Any]],
) -> tuple[str, float]:
    choices = [
        (context, float(np.linalg.norm(query - context_data[context]["centroid"])))
        for context in contexts
    ]
    return min(choices, key=lambda item: item[1])


def _evaluate_after_selection(
    examples: pd.DataFrame,
    selection: Mapping[str, Any],
    risk_config: Any,
) -> dict[str, float]:
    accepted = apply_threshold_policy(
        examples,
        risk_config.score_column,
        float(selection["threshold"]),
        risk_config.accept_if,
    )
    losses = compute_losses(examples, accepted, risk_config.loss)
    summary = summarize_policy_performance(examples, accepted, losses)
    if int(summary["num_accepted"]) != int(selection["n_accepted_from_scores"]):
        raise RuntimeError("score-only selection disagrees with post-selection coverage")
    return {"risk": float(summary["mean_loss"]), "coverage": float(summary["coverage"])}


def _example_ids(examples: pd.DataFrame) -> set[str]:
    return set(examples["example_id"].astype(str))


def _window_example_ids(examples: pd.DataFrame, windows: pd.DataFrame) -> set[str]:
    indices = [int(value) for values in windows["example_indices"] for value in values]
    return set(examples.iloc[indices]["example_id"].astype(str))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
