from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.config.loader import load_config
from contextual_crc.embeddings.context_encoder import get_embedding_columns
from contextual_crc.evaluation.leakage_free import subset_examples_and_remap_windows
from contextual_crc.evaluation.synthetic_experiment import parse_ctcrc_config, parse_risk_control_config
from contextual_crc.risk_control.ct_crc_dkw import build_dkw_validated_contexts
from contextual_crc.risk_control.losses import compute_losses, summarize_policy_performance
from contextual_crc.risk_control.policies import apply_threshold_policy, candidate_thresholds
from contextual_crc.risk_control.source_certificates import (
    SourceCalibration,
    accepted_error_risks,
    accepted_error_scores_finite,
    calibrate_crc_expected_risk,
    calibrate_dkw_source,
    calibrate_ltt_bonferroni_source,
    calibrate_ltt_fixed_sequence_source,
    coverage_at_thresholds,
    dkw_radius,
)
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


TRANSFER_FORMS = {
    "T0_source_only": (False, False, False),
    "T1_source_multiplier": (True, False, False),
    "T2_source_multiplier_margin": (True, True, False),
    "T3_theorem_form": (False, False, True),
    "T4_current_full": (True, True, True),
}


def decompose_transferred_bound(
    empirical_source_risk: float,
    source_uncertainty: float,
    *,
    multiplier: float,
    margin: float,
    l_hat: float,
    distance: float,
    epsilon_k: float,
    clip: bool = True,
) -> dict[str, float]:
    source_ucb = float(np.clip(empirical_source_risk + source_uncertainty, 0.0, 1.0))
    multiplier_increment = (float(multiplier) - 1.0) * source_ucb
    distance_penalty = float(l_hat) * float(distance)
    embedding_penalty = float(l_hat) * float(epsilon_k)
    unclipped = (
        source_ucb
        + multiplier_increment
        + float(margin)
        + distance_penalty
        + embedding_penalty
    )
    final = float(np.clip(unclipped, 0.0, 1.0)) if clip else float(unclipped)
    return {
        "empirical_source_risk": float(empirical_source_risk),
        "source_uncertainty": float(source_ucb - empirical_source_risk),
        "source_ucb": source_ucb,
        "multiplier_increment": multiplier_increment,
        "additive_margin": float(margin),
        "distance_penalty": distance_penalty,
        "embedding_penalty": embedding_penalty,
        "unclipped_final_bound": float(unclipped),
        "clipping_adjustment": float(final - unclipped),
        "final_bound": final,
    }


def select_transfer_form_threshold(
    deployment_scores: np.ndarray,
    source_scores: np.ndarray,
    source_error_scores: np.ndarray,
    *,
    source_epsilon: float,
    distance: float,
    l_hat: float,
    epsilon_k: float,
    multiplier: float,
    margin: float,
    risk_budget: float,
    form: str,
) -> dict[str, Any]:
    """Score-only target selection; deployment labels are intentionally not accepted."""
    if form not in TRANSFER_FORMS:
        raise ValueError(f"unknown transfer form: {form}")
    scores = np.asarray(deployment_scores, dtype=float).reshape(-1)
    source = np.asarray(source_scores, dtype=float).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all() or not np.isfinite(source).all():
        raise ValueError("deployment and source scores must be non-empty and finite")
    thresholds = candidate_thresholds(
        np.concatenate([scores, source]), accept_if="greater_equal", include_extremes=True
    )
    risks = accepted_error_risks(source_error_scores, thresholds)
    source_ucbs = np.clip(risks + float(source_epsilon), 0.0, 1.0)
    coverages = coverage_at_thresholds(scores, thresholds)
    accepted_counts = np.rint(coverages * len(scores)).astype(int)
    use_multiplier, use_margin, use_distance = TRANSFER_FORMS[form]
    effective_multiplier = float(multiplier) if use_multiplier else 1.0
    effective_margin = float(margin) if use_margin else 0.0
    effective_l_hat = float(l_hat) if use_distance else 0.0
    raw_bounds = np.asarray(
        [
            decompose_transferred_bound(
                float(risk),
                float(ucb - risk),
                multiplier=effective_multiplier,
                margin=effective_margin,
                l_hat=effective_l_hat,
                distance=distance,
                epsilon_k=epsilon_k,
            )["final_bound"]
            for risk, ucb in zip(risks, source_ucbs, strict=True)
        ]
    )
    feasible = (accepted_counts > 0) & (raw_bounds <= float(risk_budget) + 1e-12)
    if feasible.any():
        choices = np.flatnonzero(feasible)
        selected = min(
            choices,
            key=lambda i: (-accepted_counts[i], raw_bounds[i], thresholds[i]),
        )
        forced_none = False
        operational_bound = float(raw_bounds[selected])
    else:
        none = np.flatnonzero(accepted_counts == 0)
        if not len(none):
            raise RuntimeError("threshold candidates omitted accept-none")
        selected = int(none[0])
        forced_none = True
        operational_bound = 0.0
    decomposition = decompose_transferred_bound(
        float(risks[selected]),
        float(source_ucbs[selected] - risks[selected]),
        multiplier=effective_multiplier,
        margin=effective_margin,
        l_hat=effective_l_hat,
        distance=distance,
        epsilon_k=epsilon_k,
    )
    return {
        "form": form,
        "threshold": float(thresholds[selected]),
        "coverage_from_scores": float(coverages[selected]),
        "n_accepted_from_scores": int(accepted_counts[selected]),
        "nontrivial_feasible": not forced_none,
        "forced_accept_none": forced_none,
        "operational_bound": operational_bound,
        "num_candidates": len(thresholds),
        **decomposition,
    }


def run_source_certificate_audit(
    *,
    evaluation_dir: str | Path = "outputs/runs/leakage_free_main",
    output_dir: str | Path = "outputs/runs/source_certificate_audit",
    artifact_tables_dir: str | Path = "outputs/artifacts/tables",
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_dir)
    comparison_path = output_path / "source_certificate_comparison.csv"
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_path}")
    ensure_dir(output_path)
    evaluation_path = Path(evaluation_dir)
    evaluation_manifest = _read_json(evaluation_path / "run_manifest.json")
    started = datetime.now(timezone.utc)
    comparison_frames: list[pd.DataFrame] = []
    budget_frames: list[pd.DataFrame] = []
    transfer_frames: list[pd.DataFrame] = []
    dataset_manifests: list[dict[str, Any]] = []

    for dataset_manifest in evaluation_manifest["datasets"]:
        result = evaluate_source_certificate_dataset(dataset_manifest)
        comparison_frames.append(result["comparison"])
        budget_frames.append(result["budget"])
        transfer_frames.append(result["transfer"])
        dataset_manifests.append(result["manifest"])

    comparison = pd.concat(comparison_frames, ignore_index=True)
    budget = pd.concat(budget_frames, ignore_index=True)
    transfer = pd.concat(transfer_frames, ignore_index=True)
    guarantee_types = build_guarantee_type_table()
    comparison.to_csv(comparison_path, index=False)
    budget.to_csv(output_path / "budget_decomposition.csv", index=False)
    transfer.to_csv(output_path / "transfer_form_diagnostics.csv", index=False)
    guarantee_types.to_csv(output_path / "guarantee_types.csv", index=False)

    from contextual_crc.artifacts.source_certificate_audit import write_source_certificate_artifacts

    artifact_paths = write_source_certificate_artifacts(
        artifact_tables_dir,
        comparison_csv=comparison_path,
        budget_csv=output_path / "budget_decomposition.csv",
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "workflow": "source_certificate_audit",
        "workflow_version": 1,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": (finished - started).total_seconds(),
        "evaluation_dir": str(evaluation_path),
        "evaluation_manifest_sha256": _sha256(evaluation_path / "run_manifest.json"),
        "uses_fixed_assignments": True,
        "uses_fixed_encoder_and_embeddings": True,
        "uses_fixed_l_hat": True,
        "reruns_model_inference": False,
        "refits_representation": False,
        "retunes_transfer_parameters": False,
        "risk_budget": 0.10,
        "library_alpha": 0.05,
        "ltt_theta_grid": "linspace(0,1,101), fixed before reading calibration scores/labels",
        "one_sided_dkw_derivation": (
            "For S(theta)=P(V>=theta), S-S_n equals the one-sided empirical-CDF "
            "deviation F_n(theta-)-F(theta-), bounded by exp(-2 n epsilon^2)."
        ),
        "crc_formula": "(n * empirical_risk(theta) + B)/(n+1) <= tau, B=1",
        "crc_guarantee": "expected risk under exchangeability and bounded monotone loss",
        "references": {
            "crc_reference": "https://arxiv.org/abs/2208.02814",
            "crc_official_code": (
                "https://github.com/aangelopoulos/conformal-risk/blob/main/core/get_lhat.py"
            ),
        },
        "datasets": dataset_manifests,
        "artifact_paths": artifact_paths,
    }
    _write_json(manifest, output_path / "run_manifest.json")
    return comparison_path


def evaluate_source_certificate_dataset(dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
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
    source_audit_windows = evaluation_windows[
        evaluation_windows["context_role"] == "validated_source"
    ].reset_index(drop=True)
    calibration_examples, calibration_windows = subset_examples_and_remap_windows(
        examples, calibration_windows_original
    )
    cal_ids = set(calibration_windows_original["window_id"].astype(str))
    calibration_embeddings = embeddings[
        embeddings["window_id"].astype(str).isin(cal_ids)
    ].reset_index(drop=True)
    if set(calibration_examples["example_id"].astype(str)) & _example_ids(examples, evaluation_windows):
        raise RuntimeError("Source calibration overlaps the fixed evaluation split")

    risk_config = parse_risk_control_config(resolved.get("risk_control", {}))
    raw_ct = resolved.get("ct_crc", {})
    ct_config = replace(
        parse_ctcrc_config(raw_ct, risk_config),
        l_hat=float(dataset_manifest["dkw_resolved_l_hat"]),
    )
    library_alpha = 0.05
    context_values = [
        str(value) for value in calibration_examples["context_id"].drop_duplicates().tolist()
    ]
    per_context_delta = library_alpha / len(context_values)
    theta_grid = np.linspace(0.0, 1.0, 101)
    calibrations: dict[str, dict[str, SourceCalibration]] = {}
    source_arrays: dict[str, dict[str, np.ndarray]] = {}
    comparison_rows: list[dict[str, Any]] = []
    embedding_cols = get_embedding_columns(calibration_embeddings)
    centroids: dict[str, np.ndarray] = {}

    for context_id in context_values:
        context_examples = calibration_examples[
            calibration_examples["context_id"].astype(str) == context_id
        ].reset_index(drop=True)
        context_embeddings = calibration_embeddings[
            calibration_embeddings["context_id"].astype(str) == context_id
        ]
        centroids[context_id] = context_embeddings[embedding_cols].to_numpy(dtype=float).mean(axis=0)
        method_results = {
            "two_sided_dkw_source": calibrate_dkw_source(
                context_examples,
                risk_budget=risk_config.risk_budget,
                delta=per_context_delta,
                side="two_sided",
            ),
            "one_sided_dkw_source": calibrate_dkw_source(
                context_examples,
                risk_budget=risk_config.risk_budget,
                delta=per_context_delta,
                side="one_sided",
            ),
            "crc_expected_risk_source": calibrate_crc_expected_risk(
                context_examples, risk_budget=risk_config.risk_budget
            ),
            "ltt_bonferroni_source": calibrate_ltt_bonferroni_source(
                context_examples,
                risk_target=risk_config.risk_budget,
                delta=per_context_delta,
                theta_grid=theta_grid,
            ),
            "ltt_fixed_sequence_source": calibrate_ltt_fixed_sequence_source(
                context_examples,
                risk_target=risk_config.risk_budget,
                delta=per_context_delta,
                theta_grid=theta_grid,
            ),
        }
        calibrations[context_id] = method_results
        source_arrays[context_id] = {
            "scores": pd.to_numeric(
                context_examples[risk_config.score_column], errors="coerce"
            ).to_numpy(dtype=float),
            "error_scores": accepted_error_scores_finite(
                context_examples, score_column=risk_config.score_column
            ),
        }
        heldout = _examples_for_context_windows(examples, source_audit_windows, context_id)
        if heldout.empty:
            raise RuntimeError(f"validated context has no held-out evaluation window: {context_id}")
        for method, calibration in method_results.items():
            heldout_metrics = _evaluate_threshold(heldout, calibration.threshold, risk_config)
            applicable_bound = (
                calibration.certificate
                if method != "crc_expected_risk_source" and np.isfinite(calibration.certificate)
                else np.nan
            )
            comparison_rows.append(
                {
                    "row_type": "context",
                    "dataset": dataset,
                    "context_id": context_id,
                    "method": method,
                    "guarantee_type": calibration.guarantee_type,
                    "n_calibration": calibration.n_calibration,
                    "n_calibration_windows": len(context_embeddings),
                    "selected_threshold": calibration.threshold,
                    "empirical_source_risk": calibration.empirical_risk,
                    "source_uncertainty": calibration.uncertainty,
                    "source_certificate": calibration.certificate,
                    "source_calibration_coverage": calibration.coverage,
                    "source_feasible": calibration.feasible,
                    "source_nontrivial": calibration.nontrivial,
                    "dkw_epsilon": (
                        dkw_radius(
                            calibration.n_calibration,
                            per_context_delta,
                            side="one_sided" if method.startswith("one_sided") else "two_sided",
                        )
                        if "dkw" in method
                        else np.nan
                    ),
                    "delta_context": per_context_delta if method != "crc_expected_risk_source" else np.nan,
                    "certified_threshold_set_size": calibration.certified_set_size,
                    "selected_p_value": calibration.p_value,
                    "n_heldout": len(heldout),
                    "n_heldout_windows": len(
                        source_audit_windows[
                            source_audit_windows["context_id"].astype(str) == context_id
                        ]
                    ),
                    "heldout_source_risk": heldout_metrics["risk"],
                    "heldout_source_coverage": heldout_metrics["coverage"],
                    "certificate_violation": (
                        float(heldout_metrics["risk"] > applicable_bound + 1e-12)
                        if np.isfinite(applicable_bound)
                        else np.nan
                    ),
                }
            )

    # Rebuild the fixed two-sided context objects with the saved L-hat; no representation refit.
    dkw_result, _ = build_dkw_validated_contexts(
        calibration_examples,
        calibration_embeddings,
        ct_config,
        library_alpha=library_alpha,
    )
    two_sided_transfer = _evaluate_transfer_windows(
        examples,
        evaluation_windows,
        embeddings,
        calibrations,
        source_arrays,
        centroids,
        ct_config,
        side="two_sided",
        forms=tuple(TRANSFER_FORMS),
    )
    one_sided_transfer = _evaluate_transfer_windows(
        examples,
        evaluation_windows,
        embeddings,
        calibrations,
        source_arrays,
        centroids,
        ct_config,
        side="one_sided",
        forms=("T4_current_full",),
    )
    _assert_matches_fixed_evaluation(dataset_dir, two_sided_transfer)
    crc_target = _evaluate_nearest_source_policy(
        examples,
        evaluation_windows,
        embeddings,
        calibrations,
        centroids,
        risk_config,
        method="crc_expected_risk_source",
    )

    comparison = pd.DataFrame(comparison_rows)
    aggregate_rows = _aggregate_source_comparison(
        comparison,
        two_sided_transfer,
        one_sided_transfer,
        crc_target,
    )
    comparison = pd.concat([comparison, aggregate_rows], ignore_index=True)
    budget = _build_budget_output(
        two_sided_transfer[two_sided_transfer["form"] == "T4_current_full"].copy(),
        risk_budget=risk_config.risk_budget,
    )
    transfer = _summarize_transfer_forms(two_sided_transfer, risk_config.risk_budget)
    manifest = {
        "dataset": dataset,
        "fixed_evaluation_dataset_dir": str(dataset_dir),
        "n_calibration_examples": len(calibration_examples),
        "n_calibration_windows": len(calibration_windows_original),
        "n_source_audit_windows": len(source_audit_windows),
        "n_evaluation_windows": len(evaluation_windows),
        "validated_contexts": context_values,
        "per_context_delta": per_context_delta,
        "two_sided_radius": dkw_radius(
            int(comparison[comparison["row_type"] == "context"]["n_calibration"].iloc[0]),
            per_context_delta,
            side="two_sided",
        ),
        "one_sided_radius": dkw_radius(
            int(comparison[comparison["row_type"] == "context"]["n_calibration"].iloc[0]),
            per_context_delta,
            side="one_sided",
        ),
        "resolved_l_hat": ct_config.l_hat,
        "ct_multiplier": ct_config.bound_multiplier,
        "ct_margin": ct_config.bound_margin,
        "epsilon_k": ct_config.epsilon_k,
        "fixed_evaluation_dkw_context_count": len(dkw_result.contexts),
    }
    return {"comparison": comparison, "budget": budget, "transfer": transfer, "manifest": manifest}


def _evaluate_transfer_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embeddings: pd.DataFrame,
    calibrations: Mapping[str, Mapping[str, SourceCalibration]],
    source_arrays: Mapping[str, Mapping[str, np.ndarray]],
    centroids: Mapping[str, np.ndarray],
    ct_config: Any,
    *,
    side: str,
    forms: tuple[str, ...],
) -> pd.DataFrame:
    embedding_cols = get_embedding_columns(embeddings)
    by_window = embeddings.set_index("window_id")
    rows: list[dict[str, Any]] = []
    method = f"{side}_dkw_source"
    for _, window in windows.iterrows():
        query = by_window.loc[window["window_id"], embedding_cols].to_numpy(dtype=float)
        nearest, distance = _nearest_context(query, centroids)
        source_cal = calibrations[nearest][method]
        source_scores = np.asarray(source_arrays[nearest]["scores"], dtype=float)
        error_scores = np.asarray(source_arrays[nearest]["error_scores"], dtype=float)
        indices = [int(value) for value in window["example_indices"]]
        window_examples = examples.iloc[indices]
        deployment_scores = pd.to_numeric(
            window_examples[ct_config.risk_control.score_column], errors="coerce"
        ).to_numpy(dtype=float)
        for form in forms:
            selection = select_transfer_form_threshold(
                deployment_scores,
                source_scores,
                error_scores,
                source_epsilon=dkw_radius(
                    source_cal.n_calibration,
                    source_cal.delta,
                    side="one_sided" if side == "one_sided" else "two_sided",
                ),
                distance=distance,
                l_hat=ct_config.l_hat,
                epsilon_k=ct_config.epsilon_k,
                multiplier=ct_config.bound_multiplier,
                margin=ct_config.bound_margin,
                risk_budget=ct_config.risk_control.risk_budget,
                form=form,
            )
            # Labels are first consumed here, after score-only selection returns.
            metrics = _evaluate_threshold(window_examples, selection["threshold"], ct_config.risk_control)
            if int(round(metrics["coverage"] * len(window_examples))) != selection["n_accepted_from_scores"]:
                raise RuntimeError("score-only coverage disagrees with post-selection evaluation")
            rows.append(
                {
                    "dataset": str(window.get("dataset", examples["dataset"].iloc[0])),
                    "side": side,
                    "form": form,
                    "window_id": window["window_id"],
                    "context_id": window["context_id"],
                    "nearest_context_id": nearest,
                    "novelty_distance": distance,
                    "realized_risk": metrics["risk"],
                    "coverage": metrics["coverage"],
                    "n_examples": len(window_examples),
                    **selection,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_nearest_source_policy(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    embeddings: pd.DataFrame,
    calibrations: Mapping[str, Mapping[str, SourceCalibration]],
    centroids: Mapping[str, np.ndarray],
    risk_config: Any,
    *,
    method: str,
) -> pd.DataFrame:
    embedding_cols = get_embedding_columns(embeddings)
    by_window = embeddings.set_index("window_id")
    rows = []
    for _, window in windows.iterrows():
        query = by_window.loc[window["window_id"], embedding_cols].to_numpy(dtype=float)
        nearest, distance = _nearest_context(query, centroids)
        calibration = calibrations[nearest][method]
        window_examples = examples.iloc[[int(value) for value in window["example_indices"]]]
        metrics = _evaluate_threshold(window_examples, calibration.threshold, risk_config)
        rows.append(
            {
                "window_id": window["window_id"],
                "nearest_context_id": nearest,
                "novelty_distance": distance,
                "risk": metrics["risk"],
                "coverage": metrics["coverage"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_source_comparison(
    context_rows: pd.DataFrame,
    two_sided_transfer: pd.DataFrame,
    one_sided_transfer: pd.DataFrame,
    crc_target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dataset = str(context_rows["dataset"].iloc[0])
    for method, group in context_rows.groupby("method", sort=False):
        row = {
            "row_type": "dataset_aggregate",
            "dataset": dataset,
            "context_id": "__aggregate__",
            "method": method,
            "guarantee_type": group["guarantee_type"].iloc[0],
            "n_contexts": len(group),
            "n_calibration_min": group["n_calibration"].min(),
            "n_calibration_median": group["n_calibration"].median(),
            "n_calibration_max": group["n_calibration"].max(),
            "dkw_epsilon_min": group["dkw_epsilon"].min(),
            "dkw_epsilon_median": group["dkw_epsilon"].median(),
            "dkw_epsilon_max": group["dkw_epsilon"].max(),
            "mean_empirical_source_risk": group["empirical_source_risk"].mean(),
            "mean_source_certificate": group["source_certificate"].mean(),
            "mean_source_calibration_coverage": group["source_calibration_coverage"].mean(),
            "source_nontrivial_fraction": group["source_nontrivial"].astype(float).mean(),
            "mean_heldout_source_risk": np.average(
                group["heldout_source_risk"], weights=group["n_heldout"]
            ),
            "mean_heldout_source_coverage": np.average(
                group["heldout_source_coverage"], weights=group["n_heldout"]
            ),
            "heldout_certificate_violation_fraction": group["certificate_violation"].mean(),
            "mean_certified_threshold_set_size": group["certified_threshold_set_size"].mean(),
        }
        if method == "two_sided_dkw_source":
            target = two_sided_transfer[two_sided_transfer["form"] == "T4_current_full"]
            row.update(_target_summary(target))
        elif method == "one_sided_dkw_source":
            row.update(_target_summary(one_sided_transfer))
        elif method == "crc_expected_risk_source":
            row.update(
                {
                    "target_risk": crc_target["risk"].mean(),
                    "target_coverage": crc_target["coverage"].mean(),
                    "target_nontrivial_fraction": float((crc_target["coverage"] > 0).mean()),
                    "target_accept_none_fraction": float((crc_target["coverage"] == 0).mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _target_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "target_risk": frame["realized_risk"].mean(),
        "target_coverage": frame["coverage"].mean(),
        "target_nontrivial_fraction": frame["nontrivial_feasible"].astype(float).mean(),
        "target_accept_none_fraction": frame["forced_accept_none"].astype(float).mean(),
    }


def _build_budget_output(frame: pd.DataFrame, *, risk_budget: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["row_type"] = "window"
    components = (
        "empirical_source_risk",
        "source_uncertainty",
        "source_ucb",
        "multiplier_increment",
        "additive_margin",
        "distance_penalty",
        "embedding_penalty",
        "final_bound",
    )
    for column in components:
        frame[f"{column}_fraction_of_tau"] = frame[column] / float(risk_budget)
    summary_rows = []
    for statistic in ("mean", "median", "max"):
        values = getattr(frame[list(components)], statistic)()
        row: dict[str, Any] = {
            "row_type": f"dataset_{statistic}",
            "dataset": frame["dataset"].iloc[0],
            "side": "two_sided",
            "form": "T4_current_full",
            "window_id": f"__{statistic}__",
            "n_windows": len(frame),
        }
        for column in components:
            row[column] = values[column]
            row[f"{column}_fraction_of_tau"] = values[column] / float(risk_budget)
        summary_rows.append(row)
    return pd.concat([frame, pd.DataFrame(summary_rows)], ignore_index=True)


def _summarize_transfer_forms(frame: pd.DataFrame, risk_budget: float) -> pd.DataFrame:
    rows = []
    for (dataset, form), group in frame.groupby(["dataset", "form"], sort=False):
        bounds = group["operational_bound"]
        rows.append(
            {
                "dataset": dataset,
                "form": form,
                "risk": group["realized_risk"].mean(),
                "coverage": group["coverage"].mean(),
                "bound_underestimation": float((bounds + 1e-12 < group["realized_risk"]).mean()),
                "mean_bound_gap": (bounds - group["realized_risk"]).mean(),
                "nontrivial_feasible_fraction": group["nontrivial_feasible"].astype(float).mean(),
                "accept_none_fraction": group["forced_accept_none"].astype(float).mean(),
                "mean_raw_final_bound": group["final_bound"].mean(),
                "mean_operational_bound": bounds.mean(),
                "all_nontrivial_bounds_feasible": bool(
                    (bounds[~group["forced_accept_none"].astype(bool)] <= risk_budget + 1e-12).all()
                ),
                "diagnostic_only": form != "T4_current_full",
            }
        )
    return pd.DataFrame(rows)


def build_guarantee_type_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "pointwise_empirical_bernstein",
                "guarantee_type": "pointwise high-probability only; diagnostic after adaptive selection",
                "adaptive_threshold_selection_allowed": False,
                "simultaneous_contexts_handled": "only with context correction; theta selection remains invalid",
                "numeric_upper_bound_available": True,
                "appropriate_for_direct_ct_transfer": False,
            },
            {
                "method": "two_sided_dkw_source",
                "guarantee_type": "1-alpha high-probability, uniform over all thresholds",
                "adaptive_threshold_selection_allowed": True,
                "simultaneous_contexts_handled": "yes, delta_j=alpha/m",
                "numeric_upper_bound_available": True,
                "appropriate_for_direct_ct_transfer": True,
            },
            {
                "method": "one_sided_dkw_source",
                "guarantee_type": "1-alpha high-probability, one-sided uniform upper envelope",
                "adaptive_threshold_selection_allowed": True,
                "simultaneous_contexts_handled": "yes, delta_j=alpha/m",
                "numeric_upper_bound_available": True,
                "appropriate_for_direct_ct_transfer": True,
            },
            {
                "method": "crc_expected_risk_source",
                "guarantee_type": "expected-risk control under exchangeability and monotonicity",
                "adaptive_threshold_selection_allowed": True,
                "simultaneous_contexts_handled": "not as a simultaneous high-probability event",
                "numeric_upper_bound_available": False,
                "appropriate_for_direct_ct_transfer": False,
            },
            {
                "method": "ltt_bonferroni_source",
                "guarantee_type": "1-alpha high-probability finite-family FWER",
                "adaptive_threshold_selection_allowed": True,
                "simultaneous_contexts_handled": "yes, context and grid Bonferroni",
                "numeric_upper_bound_available": False,
                "appropriate_for_direct_ct_transfer": False,
            },
            {
                "method": "ltt_fixed_sequence_source",
                "guarantee_type": "1-alpha high-probability finite-family fixed-sequence FWER",
                "adaptive_threshold_selection_allowed": True,
                "simultaneous_contexts_handled": "yes, context Bonferroni; within-context fixed sequence",
                "numeric_upper_bound_available": False,
                "appropriate_for_direct_ct_transfer": False,
            },
        ]
    )


def _assert_matches_fixed_evaluation(dataset_dir: Path, transfer: pd.DataFrame) -> None:
    saved = pd.read_csv(dataset_dir / "results" / "dkw_target_feasible_ct_windows.csv")
    current = transfer[transfer["form"] == "T4_current_full"].sort_values("window_id")
    saved = saved.sort_values("window_id")
    if not np.allclose(current["threshold"], saved["threshold"], atol=1e-12, rtol=0):
        raise RuntimeError("Recomputed T4 thresholds differ from the fixed evaluation selections")
    if not np.allclose(current["coverage"], saved["coverage"], atol=1e-12, rtol=0):
        raise RuntimeError("Recomputed T4 coverage differs from the fixed evaluation selections")


def _nearest_context(query: np.ndarray, centroids: Mapping[str, np.ndarray]) -> tuple[str, float]:
    choices = [(context, float(np.linalg.norm(query - centroid))) for context, centroid in centroids.items()]
    return min(choices, key=lambda item: item[1])


def _evaluate_threshold(examples: pd.DataFrame, threshold: float, risk_config: Any) -> dict[str, float]:
    accepted = apply_threshold_policy(
        examples, risk_config.score_column, float(threshold), risk_config.accept_if
    )
    losses = compute_losses(examples, accepted, risk_config.loss)
    summary = summarize_policy_performance(examples, accepted, losses)
    return {"risk": float(summary["mean_loss"]), "coverage": float(summary["coverage"])}


def _examples_for_context_windows(
    examples: pd.DataFrame, windows: pd.DataFrame, context_id: str
) -> pd.DataFrame:
    selected = windows[windows["context_id"].astype(str) == str(context_id)]
    indices = [int(value) for values in selected["example_indices"] for value in values]
    return examples.iloc[indices].reset_index(drop=True)


def _example_ids(examples: pd.DataFrame, windows: pd.DataFrame) -> set[str]:
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
