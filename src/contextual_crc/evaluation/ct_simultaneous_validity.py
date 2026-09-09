from __future__ import annotations

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
from contextual_crc.evaluation.synthetic_experiment import (
    parse_ctcrc_config,
    parse_risk_control_config,
)
from contextual_crc.risk_control.ct_crc import (
    CTCRCConfig,
    estimate_lipschitz_from_contexts,
    evaluate_ctcrc_windows,
    validated_contexts_to_frame,
)
from contextual_crc.risk_control.ct_crc_simultaneous import (
    ALLOWED_SIMULTANEOUS_VARIANTS,
    allocate_context_delta,
    build_simultaneously_validated_contexts,
    count_candidate_thresholds,
)
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


SUMMARY_FILENAME = "ct_simultaneous_validity.csv"
CONTEXT_AUDIT_FILENAME = "ct_simultaneous_validity_contexts.csv"


def run_ct_simultaneous_validity(
    config: dict[str, Any],
    *,
    library_delta: float = 0.05,
    overwrite: bool = False,
    output_dir: str | Path = "outputs/runs/ct_simultaneous_validity",
) -> Path:
    """Evaluate corrected CT variants from existing prediction and run caches."""
    output_path = Path(output_dir)
    summary_path = output_path / SUMMARY_FILENAME
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {summary_path}")

    run_dirs = [Path(path) for path in config.get("paths", {}).get("run_dirs", [])]
    if not run_dirs:
        raise ValueError("artifact config must define paths.run_dirs")

    ensure_dir(output_path)
    windows_dir = ensure_dir(output_path / "windows")
    summary_rows: list[dict[str, Any]] = []
    context_frames: list[pd.DataFrame] = []
    started_at = datetime.now(timezone.utc)

    for run_dir in run_dirs:
        rows, contexts, variant_windows = evaluate_run_from_cached_data(
            run_dir,
            library_delta=library_delta,
        )
        summary_rows.extend(rows)
        context_frames.append(contexts)
        for variant, windows in variant_windows.items():
            dataset = str(rows[0]["dataset"])
            safe_dataset = dataset.replace("/", "_").replace("\\", "_")
            windows.to_csv(windows_dir / f"{safe_dataset}__{variant}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    context_audit = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    context_audit.to_csv(output_path / CONTEXT_AUDIT_FILENAME, index=False)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime_seconds": (finished_at - started_at).total_seconds(),
        "library_delta": library_delta,
        "run_dirs": [str(path) for path in run_dirs],
        "variants": ["original_ct_crc", *ALLOWED_SIMULTANEOUS_VARIANTS],
        "uses_cached_predictions": True,
        "reruns_model_inference": False,
        "validity_audit": {
            "empirical_bernstein_is_pointwise": True,
            "uniform_over_candidate_thresholds": False,
            "adaptive_threshold_selection_uses_same_calibration_data": True,
            "threshold_level_correction_required": True,
        },
        "metric_definitions": {
            "feasible_window_fraction": "fraction with transferred_upper_bound <= risk_budget",
            "candidate_threshold_count": "finite unique scores plus accept-all and accept-none extremes",
        },
        "summary_csv": str(summary_path),
        "context_audit_csv": str(output_path / CONTEXT_AUDIT_FILENAME),
    }
    _write_json(manifest, output_path / "run_manifest.json")
    return summary_path


def evaluate_run_from_cached_data(
    run_dir: str | Path,
    *,
    library_delta: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate one saved run without rebuilding features, embeddings, or inference."""
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "results" / "run_manifest.json")
    resolved_config = load_config(run_path / "configs" / "resolved_config.yaml")
    cache_dir = Path(manifest["cache_dir"])
    if not cache_exists(cache_dir):
        raise FileNotFoundError(f"Prediction cache is missing or incomplete: {cache_dir}")

    examples = pd.read_parquet(cache_dir / "examples.parquet", engine="pyarrow")
    windows = load_windows(run_path / "intermediate" / "windows.parquet")
    embeddings = pd.read_parquet(run_path / "intermediate" / "embeddings.parquet", engine="pyarrow")
    calibration_contexts = [str(value) for value in manifest.get("calibration_contexts", [])]
    evaluation_contexts = [str(value) for value in manifest.get("evaluation_contexts", [])]
    if not calibration_contexts or not evaluation_contexts:
        raise ValueError(f"Saved run manifest lacks calibration/evaluation contexts: {run_path}")

    calibration_examples = examples[examples["context_id"].isin(calibration_contexts)].reset_index(drop=True)
    calibration_embeddings = embeddings[
        embeddings["context_id"].isin(calibration_contexts)
    ].reset_index(drop=True)
    evaluation_windows = windows[windows["context_id"].isin(evaluation_contexts)].reset_index(drop=True)
    evaluation_window_ids = set(evaluation_windows["window_id"])
    evaluation_embeddings = embeddings[
        embeddings["window_id"].isin(evaluation_window_ids)
    ].reset_index(drop=True)

    risk_config = parse_risk_control_config(resolved_config.get("risk_control", {}))
    base_ct_config = parse_ctcrc_config(resolved_config.get("ct_crc", {}), risk_config)
    candidate_counts = count_candidate_thresholds(
        calibration_examples,
        score_column=risk_config.score_column,
        accept_if=risk_config.accept_if,
    )
    dataset = _dataset_name(manifest, run_path.name)

    original_windows = pd.read_csv(run_path / "results" / "ct_crc_windows.csv")
    rows = [
        _summary_row(
            dataset=dataset,
            run_name=run_path.name,
            variant="original_ct_crc",
            windows=original_windows,
            candidate_counts=candidate_counts,
            risk_budget=risk_config.risk_budget,
            overall_library_delta=float("nan"),
            per_context_delta=risk_config.delta,
            threshold_selection_correction="none",
            notes=(
                "Saved operating point; delta is pointwise per context and adaptive "
                "threshold selection is uncorrected."
            ),
        )
    ]

    context_frames: list[pd.DataFrame] = []
    variant_windows: dict[str, pd.DataFrame] = {}
    for variant in ALLOWED_SIMULTANEOUS_VARIANTS:
        ct_result, context_audit = _build_variant_with_auto_l_hat(
            calibration_examples,
            calibration_embeddings,
            base_ct_config,
            raw_ct_config=resolved_config.get("ct_crc", {}),
            library_delta=library_delta,
            variant=variant,
        )
        evaluated = evaluate_ctcrc_windows(
            examples,
            evaluation_windows,
            evaluation_embeddings,
            ct_result,
        )
        evaluated.insert(0, "variant", variant)
        evaluated.insert(0, "dataset", dataset)
        variant_windows[variant] = evaluated

        context_audit.insert(0, "run_name", run_path.name)
        context_audit.insert(0, "dataset", dataset)
        context_audit["l_hat"] = ct_result.config.l_hat
        context_frames.append(context_audit)

        correction = (
            "none; library-only Bonferroni (threshold selection remains pointwise)"
            if variant == "ct_crc_library_bonferroni"
            else "Bonferroni within each context over its exact T_j candidates"
        )
        notes = (
            f"Overall {library_delta:g} budget split across contexts; selected-threshold validity is conditional."
            if variant == "ct_crc_library_bonferroni"
            else (
                f"Overall {library_delta:g} budget split across contexts and all adaptively "
                "searched thresholds."
            )
        )
        rows.append(
            _summary_row(
                dataset=dataset,
                run_name=run_path.name,
                variant=variant,
                windows=evaluated,
                candidate_counts=candidate_counts,
                risk_budget=risk_config.risk_budget,
                overall_library_delta=library_delta,
                per_context_delta=allocate_context_delta(library_delta, len(candidate_counts)),
                threshold_selection_correction=correction,
                notes=notes,
            )
        )

    return rows, pd.concat(context_frames, ignore_index=True), variant_windows


def _build_variant_with_auto_l_hat(
    calibration_examples: pd.DataFrame,
    calibration_embeddings: pd.DataFrame,
    base_config: CTCRCConfig,
    *,
    raw_ct_config: dict[str, Any],
    library_delta: float,
    variant: str,
) -> tuple[Any, pd.DataFrame]:
    initial_result, _ = build_simultaneously_validated_contexts(
        calibration_examples,
        calibration_embeddings,
        base_config,
        library_delta=library_delta,
        variant=variant,
    )
    if raw_ct_config.get("l_hat") != "auto":
        return build_simultaneously_validated_contexts(
            calibration_examples,
            calibration_embeddings,
            base_config,
            library_delta=library_delta,
            variant=variant,
        )

    context_frame = validated_contexts_to_frame(initial_result.contexts)
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
    final_config = replace(base_config, l_hat=l_hat)
    return build_simultaneously_validated_contexts(
        calibration_examples,
        calibration_embeddings,
        final_config,
        library_delta=library_delta,
        variant=variant,
    )


def _summary_row(
    *,
    dataset: str,
    run_name: str,
    variant: str,
    windows: pd.DataFrame,
    candidate_counts: dict[str, int],
    risk_budget: float,
    overall_library_delta: float,
    per_context_delta: float,
    threshold_selection_correction: str,
    notes: str,
) -> dict[str, Any]:
    summary = summarize_window_results(
        windows,
        risk_budget=risk_budget,
        upper_bound_col="transferred_upper_bound",
        method_name=variant,
    )
    counts = np.asarray(list(candidate_counts.values()), dtype=float)
    bounds = pd.to_numeric(windows["transferred_upper_bound"], errors="coerce")
    return {
        "dataset": dataset,
        "run_name": run_name,
        "variant": variant,
        "overall_library_delta": overall_library_delta,
        "n_validated_contexts": len(candidate_counts),
        "per_context_delta": per_context_delta,
        "threshold_selection_correction": threshold_selection_correction,
        "mean_candidate_threshold_count": float(np.mean(counts)),
        "min_candidate_threshold_count": int(np.min(counts)),
        "max_candidate_threshold_count": int(np.max(counts)),
        "risk": summary["mean_risk"],
        "coverage": summary["mean_coverage"],
        "bound_underestimation": summary["bound_underestimation_rate"],
        "mean_bound_gap": summary["mean_bound_gap"],
        "feasible_window_fraction": safe_mean(bounds <= risk_budget),
        "n_evaluation_windows": int(len(windows)),
        "mean_selected_threshold": safe_mean(windows["threshold"]),
        "risk_budget": risk_budget,
        "notes": notes,
    }


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
