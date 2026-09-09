#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.artifacts.export import (
    artifact_timestamp,
    ensure_artifact_dirs,
    load_many_run_outputs,
    load_run_outputs,
    write_json,
)
from contextual_crc.artifacts.cw_sensitivity import write_cw_sensitivity_artifacts
from contextual_crc.artifacts.ct_simultaneous_validity import write_ct_simultaneous_validity_artifacts
from contextual_crc.artifacts.ct_selection_valid import write_ct_selection_valid_artifacts
from contextual_crc.artifacts.leakage_free import write_leakage_free_artifacts
from contextual_crc.artifacts.source_certificate_audit import write_source_certificate_artifacts
from contextual_crc.artifacts.transfer_penalty_audit import write_transfer_penalty_artifacts
from contextual_crc.artifacts.embedding_ablations import write_embedding_ablation_artifacts
from contextual_crc.artifacts.make_figures import write_figures
from contextual_crc.artifacts.make_tables import write_tables
from contextual_crc.artifacts.risk_coverage import write_risk_coverage_artifacts
from contextual_crc.artifacts.seed_variability import write_seed_variability_artifacts
from contextual_crc.config.loader import load_config
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tables and figures from experiment runs.")
    parser.add_argument("--config", required=True, help="Path to artifact YAML config.")
    parser.add_argument("--output-root", default=None, help="Override artifact output root.")
    parser.add_argument(
        "--include-missing-runs",
        action="store_true",
        help="Record missing run directories in loaded outputs instead of skipping them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 123)))
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))

    paths_config = config.get("paths", {})
    artifacts_config = config.get("artifacts", {})
    requested_run_dirs = [Path(path) for path in paths_config.get("run_dirs", [])]
    requested_label_efficiency_dirs = [Path(path) for path in paths_config.get("label_efficiency_dirs", [])]
    requested_online_trace_dirs = [Path(path) for path in paths_config.get("online_trace_dirs", [])]
    include_missing = bool(artifacts_config.get("include_missing_runs", False)) or bool(args.include_missing_runs)
    output_root = Path(args.output_root or paths_config.get("output_root", "outputs/artifacts"))

    run_outputs = load_many_run_outputs(requested_run_dirs, include_missing=include_missing)
    valid_outputs = [output for output in run_outputs if bool(output.get("valid"))]
    online_trace_outputs, found_online_trace_dirs, missing_online_trace_dirs = _load_online_trace_outputs(
        requested_online_trace_dirs,
        include_missing=include_missing,
    )
    found_run_dirs = [str(output["run_dir"]) for output in valid_outputs]
    found_set = set(found_run_dirs)
    missing_run_dirs = [
        str(path)
        for path in requested_run_dirs
        if str(path) not in found_set and not (path / "results" / "method_comparison.csv").is_file()
    ]

    artifact_dirs = ensure_artifact_dirs(output_root)
    precision = int(artifacts_config.get("precision", 3))
    label_efficiency_frames, found_label_efficiency_dirs, missing_label_efficiency_dirs = _load_label_efficiency(
        requested_label_efficiency_dirs
    )
    table_paths = write_tables(
        valid_outputs,
        artifact_dirs["tables"],
        precision=precision,
        label_efficiency_frames=label_efficiency_frames,
        make_label_efficiency=bool(config.get("tables", {}).get("make_label_efficiency", False)),
    )
    figure_paths = write_figures(
        [*online_trace_outputs, *valid_outputs],
        artifact_dirs["figures"],
        figure_format=str(artifacts_config.get("figure_format", "pdf")),
        precision=precision,
        label_efficiency_frames=label_efficiency_frames,
        make_label_efficiency=bool(config.get("figures", {}).get("make_label_efficiency", False)),
    )
    risk_coverage_paths = write_risk_coverage_artifacts(
        valid_outputs,
        artifact_dirs["tables"],
        artifact_dirs["figures"],
        figure_format=str(artifacts_config.get("figure_format", "pdf")),
        precision=precision,
    )
    embedding_ablation_paths = write_embedding_ablation_artifacts(
        valid_outputs,
        artifact_dirs["tables"],
        artifact_dirs["figures"],
        figure_format=str(artifacts_config.get("figure_format", "pdf")),
        precision=precision,
    )
    cw_sensitivity_paths = write_cw_sensitivity_artifacts(
        artifact_dirs["tables"],
        artifact_dirs["figures"],
        sensitivity_csv=artifacts_config.get(
            "cw_sensitivity_csv",
            "outputs/runs/cw_sensitivity/cw_sensitivity.csv",
        ),
        figure_format=str(artifacts_config.get("figure_format", "pdf")),
        precision=precision,
    )
    seed_variability_paths = write_seed_variability_artifacts(
        artifact_dirs["tables"],
        artifact_dirs["figures"],
        metrics_csv=artifacts_config.get(
            "seed_variability_metrics_csv",
            "outputs/runs/seed_variability/seed_variability_metrics.csv",
        ),
        smoothness_csv=artifacts_config.get(
            "seed_variability_smoothness_csv",
            "outputs/runs/seed_variability/seed_variability_smoothness.csv",
        ),
        figure_format=str(artifacts_config.get("figure_format", "pdf")),
        precision=precision,
    )
    ct_simultaneous_validity_paths = write_ct_simultaneous_validity_artifacts(
        artifact_dirs["tables"],
        results_csv=artifacts_config.get(
            "ct_simultaneous_validity_csv",
            "outputs/runs/ct_simultaneous_validity/ct_simultaneous_validity.csv",
        ),
        precision=precision,
    )
    ct_selection_valid_paths = write_ct_selection_valid_artifacts(
        artifact_dirs["tables"],
        results_csv=artifacts_config.get(
            "ct_selection_valid_csv",
            "outputs/runs/ct_selection_valid/ct_selection_valid_results.csv",
        ),
        precision=precision,
    )
    leakage_free_paths = write_leakage_free_artifacts(
        artifact_dirs["tables"],
        main_results_csv=artifacts_config.get(
            "leakage_free_main_csv",
            "outputs/runs/leakage_free_main/main_results.csv",
        ),
        matched_frontier_csv=artifacts_config.get(
            "leakage_free_matched_csv",
            "outputs/runs/leakage_free_main/matched_frontier.csv",
        ),
        precision=precision,
    )
    source_certificate_paths = write_source_certificate_artifacts(
        artifact_dirs["tables"],
        comparison_csv=artifacts_config.get(
            "source_certificate_comparison_csv",
            "outputs/runs/source_certificate_audit/source_certificate_comparison.csv",
        ),
        budget_csv=artifacts_config.get(
            "ct_budget_decomposition_csv",
            "outputs/runs/source_certificate_audit/budget_decomposition.csv",
        ),
        precision=precision,
    )
    transfer_penalty_paths = write_transfer_penalty_artifacts(
        artifact_dirs["tables"],
        final_results_csv=artifacts_config.get(
            "transfer_penalty_results_csv",
            "outputs/runs/transfer_penalty_audit/final_transfer_results.csv",
        ),
        certifiability_csv=artifacts_config.get(
            "transfer_penalty_certifiability_csv",
            "outputs/runs/transfer_penalty_audit/certifiability.csv",
        ),
        precision=precision,
    )

    manifest = {
        "timestamp": artifact_timestamp(),
        "config_path": str(args.config),
        "output_root": str(output_root),
        "run_dirs_found": found_run_dirs,
        "run_dirs_missing": missing_run_dirs,
        "online_trace_dirs_found": found_online_trace_dirs,
        "online_trace_dirs_missing": missing_online_trace_dirs,
        "label_efficiency_dirs_found": found_label_efficiency_dirs,
        "label_efficiency_dirs_missing": missing_label_efficiency_dirs,
        "tables_written": table_paths,
        "figures_written": figure_paths,
        "risk_coverage_written": risk_coverage_paths,
        "embedding_ablations_written": embedding_ablation_paths,
        "cw_sensitivity_written": cw_sensitivity_paths,
        "seed_variability_written": seed_variability_paths,
        "ct_simultaneous_validity_written": ct_simultaneous_validity_paths,
        "ct_selection_valid_written": ct_selection_valid_paths,
        "leakage_free_written": leakage_free_paths,
        "source_certificate_audit_written": source_certificate_paths,
        "transfer_penalty_audit_written": transfer_penalty_paths,
    }
    write_json(manifest, artifact_dirs["root"] / "artifact_manifest.json")
    logger.info("Artifacts written to: %s", output_root)
    return 0


def _load_online_trace_outputs(
    online_trace_dirs: list[Path],
    *,
    include_missing: bool,
) -> tuple[list[dict], list[str], list[str]]:
    outputs: list[dict] = []
    found: list[str] = []
    missing: list[str] = []
    for directory in online_trace_dirs:
        path = directory / "results" / "online_trace.csv"
        if not path.is_file():
            missing.append(str(directory))
            if include_missing:
                outputs.append(load_run_outputs(directory))
            continue
        output = load_run_outputs(directory)
        trace = output.get("online_trace")
        if trace is None or trace.empty:
            missing.append(str(directory))
            continue
        outputs.append(output)
        found.append(str(directory))
    return outputs, found, missing


def _load_label_efficiency(label_efficiency_dirs: list[Path]) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    found: list[str] = []
    missing: list[str] = []
    for directory in label_efficiency_dirs:
        path = directory / "label_efficiency.csv"
        if not path.is_file():
            missing.append(str(directory))
            continue
        try:
            frames.append(pd.read_csv(path))
            found.append(str(directory))
        except (OSError, pd.errors.EmptyDataError):
            missing.append(str(directory))
    return frames, found, missing


if __name__ == "__main__":
    raise SystemExit(main())
