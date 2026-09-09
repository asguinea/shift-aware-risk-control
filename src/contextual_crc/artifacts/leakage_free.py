from __future__ import annotations

from pathlib import Path

import pandas as pd

from contextual_crc.artifacts.make_tables import dataframe_to_latex_tabular
from contextual_crc.utils.paths import ensure_dir


DATASET_DISPLAY = {
    "synthetic_stronger_contextual_shift": "Synthetic",
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "wilds_camelyon17": "Camelyon17-WILDS",
}


def make_leakage_free_main_table(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "method",
        "risk",
        "coverage",
        "bound_underestimation",
        "mean_bound_gap",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"leakage-free results missing columns: {missing}")
    output = results[
        [
            "dataset",
            "method",
            "risk",
            "coverage",
            "bound_underestimation",
            "mean_bound_gap",
            "n_calibration_examples",
            "n_calibration_windows",
            "n_evaluation_examples",
            "n_evaluation_windows",
        ]
    ].copy()
    output["dataset"] = output["dataset"].map(
        lambda value: DATASET_DISPLAY.get(str(value), str(value))
    )
    return output.rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "risk": "Risk",
            "coverage": "Coverage",
            "bound_underestimation": "Bound underestimation",
            "mean_bound_gap": "Mean bound gap",
            "n_calibration_examples": "Cal. examples",
            "n_calibration_windows": "Cal. windows",
            "n_evaluation_examples": "Eval. examples",
            "n_evaluation_windows": "Eval. windows",
        }
    )


def make_leakage_free_matched_table(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "method",
        "global_coverage_target",
        "risk_at_global_coverage",
        "coverage_at_tau_0_10",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"matched-frontier results missing columns: {missing}")
    output = results[
        [
            "dataset",
            "method",
            "global_coverage_target",
            "risk_at_global_coverage",
            "coverage_at_tau_0_10",
            "post_hoc_evaluation",
        ]
    ].copy()
    output["dataset"] = output["dataset"].map(
        lambda value: DATASET_DISPLAY.get(str(value), str(value))
    )
    return output.rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "global_coverage_target": "Global coverage target",
            "risk_at_global_coverage": "Risk at Global coverage",
            "coverage_at_tau_0_10": "Coverage at risk 0.10",
            "post_hoc_evaluation": "Post-hoc evaluation",
        }
    )


def write_leakage_free_artifacts(
    tables_dir: str | Path,
    *,
    main_results_csv: str | Path = "outputs/runs/leakage_free_main/main_results.csv",
    matched_frontier_csv: str | Path = "outputs/runs/leakage_free_main/matched_frontier.csv",
    precision: int = 3,
) -> dict[str, str]:
    """Write tables when leakage-free evaluation files are available."""
    main_path = Path(main_results_csv)
    matched_path = Path(matched_frontier_csv)
    if not main_path.is_file() or not matched_path.is_file():
        return {}
    try:
        main = pd.read_csv(main_path)
        matched = pd.read_csv(matched_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if main.empty or matched.empty:
        return {}

    output_dir = ensure_dir(tables_dir)
    main_table = make_leakage_free_main_table(main)
    matched_table = make_leakage_free_matched_table(matched)
    main_csv = output_dir / "leakage_free_main_results.csv"
    main_tex = output_dir / "leakage_free_main_results.tex"
    matched_csv = output_dir / "leakage_free_matched_frontier.csv"
    matched_tex = output_dir / "leakage_free_matched_frontier.tex"
    main_table.to_csv(main_csv, index=False)
    matched_table.to_csv(matched_csv, index=False)
    main_tex.write_text(
        dataframe_to_latex_tabular(
            main_table,
            caption="Main results on disjoint, leakage-free calibration/evaluation windows.",
            label="tab:leakage-free-main",
            precision=precision,
        ),
        encoding="utf-8",
    )
    matched_tex.write_text(
        dataframe_to_latex_tabular(
            matched_table,
            caption=(
                "Post-hoc risk--coverage comparisons on the common leakage-free evaluation set; "
                "evaluation labels are used only for these descriptive frontier metrics."
            ),
            label="tab:leakage-free-matched",
            precision=precision,
        ),
        encoding="utf-8",
    )
    return {
        "leakage_free_main_results_csv": str(main_csv),
        "leakage_free_main_results_tex": str(main_tex),
        "leakage_free_matched_frontier_csv": str(matched_csv),
        "leakage_free_matched_frontier_tex": str(matched_tex),
    }
