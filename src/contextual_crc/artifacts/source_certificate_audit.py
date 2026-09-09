from __future__ import annotations

from pathlib import Path

import pandas as pd

from contextual_crc.artifacts.make_tables import dataframe_to_latex_tabular
from contextual_crc.utils.paths import ensure_dir


def make_source_certificate_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = comparison[comparison["row_type"] == "dataset_aggregate"].copy()
    columns = [
        "dataset",
        "method",
        "mean_empirical_source_risk",
        "mean_source_certificate",
        "mean_source_calibration_coverage",
        "source_nontrivial_fraction",
        "mean_heldout_source_risk",
        "mean_heldout_source_coverage",
        "target_risk",
        "target_coverage",
    ]
    return rows.reindex(columns=columns).rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "mean_empirical_source_risk": "Cal. risk",
            "mean_source_certificate": "Source certificate",
            "mean_source_calibration_coverage": "Cal. coverage",
            "source_nontrivial_fraction": "Nontrivial fraction",
            "mean_heldout_source_risk": "Held-out risk",
            "mean_heldout_source_coverage": "Held-out coverage",
            "target_risk": "Target risk",
            "target_coverage": "Target coverage",
        }
    )


def make_budget_decomposition_table(budget: pd.DataFrame) -> pd.DataFrame:
    rows = budget[budget["row_type"] == "dataset_mean"].copy()
    columns = [
        "dataset",
        "empirical_source_risk",
        "source_uncertainty",
        "multiplier_increment",
        "additive_margin",
        "distance_penalty",
        "embedding_penalty",
        "final_bound",
    ]
    return rows.reindex(columns=columns).rename(
        columns={
            "dataset": "Dataset",
            "empirical_source_risk": "Source risk",
            "source_uncertainty": "Uncertainty",
            "multiplier_increment": "Multiplier",
            "additive_margin": "Margin",
            "distance_penalty": "Distance",
            "embedding_penalty": "Embedding",
            "final_bound": "Final U",
        }
    )


def write_source_certificate_artifacts(
    tables_dir: str | Path,
    *,
    comparison_csv: str | Path = (
        "outputs/runs/source_certificate_audit/source_certificate_comparison.csv"
    ),
    budget_csv: str | Path = "outputs/runs/source_certificate_audit/budget_decomposition.csv",
    precision: int = 3,
) -> dict[str, str]:
    comparison_path = Path(comparison_csv)
    budget_path = Path(budget_csv)
    if not comparison_path.is_file() or not budget_path.is_file():
        return {}
    try:
        comparison = pd.read_csv(comparison_path)
        budget = pd.read_csv(budget_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if comparison.empty or budget.empty:
        return {}
    output_dir = ensure_dir(tables_dir)
    comparison_table = make_source_certificate_table(comparison)
    budget_table = make_budget_decomposition_table(budget)
    comparison_out = output_dir / "source_certificate_comparison.csv"
    comparison_tex = output_dir / "source_certificate_comparison.tex"
    budget_out = output_dir / "ct_budget_decomposition.csv"
    budget_tex = output_dir / "ct_budget_decomposition.tex"
    comparison_table.to_csv(comparison_out, index=False)
    budget_table.to_csv(budget_out, index=False)
    comparison_tex.write_text(
        dataframe_to_latex_tabular(
            comparison_table,
            caption="Source-certificate comparison on the fixed evaluation split.",
            label="tab:source-certificate-comparison",
            precision=precision,
        ),
        encoding="utf-8",
    )
    budget_tex.write_text(
        dataframe_to_latex_tabular(
            budget_table,
            caption="Decomposition of the current CT transferred risk bound.",
            label="tab:ct-budget-decomposition",
            precision=precision,
        ),
        encoding="utf-8",
    )
    return {
        "source_certificate_comparison_csv": str(comparison_out),
        "source_certificate_comparison_tex": str(comparison_tex),
        "ct_budget_decomposition_csv": str(budget_out),
        "ct_budget_decomposition_tex": str(budget_tex),
    }
