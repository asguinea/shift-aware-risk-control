from __future__ import annotations

from pathlib import Path

import pandas as pd

from contextual_crc.artifacts.make_tables import dataframe_to_latex_tabular
from contextual_crc.utils.paths import ensure_dir


def make_transfer_penalty_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "family",
        "is_primary",
        "risk",
        "coverage",
        "bound_underestimation",
        "mean_bound_gap",
        "nontrivial_feasible_fraction",
        "accept_none_fraction",
        "mean_gamma",
        "mean_source_ucb",
        "mean_final_ucb",
    ]
    return results.reindex(columns=columns).rename(
        columns={
            "dataset": "Dataset",
            "family": "Penalty family",
            "is_primary": "Primary",
            "risk": "Risk",
            "coverage": "Coverage",
            "bound_underestimation": "Bound underestimation",
            "mean_bound_gap": "Mean bound gap",
            "nontrivial_feasible_fraction": "Nontrivial fraction",
            "accept_none_fraction": "Accept-none fraction",
            "mean_gamma": "Mean Gamma",
            "mean_source_ucb": "Mean source UCB",
            "mean_final_ucb": "Mean final UCB",
        }
    )


def make_certifiability_table(certifiability: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "family",
        "source_certifiable_fraction",
        "target_certifiable_fraction",
        "n_validated_contexts",
        "n_evaluation_windows",
    ]
    return certifiability.reindex(columns=columns).rename(
        columns={
            "dataset": "Dataset",
            "family": "Penalty family",
            "source_certifiable_fraction": "Source-certifiable fraction",
            "target_certifiable_fraction": "Target-certifiable fraction",
            "n_validated_contexts": "Validated contexts",
            "n_evaluation_windows": "Evaluation windows",
        }
    )


def write_transfer_penalty_artifacts(
    tables_dir: str | Path,
    *,
    final_results_csv: str | Path = "outputs/runs/transfer_penalty_audit/final_transfer_results.csv",
    certifiability_csv: str | Path = "outputs/runs/transfer_penalty_audit/certifiability.csv",
    precision: int = 3,
) -> dict[str, str]:
    final_path = Path(final_results_csv)
    cert_path = Path(certifiability_csv)
    if not final_path.is_file() or not cert_path.is_file():
        return {}
    try:
        final = pd.read_csv(final_path)
        cert = pd.read_csv(cert_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if final.empty or cert.empty:
        return {}
    output_dir = ensure_dir(tables_dir)
    result_table = make_transfer_penalty_table(final)
    cert_table = make_certifiability_table(cert)
    result_csv = output_dir / "transfer_penalty_results.csv"
    result_tex = output_dir / "transfer_penalty_results.tex"
    cert_csv = output_dir / "certifiability.csv"
    cert_tex = output_dir / "certifiability.tex"
    result_table.to_csv(result_csv, index=False)
    cert_table.to_csv(cert_csv, index=False)
    result_tex.write_text(
        dataframe_to_latex_tabular(
            result_table,
            caption="Transfer-penalty families on the frozen leakage-free evaluation set.",
            label="tab:transfer-penalty-results",
            precision=precision,
        ),
        encoding="utf-8",
    )
    cert_tex.write_text(
        dataframe_to_latex_tabular(
            cert_table,
            caption="Source and target certifiability under one-sided DKW calibration.",
            label="tab:transfer-certifiability",
            precision=precision,
        ),
        encoding="utf-8",
    )
    return {
        "transfer_penalty_results_csv": str(result_csv),
        "transfer_penalty_results_tex": str(result_tex),
        "certifiability_csv": str(cert_csv),
        "certifiability_tex": str(cert_tex),
    }
