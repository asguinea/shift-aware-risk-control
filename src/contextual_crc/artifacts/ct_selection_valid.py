from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from contextual_crc.artifacts.make_tables import dataframe_to_latex_tabular
from contextual_crc.utils.paths import ensure_dir


VARIANT_COLUMNS = {
    "original_ct": "Original CT",
    "library_bonferroni_ct": "Library-corrected CT",
    "fully_threshold_bonferroni_ct": "Threshold-Bonferroni CT",
    "dkw_target_feasible_ct": "DKW target-feasible CT",
}

DATASET_DISPLAY = {
    "synthetic_stronger_contextual_shift": "Synthetic",
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "wilds_camelyon17": "Camelyon17-WILDS",
}


def make_ct_selection_valid_table(
    results: pd.DataFrame,
    *,
    precision: int = 3,
) -> pd.DataFrame:
    """Make the compact risk/coverage/underestimation/nontrivial-feasibility table."""
    required = {
        "dataset",
        "variant",
        "risk",
        "coverage",
        "bound_underestimation",
        "nontrivial_feasible_fraction",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"CT selection-valid results missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for dataset, group in results.groupby("dataset", sort=False, dropna=False):
        row: dict[str, Any] = {"Dataset": DATASET_DISPLAY.get(str(dataset), str(dataset))}
        indexed = group.set_index("variant", drop=False)
        for variant, column in VARIANT_COLUMNS.items():
            if variant not in indexed.index:
                row[column] = "--"
                continue
            result = indexed.loc[variant]
            if isinstance(result, pd.DataFrame):
                result = result.iloc[0]
            row[column] = " / ".join(
                f"{float(result[name]):.{precision}f}"
                for name in (
                    "risk",
                    "coverage",
                    "bound_underestimation",
                    "nontrivial_feasible_fraction",
                )
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["Dataset", *VARIANT_COLUMNS.values()])


def write_ct_selection_valid_artifacts(
    tables_dir: str | Path,
    *,
    results_csv: str | Path = "outputs/runs/ct_selection_valid/ct_selection_valid_results.csv",
    precision: int = 3,
) -> dict[str, str]:
    """Write optional selection-validity artifacts when raw results exist."""
    source_path = Path(results_csv)
    if not source_path.is_file():
        return {}
    try:
        results = pd.read_csv(source_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if results.empty:
        return {}

    output_dir = ensure_dir(tables_dir)
    table = make_ct_selection_valid_table(results, precision=precision)
    csv_path = output_dir / "ct_selection_valid.csv"
    tex_path = output_dir / "ct_selection_valid.tex"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        dataframe_to_latex_tabular(
            table,
            caption=(
                "Selection-valid and target-feasible CT-CRC. Entries are risk / coverage / "
                "bound-underestimation / nontrivial-feasible-window fraction."
            ),
            label="tab:ct-selection-valid",
            precision=precision,
        ),
        encoding="utf-8",
    )
    return {
        "ct_selection_valid_csv": str(csv_path),
        "ct_selection_valid_tex": str(tex_path),
    }
