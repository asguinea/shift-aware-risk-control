from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from contextual_crc.utils.paths import ensure_dir


MAIN_RESULT_COLUMNS = (
    "dataset",
    "method",
    "mean_risk",
    "violation_rate",
    "mean_coverage",
    "mean_bound",
    "bound_underestimation_rate",
)

BOUND_DIAGNOSTIC_COLUMNS = (
    "dataset",
    "method",
    "dataset_display",
    "method_display",
    "mean_risk",
    "mean_bound",
    "violation_rate",
    "bound_underestimation_rate",
    "mean_bound_gap",
)

DATASET_SUMMARY_COLUMNS = (
    "dataset",
    "run_name",
    "num_examples",
    "num_windows",
    "num_signature_features",
    "num_calibration_contexts",
    "num_evaluation_contexts",
)

LABEL_EFFICIENCY_COLUMNS = (
    "dataset",
    "budget_windows_per_context",
    "actual_labeled_examples",
    "method",
    "mean_risk",
    "mean_coverage",
    "violation_rate",
    "mean_bound",
    "bound_underestimation_rate",
    "seed",
)

DATASET_DISPLAY_NAMES = {
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "synthetic_contextual_shift": "Synthetic",
    "synthetic_stronger_contextual_shift": "Synthetic",
    "officehome": "OfficeHome",
    "wilds_camelyon17": "Camelyon17-WILDS",
    "wilds_fmow": "FMoW-WILDS",
}


def format_float(value: Any, precision: int = 3) -> str:
    """Format values for stable human-readable tables."""
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "--"
    return f"{numeric:.{precision}f}"


def dataframe_to_latex_tabular(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
    column_format: str | None = None,
) -> str:
    """Render a simple LaTeX tabular/table string without pandas Styler."""
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: format_float(value, precision))
        else:
            display[column] = display[column].where(display[column].notna(), "--")

    kwargs: dict[str, Any] = {
        "index": False,
        "escape": True,
        "na_rep": "--",
    }
    if column_format is not None:
        kwargs["column_format"] = column_format
    if caption is not None:
        kwargs["caption"] = caption
    if label is not None:
        kwargs["label"] = label
    try:
        return display.to_latex(**kwargs)
    except ImportError:
        return _manual_latex_table(
            display,
            caption=caption,
            label=label,
            column_format=column_format,
        )


def make_main_results_table(
    run_outputs: Sequence[dict[str, Any]],
    *,
    precision: int = 3,
) -> pd.DataFrame:
    """Combine method-comparison outputs across runs."""
    del precision
    frames: list[pd.DataFrame] = []
    for output in run_outputs:
        frame = output.get("method_comparison")
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        working["dataset"] = output.get("dataset") or output.get("run_name") or "unknown"
        for column in MAIN_RESULT_COLUMNS:
            if column not in working.columns:
                working[column] = np.nan
        frames.append(working.loc[:, list(MAIN_RESULT_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=MAIN_RESULT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["dataset", "method"], kind="mergesort").reset_index(drop=True)


def make_bound_diagnostics_table(
    run_outputs: Sequence[dict[str, Any]],
    *,
    precision: int = 3,
) -> pd.DataFrame:
    """Create one CT/CW bound-diagnostic row per run and method."""
    del precision
    rows: list[dict[str, Any]] = []
    for output in run_outputs:
        dataset = output.get("dataset") or output.get("run_name") or "unknown"
        method_comparison = output.get("method_comparison")
        bound_diagnostics = output.get("bound_diagnostics") or {}
        for method, summary_key, diag_key, bound_col in (
            ("ct_crc", "ct_crc_summary", "ct_crc", "transferred_upper_bound"),
            ("cw_crc", "cw_crc_summary", "cw_crc", "weighted_risk_upper_bound"),
        ):
            summary = output.get(summary_key) or {}
            diagnostics = bound_diagnostics.get(diag_key, {}) if isinstance(bound_diagnostics, dict) else {}
            comparison_row = _method_comparison_row(method_comparison, method)
            if not summary and comparison_row is None and not diagnostics:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "dataset_display": _display_dataset(dataset),
                    "method_display": _display_method(method),
                    "mean_risk": _first_available(
                        summary.get("mean_risk"),
                        None if comparison_row is None else comparison_row.get("mean_risk"),
                    ),
                    "mean_bound": _first_available(
                        summary.get("mean_upper_bound"),
                        summary.get("mean_bound"),
                        None if comparison_row is None else comparison_row.get("mean_bound"),
                    ),
                    "violation_rate": _first_available(
                        summary.get("violation_rate"),
                        None if comparison_row is None else comparison_row.get("violation_rate"),
                    ),
                    "bound_underestimation_rate": _first_available(
                        diagnostics.get("bound_underestimation_rate"),
                        summary.get("bound_underestimation_rate"),
                        None if comparison_row is None else comparison_row.get("bound_underestimation_rate"),
                    ),
                    "mean_bound_gap": _first_available(diagnostics.get("mean_bound_gap")),
                    "bound_column": bound_col,
                }
            )
    if not rows:
        return pd.DataFrame(columns=BOUND_DIAGNOSTIC_COLUMNS)
    frame = pd.DataFrame(rows)
    return frame.loc[:, list(BOUND_DIAGNOSTIC_COLUMNS)].sort_values(
        ["dataset", "method"],
        kind="mergesort",
    ).reset_index(drop=True)


def make_dataset_summary_table(
    run_outputs: Sequence[dict[str, Any]],
    *,
    precision: int = 3,
) -> pd.DataFrame:
    """Summarize run sizes from manifests."""
    del precision
    rows: list[dict[str, Any]] = []
    for output in run_outputs:
        manifest = output.get("manifest") or {}
        if not manifest and not output.get("valid"):
            continue
        calibration_contexts = manifest.get("calibration_contexts")
        evaluation_contexts = manifest.get("evaluation_contexts")
        rows.append(
            {
                "dataset": output.get("dataset") or output.get("run_name") or "unknown",
                "run_name": output.get("run_name") or manifest.get("run_name") or "unknown",
                "num_examples": manifest.get("num_examples", np.nan),
                "num_windows": manifest.get("num_windows", np.nan),
                "num_signature_features": _first_available(
                    manifest.get("num_signature_features"),
                    _safe_len(manifest.get("signature_feature_columns")),
                ),
                "num_calibration_contexts": _first_available(
                    manifest.get("num_calibration_contexts"),
                    _safe_len(calibration_contexts),
                ),
                "num_evaluation_contexts": _first_available(
                    manifest.get("num_evaluation_contexts"),
                    _safe_len(evaluation_contexts),
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=DATASET_SUMMARY_COLUMNS)
    return pd.DataFrame(rows).loc[:, list(DATASET_SUMMARY_COLUMNS)].sort_values(
        ["dataset", "run_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def make_label_efficiency_table(
    label_efficiency_frames: Sequence[pd.DataFrame] | None,
    *,
    precision: int = 3,
) -> pd.DataFrame:
    """Combine optional label-efficiency outputs across runs."""
    del precision
    frames = [frame.copy() for frame in (label_efficiency_frames or []) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=LABEL_EFFICIENCY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    for column in LABEL_EFFICIENCY_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    return combined.loc[:, list(LABEL_EFFICIENCY_COLUMNS)].sort_values(
        ["dataset", "budget_windows_per_context", "method"],
        kind="mergesort",
    ).reset_index(drop=True)


def write_tables(
    run_outputs: Sequence[dict[str, Any]],
    output_tables_dir: str | Path,
    *,
    precision: int = 3,
    label_efficiency_frames: Sequence[pd.DataFrame] | None = None,
    make_label_efficiency: bool = False,
) -> dict[str, str]:
    """Write configured LaTeX and CSV tables."""
    tables_dir = ensure_dir(output_tables_dir)
    tables = {
        "main_results": (
            make_main_results_table(run_outputs, precision=precision),
            "Main Results",
            "tab:main-results",
        ),
        "bound_diagnostics": (
            make_bound_diagnostics_table(run_outputs, precision=precision),
            "Bound Diagnostics",
            "tab:bound-diagnostics",
        ),
        "dataset_summary": (
            make_dataset_summary_table(run_outputs, precision=precision),
            "Dataset Summary",
            "tab:dataset-summary",
        ),
    }
    if make_label_efficiency:
        tables["label_efficiency"] = (
            make_label_efficiency_table(label_efficiency_frames, precision=precision),
            "Label Efficiency",
            "tab:label-efficiency",
        )

    written: dict[str, str] = {}
    for name, (frame, caption, label) in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        tex_path = tables_dir / f"{name}.tex"
        frame.to_csv(csv_path, index=False)
        tex_path.write_text(
            dataframe_to_latex_tabular(frame, caption=caption, label=label, precision=precision),
            encoding="utf-8",
        )
        written[f"{name}_csv"] = str(csv_path)
        written[f"{name}_tex"] = str(tex_path)
    return written


def _method_comparison_row(frame: pd.DataFrame | None, method: str) -> dict[str, Any] | None:
    if frame is None or frame.empty or "method" not in frame.columns:
        return None
    rows = frame[frame["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _first_available(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return np.nan


def _safe_len(value: Any) -> int | float:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if value in (None, "all"):
        return np.nan
    return np.nan


def _display_dataset(dataset: Any) -> str:
    value = str(dataset)
    return DATASET_DISPLAY_NAMES.get(value, value.replace("_", " "))


def _display_method(method: Any) -> str:
    display_names = {
        "ct_crc": "CT-CRC",
        "cw_crc": "CW-CRC",
    }
    value = str(method)
    return display_names.get(value, value.replace("_", " "))


def _manual_latex_table(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
) -> str:
    """Fallback renderer for environments without pandas' LaTeX extras."""
    alignment = column_format or ("l" * len(df.columns))
    lines: list[str] = []
    if caption is not None or label is not None:
        lines.append("\\begin{table}")
        lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{alignment}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(str(column)) for column in df.columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(str(value)) for value in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption is not None:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label is not None:
        lines.append(f"\\label{{{_latex_escape(label)}}}")
    if caption is not None or label is not None:
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash ",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde ",
        "^": r"\textasciicircum ",
    }
    return "".join(replacements.get(character, character) for character in value)


def _manual_latex_table(
    df: pd.DataFrame,
    *,
    caption: str | None,
    label: str | None,
    column_format: str | None,
) -> str:
    column_format = column_format or ("l" * len(df.columns))
    lines = ["\\begin{table}", "\\centering"]
    if caption is not None:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label is not None:
        lines.append(f"\\label{{{_latex_escape(label)}}}")
    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(str(column)) for column in df.columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        values = [_latex_escape(str(value)) if str(value) else "--" for value in row.tolist()]
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)
