from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_WINDOW_COLUMNS = ("dataset", "split", "context_id", "context_name")


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 100
    stride: int = 50
    group_by_context: bool = True
    sort_by: str | None = "time_index"
    drop_incomplete: bool = True


def make_windows(
    examples: pd.DataFrame,
    config: WindowConfig,
) -> pd.DataFrame:
    """Construct deployment windows over cached prediction examples."""
    _validate_window_inputs(examples, config)

    working = examples.copy()
    working["_original_position"] = range(len(working))

    rows: list[dict[str, Any]] = []
    if config.group_by_context:
        grouped = working.groupby("context_id", sort=False, dropna=False)
        for _, group in grouped:
            sorted_group = _sort_examples(group, config)
            rows.extend(_windows_for_group(sorted_group, config))
    else:
        sorted_examples = _sort_examples(working, config)
        rows.extend(_windows_for_group(sorted_examples, config, force_mixed_context=True))

    return pd.DataFrame(
        rows,
        columns=[
            "window_id",
            "dataset",
            "split",
            "context_id",
            "context_name",
            "start_pos",
            "end_pos",
            "n_examples",
            "example_indices",
        ],
    )


def explode_window_indices(windows: pd.DataFrame) -> pd.DataFrame:
    """Expand list-valued window indices into a long debug-friendly table."""
    rows: list[dict[str, Any]] = []
    for _, row in windows.iterrows():
        for position, example_index in enumerate(row["example_indices"]):
            rows.append(
                {
                    "window_id": row["window_id"],
                    "example_index": int(example_index),
                    "position_in_window": position,
                }
            )
    return pd.DataFrame(rows, columns=["window_id", "example_index", "position_in_window"])


def _validate_window_inputs(examples: pd.DataFrame, config: WindowConfig) -> None:
    if config.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if config.stride <= 0:
        raise ValueError("stride must be > 0")

    missing = [column for column in REQUIRED_WINDOW_COLUMNS if column not in examples.columns]
    if missing:
        raise ValueError(f"examples is missing required columns for windowing: {missing}")


def _sort_examples(examples: pd.DataFrame, config: WindowConfig) -> pd.DataFrame:
    if config.sort_by is None or config.sort_by not in examples.columns:
        return examples

    sort_columns = [config.sort_by]
    if "example_id" in examples.columns and config.sort_by != "example_id":
        sort_columns.append("example_id")
    return examples.sort_values(sort_columns, kind="mergesort")


def _windows_for_group(
    group: pd.DataFrame,
    config: WindowConfig,
    force_mixed_context: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group = group.reset_index(drop=True)
    n = len(group)
    start = 0
    local_window_index = 0

    while start < n:
        end = min(start + config.window_size, n)
        if end - start < config.window_size and config.drop_incomplete:
            break

        window = group.iloc[start:end]
        metadata = _window_metadata(window, force_mixed_context=force_mixed_context)
        context_for_id = metadata["context_id"]
        rows.append(
            {
                "window_id": (
                    f"{metadata['dataset']}/{metadata['split']}/"
                    f"{context_for_id}/w{local_window_index:05d}"
                ),
                "dataset": metadata["dataset"],
                "split": metadata["split"],
                "context_id": metadata["context_id"],
                "context_name": metadata["context_name"],
                "start_pos": start,
                "end_pos": end,
                "n_examples": end - start,
                "example_indices": [int(value) for value in window["_original_position"].tolist()],
            }
        )
        local_window_index += 1
        start += config.stride

    return rows


def _window_metadata(window: pd.DataFrame, force_mixed_context: bool) -> dict[str, str]:
    dataset = _consistent_value_or_mixed(window["dataset"])
    split = _consistent_value_or_mixed(window["split"])
    context_id = _consistent_value_or_mixed(window["context_id"])
    context_name = _consistent_value_or_mixed(window["context_name"])

    if force_mixed_context and window["context_id"].nunique(dropna=False) > 1:
        context_id = "mixed"
        context_name = "mixed"

    return {
        "dataset": dataset,
        "split": split,
        "context_id": context_id,
        "context_name": context_name,
    }


def _consistent_value_or_mixed(values: pd.Series) -> str:
    unique_values = list(values.drop_duplicates())
    if len(unique_values) == 1:
        return str(unique_values[0])
    return "mixed"
