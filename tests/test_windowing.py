from __future__ import annotations

import pandas as pd
import pytest

from contextual_crc.signatures.windowing import (
    WindowConfig,
    explode_window_indices,
    make_windows,
)


def make_examples(num_per_context: int = 6) -> pd.DataFrame:
    rows = []
    for context_index, context_id in enumerate(("ctx_a", "ctx_b")):
        for local_index in range(num_per_context):
            rows.append(
                {
                    "example_id": f"{context_id}_{local_index}",
                    "dataset": "fixture",
                    "split": "eval",
                    "context_id": context_id,
                    "context_name": f"Context {context_index}",
                    "time_index": local_index,
                }
            )
    return pd.DataFrame(rows)


def test_make_windows_grouped_complete() -> None:
    examples = make_examples(num_per_context=6)
    windows = make_windows(
        examples,
        WindowConfig(window_size=3, stride=3, group_by_context=True),
    )

    assert len(windows) == 4
    assert all(window_indices == 3 for window_indices in windows["n_examples"])
    assert all(context_id in {"ctx_a", "ctx_b"} for context_id in windows["context_id"])
    assert windows.iloc[0]["example_indices"] == [0, 1, 2]
    assert windows.iloc[1]["example_indices"] == [3, 4, 5]
    assert windows.iloc[2]["example_indices"] == [6, 7, 8]
    assert windows.iloc[3]["example_indices"] == [9, 10, 11]


def test_make_windows_drop_incomplete_false() -> None:
    examples = make_examples(num_per_context=5).iloc[:5].copy()
    windows = make_windows(
        examples,
        WindowConfig(
            window_size=3,
            stride=3,
            group_by_context=True,
            drop_incomplete=False,
        ),
    )

    assert len(windows) == 2
    assert windows.iloc[0]["n_examples"] == 3
    assert windows.iloc[1]["n_examples"] == 2
    assert windows.iloc[1]["example_indices"] == [3, 4]


def test_make_windows_mixed_context() -> None:
    examples = make_examples(num_per_context=3)
    windows = make_windows(
        examples,
        WindowConfig(
            window_size=4,
            stride=4,
            group_by_context=False,
            sort_by=None,
            drop_incomplete=False,
        ),
    )

    assert len(windows) == 2
    assert windows.iloc[0]["context_id"] == "mixed"
    assert windows.iloc[0]["context_name"] == "mixed"
    assert windows.iloc[1]["context_id"] == "ctx_b"


def test_explode_window_indices() -> None:
    examples = make_examples(num_per_context=3)
    windows = make_windows(
        examples,
        WindowConfig(window_size=2, stride=2, group_by_context=True, drop_incomplete=False),
    )

    exploded = explode_window_indices(windows)

    assert len(exploded) == int(windows["n_examples"].sum())
    assert set(exploded.columns) == {"window_id", "example_index", "position_in_window"}
    assert exploded.groupby("window_id")["position_in_window"].min().eq(0).all()


def test_invalid_window_config() -> None:
    examples = make_examples(num_per_context=3)

    with pytest.raises(ValueError, match="window_size"):
        make_windows(examples, WindowConfig(window_size=0))
    with pytest.raises(ValueError, match="stride"):
        make_windows(examples, WindowConfig(stride=0))
