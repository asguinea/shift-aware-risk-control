from __future__ import annotations

import pandas as pd

from contextual_crc.signatures.io import (
    load_signatures,
    load_windows,
    save_signatures,
    save_windows,
)


def test_save_load_windows_round_trips_example_indices(tmp_path) -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["w0", "w1"],
            "dataset": ["fixture", "fixture"],
            "split": ["eval", "eval"],
            "context_id": ["ctx", "ctx"],
            "context_name": ["Context", "Context"],
            "start_pos": [0, 2],
            "end_pos": [2, 4],
            "n_examples": [2, 2],
            "example_indices": [[0, 1], [2, 3]],
        }
    )
    path = tmp_path / "nested" / "windows.parquet"

    saved_path = save_windows(windows, path)
    loaded = load_windows(saved_path)

    assert saved_path == path
    assert loaded["example_indices"].tolist() == [[0, 1], [2, 3]]
    assert loaded.shape == windows.shape


def test_save_load_signatures_round_trips_shape_and_columns(tmp_path) -> None:
    signatures = pd.DataFrame(
        {
            "window_id": ["w0", "w1"],
            "dataset": ["fixture", "fixture"],
            "split": ["eval", "eval"],
            "context_id": ["ctx", "ctx"],
            "context_name": ["Context", "Context"],
            "start_pos": [0, 2],
            "end_pos": [2, 4],
            "n_examples": [2, 2],
            "confidence_mean": [0.8, 0.7],
        }
    )
    path = tmp_path / "signatures.parquet"

    saved_path = save_signatures(signatures, path)
    loaded = load_signatures(saved_path)

    assert saved_path == path
    assert loaded.shape == signatures.shape
    assert loaded.columns.tolist() == signatures.columns.tolist()
