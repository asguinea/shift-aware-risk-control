from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.cache.io import cache_exists, load_prediction_cache, save_prediction_cache


def make_valid_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["ex0", "ex1", "ex2", "ex3"],
            "dataset": ["fixture"] * 4,
            "split": ["test"] * 4,
            "context_id": ["clean", "clean", "shift", "shift"],
            "context_name": ["Clean", "Clean", "Shift", "Shift"],
            "label": [0, 1, 2, 1],
            "pred": [0, 2, 2, 1],
            "correct": [True, False, True, True],
            "confidence": [0.8, 0.6, 0.9, 0.7],
            "entropy": [0.5, 0.8, 0.3, 0.6],
            "margin": [1.2, 0.2, 1.5, 0.7],
            "energy": [-2.0, -1.5, -2.4, -1.8],
            "severity": [0, 1, 2, 3],
        }
    )


def make_valid_arrays() -> dict[str, np.ndarray]:
    return {
        "logits": np.array(
            [
                [2.0, 0.5, -1.0],
                [0.2, 1.0, 1.4],
                [-0.5, 0.3, 2.2],
                [0.1, 1.8, 0.4],
            ],
            dtype=float,
        )
    }


def make_valid_metadata() -> dict[str, object]:
    return {
        "dataset": "fixture",
        "split": "test",
        "context_fields": ["context_id", "context_name"],
    }


def test_save_and_load_prediction_cache_round_trip(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    saved_path = save_prediction_cache(
        cache_dir,
        make_valid_examples(),
        make_valid_arrays(),
        make_valid_metadata(),
    )

    assert saved_path == cache_dir
    assert cache_exists(cache_dir)
    assert (cache_dir / "metadata.json").is_file()
    assert (cache_dir / "examples.parquet").is_file()
    assert (cache_dir / "arrays.npz").is_file()

    examples, arrays, metadata = load_prediction_cache(cache_dir)

    assert len(examples) == 4
    assert arrays["logits"].shape == (4, 3)
    assert metadata["dataset"] == "fixture"
    assert metadata["split"] == "test"
    assert metadata["num_examples"] == 4
    assert metadata["num_classes"] == 3
    assert metadata["schema_version"] == "1.0"
    assert metadata["array_keys"] == ["logits"]
    assert "severity" in metadata["optional_columns"]


def test_save_prediction_cache_without_overwrite_raises_on_non_empty_dir(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    save_prediction_cache(cache_dir, make_valid_examples(), make_valid_arrays(), make_valid_metadata())

    with pytest.raises(FileExistsError):
        save_prediction_cache(
            cache_dir,
            make_valid_examples(),
            make_valid_arrays(),
            make_valid_metadata(),
            overwrite=False,
        )


def test_save_prediction_cache_with_overwrite_succeeds(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    save_prediction_cache(cache_dir, make_valid_examples(), make_valid_arrays(), make_valid_metadata())

    save_prediction_cache(
        cache_dir,
        make_valid_examples(),
        make_valid_arrays(),
        make_valid_metadata(),
        overwrite=True,
    )

    assert cache_exists(cache_dir)
