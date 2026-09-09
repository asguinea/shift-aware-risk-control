from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contextual_crc.cache.validate import (
    validate_prediction_cache,
    validate_prediction_cache_dir,
)


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
        "num_examples": 4,
        "num_classes": 3,
        "created_at": "2026-04-24T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits"],
        "required_columns": [
            "example_id",
            "dataset",
            "split",
            "context_id",
            "context_name",
            "label",
            "pred",
            "correct",
            "confidence",
            "entropy",
            "margin",
            "energy",
        ],
        "optional_columns": [],
        "context_fields": ["context_id", "context_name"],
    }


def test_valid_fixture_passes_validation() -> None:
    validate_prediction_cache(make_valid_examples(), make_valid_arrays(), make_valid_metadata())


def test_missing_required_column_raises_value_error() -> None:
    examples = make_valid_examples().drop(columns=["confidence"])

    with pytest.raises(ValueError, match="confidence"):
        validate_prediction_cache(examples, make_valid_arrays(), make_valid_metadata())


def test_wrong_logits_shape_raises_value_error() -> None:
    arrays = {"logits": np.array([1.0, 2.0, 3.0])}

    with pytest.raises(ValueError, match="logits"):
        validate_prediction_cache(make_valid_examples(), arrays, make_valid_metadata())


def test_incorrect_correct_column_raises_value_error() -> None:
    examples = make_valid_examples()
    examples.loc[0, "correct"] = False

    with pytest.raises(ValueError, match="correct"):
        validate_prediction_cache(examples, make_valid_arrays(), make_valid_metadata())


def test_confidence_outside_unit_interval_raises_value_error() -> None:
    examples = make_valid_examples()
    examples.loc[0, "confidence"] = 1.2

    with pytest.raises(ValueError, match="confidence"):
        validate_prediction_cache(examples, make_valid_arrays(), make_valid_metadata())


def test_metadata_num_examples_mismatch_raises_value_error() -> None:
    metadata = make_valid_metadata()
    metadata["num_examples"] = 5

    with pytest.raises(ValueError, match="num_examples"):
        validate_prediction_cache(make_valid_examples(), make_valid_arrays(), metadata)


def test_validate_prediction_cache_dir_raises_file_not_found_on_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_prediction_cache_dir(tmp_path / "missing_cache")
