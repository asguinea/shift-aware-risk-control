from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contextual_crc.cache.schema import (
    OPTIONAL_ARRAY_KEYS,
    OPTIONAL_EXAMPLE_COLUMNS,
    REQUIRED_ARRAY_KEYS,
    REQUIRED_EXAMPLE_COLUMNS,
    REQUIRED_METADATA_FIELDS,
    SCHEMA_VERSION,
)

CACHE_FILENAMES = ("metadata.json", "examples.parquet", "arrays.npz")


def validate_prediction_cache(
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Validate in-memory prediction cache components."""
    _validate_metadata(metadata)
    _validate_examples_columns(examples)
    _validate_arrays(arrays)

    logits = arrays["logits"]
    num_examples, num_classes = logits.shape

    if len(examples) != num_examples:
        raise ValueError(
            f"len(examples) must match logits.shape[0]: {len(examples)} != {num_examples}"
        )

    if metadata["num_examples"] != len(examples):
        raise ValueError(
            "metadata.num_examples must match len(examples): "
            f"{metadata['num_examples']} != {len(examples)}"
        )

    if metadata["num_classes"] != num_classes:
        raise ValueError(
            "metadata.num_classes must match logits.shape[1]: "
            f"{metadata['num_classes']} != {num_classes}"
        )

    metadata_array_keys = list(metadata["array_keys"])
    if set(metadata_array_keys) != set(arrays.keys()):
        raise ValueError(
            "metadata.array_keys must match arrays.npz keys: "
            f"{metadata_array_keys} != {sorted(arrays.keys())}"
        )

    if list(metadata["required_columns"]) != list(REQUIRED_EXAMPLE_COLUMNS):
        raise ValueError(
            "metadata.required_columns must match required example columns: "
            f"{metadata['required_columns']} != {list(REQUIRED_EXAMPLE_COLUMNS)}"
        )

    expected_optional_columns = [
        column for column in OPTIONAL_EXAMPLE_COLUMNS if column in examples.columns
    ]
    if list(metadata["optional_columns"]) != expected_optional_columns:
        raise ValueError(
            "metadata.optional_columns must match optional columns present in examples: "
            f"{metadata['optional_columns']} != {expected_optional_columns}"
        )

    _validate_context_fields(examples, metadata)
    _validate_dataset_split_consistency(examples, metadata)
    _validate_labels_predictions_correctness(examples)
    _validate_scalar_scores(examples)


def validate_prediction_cache_dir(cache_dir: str | Path) -> None:
    """Validate a prediction cache directory on disk."""
    cache_path = Path(cache_dir)
    missing = [name for name in CACHE_FILENAMES if not (cache_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prediction cache directory is missing required files: {missing}"
        )

    with (cache_path / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    examples = pd.read_parquet(cache_path / "examples.parquet", engine="pyarrow")
    with np.load(cache_path / "arrays.npz") as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    validate_prediction_cache(examples, arrays, metadata)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"metadata is missing required fields: {missing}")

    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"metadata.schema_version must be {SCHEMA_VERSION!r}: {metadata['schema_version']!r}"
        )

    for field in ("dataset", "split", "created_at"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"metadata.{field} must be a non-empty string")

    try:
        datetime.fromisoformat(metadata["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("metadata.created_at must be an ISO timestamp string") from exc

    for field in ("num_examples", "num_classes"):
        if not isinstance(metadata[field], int) or metadata[field] < 0:
            raise ValueError(f"metadata.{field} must be a non-negative integer")

    for field in ("array_keys", "required_columns", "optional_columns", "context_fields"):
        if not isinstance(metadata[field], list) or not all(
            isinstance(item, str) for item in metadata[field]
        ):
            raise ValueError(f"metadata.{field} must be a list of strings")


def _validate_examples_columns(examples: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_EXAMPLE_COLUMNS if column not in examples.columns]
    if missing:
        raise ValueError(f"examples is missing required columns: {missing}")


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_ARRAY_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"arrays is missing required keys: {missing}")

    allowed_keys = set(REQUIRED_ARRAY_KEYS) | set(OPTIONAL_ARRAY_KEYS)
    unknown = sorted(set(arrays) - allowed_keys)
    if unknown:
        raise ValueError(f"arrays contains unknown keys: {unknown}")

    logits = arrays["logits"]
    if not isinstance(logits, np.ndarray):
        raise ValueError("arrays.logits must be a numpy.ndarray")
    if logits.ndim != 2:
        raise ValueError(f"arrays.logits must be 2D, got shape {logits.shape}")

    num_examples = logits.shape[0]
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"arrays.{key} must be a numpy.ndarray")
        if value.shape[0] != num_examples:
            raise ValueError(
                f"arrays.{key}.shape[0] must match logits.shape[0]: "
                f"{value.shape[0]} != {num_examples}"
            )


def _validate_context_fields(examples: pd.DataFrame, metadata: dict[str, Any]) -> None:
    missing = [field for field in metadata["context_fields"] if field not in examples.columns]
    if missing:
        raise ValueError(f"metadata.context_fields are missing from examples: {missing}")


def _validate_dataset_split_consistency(
    examples: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    for field in ("dataset", "split"):
        values = set(examples[field].dropna().unique())
        expected = metadata[field]
        if values != {expected}:
            raise ValueError(
                f"examples.{field} must be internally consistent with metadata.{field}: "
                f"{sorted(values)} != {expected!r}"
            )


def _validate_labels_predictions_correctness(examples: pd.DataFrame) -> None:
    for column in ("label", "pred"):
        numeric = pd.to_numeric(examples[column], errors="coerce")
        if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"examples.{column} must be integer-compatible")

    correct_values = examples["correct"]
    if not correct_values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("examples.correct must be boolean")

    expected_correct = examples["pred"].astype(int) == examples["label"].astype(int)
    actual_correct = correct_values.astype(bool)
    if not actual_correct.equals(expected_correct):
        raise ValueError("examples.correct must equal examples.pred == examples.label")


def _validate_scalar_scores(examples: pd.DataFrame) -> None:
    confidence = pd.to_numeric(examples["confidence"], errors="coerce")
    if confidence.isna().any() or not np.isfinite(confidence).all():
        raise ValueError("examples.confidence must be finite")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("examples.confidence must be in [0, 1]")

    for column in ("entropy", "margin", "energy"):
        values = pd.to_numeric(examples[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"examples.{column} must be finite")
