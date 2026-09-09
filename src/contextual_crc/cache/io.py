from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contextual_crc.cache.schema import (
    OPTIONAL_EXAMPLE_COLUMNS,
    REQUIRED_EXAMPLE_COLUMNS,
    SCHEMA_VERSION,
)
from contextual_crc.cache.validate import validate_prediction_cache, validate_prediction_cache_dir
from contextual_crc.utils.paths import ensure_dir


def save_prediction_cache(
    cache_dir: str | Path,
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Save prediction cache components to a directory-based cache format."""
    cache_path = Path(cache_dir)
    if cache_path.exists() and any(cache_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Cache directory already exists and is not empty: {cache_path}")

    ensure_dir(cache_path)

    enriched_metadata = _enrich_metadata(examples, arrays, metadata)
    validate_prediction_cache(examples, arrays, enriched_metadata)

    examples.to_parquet(cache_path / "examples.parquet", engine="pyarrow", index=False)
    np.savez_compressed(cache_path / "arrays.npz", **arrays)
    with (cache_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(enriched_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    validate_prediction_cache_dir(cache_path)
    return cache_path


def load_prediction_cache(
    cache_dir: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a prediction cache directory."""
    cache_path = Path(cache_dir)
    validate_prediction_cache_dir(cache_path)

    with (cache_path / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    examples = pd.read_parquet(cache_path / "examples.parquet", engine="pyarrow")
    with np.load(cache_path / "arrays.npz") as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    return examples, arrays, metadata


def cache_exists(cache_dir: str | Path) -> bool:
    """Return True when all required cache files exist."""
    cache_path = Path(cache_dir)
    return all((cache_path / filename).is_file() for filename in _cache_filenames())


def _enrich_metadata(
    examples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(metadata)
    logits = arrays.get("logits")

    enriched.setdefault("schema_version", SCHEMA_VERSION)
    enriched.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    enriched.setdefault("num_examples", len(examples))
    if logits is not None and logits.ndim >= 2:
        enriched.setdefault("num_classes", int(logits.shape[1]))
    enriched["array_keys"] = list(arrays.keys())
    enriched["required_columns"] = list(REQUIRED_EXAMPLE_COLUMNS)
    enriched["optional_columns"] = [
        column for column in OPTIONAL_EXAMPLE_COLUMNS if column in examples.columns
    ]
    enriched.setdefault("context_fields", ["context_id", "context_name"])

    return enriched


def _cache_filenames() -> tuple[str, str, str]:
    return ("metadata.json", "examples.parquet", "arrays.npz")
