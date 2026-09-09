from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from contextual_crc.utils.paths import ensure_dir


def save_windows(windows: pd.DataFrame, path: str | Path) -> Path:
    """Save windows to Parquet, preserving example_indices as JSON strings."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    serializable = windows.copy()
    if "example_indices" in serializable.columns:
        serializable["example_indices"] = serializable["example_indices"].map(
            lambda values: json.dumps([int(value) for value in values])
        )
    serializable.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def load_windows(path: str | Path) -> pd.DataFrame:
    """Load windows from Parquet and restore example_indices lists."""
    input_path = Path(path)
    windows = pd.read_parquet(input_path, engine="pyarrow")
    if "example_indices" in windows.columns:
        windows["example_indices"] = windows["example_indices"].map(_decode_indices)
    return windows


def save_signatures(signatures: pd.DataFrame, path: str | Path) -> Path:
    """Save behavioral signatures to Parquet."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    signatures.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def load_signatures(path: str | Path) -> pd.DataFrame:
    """Load behavioral signatures from Parquet."""
    return pd.read_parquet(Path(path), engine="pyarrow")


def _decode_indices(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    if isinstance(value, list):
        return [int(item) for item in value]
    if hasattr(value, "tolist"):
        return [int(item) for item in value.tolist()]
    raise ValueError(f"Cannot decode example_indices value: {value!r}")
