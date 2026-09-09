from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "1.0"

REQUIRED_EXAMPLE_COLUMNS: tuple[str, ...] = (
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
)

OPTIONAL_EXAMPLE_COLUMNS: tuple[str, ...] = (
    "severity",
    "corruption_type",
    "domain_id",
    "group_id",
    "time_index",
    "source",
)

REQUIRED_ARRAY_KEYS: tuple[str, ...] = ("logits",)

OPTIONAL_ARRAY_KEYS: tuple[str, ...] = (
    "probabilities",
    "feature_embedding",
    "ood_score",
)

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "dataset",
    "split",
    "num_examples",
    "num_classes",
    "created_at",
    "schema_version",
    "array_keys",
    "required_columns",
    "optional_columns",
    "context_fields",
)


@dataclass(frozen=True)
class CacheSchema:
    schema_version: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    required_array_keys: tuple[str, ...]
    optional_array_keys: tuple[str, ...]
    required_metadata_fields: tuple[str, ...]


def default_schema() -> CacheSchema:
    """Return the default prediction cache schema for this repository."""
    return CacheSchema(
        schema_version=SCHEMA_VERSION,
        required_columns=REQUIRED_EXAMPLE_COLUMNS,
        optional_columns=OPTIONAL_EXAMPLE_COLUMNS,
        required_array_keys=REQUIRED_ARRAY_KEYS,
        optional_array_keys=OPTIONAL_ARRAY_KEYS,
        required_metadata_fields=REQUIRED_METADATA_FIELDS,
    )
