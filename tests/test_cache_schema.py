from __future__ import annotations

from contextual_crc.cache.schema import (
    OPTIONAL_ARRAY_KEYS,
    REQUIRED_ARRAY_KEYS,
    REQUIRED_EXAMPLE_COLUMNS,
    REQUIRED_METADATA_FIELDS,
    SCHEMA_VERSION,
    default_schema,
)


def test_default_schema_returns_expected_constants() -> None:
    schema = default_schema()

    assert schema.schema_version == SCHEMA_VERSION
    assert schema.required_columns == REQUIRED_EXAMPLE_COLUMNS
    assert schema.required_array_keys == REQUIRED_ARRAY_KEYS
    assert schema.optional_array_keys == OPTIONAL_ARRAY_KEYS
    assert schema.required_metadata_fields == REQUIRED_METADATA_FIELDS


def test_required_columns_include_key_fields() -> None:
    schema = default_schema()

    for column in ("example_id", "label", "pred", "confidence", "entropy", "margin", "energy"):
        assert column in schema.required_columns
