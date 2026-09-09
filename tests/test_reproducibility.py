from __future__ import annotations

import json

import pytest

from contextual_crc.utils.reproducibility import (
    collect_reproducibility_metadata,
    get_python_environment,
    get_selected_package_versions,
    require_config_keys,
    write_reproducibility_metadata,
    write_requirements_snapshot,
)


def test_get_python_environment() -> None:
    environment = get_python_environment()

    assert environment["python_version"]
    assert environment["executable"]
    assert "platform" in environment


def test_get_selected_package_versions() -> None:
    versions = get_selected_package_versions(["numpy", "definitely_missing_package_xyz"])

    assert "numpy" in versions
    assert "definitely_missing_package_xyz" in versions
    assert versions["definitely_missing_package_xyz"] is None


def test_collect_reproducibility_metadata() -> None:
    metadata = collect_reproducibility_metadata(config_path="configs/main.yaml")

    assert metadata["timestamp_utc"]
    assert "git" in metadata
    assert "python" in metadata
    assert "selected_packages" in metadata


def test_write_reproducibility_metadata(tmp_path) -> None:
    output_path = write_reproducibility_metadata(tmp_path / "repro.json", config_path="config.yaml")

    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["python"]["python_version"]
    assert payload["config_path"] == "config.yaml"


def test_write_requirements_snapshot(tmp_path) -> None:
    output_path = write_requirements_snapshot(tmp_path / "requirements_snapshot.txt")

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8").strip()


def test_require_config_keys() -> None:
    config = {"pipeline": {"stages": {"smoke": True}}}

    require_config_keys(config, ["pipeline.stages.smoke"])
    with pytest.raises(KeyError, match="pipeline.stages.synthetic_cache"):
        require_config_keys(config, ["pipeline.stages.synthetic_cache"])
