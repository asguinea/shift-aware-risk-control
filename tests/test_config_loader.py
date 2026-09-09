from __future__ import annotations

from pathlib import Path

import pytest

from contextual_crc.config.loader import load_config


def test_smoke_config_loads() -> None:
    config = load_config(Path("configs/smoke.yaml"))

    assert config["project"]["name"] == "shift-aware-risk-control"
    assert config["project"]["mode"] == "smoke"
    assert config["project"]["seed"] == 123


def test_smoke_config_required_top_level_keys_exist() -> None:
    config = load_config(Path("configs/smoke.yaml"))

    assert {"project", "paths", "logging"}.issubset(config.keys())


def test_missing_config_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("configs/does_not_exist.yaml"))


def test_cifar100c_resnet18_cw_sweep_selection_config() -> None:
    config = load_config(Path("configs/experiments/cifar100c_resnet18_main.yaml"))
    cw_config = config["cw_crc"]

    assert cw_config["target_effective_sample_size"] == 6
    assert cw_config["bound_margin"] == pytest.approx(0.03)
    assert cw_config["bound_multiplier"] == pytest.approx(1.0)
