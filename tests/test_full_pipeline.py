from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.pipeline import (
    PIPELINE_STAGE_ORDER,
    make_pipeline_run_dir,
    resolve_stages,
    run_pipeline,
)


def _base_config() -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "full", "seed": 123},
        "logging": {"level": "INFO"},
        "pipeline": {
            "output_root": "outputs/pipeline_runs",
            "run_name": "local_main",
            "stop_on_error": True,
            "stages": {
                "smoke": True,
                "synthetic_cache": True,
                "synthetic_experiment": True,
                "cifar10c_cache": False,
                "cifar10c_experiment": False,
                "cifar100c_cache": False,
                "cifar100c_experiment": False,
                "wilds_cache": False,
                "wilds_experiment": False,
                "artifacts": True,
                "tests": False,
            },
            "configs": {"smoke": "configs/smoke.yaml"},
            "overwrite": {},
            "download": {},
        },
    }


def test_resolve_stages_configured() -> None:
    resolved = resolve_stages(_base_config(), "configured")

    assert resolved["synthetic_cache"] is True
    assert resolved["synthetic_experiment"] is True
    assert resolved["artifacts"] is True
    assert resolved["cifar10c_cache"] is False
    assert resolved["wilds_experiment"] is False
    assert set(resolved) == set(PIPELINE_STAGE_ORDER)


def test_resolve_stages_synthetic() -> None:
    resolved = resolve_stages(_base_config(), "synthetic")

    assert resolved["synthetic_cache"] is True
    assert resolved["synthetic_experiment"] is True
    assert sum(resolved.values()) == 2


def test_resolve_stages_enable_cifar() -> None:
    resolved = resolve_stages(_base_config(), "all", enable_cifar10c=True)

    assert resolved["cifar10c_cache"] is True
    assert resolved["cifar10c_experiment"] is True
    assert resolved["cifar100c_cache"] is False
    assert resolved["cifar100c_experiment"] is False


def test_make_pipeline_run_dir(tmp_path) -> None:
    run_dir = make_pipeline_run_dir(tmp_path, "pipe")

    assert run_dir.is_dir()
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "manifests").is_dir()
    assert (run_dir / "configs").is_dir()


def test_run_pipeline_smoke_only(tmp_path) -> None:
    config = load_config("configs/main.yaml")
    run_dir = run_pipeline(
        config,
        config_path="configs/main.yaml",
        stage="smoke",
        run_name_override="smoke_only",
        output_root_override=tmp_path,
    )
    manifest = _read_manifest(run_dir)

    statuses = {result["stage"]: result["status"] for result in manifest["stage_results"]}
    assert statuses["smoke"] == "success"
    assert statuses["synthetic_cache"] == "skipped"
    assert statuses["artifacts"] == "skipped"
    assert manifest["summary"]["success"] == 1
    assert "reproducibility" in manifest
    assert (run_dir / "manifests" / "requirements_snapshot.txt").is_file()


def test_full_pipeline_script_smoke(tmp_path) -> None:
    run_name = "script_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/full_pipeline.py",
            "--config",
            "configs/main.yaml",
            "--stage",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--run-name",
            run_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / run_name / "manifests" / "pipeline_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statuses = {stage["stage"]: stage["status"] for stage in manifest["stage_results"]}
    assert statuses["smoke"] == "success"
    assert manifest["reproducibility"]["selected_packages"]


def test_full_pipeline_script_returns_nonzero_for_failed_stage(tmp_path) -> None:
    config = _base_config()
    config["pipeline"]["configs"]["smoke"] = "configs/does_not_exist.yaml"
    config_path = tmp_path / "failing_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/full_pipeline.py",
            "--config",
            str(config_path),
            "--stage",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--run-name",
            "expected_failure",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    manifest = _read_manifest(tmp_path / "expected_failure")
    assert manifest["summary"]["failed"] == 1


def test_resolved_config_written(tmp_path) -> None:
    config = load_config("configs/main.yaml")
    run_dir = run_pipeline(
        config,
        config_path="configs/main.yaml",
        stage="smoke",
        run_name_override="resolved_config",
        output_root_override=tmp_path,
    )

    resolved = yaml.safe_load((run_dir / "configs" / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["pipeline"]["resolved_stages"]["smoke"] is True
    assert resolved["pipeline"]["resolved_stages"]["synthetic_cache"] is False


def _read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifests" / "pipeline_manifest.json"
    assert manifest_path.is_file()
    return json.loads(manifest_path.read_text(encoding="utf-8"))
