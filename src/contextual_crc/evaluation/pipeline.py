from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from contextual_crc.utils.paths import ensure_dir
from contextual_crc.utils.reproducibility import (
    collect_reproducibility_metadata,
    write_requirements_snapshot,
)


PIPELINE_STAGE_ORDER = (
    "smoke",
    "synthetic_cache",
    "synthetic_experiment",
    "cifar10c_cache",
    "cifar10c_experiment",
    "cifar100c_cache",
    "cifar100c_experiment",
    "wilds_cache",
    "wilds_experiment",
    "artifacts",
    "tests",
)

STAGE_SELECTORS = {
    "configured",
    "smoke",
    "synthetic",
    "cifar10c",
    "cifar100c",
    "wilds",
    "artifacts",
    "tests",
    "all",
}


@dataclass
class StageResult:
    stage: str
    status: str
    command: str | None = None
    output_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None


def resolve_stages(
    config: dict,
    stage: str,
    *,
    enable_cifar10c: bool = False,
    enable_cifar100c: bool = False,
    enable_wilds: bool = False,
    enable_tests: bool = False,
) -> dict[str, bool]:
    """Resolve requested pipeline stages into a complete stage map."""
    if stage not in STAGE_SELECTORS:
        raise ValueError(f"Unknown pipeline stage selector: {stage}")

    configured = config.get("pipeline", {}).get("stages", {})
    resolved = {name: bool(configured.get(name, False)) for name in PIPELINE_STAGE_ORDER}

    if stage not in ("configured", "all"):
        resolved = dict.fromkeys(PIPELINE_STAGE_ORDER, False)
        if stage == "smoke":
            resolved["smoke"] = True
        elif stage == "synthetic":
            resolved["synthetic_cache"] = True
            resolved["synthetic_experiment"] = True
        elif stage == "cifar10c":
            resolved["cifar10c_cache"] = True
            resolved["cifar10c_experiment"] = True
        elif stage == "cifar100c":
            resolved["cifar100c_cache"] = True
            resolved["cifar100c_experiment"] = True
        elif stage == "wilds":
            resolved["wilds_cache"] = True
            resolved["wilds_experiment"] = True
        elif stage == "artifacts":
            resolved["artifacts"] = True
        elif stage == "tests":
            resolved["tests"] = True

    if enable_cifar10c:
        resolved["cifar10c_cache"] = True
        resolved["cifar10c_experiment"] = True
    if enable_cifar100c:
        resolved["cifar100c_cache"] = True
        resolved["cifar100c_experiment"] = True
    if enable_wilds:
        resolved["wilds_cache"] = True
        resolved["wilds_experiment"] = True
    if enable_tests:
        resolved["tests"] = True

    return resolved


def make_pipeline_run_dir(output_root: str | Path, run_name: str) -> Path:
    """Create the pipeline run directory and standard subdirectories."""
    run_dir = ensure_dir(Path(output_root) / run_name)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "manifests")
    ensure_dir(run_dir / "configs")
    return run_dir


def run_subprocess_stage(
    stage: str,
    command: list[str],
    *,
    cwd: str | Path | None = None,
) -> StageResult:
    """Run one subprocess-backed stage and summarize its outcome."""
    started = time.monotonic()
    started_at = _timestamp()
    command_string = " ".join(command)
    result = StageResult(stage=stage, status="failed", command=command_string, started_at=started_at)
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else Path(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        result.error = str(exc)
    else:
        if completed.returncode == 0:
            result.status = "success"
        else:
            result.error = _format_subprocess_error(completed.returncode, completed.stdout, completed.stderr)
    finally:
        result.finished_at = _timestamp()
        result.duration_seconds = round(time.monotonic() - started, 6)
    return result


def write_pipeline_manifest(
    run_dir: str | Path,
    *,
    config: dict,
    stage_results: Sequence[StageResult],
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Write a JSON manifest for a pipeline run."""
    manifest_dir = ensure_dir(Path(run_dir) / "manifests")
    requirements_snapshot_path = write_requirements_snapshot(manifest_dir / "requirements_snapshot.txt")
    summary = {
        "success": sum(result.status == "success" for result in stage_results),
        "skipped": sum(result.status == "skipped" for result in stage_results),
        "failed": sum(result.status == "failed" for result in stage_results),
    }
    manifest = {
        "timestamp": _timestamp(),
        "run_dir": str(run_dir),
        "config": config,
        "stage_results": [asdict(result) for result in stage_results],
        "summary": summary,
        "reproducibility": collect_reproducibility_metadata(
            repo_root=repo_root,
            config_path=config_path,
            extra={"requirements_snapshot": str(requirements_snapshot_path)},
        ),
    }
    path = manifest_dir / "pipeline_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def run_pipeline(
    config: dict,
    *,
    config_path: str | Path,
    stage: str = "configured",
    run_name_override: str | None = None,
    output_root_override: str | None = None,
    stop_on_error: bool | None = None,
    enable_cifar10c: bool = False,
    enable_cifar100c: bool = False,
    enable_wilds: bool = False,
    enable_tests: bool = False,
    download_cifar: bool = False,
    download_wilds: bool = False,
    no_overwrite: bool = False,
) -> Path:
    """Run configured pipeline stages and write a manifest."""
    logger = logging.getLogger("contextual_crc")
    pipeline_config = config.get("pipeline", {})
    resolved_stages = resolve_stages(
        config,
        stage,
        enable_cifar10c=enable_cifar10c,
        enable_cifar100c=enable_cifar100c,
        enable_wilds=enable_wilds,
        enable_tests=enable_tests,
    )
    run_name = run_name_override or str(pipeline_config.get("run_name", "local_main"))
    output_root = output_root_override or str(pipeline_config.get("output_root", "outputs/pipeline_runs"))
    should_stop = bool(pipeline_config.get("stop_on_error", True)) if stop_on_error is None else stop_on_error
    run_dir = make_pipeline_run_dir(output_root, run_name)

    resolved_config = dict(config)
    resolved_config["pipeline"] = dict(pipeline_config)
    resolved_config["pipeline"]["resolved_stages"] = resolved_stages
    resolved_config["pipeline"]["run_name"] = run_name
    resolved_config["pipeline"]["output_root"] = str(output_root)
    resolved_config_path = run_dir / "configs" / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(resolved_config), handle, sort_keys=False)

    stage_results: list[StageResult] = []
    prior_failure = False
    for stage_name in PIPELINE_STAGE_ORDER:
        if not resolved_stages.get(stage_name, False):
            stage_results.append(StageResult(stage=stage_name, status="skipped", error="disabled"))
            continue
        if prior_failure and should_stop:
            stage_results.append(StageResult(stage=stage_name, status="skipped", error="skipped after prior failure"))
            continue

        command = _stage_command(
            stage_name,
            config,
            config_path=config_path,
            download_cifar=download_cifar,
            download_wilds=download_wilds,
            no_overwrite=no_overwrite,
        )
        logger.info("Running pipeline stage %s: %s", stage_name, " ".join(command))
        result = run_subprocess_stage(stage_name, command, cwd=_repo_root())
        stage_results.append(result)
        if result.status == "failed":
            logger.error("Pipeline stage failed: %s", stage_name)
            prior_failure = True

    repo_root = _repo_root()
    manifest_path = write_pipeline_manifest(
        run_dir,
        config=resolved_config,
        stage_results=stage_results,
        config_path=config_path,
        repo_root=repo_root,
    )
    logger.info("Pipeline manifest written to: %s", manifest_path)
    logger.info("Pipeline summary: %s", _stage_summary(stage_results))
    return run_dir


def _stage_command(
    stage: str,
    config: dict,
    *,
    config_path: str | Path,
    download_cifar: bool,
    download_wilds: bool,
    no_overwrite: bool,
) -> list[str]:
    pipeline_config = config.get("pipeline", {})
    config_paths = pipeline_config.get("configs", {})
    overwrite = pipeline_config.get("overwrite", {})
    download = pipeline_config.get("download", {})
    py = sys.executable

    if stage == "smoke":
        return [py, "scripts/smoke_test.py", "--config", str(config_paths.get("smoke", "configs/smoke.yaml"))]
    if stage == "synthetic_cache":
        command = [py, "scripts/build_synthetic_cache.py", "--config", str(config_paths.get(stage, "configs/synthetic.yaml"))]
        return _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
    if stage == "synthetic_experiment":
        command = [
            py,
            "scripts/run_synthetic_experiment.py",
            "--config",
            str(config_paths.get(stage, "configs/experiments/synthetic_main.yaml")),
        ]
        return _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
    if stage == "cifar10c_cache":
        command = [py, "scripts/build_cifar_c_cache.py", "--config", str(config_paths.get(stage, "configs/cifar10c.yaml"))]
        command = _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
        if download_cifar or bool(download.get(stage, False)):
            command.append("--download")
        return command
    if stage == "cifar10c_experiment":
        command = [
            py,
            "scripts/run_cifar_c_experiment.py",
            "--config",
            str(config_paths.get(stage, "configs/experiments/cifar10c_main.yaml")),
        ]
        return _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
    if stage == "cifar100c_cache":
        command = [py, "scripts/build_cifar_c_cache.py", "--config", str(config_paths.get(stage, "configs/cifar100c.yaml"))]
        command = _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
        if download_cifar or bool(download.get(stage, False)):
            command.append("--download")
        return command
    if stage == "cifar100c_experiment":
        command = [
            py,
            "scripts/run_cifar_c_experiment.py",
            "--config",
            str(config_paths.get(stage, "configs/experiments/cifar100c_main.yaml")),
        ]
        return _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
    if stage == "wilds_cache":
        command = [py, "scripts/build_wilds_cache.py", "--config", str(config_paths.get(stage, "configs/wilds_civilcomments.yaml"))]
        command = _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
        if download_wilds or bool(download.get(stage, False)):
            command.append("--download")
        return command
    if stage == "wilds_experiment":
        command = [
            py,
            "scripts/run_wilds_experiment.py",
            "--config",
            str(config_paths.get(stage, "configs/experiments/wilds_civilcomments_main.yaml")),
        ]
        return _with_overwrite(command, overwrite.get(stage, False), no_overwrite)
    if stage == "artifacts":
        return [py, "scripts/make_artifacts.py", "--config", str(config_paths.get("artifacts", "configs/artifacts.yaml"))]
    if stage == "tests":
        return [py, "-m", "pytest"]
    raise ValueError(f"Unknown pipeline stage: {stage}")


def _with_overwrite(command: list[str], enabled: Any, no_overwrite: bool) -> list[str]:
    if bool(enabled) and not no_overwrite:
        return [*command, "--overwrite"]
    return command


def _format_subprocess_error(returncode: int, stdout: str, stderr: str) -> str:
    chunks = [f"return code {returncode}"]
    if stdout.strip():
        chunks.append(f"stdout:\n{_tail(stdout)}")
    if stderr.strip():
        chunks.append(f"stderr:\n{_tail(stderr)}")
    return "\n".join(chunks)


def _tail(text: str, max_chars: int = 4000) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def _stage_summary(stage_results: Sequence[StageResult]) -> dict[str, int]:
    return {
        "success": sum(result.status == "success" for result in stage_results),
        "skipped": sum(result.status == "skipped" for result in stage_results),
        "failed": sum(result.status == "failed" for result in stage_results),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
