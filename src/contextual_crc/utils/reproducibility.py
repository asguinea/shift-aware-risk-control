from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from contextual_crc.utils.paths import ensure_dir


DEFAULT_SELECTED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pyarrow",
    "pyyaml",
    "pytest",
)


def get_git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if repo_root is None else Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def get_git_status(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return commit and short working-tree status without raising."""
    commit = get_git_commit(repo_root)
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=None if repo_root is None else Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return {"commit": commit, "is_dirty": None, "status_short": None}
    if completed.returncode != 0:
        return {"commit": commit, "is_dirty": None, "status_short": None}
    status_short = completed.stdout.rstrip()
    return {
        "commit": commit,
        "is_dirty": bool(status_short),
        "status_short": status_short,
    }


def get_python_environment() -> dict[str, Any]:
    """Collect Python and OS metadata for reproducibility manifests."""
    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }


def get_installed_packages() -> dict[str, str]:
    """Return installed distribution versions keyed by normalized package name."""
    packages: dict[str, str] = {}
    try:
        distributions = metadata.distributions()
        for distribution in distributions:
            name = distribution.metadata.get("Name")
            version = distribution.version
            if name and version:
                packages[_normalize_package_name(name)] = version
    except Exception:
        return {}
    return packages


def get_selected_package_versions(package_names: Sequence[str]) -> dict[str, str | None]:
    """Return versions for requested packages, preserving requested keys."""
    installed = get_installed_packages()
    selected: dict[str, str | None] = {}
    for package_name in package_names:
        selected[package_name] = installed.get(_normalize_package_name(package_name))
    return selected


def collect_reproducibility_metadata(
    *,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect git, Python, and selected dependency metadata."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": None if config_path is None else str(config_path),
        "git": get_git_status(repo_root),
        "python": get_python_environment(),
        "selected_packages": get_selected_package_versions(DEFAULT_SELECTED_PACKAGES),
        "extra": dict(extra or {}),
    }


def write_reproducibility_metadata(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write reproducibility metadata as stable JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    payload = collect_reproducibility_metadata(
        repo_root=repo_root,
        config_path=config_path,
        extra=extra,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def write_requirements_snapshot(path: str | Path) -> Path:
    """Write a requirements-style snapshot of installed distributions."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    packages = get_installed_packages()
    with output_path.open("w", encoding="utf-8") as handle:
        if not packages:
            handle.write("# No installed packages could be collected.\n")
        else:
            for name in sorted(packages):
                handle.write(f"{name}=={packages[name]}\n")
    return output_path


def require_config_keys(config: Mapping[str, Any], required_paths: Sequence[str]) -> None:
    """Require dot-notated paths in a nested config mapping."""
    for required_path in required_paths:
        current: Any = config
        for part in required_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(f"Missing required config key: {required_path}")
            current = current[part]


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
