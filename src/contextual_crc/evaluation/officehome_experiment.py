from __future__ import annotations

from pathlib import Path
from typing import Any

from contextual_crc.cache.io import cache_exists
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.config.loader import load_config
from contextual_crc.datasets.officehome import OfficeHomeConfig, build_officehome_prediction_cache
from contextual_crc.evaluation.wilds_experiment import run_wilds_experiment


def run_officehome_experiment(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    run_name_override: str | None = None,
    build_cache_override: bool | None = None,
    cache_dir_override: str | None = None,
) -> Path:
    """Run the OfficeHome cache-backed contextual-CRC experiment."""
    resolved_config = dict(config)
    paths = resolved_config.setdefault("paths", {})
    if cache_dir_override is not None:
        paths["cache_dir"] = cache_dir_override

    cache_dir = _ensure_officehome_cache(
        resolved_config,
        build_cache_override=build_cache_override,
    )
    validate_prediction_cache_dir(cache_dir)

    delegated_config = dict(resolved_config)
    delegated_config["wilds_cache"] = {
        "build_if_missing": False,
        "overwrite_cache": False,
        "config_path": delegated_config.get("officehome_cache", {}).get(
            "config_path",
            "configs/officehome_resnet18.yaml",
        ),
    }
    return run_wilds_experiment(
        delegated_config,
        overwrite=overwrite,
        run_name_override=run_name_override,
        build_cache_override=False,
        cache_dir_override=str(cache_dir),
    )


def _ensure_officehome_cache(
    config: dict[str, Any],
    *,
    build_cache_override: bool | None,
) -> Path:
    paths = config.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "outputs/caches/officehome_resnet18_subset"))
    cache_config = config.get("officehome_cache", {})
    build_if_missing = bool(cache_config.get("build_if_missing", False))
    if build_cache_override is not None:
        build_if_missing = bool(build_cache_override)
    overwrite_cache = bool(cache_config.get("overwrite_cache", False))

    if cache_exists(cache_dir) and not overwrite_cache and not build_cache_override:
        return cache_dir
    if not cache_exists(cache_dir) and not build_if_missing:
        raise FileNotFoundError(
            f"OfficeHome cache not found: {cache_dir}. Build it with scripts/build_officehome_cache.py "
            "or set officehome_cache.build_if_missing=true / pass --build-cache."
        )

    raw_cache_config = load_config(cache_config.get("config_path", "configs/officehome_resnet18.yaml"))
    officehome_raw = dict(raw_cache_config.get("officehome", {}))
    officehome_paths = raw_cache_config.get("paths", {})
    officehome_raw["data_root"] = officehome_paths.get(
        "data_root",
        officehome_raw.get("data_root", "data/officehome"),
    )
    officehome_raw["cache_dir"] = str(cache_dir)
    officehome_config = OfficeHomeConfig(**officehome_raw)
    build_officehome_prediction_cache(
        cache_dir,
        officehome_config,
        overwrite=overwrite_cache or bool(build_cache_override) or not cache_exists(cache_dir),
    )
    return cache_dir
