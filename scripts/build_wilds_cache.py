#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.config.loader import load_config
from contextual_crc.datasets.wilds_dataset import WILDSConfig, build_wilds_prediction_cache
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WILDS prediction cache.")
    parser.add_argument("--config", required=True, help="Path to WILDS YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory.")
    parser.add_argument("--download", action="store_true", help="Allow optional real-WILDS download.")
    parser.add_argument("--source", choices=("csv", "wilds"), default=None, help="Override data source.")
    parser.add_argument("--data-root", default=None, help="Override WILDS data root.")
    parser.add_argument("--cache-dir", default=None, help="Override output cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config)
    project_config = raw_config.get("project", {})
    logging_config = raw_config.get("logging", {})
    paths_config = raw_config.get("paths", {})
    wilds_raw = dict(raw_config.get("wilds", {}))

    seed = int(project_config.get("seed", wilds_raw.get("random_seed", 123)))
    set_seed(seed)
    logger = setup_logging(str(logging_config.get("level", "INFO")))

    wilds_raw["data_root"] = args.data_root or paths_config.get(
        "data_root",
        wilds_raw.get("data_root", "data/wilds"),
    )
    if args.source is not None:
        wilds_raw["source"] = args.source
    if args.download:
        wilds_raw["download"] = True

    cache_dir = Path(args.cache_dir or paths_config.get("cache_dir", "outputs/caches/wilds_civilcomments_subset"))
    config = WILDSConfig(**wilds_raw)
    saved_path = build_wilds_prediction_cache(cache_dir, config, overwrite=args.overwrite)
    validate_prediction_cache_dir(saved_path)
    logger.info("WILDS prediction cache written to: %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
