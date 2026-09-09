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
from contextual_crc.datasets.synthetic import SyntheticConfig, build_synthetic_prediction_cache
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.paths import ensure_dir
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the synthetic prediction cache.")
    parser.add_argument("--config", required=True, help="Path to synthetic YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config)

    project_config = raw_config.get("project", {})
    logging_config = raw_config.get("logging", {})
    paths_config = raw_config.get("paths", {})
    synthetic_config = raw_config.get("synthetic", {})

    seed = int(project_config.get("seed", synthetic_config.get("random_seed", 123)))
    set_seed(seed)
    logger = setup_logging(str(logging_config.get("level", "INFO")))

    ensure_dir(paths_config.get("output_root", "outputs"))
    cache_dir = paths_config.get("cache_dir", "outputs/caches/synthetic_default")
    config = SyntheticConfig(**synthetic_config)

    saved_path = build_synthetic_prediction_cache(
        cache_dir,
        config,
        overwrite=args.overwrite,
    )
    validate_prediction_cache_dir(saved_path)
    logger.info("Synthetic prediction cache written to: %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
