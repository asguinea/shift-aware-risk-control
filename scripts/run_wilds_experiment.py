#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.wilds_experiment import run_wilds_experiment
from contextual_crc.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a WILDS contextual-CRC experiment.")
    parser.add_argument("--config", required=True, help="Path to WILDS experiment YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run directory.")
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    parser.add_argument("--build-cache", action="store_true", help="Build cache if missing or rebuild if requested.")
    parser.add_argument("--source", choices=("csv", "wilds"), default=None, help="Override cache data source.")
    parser.add_argument("--download", action="store_true", help="Allow optional WILDS download during cache build.")
    parser.add_argument("--cache-dir", default=None, help="Override prediction cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    run_dir = run_wilds_experiment(
        config,
        overwrite=args.overwrite,
        run_name_override=args.run_name,
        build_cache_override=True if args.build_cache else None,
        source_override=args.source,
        download_override=True if args.download else None,
        cache_dir_override=args.cache_dir,
    )
    logger.info("WILDS experiment completed: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
