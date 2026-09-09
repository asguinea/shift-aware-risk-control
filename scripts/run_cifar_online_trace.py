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
from contextual_crc.evaluation.online_trace import run_cifar_online_trace
from contextual_crc.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dedicated CIFAR-C online severity trace.")
    parser.add_argument("--config", required=True, help="Path to CIFAR online trace YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing trace run directory.")
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    parser.add_argument("--cache-dir", default=None, help="Override prediction cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    run_dir = run_cifar_online_trace(
        config,
        overwrite=args.overwrite,
        run_name_override=args.run_name,
        cache_dir_override=args.cache_dir,
    )
    logger.info("CIFAR online severity trace completed: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
