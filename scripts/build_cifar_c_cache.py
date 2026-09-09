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
from contextual_crc.datasets.cifar_c import CIFARCConfig, build_cifar_c_prediction_cache
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CIFAR-C prediction cache.")
    parser.add_argument("--config", required=True, help="Path to CIFAR-C YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory.")
    parser.add_argument("--download", action="store_true", help="Download/extract official CIFAR-C archive first.")
    parser.add_argument("--data-root", default=None, help="Override data root.")
    parser.add_argument("--cache-dir", default=None, help="Override output cache directory.")
    parser.add_argument("--dataset", choices=("cifar10c", "cifar100c"), default=None, help="Override dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config)
    set_seed(int(raw_config.get("project", {}).get("seed", 123)))
    logger = setup_logging(str(raw_config.get("logging", {}).get("level", "INFO")))

    cifar_raw = dict(raw_config.get("cifar_c", {}))
    paths = raw_config.get("paths", {})
    cifar_raw["data_root"] = args.data_root or paths.get("data_root", cifar_raw.get("data_root", "data"))
    if args.dataset is not None:
        cifar_raw["dataset"] = args.dataset
    if args.download:
        cifar_raw["download"] = True

    cache_dir = Path(args.cache_dir or paths.get("cache_dir", "outputs/caches/cifar_c"))
    config = CIFARCConfig(**cifar_raw)
    saved_path = build_cifar_c_prediction_cache(cache_dir, config, overwrite=args.overwrite)
    validate_prediction_cache_dir(saved_path)
    logger.info("CIFAR-C prediction cache written to: %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
