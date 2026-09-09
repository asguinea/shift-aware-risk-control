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
from contextual_crc.datasets.officehome import OfficeHomeConfig, build_officehome_prediction_cache
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an OfficeHome prediction cache.")
    parser.add_argument("--config", required=True, help="Path to OfficeHome YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory.")
    parser.add_argument("--data-root", default=None, help="Override OfficeHome data root.")
    parser.add_argument("--cache-dir", default=None, help="Override output cache directory.")
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"), default=None, help="Override torch device.")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Override max training examples.")
    parser.add_argument("--max-eval-examples", type=int, default=None, help="Override max eval examples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = load_config(args.config)
    project_config = raw_config.get("project", {})
    logging_config = raw_config.get("logging", {})
    paths_config = raw_config.get("paths", {})
    officehome_raw = dict(raw_config.get("officehome", {}))

    seed = int(project_config.get("seed", officehome_raw.get("random_seed", 123)))
    set_seed(seed)
    logger = setup_logging(str(logging_config.get("level", "INFO")))

    officehome_raw["data_root"] = args.data_root or paths_config.get(
        "data_root",
        officehome_raw.get("data_root", "data/officehome"),
    )
    officehome_raw["cache_dir"] = args.cache_dir or paths_config.get(
        "cache_dir",
        officehome_raw.get("cache_dir", "outputs/caches/officehome_resnet18_subset"),
    )
    if args.torch_device is not None:
        officehome_raw["torch_device"] = args.torch_device
    if args.max_train_examples is not None:
        officehome_raw["max_train_examples"] = args.max_train_examples
    if args.max_eval_examples is not None:
        officehome_raw["max_eval_examples"] = args.max_eval_examples

    config = OfficeHomeConfig(**officehome_raw)
    cache_dir = Path(config.cache_dir)
    saved_path = build_officehome_prediction_cache(cache_dir, config, overwrite=args.overwrite)
    validate_prediction_cache_dir(saved_path)
    logger.info("OfficeHome prediction cache written to: %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
