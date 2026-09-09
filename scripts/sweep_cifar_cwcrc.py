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
from contextual_crc.evaluation.tuning import run_cifar_cwcrc_sweep
from contextual_crc.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CIFAR-C CW-CRC hyperparameters.")
    parser.add_argument("--config", required=True, help="Path to CIFAR-C experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on sweep variants.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name for sweep summary rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    output_dir = run_cifar_cwcrc_sweep(
        base_config=config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_runs=args.max_runs,
        dataset_name=args.dataset_name,
    )
    logger.info("CIFAR CW-CRC sweep written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
