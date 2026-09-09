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
from contextual_crc.evaluation.label_efficiency import run_label_efficiency_from_cache
from contextual_crc.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run window-level label-efficiency evaluation.")
    parser.add_argument("--config", required=True, help="Path to synthetic or CIFAR experiment YAML config.")
    parser.add_argument("--dataset-kind", required=True, choices=("synthetic", "cifar"), help="Experiment family.")
    parser.add_argument("--output-dir", required=True, help="Directory for label-efficiency outputs.")
    parser.add_argument("--budgets", required=True, help="Comma-separated windows-per-context budgets.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]
    output_dir = run_label_efficiency_from_cache(
        experiment_config=config,
        budgets=budgets,
        output_dir=args.output_dir,
        dataset_kind=args.dataset_kind,
        overwrite=args.overwrite,
    )
    logger.info("Label-efficiency outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
