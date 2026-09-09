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
from contextual_crc.evaluation.leakage_free import run_leakage_free_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-free evaluation.")
    parser.add_argument("--config", required=True, help="Artifact YAML containing source run_dirs.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--library-alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default="outputs/runs/leakage_free_main")
    parser.add_argument("--artifact-tables-dir", default="outputs/artifacts/tables")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_leakage_free_evaluation(
        load_config(args.config),
        seed=args.seed,
        library_alpha=args.library_alpha,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        artifact_tables_dir=args.artifact_tables_dir,
    )
    print(f"Leakage-free main results written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
