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
from contextual_crc.evaluation.ct_selection_valid import run_ct_selection_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selection-valid, target-feasible CT-CRC from saved caches."
    )
    parser.add_argument("--config", required=True, help="Artifact YAML containing source run_dirs.")
    parser.add_argument("--library-alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default="outputs/runs/ct_selection_valid")
    parser.add_argument(
        "--simultaneous-results",
        default="outputs/runs/ct_simultaneous_validity/ct_simultaneous_validity.csv",
    )
    parser.add_argument(
        "--frontier-results",
        default="outputs/artifacts/tables/risk_coverage_frontiers.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    results_path = run_ct_selection_valid(
        config,
        library_alpha=args.library_alpha,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        simultaneous_results=args.simultaneous_results,
        frontier_results=args.frontier_results,
    )
    print(f"CT selection-valid results written to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
