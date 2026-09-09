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
from contextual_crc.evaluation.ct_simultaneous_validity import run_ct_simultaneous_validity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate theorem-aligned CT-CRC source-certificate corrections from caches."
    )
    parser.add_argument("--config", required=True, help="Artifact YAML containing source run_dirs.")
    parser.add_argument("--library-delta", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/ct_simultaneous_validity",
        help="Dedicated output directory; existing main runs are never modified.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    summary_path = run_ct_simultaneous_validity(
        config,
        library_delta=args.library_delta,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
    )
    print(f"CT simultaneous-validity results written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
