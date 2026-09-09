#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.evaluation.source_certificate_audit import run_source_certificate_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the source-certificate audit.")
    parser.add_argument("--evaluation-dir", default="outputs/runs/leakage_free_main")
    parser.add_argument("--output-dir", default="outputs/runs/source_certificate_audit")
    parser.add_argument("--artifact-tables-dir", default="outputs/artifacts/tables")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_source_certificate_audit(
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        artifact_tables_dir=args.artifact_tables_dir,
        overwrite=args.overwrite,
    )
    print(f"Source-certificate comparison written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
