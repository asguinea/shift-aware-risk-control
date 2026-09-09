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
from contextual_crc.evaluation.cw_sensitivity import (
    DEFAULT_ESS_GRID,
    DEFAULT_MARGIN_GRID,
    run_cw_sensitivity,
)
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CW-CRC sensitivity from saved run outputs and caches.")
    parser.add_argument("--config", default="configs/artifacts.yaml", help="Artifact YAML config with default run_dirs.")
    parser.add_argument("--output-dir", default="outputs/runs/cw_sensitivity", help="Sensitivity output directory.")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Specific saved run directory to process. May be passed multiple times; defaults to config paths.run_dirs.",
    )
    parser.add_argument("--ess-grid", nargs="*", type=float, default=list(DEFAULT_ESS_GRID), help="ESS sweep values.")
    parser.add_argument(
        "--margin-grid",
        nargs="*",
        type=float,
        default=list(DEFAULT_MARGIN_GRID),
        help="Bound-margin sweep values.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 123)))
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    run_dirs = [Path(path) for path in (args.run_dir or config.get("paths", {}).get("run_dirs", []))]
    if not run_dirs:
        raise ValueError("No paths.run_dirs configured for CW sensitivity")
    output_dir = run_cw_sensitivity(
        run_dirs,
        args.output_dir,
        ess_grid=args.ess_grid,
        margin_grid=args.margin_grid,
    )
    logger.info("CW sensitivity written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
