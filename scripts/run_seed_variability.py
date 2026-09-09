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
from contextual_crc.evaluation.seed_variability import DEFAULT_SEEDS, normalize_seed_list, run_seed_variability
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed/window-bootstrap variability diagnostics from saved outputs.")
    parser.add_argument("--config", default="configs/artifacts.yaml", help="Artifact YAML config with default run_dirs.")
    parser.add_argument("--output-dir", default="outputs/runs/seed_variability", help="Variability output directory.")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS), help="Seeds to evaluate.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["all"],
        help="Dataset ids/run-name fragments to include, or all.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Specific saved run directory to process. May be passed multiple times; defaults to config paths.run_dirs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing seed variability outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 123)))
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    run_dirs = [Path(path) for path in (args.run_dir or config.get("paths", {}).get("run_dirs", []))]
    run_dirs = _filter_run_dirs(run_dirs, args.datasets)
    if not run_dirs:
        raise ValueError("No run directories selected for seed variability")
    metrics_path, smoothness_path = run_seed_variability(
        run_dirs,
        args.output_dir,
        seeds=normalize_seed_list(args.seeds),
        overwrite=bool(args.overwrite),
    )
    logger.info("Seed variability metrics written to: %s", metrics_path)
    logger.info("Seed variability smoothness written to: %s", smoothness_path)
    return 0


def _filter_run_dirs(run_dirs: list[Path], datasets: list[str]) -> list[Path]:
    requested = {str(value).lower() for value in datasets}
    if not requested or "all" in requested:
        return run_dirs
    selected = []
    for path in run_dirs:
        haystack = str(path).replace("\\", "/").lower()
        if any(token in haystack for token in requested):
            selected.append(path)
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
