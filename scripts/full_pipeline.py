#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.pipeline import STAGE_SELECTORS, run_pipeline
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the contextual-CRC staged pipeline.")
    parser.add_argument("--config", default="configs/main.yaml", help="Path to main pipeline YAML config.")
    parser.add_argument("--stage", default="configured", choices=sorted(STAGE_SELECTORS), help="Stage selector.")
    parser.add_argument("--run-name", default=None, help="Override pipeline run name.")
    parser.add_argument("--output-root", default=None, help="Override pipeline output root.")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", help="Stop after the first failed enabled stage.")
    error_group.add_argument("--continue-on-error", dest="stop_on_error", action="store_false", help="Continue after failed stages.")
    parser.set_defaults(stop_on_error=None)
    parser.add_argument("--enable-cifar10c", action="store_true", help="Enable CIFAR-10-C cache and experiment stages.")
    parser.add_argument("--enable-cifar100c", action="store_true", help="Enable CIFAR-100-C cache and experiment stages.")
    parser.add_argument("--enable-wilds", action="store_true", help="Enable WILDS cache and experiment stages.")
    parser.add_argument("--enable-tests", action="store_true", help="Enable the pytest stage.")
    parser.add_argument("--download-cifar", action="store_true", help="Allow CIFAR-C download during enabled cache stages.")
    parser.add_argument("--download-wilds", action="store_true", help="Allow WILDS download during enabled cache stage.")
    parser.add_argument("--no-overwrite", action="store_true", help="Disable configured overwrite flags.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 123)))
    logger = setup_logging(str(config.get("logging", {}).get("level", "INFO")))

    run_dir = run_pipeline(
        config,
        config_path=args.config,
        stage=args.stage,
        run_name_override=args.run_name,
        output_root_override=args.output_root,
        stop_on_error=args.stop_on_error,
        enable_cifar10c=args.enable_cifar10c,
        enable_cifar100c=args.enable_cifar100c,
        enable_wilds=args.enable_wilds,
        enable_tests=args.enable_tests,
        download_cifar=args.download_cifar,
        download_wilds=args.download_wilds,
        no_overwrite=args.no_overwrite,
    )
    logger.info("Pipeline run directory: %s", run_dir)
    manifest_path = run_dir / "manifests" / "pipeline_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    failed_stages = int(manifest.get("summary", {}).get("failed", 0))
    if failed_stages:
        logger.error("Pipeline completed with %d failed stage(s)", failed_stages)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
