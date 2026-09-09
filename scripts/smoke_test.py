#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextual_crc.config.loader import load_config
from contextual_crc.utils.logging import setup_logging
from contextual_crc.utils.paths import make_run_dir
from contextual_crc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shift-aware-risk-control smoke test.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    project = config.get("project", {})
    paths = config.get("paths", {})
    logging_config = config.get("logging", {})

    seed = int(project.get("seed", 0))
    set_seed(seed)

    logger = setup_logging(str(logging_config.get("level", "INFO")))

    output_root = paths.get("output_root", "outputs")
    run_name = paths.get("run_name", "smoke_run")
    run_dir = make_run_dir(output_root, run_name)

    resolved_config_path = run_dir / "configs" / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = {
        "status": "ok",
        "project": project.get("name"),
        "mode": project.get("mode"),
        "seed": seed,
    }
    result_path = run_dir / "results" / "smoke_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    logger.info("Smoke test completed successfully: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
