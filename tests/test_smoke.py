from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_smoke_script_runs_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--config", "configs/smoke.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    run_dir = Path("outputs/smoke_run")
    resolved_config = run_dir / "configs" / "resolved_config.yaml"
    smoke_result = run_dir / "results" / "smoke_result.json"

    assert run_dir.is_dir()
    assert resolved_config.is_file()
    assert smoke_result.is_file()

    with smoke_result.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["status"] == "ok"
