from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from contextual_crc.evaluation.tuning import compute_officehome_selection_score
from scripts import sweep_officehome_bounds


def test_officehome_bound_sweep_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_officehome_bounds, "run_officehome_experiment", _fake_officehome_runner)
    output_dir = tmp_path / "officehome_sweep"

    written = sweep_officehome_bounds.run_officehome_bound_sweep(
        base_config=_base_config(tmp_path),
        output_dir=output_dir,
        overwrite=True,
        max_runs=3,
        method="both",
    )

    results = pd.read_csv(written / "sweep_results.csv")
    assert written == output_dir
    assert (written / "sweep_manifest.json").is_file()
    assert (written / "best_ct_config.yaml").is_file()
    assert (written / "best_cw_config.yaml").is_file()
    assert {"ct_crc", "cw_crc"}.issubset(set(results["sweep_method"]))
    assert {"global_crc", "metadata_stratified_context", "ct_crc", "cw_crc"}.issubset(set(results["method"]))
    assert "ct_crc.bound_margin" in results.columns
    assert "cw_crc.target_effective_sample_size" in results.columns

    with (written / "best_ct_config.yaml").open("r", encoding="utf-8") as handle:
        best_ct = yaml.safe_load(handle)
    with (written / "best_cw_config.yaml").open("r", encoding="utf-8") as handle:
        best_cw = yaml.safe_load(handle)
    assert best_ct["ct_crc"]["bound_margin"] in {0.03, 0.05, 0.08, 0.10, 0.15}
    assert best_cw["cw_crc"]["target_effective_sample_size"] in {4, 6, 8, 12}


def test_officehome_bound_sweep_respects_method_and_max_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_officehome_bounds, "run_officehome_experiment", _fake_officehome_runner)

    written = sweep_officehome_bounds.run_officehome_bound_sweep(
        base_config=_base_config(tmp_path),
        output_dir=tmp_path / "cw_only",
        overwrite=True,
        max_runs=2,
        method="cw_crc",
    )

    results = pd.read_csv(written / "sweep_results.csv")
    assert set(results["sweep_method"]) == {"cw_crc"}
    assert results["variant_index"].nunique() == 2
    assert not (written / "best_ct_config.yaml").exists()
    assert (written / "best_cw_config.yaml").is_file()


def test_officehome_bound_sweep_existing_dir_requires_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "exists"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sweep_officehome_bounds.run_officehome_bound_sweep(
            base_config=_base_config(tmp_path),
            output_dir=output_dir,
            overwrite=False,
            max_runs=1,
            method="ct_crc",
        )


def test_officehome_selection_score_prioritizes_safe_useful_rows() -> None:
    good = compute_officehome_selection_score(
        mean_risk=0.15,
        mean_coverage=0.72,
        bound_underestimation_rate=0.08,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    zero_coverage = compute_officehome_selection_score(
        mean_risk=0.01,
        mean_coverage=0.0,
        bound_underestimation_rate=0.0,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    high_underestimation = compute_officehome_selection_score(
        mean_risk=0.15,
        mean_coverage=0.72,
        bound_underestimation_rate=0.5,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    lower_risk = compute_officehome_selection_score(
        mean_risk=0.12,
        mean_coverage=0.72,
        bound_underestimation_rate=0.08,
        global_risk=0.25,
        metadata_risk=0.24,
    )

    assert zero_coverage > good
    assert high_underestimation > good
    assert lower_risk < good


def _base_config(tmp_path: Path) -> dict:
    return {
        "project": {"name": "shift-aware-risk-control", "mode": "officehome_experiment", "seed": 123},
        "paths": {
            "output_root": str(tmp_path / "runs"),
            "run_name": "officehome_resnet18_main",
            "cache_dir": str(tmp_path / "cache"),
        },
        "officehome_cache": {
            "build_if_missing": False,
            "overwrite_cache": False,
            "config_path": "configs/officehome_resnet18.yaml",
        },
        "risk_control": {"risk_budget": 0.10},
        "ct_crc": {"bound_margin": 0.03, "bound_multiplier": 1.25, "l_hat_quantile": 0.95},
        "cw_crc": {
            "bound_margin": 0.03,
            "bound_multiplier": 1.0,
            "target_effective_sample_size": 6,
        },
    }


def _fake_officehome_runner(config, *, overwrite=False, run_name_override=None, **kwargs):
    del overwrite, kwargs
    run_name = str(run_name_override or config["paths"]["run_name"])
    run_dir = Path(config["paths"]["output_root"]) / run_name
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ct_margin = float(config["ct_crc"]["bound_margin"])
    ct_multiplier = float(config["ct_crc"]["bound_multiplier"])
    cw_margin = float(config["cw_crc"]["bound_margin"])
    cw_multiplier = float(config["cw_crc"]["bound_multiplier"])
    cw_ess = float(config["cw_crc"]["target_effective_sample_size"])
    ct_under = max(0.03, 0.55 - 2.0 * ct_margin - 0.08 * ct_multiplier)
    cw_under = max(0.03, 0.55 - 2.0 * cw_margin - 0.10 * cw_multiplier - 0.01 * cw_ess)
    ct_coverage = max(0.0, 0.92 - 1.5 * ct_margin - 0.05 * ct_multiplier)
    cw_coverage = max(0.0, 0.86 - 1.2 * cw_margin - 0.04 * cw_multiplier - 0.01 * cw_ess)
    pd.DataFrame(
        [
            {
                "method": "global_crc",
                "mean_risk": 0.255,
                "mean_coverage": 0.948,
                "violation_rate": 0.0,
                "mean_bound": float("nan"),
                "bound_underestimation_rate": float("nan"),
            },
            {
                "method": "metadata_stratified_context",
                "mean_risk": 0.251,
                "mean_coverage": 0.941,
                "violation_rate": 0.0,
                "mean_bound": float("nan"),
                "bound_underestimation_rate": float("nan"),
            },
            {
                "method": "ct_crc",
                "mean_risk": 0.19 + 0.2 * ct_margin,
                "mean_coverage": ct_coverage,
                "violation_rate": 0.0,
                "mean_bound": 0.25 + ct_margin,
                "bound_underestimation_rate": ct_under,
            },
            {
                "method": "cw_crc",
                "mean_risk": 0.14 + 0.15 * cw_margin,
                "mean_coverage": cw_coverage,
                "violation_rate": 0.0,
                "mean_bound": 0.20 + cw_margin,
                "bound_underestimation_rate": cw_under,
            },
        ]
    ).to_csv(results_dir / "method_comparison.csv", index=False)
    return run_dir
