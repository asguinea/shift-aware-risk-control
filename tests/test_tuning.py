from __future__ import annotations

import pandas as pd
import pytest

from contextual_crc.evaluation.tuning import (
    compute_officehome_selection_score,
    compute_selection_score,
    make_config_variants,
    run_cifar_cwcrc_sweep,
    summarize_sweep_results,
)


def test_make_config_variants_does_not_mutate_base() -> None:
    base = {"cw_crc": {"target_effective_sample_size": 8, "bound_margin": 0.03}}
    sweep = {
        "cw_crc.target_effective_sample_size": [4, 8],
        "cw_crc.bound_margin": [0.0, 0.01],
    }

    variants = make_config_variants(base, sweep)

    assert len(variants) == 4
    assert base["cw_crc"]["target_effective_sample_size"] == 8
    assert base["cw_crc"]["bound_margin"] == 0.03
    assert [variant["cw_crc"]["target_effective_sample_size"] for variant in variants] == [4, 4, 8, 8]
    assert [variant["cw_crc"]["bound_margin"] for variant in variants] == [0.0, 0.01, 0.0, 0.01]


def test_compute_selection_score_penalizes_zero_coverage() -> None:
    zero_coverage = compute_selection_score(
        mean_risk=0.0,
        mean_coverage=0.0,
        bound_underestimation_rate=0.0,
        risk_budget=0.1,
    )
    useful_coverage = compute_selection_score(
        mean_risk=0.05,
        mean_coverage=0.4,
        bound_underestimation_rate=0.0,
        risk_budget=0.1,
    )

    assert zero_coverage > useful_coverage
    assert zero_coverage == pytest.approx(float(zero_coverage))


def test_compute_officehome_selection_score_penalizes_bad_candidates() -> None:
    useful = compute_officehome_selection_score(
        mean_risk=0.16,
        mean_coverage=0.7,
        bound_underestimation_rate=0.08,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    zero_coverage = compute_officehome_selection_score(
        mean_risk=0.0,
        mean_coverage=0.0,
        bound_underestimation_rate=0.0,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    high_underestimation = compute_officehome_selection_score(
        mean_risk=0.16,
        mean_coverage=0.7,
        bound_underestimation_rate=0.45,
        global_risk=0.25,
        metadata_risk=0.24,
    )
    higher_risk = compute_officehome_selection_score(
        mean_risk=0.22,
        mean_coverage=0.7,
        bound_underestimation_rate=0.08,
        global_risk=0.25,
        metadata_risk=0.24,
    )

    assert zero_coverage > useful
    assert high_underestimation > useful
    assert higher_risk > useful
    assert useful == pytest.approx(float(useful))


def test_summarize_sweep_results() -> None:
    rows = [
        {"method": "cw_crc", "score": 2.0, "mean_coverage": 0.0},
        {"method": "cw_crc", "score": 0.5, "mean_coverage": 0.4},
        {"method": "ct_crc", "mean_coverage": 0.5},
    ]

    summary = summarize_sweep_results(rows)

    assert isinstance(summary, pd.DataFrame)
    assert summary.iloc[0]["score"] == pytest.approx(0.5)
    assert set(summary["method"]) == {"cw_crc", "ct_crc"}


def test_run_cifar_cwcrc_sweep_with_fake_runner(tmp_path) -> None:
    base_config = {
        "paths": {"cache_dir": "outputs/caches/cifar100c_resnet18_subset"},
        "risk_control": {"risk_budget": 0.1},
        "cw_crc": {"target_effective_sample_size": 8, "bound_margin": 0.03},
    }
    sweep = {
        "cw_crc.target_effective_sample_size": [4, 8],
        "cw_crc.bound_margin": [0.0],
    }

    def fake_runner(config, *, overwrite=False, run_name_override=None, **kwargs):
        del overwrite, kwargs
        run_dir = tmp_path / "sweep" / "runs" / str(run_name_override)
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        coverage = 0.2 if config["cw_crc"]["target_effective_sample_size"] == 4 else 0.0
        pd.DataFrame(
            [
                {
                    "method": "cw_crc",
                    "mean_risk": 0.08,
                    "mean_coverage": coverage,
                    "bound_underestimation_rate": 0.0,
                },
                {
                    "method": "ct_crc",
                    "mean_risk": 0.09,
                    "mean_coverage": 0.3,
                    "bound_underestimation_rate": 0.0,
                },
            ]
        ).to_csv(results_dir / "method_comparison.csv", index=False)
        return run_dir

    output_dir = run_cifar_cwcrc_sweep(
        base_config=base_config,
        output_dir=tmp_path / "sweep",
        sweep=sweep,
        overwrite=True,
        run_experiment=fake_runner,
    )

    results = pd.read_csv(output_dir / "sweep_results.csv")
    assert (output_dir / "best_config.yaml").is_file()
    assert (output_dir / "sweep_manifest.json").is_file()
    assert len(results[results["method"] == "cw_crc"]) == 2
    assert results[results["method"] == "cw_crc"].iloc[0]["mean_coverage"] == pytest.approx(0.2)
