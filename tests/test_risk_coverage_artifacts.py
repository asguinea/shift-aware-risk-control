from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from contextual_crc.artifacts.export import load_run_outputs
from contextual_crc.artifacts.risk_coverage import (
    build_risk_coverage_frontiers,
    compute_risk_coverage_curve,
    make_matched_coverage_table,
    make_matched_risk_table,
    matched_coverage_point,
    matched_risk_point,
    write_risk_coverage_artifacts,
)
from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.signatures.io import save_windows


def test_compute_risk_coverage_curve_from_scores_and_correctness() -> None:
    curve = compute_risk_coverage_curve(
        [0.9, 0.8, 0.6, 0.2],
        [False, True, False, True],
        selected_threshold=0.7,
    )

    selected = curve[curve["selected_operating_point"]].iloc[0]
    accept_all = curve[curve["coverage"] == 1.0].iloc[0]

    assert selected["coverage"] == pytest.approx(0.5)
    assert selected["accepted_error_risk"] == pytest.approx(0.25)
    assert selected["n_accepted"] == pytest.approx(2.0)
    assert accept_all["accepted_error_risk"] == pytest.approx(0.5)


def test_matched_coverage_selection_interpolates() -> None:
    curve = pd.DataFrame(
        {
            "coverage": [0.0, 0.5, 1.0],
            "accepted_error_risk": [0.0, 0.1, 0.4],
        }
    )

    point = matched_coverage_point(curve, 0.75)

    assert point is not None
    assert point["interpolated_or_nearest"] == "interpolated"
    assert point["achieved_coverage"] == pytest.approx(0.75)
    assert point["risk_at_coverage"] == pytest.approx(0.25)


def test_matched_risk_selection_interpolates() -> None:
    curve = pd.DataFrame(
        {
            "coverage": [0.0, 0.5, 1.0],
            "accepted_error_risk": [0.0, 0.1, 0.4],
        }
    )

    point = matched_risk_point(curve, 0.25)

    assert point is not None
    assert point["interpolated_or_nearest"] == "interpolated"
    assert point["achieved_risk"] == pytest.approx(0.25)
    assert point["coverage_at_risk"] == pytest.approx(0.75)


def test_missing_curve_data_is_handled_gracefully(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing_details"
    results = run_dir / "results"
    results.mkdir(parents=True)
    _write_json_manifest(results / "run_manifest.json", dataset="fixture", run_name="missing_details")

    frontiers = build_risk_coverage_frontiers([load_run_outputs(run_dir)])
    written = write_risk_coverage_artifacts(
        [load_run_outputs(run_dir)],
        tmp_path / "artifacts" / "tables",
        tmp_path / "artifacts" / "figures",
    )

    assert frontiers.empty
    for path in written.values():
        assert Path(path).is_file()


def test_risk_coverage_artifact_generation_with_minimal_run(tmp_path: Path) -> None:
    run_dir = _create_minimal_frontier_run(tmp_path)
    outputs = [load_run_outputs(run_dir)]
    tables_dir = tmp_path / "artifacts" / "tables"
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_risk_coverage_artifacts(outputs, tables_dir, figures_dir)
    frontiers = pd.read_csv(written["risk_coverage_frontiers_csv"])
    matched_coverage = pd.read_csv(written["matched_coverage_csv"])
    matched_risk = pd.read_csv(written["matched_risk_csv"])

    assert {"global_crc", "metadata_stratified", "ct_crc", "cw_crc", "oracle_context_crc"}.issubset(
        set(frontiers["method"])
    )
    assert (figures_dir / "risk_coverage_frontiers.pdf").is_file()
    assert (tables_dir / "matched_coverage.tex").is_file()
    assert (tables_dir / "matched_risk.tex").is_file()
    assert not matched_coverage.empty
    assert not matched_risk.empty
    assert {"delta_risk_vs_global_at_same_coverage", "delta_risk_vs_metadata_at_same_coverage"}.issubset(
        matched_coverage.columns
    )
    assert {"delta_coverage_vs_global_at_same_risk", "delta_coverage_vs_metadata_at_same_risk"}.issubset(
        matched_risk.columns
    )


def test_matched_tables_skip_absent_methods() -> None:
    frontiers = pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "run_name": ["run", "run"],
            "method": ["global_crc", "global_crc"],
            "threshold": [1.0, 0.0],
            "coverage": [0.0, 1.0],
            "accepted_error_risk": [0.0, 0.2],
            "n_units": [10.0, 10.0],
            "n_accepted": [0.0, 10.0],
            "n_accepted_errors": [0.0, 2.0],
            "selected_operating_point": [False, True],
            "unit": ["window_example", "window_example"],
            "reconstruction_status": ["reconstructed", "reconstructed"],
        }
    )

    matched_coverage = make_matched_coverage_table(frontiers)
    matched_risk = make_matched_risk_table(frontiers)

    assert set(matched_coverage["method"]) == {"global_crc"}
    assert set(matched_risk["method"]) == {"global_crc"}


def _create_minimal_frontier_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "frontier_fixture"
    results = run_dir / "results"
    intermediate = run_dir / "intermediate"
    configs = run_dir / "configs"
    results.mkdir(parents=True)
    intermediate.mkdir(parents=True)
    configs.mkdir(parents=True)

    examples = _examples()
    cache_dir = tmp_path / "cache"
    save_prediction_cache(
        cache_dir,
        examples,
        {"logits": np.zeros((len(examples), 2), dtype=float)},
        {"dataset": "fixture", "split": "test", "context_fields": ["context_id", "context_name"]},
    )
    save_windows(
        pd.DataFrame(
            {
                "window_id": ["w0", "w1"],
                "dataset": ["fixture", "fixture"],
                "split": ["test", "test"],
                "context_id": ["a", "b"],
                "context_name": ["A", "B"],
                "start_pos": [0, 0],
                "end_pos": [2, 2],
                "n_examples": [2, 2],
                "example_indices": [[0, 1], [2, 3]],
            }
        ),
        intermediate / "windows.parquet",
    )
    config = {
        "paths": {"cache_dir": str(cache_dir)},
        "splits": {"evaluation_contexts": "all"},
        "risk_control": {"score_column": "confidence", "accept_if": "greater_equal"},
    }
    with (configs / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    pd.DataFrame({"context_id": ["a", "b"], "threshold": [0.5, 0.5]}).to_csv(
        results / "global_by_context.csv",
        index=False,
    )
    pd.DataFrame({"context_id": ["a", "b"], "threshold": [0.7, 0.4]}).to_csv(
        results / "stratified_by_context.csv",
        index=False,
    )
    pd.DataFrame({"context_id": ["a"], "threshold": [0.7]}).to_csv(
        results / "oracle_by_context.csv",
        index=False,
    )
    pd.DataFrame({"window_id": ["w0", "w1"], "threshold": [0.6, 0.6]}).to_csv(
        results / "ct_crc_windows.csv",
        index=False,
    )
    pd.DataFrame({"window_id": ["w0", "w1"], "threshold": [0.8, 0.3]}).to_csv(
        results / "cw_crc_windows.csv",
        index=False,
    )
    _write_json_manifest(results / "run_manifest.json", dataset="fixture", run_name="frontier_fixture")
    return run_dir


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["e0", "e1", "e2", "e3"],
            "dataset": ["fixture"] * 4,
            "split": ["test"] * 4,
            "context_id": ["a", "a", "b", "b"],
            "context_name": ["A", "A", "B", "B"],
            "label": [0, 1, 0, 1],
            "pred": [0, 0, 0, 1],
            "correct": [True, False, True, True],
            "confidence": [0.9, 0.6, 0.4, 0.2],
            "entropy": [0.1, 0.2, 0.3, 0.4],
            "margin": [0.8, 0.5, 0.3, 0.1],
            "energy": [-2.0, -1.5, -1.0, -0.5],
            "time_index": [0, 1, 2, 3],
        }
    )


def _write_json_manifest(path: Path, *, dataset: str, run_name: str) -> None:
    path.write_text(
        f'{{"dataset": "{dataset}", "run_name": "{run_name}", "num_examples": 4, "num_windows": 2}}\n',
        encoding="utf-8",
    )
