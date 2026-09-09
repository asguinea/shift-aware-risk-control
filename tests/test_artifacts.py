from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from contextual_crc.artifacts.export import load_many_run_outputs, load_run_outputs
from contextual_crc.artifacts.make_figures import (
    _context_smoothness_grid,
    _display_dataset as _display_figure_dataset,
    _select_method_comparison_rows,
    _select_online_trace_output,
    write_figures,
)
from contextual_crc.artifacts.make_tables import (
    _display_dataset as _display_table_dataset,
    make_bound_diagnostics_table,
    make_label_efficiency_table,
    make_main_results_table,
    write_tables,
)


def create_fake_run(root: Path, name: str = "synthetic_fake", dataset: str = "synthetic_contextual_shift") -> Path:
    run_dir = root / "runs" / name
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "method": [
                "global_crc",
                "metadata_stratified_context",
                "metadata_stratified_corruption",
                "metadata_stratified_severity",
                "ct_crc",
                "cw_crc",
                "oracle_context_crc",
            ],
            "mean_risk": [0.2, 0.18, 0.19, 0.21, 0.15, 0.12, 0.1],
            "violation_rate": [float("nan"), float("nan"), float("nan"), float("nan"), 0.1, 0.0, float("nan")],
            "mean_coverage": [0.8, 0.78, 0.77, 0.76, 0.75, 0.7, 0.82],
            "mean_bound": [float("nan"), float("nan"), float("nan"), float("nan"), 0.3, 0.25, float("nan")],
            "bound_underestimation_rate": [float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0.0, float("nan")],
            "n_windows": [float("nan"), float("nan"), float("nan"), float("nan"), 3, 3, float("nan")],
        }
    ).to_csv(results / "method_comparison.csv", index=False)
    pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "context_id": ["a", "b", "c"],
            "novelty_distance": [0.1, 0.4, 0.9],
            "realized_risk": [0.1, 0.2, 0.15],
            "transferred_upper_bound": [0.25, 0.35, 0.3],
            "coverage": [0.85, 0.75, 0.8],
        }
    ).to_csv(results / "ct_crc_windows.csv", index=False)
    pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "context_id": ["a", "b", "c"],
            "novelty_distance": [0.1, 0.4, 0.9],
            "realized_risk": [0.08, 0.12, 0.1],
            "mean_loss": [0.08, 0.12, 0.1],
            "weighted_risk_upper_bound": [0.2, 0.22, 0.21],
            "coverage": [0.8, 0.7, 0.72],
        }
    ).to_csv(results / "cw_crc_windows.csv", index=False)
    pd.DataFrame(
        {
            "window_id_i": ["w0", "w0", "w0", "w1", "w1", "w2"],
            "window_id_j": ["w1", "w2", "w3", "w2", "w3", "w3"],
            "distance": [0.1, 0.3, 0.6, 0.2, 0.5, 0.25],
            "risk_difference": [0.02, 0.05, 0.08, 0.03, 0.06, 0.02],
            "risk_i": [0.1, 0.1, 0.1, 0.12, 0.12, 0.15],
            "risk_j": [0.12, 0.15, 0.18, 0.15, 0.18, 0.18],
        }
    ).to_csv(results / "context_smoothness.csv", index=False)

    write_json(results / "ct_crc_summary.json", {
        "method": "ct_crc",
        "mean_risk": 0.15,
        "mean_upper_bound": 0.3,
        "violation_rate": 0.1,
        "bound_underestimation_rate": 0.0,
        "n_windows": 3,
    })
    write_json(results / "cw_crc_summary.json", {
        "method": "cw_crc",
        "mean_risk": 0.12,
        "mean_upper_bound": 0.25,
        "violation_rate": 0.0,
        "bound_underestimation_rate": 0.0,
        "n_windows": 3,
    })
    write_json(results / "bound_diagnostics.json", {
        "ct_crc": {"bound_underestimation_rate": 0.0, "mean_bound_gap": 0.15},
        "cw_crc": {"bound_underestimation_rate": 0.0, "mean_bound_gap": 0.13},
    })
    write_json(results / "run_manifest.json", {
        "run_name": name,
        "dataset": dataset,
        "num_examples": 60,
        "num_windows": 3,
        "num_signature_features": 12,
        "calibration_contexts": ["source", "shift_a"],
        "evaluation_contexts": ["source", "shift_a", "shift_b"],
    })
    return run_dir


def create_fake_online_trace_run(root: Path, name: str = "cifar10c_resnet18_online_trace") -> Path:
    run_dir = root / "runs" / name
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trace_index": [0, 1, 2, 3],
            "window_id": ["w0", "w1", "w2", "w3"],
            "context_id": ["gaussian_noise_s1", "gaussian_noise_s1", "gaussian_noise_s3", "gaussian_noise_s5"],
            "corruption_type": ["gaussian_noise"] * 4,
            "severity": [1, 1, 3, 5],
            "novelty_distance": [0.05, 0.07, 0.3, 0.6],
            "ct_realized_risk": [0.08, 0.1, 0.16, 0.22],
            "ct_bound": [0.15, 0.16, 0.25, 0.32],
            "ct_coverage": [0.9, 0.88, 0.8, 0.72],
            "ct_violation": [False, False, True, True],
            "ct_bound_underestimated": [False, False, False, False],
            "cw_realized_risk": [0.07, 0.09, 0.14, 0.2],
            "cw_bound": [0.14, 0.15, 0.22, 0.28],
            "cw_coverage": [0.86, 0.84, 0.76, 0.68],
            "cw_violation": [False, False, True, True],
            "cw_bound_underestimated": [False, False, False, False],
        }
    ).to_csv(results / "online_trace.csv", index=False)
    write_json(
        results / "run_manifest.json",
        {
            "run_name": name,
            "dataset": "cifar10c",
            "corruption_type": "gaussian_noise",
            "num_trace_windows": 4,
        },
    )
    return run_dir


def test_load_run_outputs(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)

    output = load_run_outputs(run_dir)

    assert output["valid"] is True
    assert output["dataset"] == "synthetic_contextual_shift"
    assert output["method_comparison"] is not None
    assert output["manifest"]["run_name"] == "synthetic_fake"


def test_make_tables(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    run_outputs = load_many_run_outputs([run_dir])
    tables_dir = tmp_path / "artifacts" / "tables"

    written = write_tables(run_outputs, tables_dir, precision=3)
    main_results = make_main_results_table(run_outputs)
    bound_diagnostics = make_bound_diagnostics_table(run_outputs)

    assert (tables_dir / "main_results.tex").is_file()
    assert (tables_dir / "main_results.csv").is_file()
    assert (tables_dir / "bound_diagnostics.tex").is_file()
    assert (tables_dir / "dataset_summary.csv").is_file()
    assert {"global_crc", "ct_crc", "cw_crc"}.issubset(set(main_results["method"]))
    assert {"dataset_display", "method_display", "mean_bound_gap"}.issubset(bound_diagnostics.columns)
    assert {"CT-CRC", "CW-CRC"}.issubset(set(bound_diagnostics["method_display"]))
    assert "main_results_tex" in written


def test_dataset_display_names_are_publication_friendly() -> None:
    shared_expected = {
        "wilds_camelyon17": "Camelyon17-WILDS",
        "wilds_fmow": "FMoW-WILDS",
        "officehome": "OfficeHome",
        "cifar10c": "CIFAR-10-C",
        "cifar100c": "CIFAR-100-C",
        "synthetic_stronger_contextual_shift": "Synthetic",
    }

    for dataset, display_name in shared_expected.items():
        assert _display_figure_dataset(dataset) == display_name
        assert _display_table_dataset(dataset) == display_name
    assert _display_figure_dataset("synthetic_contextual_shift") == "Synthetic"


def test_context_smoothness_grid_prefers_multirow_layout() -> None:
    assert _context_smoothness_grid(1)[:2] == (1, 1)
    assert _context_smoothness_grid(2)[:2] == (1, 2)
    assert _context_smoothness_grid(4)[:2] == (2, 2)
    assert _context_smoothness_grid(5)[:2] == (2, 3)


def test_write_figures(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    run_outputs = load_many_run_outputs([run_dir])
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_figures(run_outputs, figures_dir, figure_format="pdf")

    for name in ("method_comparison", "context_smoothness", "online_shift_trace", "bound_diagnostics"):
        path = Path(written[name])
        assert path.is_file()
        assert path.stat().st_size > 0


def test_write_figures_prefers_dedicated_online_trace(tmp_path) -> None:
    generic_run = create_fake_run(tmp_path, name="cifar10c_generic", dataset="cifar10c")
    trace_run = create_fake_online_trace_run(tmp_path)
    run_outputs = load_many_run_outputs([generic_run, trace_run])
    figures_dir = tmp_path / "artifacts" / "figures"

    selected = _select_online_trace_output(run_outputs)
    written = write_figures(run_outputs, figures_dir, figure_format="pdf")
    path = Path(written["online_shift_trace"])

    assert selected is not None
    assert selected["run_name"] == "cifar10c_resnet18_online_trace"
    assert selected["online_trace"] is not None
    assert path.is_file()
    assert path.stat().st_size > 0


def test_method_comparison_selects_one_metadata_baseline() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["cifar10c"] * 6 + ["synthetic_stronger_contextual_shift"] * 5,
            "method": [
                "global_crc",
                "metadata_stratified_context",
                "metadata_stratified_corruption",
                "ct_crc",
                "cw_crc",
                "oracle_context_crc",
                "global_crc",
                "metadata_stratified_corruption",
                "metadata_stratified_crc",
                "ct_crc",
                "cw_crc",
            ],
            "mean_risk": [0.2, 0.18, 0.19, 0.12, 0.11, 0.1, 0.3, 0.26, 0.27, 0.2, 0.21],
        }
    )

    selected = _select_method_comparison_rows(frame, metric="mean_risk")
    metadata_rows = selected[selected["plot_method"] == "metadata_stratified"]

    assert selected["plot_method"].tolist() == [
        "global_crc",
        "metadata_stratified",
        "ct_crc",
        "cw_crc",
        "oracle_context_crc",
        "global_crc",
        "metadata_stratified",
        "ct_crc",
        "cw_crc",
    ]
    assert metadata_rows["method"].tolist() == [
        "metadata_stratified_context",
        "metadata_stratified_corruption",
    ]
    assert set(metadata_rows["method_display"]) == {"Metadata-stratified"}


def test_context_smoothness_aggregated_format_does_not_crash(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    pd.DataFrame(
        {
            "n_pairs": [6.0],
            "mean_pairwise_distance": [0.4],
            "mean_value_difference": [0.08],
            "distance_value_correlation": [0.5],
        }
    ).to_csv(run_dir / "results" / "context_smoothness.csv", index=False)
    run_outputs = load_many_run_outputs([run_dir])
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_figures(run_outputs, figures_dir, figure_format="pdf")
    path = Path(written["context_smoothness"])

    assert path.is_file()
    assert path.stat().st_size > 0


def test_context_smoothness_four_dataset_artifact_written(tmp_path) -> None:
    datasets = (
        "cifar10c",
        "cifar100c",
        "synthetic_stronger_contextual_shift",
        "wilds_camelyon17",
    )
    run_dirs = [
        create_fake_run(tmp_path, name=f"{dataset}_fake", dataset=dataset)
        for dataset in datasets
    ]
    run_outputs = load_many_run_outputs(run_dirs)
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_figures(run_outputs, figures_dir, figure_format="pdf")
    path = Path(written["context_smoothness"])

    assert path.is_file()
    assert path.stat().st_size > 0


def test_online_shift_trace_missing_novelty_does_not_crash(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    ct_windows = pd.read_csv(run_dir / "results" / "ct_crc_windows.csv").drop(columns=["novelty_distance"])
    cw_windows = pd.read_csv(run_dir / "results" / "cw_crc_windows.csv").drop(columns=["novelty_distance"])
    ct_windows.to_csv(run_dir / "results" / "ct_crc_windows.csv", index=False)
    cw_windows.to_csv(run_dir / "results" / "cw_crc_windows.csv", index=False)
    run_outputs = load_many_run_outputs([run_dir])
    figures_dir = tmp_path / "artifacts" / "figures"

    written = write_figures(run_outputs, figures_dir, figure_format="pdf")
    path = Path(written["online_shift_trace"])

    assert path.is_file()
    assert path.stat().st_size > 0


def test_online_shift_trace_prefers_cifar10c() -> None:
    windows = pd.DataFrame({"realized_risk": [0.1], "coverage": [0.8]})
    selected = _select_online_trace_output(
        [
            {"dataset": "synthetic_stronger_contextual_shift", "ct_crc_windows": windows},
            {"dataset": "cifar100c", "ct_crc_windows": windows},
            {"dataset": "cifar10c", "ct_crc_windows": windows},
        ]
    )

    assert selected is not None
    assert selected["dataset"] == "cifar10c"


def test_make_artifacts_script(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    output_root = tmp_path / "artifacts"
    config = {
        "project": {"name": "shift-aware-risk-control", "mode": "artifacts", "seed": 123},
        "paths": {
            "output_root": str(output_root),
            "run_dirs": [str(run_dir)],
            "online_trace_dirs": [str(tmp_path / "runs" / "missing_online_trace")],
        },
        "artifacts": {
            "figure_format": "pdf",
            "table_format": "tex",
            "include_missing_runs": False,
            "precision": 3,
        },
    }
    config_path = tmp_path / "artifacts.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    result = subprocess.run(
        [sys.executable, "scripts/make_artifacts.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "artifact_manifest.json").is_file()
    assert (output_root / "tables" / "main_results.tex").is_file()
    assert (output_root / "figures" / "method_comparison.pdf").is_file()


def test_default_artifact_config_matches_self_contained_pipeline() -> None:
    with Path("configs/artifacts.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    run_dirs = config["paths"]["run_dirs"]

    assert run_dirs == ["outputs/runs/synthetic_main"]
    assert config["artifacts"]["include_missing_runs"] is False
    assert config["paths"]["label_efficiency_dirs"] == []
    assert config["paths"]["online_trace_dirs"] == []


def test_label_efficiency_artifacts_written(tmp_path) -> None:
    label_frame = pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "budget_windows_per_context": [1, 2],
            "actual_labeled_examples": [20, 40],
            "method": ["ct_crc", "ct_crc"],
            "mean_risk": [0.2, 0.15],
            "mean_coverage": [0.5, 0.6],
            "violation_rate": [0.1, 0.05],
            "mean_bound": [0.3, 0.25],
            "bound_underestimation_rate": [0.0, 0.0],
            "seed": [123, 123],
        }
    )
    run_dir = create_fake_run(tmp_path)
    run_outputs = load_many_run_outputs([run_dir])
    tables_dir = tmp_path / "artifacts" / "tables"
    figures_dir = tmp_path / "artifacts" / "figures"

    written_tables = write_tables(
        run_outputs,
        tables_dir,
        label_efficiency_frames=[label_frame],
        make_label_efficiency=True,
    )
    written_figures = write_figures(
        run_outputs,
        figures_dir,
        label_efficiency_frames=[label_frame],
        make_label_efficiency=True,
    )
    combined = make_label_efficiency_table([label_frame])

    assert (tables_dir / "label_efficiency.csv").is_file()
    assert (tables_dir / "label_efficiency.tex").is_file()
    label_efficiency_pdf = Path(written_figures["label_efficiency"])
    risk_only_pdf = Path(written_figures["label_efficiency_risk_only"])
    assert label_efficiency_pdf.is_file()
    assert label_efficiency_pdf.stat().st_size > 0
    assert risk_only_pdf.is_file()
    assert risk_only_pdf.stat().st_size > 0
    assert "label_efficiency_csv" in written_tables
    assert len(combined) == 2


def test_missing_runs_skipped(tmp_path) -> None:
    run_dir = create_fake_run(tmp_path)
    missing = tmp_path / "runs" / "missing"

    outputs = load_many_run_outputs([missing, run_dir], include_missing=False)
    tables_dir = tmp_path / "artifacts" / "tables"
    figures_dir = tmp_path / "artifacts" / "figures"

    write_tables(outputs, tables_dir)
    write_figures(outputs, figures_dir)

    assert len(outputs) == 1
    assert (tables_dir / "main_results.tex").is_file()
    assert (figures_dir / "method_comparison.pdf").is_file()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
