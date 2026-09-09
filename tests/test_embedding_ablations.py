from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.embedding_ablations import (
    ABLATION_ORDER,
    build_metadata_frame,
    build_representation,
    compute_pairwise_geometry_metrics,
    deterministic_random_embedding,
    make_embedding_ablation_table,
    select_signature_columns,
    write_embedding_ablation_artifacts,
)
from contextual_crc.artifacts.export import load_run_outputs


def test_column_group_selection_excludes_diagnostics() -> None:
    signatures = _fake_signatures()

    raw = select_signature_columns(signatures, "raw_signature_all")
    confidence = select_signature_columns(signatures, "confidence_uncertainty_only")
    probability = select_signature_columns(signatures, "probability_logit_only")
    feature = select_signature_columns(signatures, "feature_only")

    assert "diag_error_rate" not in raw
    assert {"confidence_mean", "entropy_mean", "margin_q50", "prob_max_mean"}.issubset(confidence)
    assert {"prob_class_0_mean", "logit_mean"}.issubset(probability)
    assert feature == ["feature_norm_mean"]


def test_build_representations_for_available_variants() -> None:
    signatures = _fake_signatures()
    embeddings = _fake_embeddings()

    full = build_representation(ablation="full_embedding", dataset="fixture", signatures=signatures, embeddings=embeddings)
    raw = build_representation(ablation="raw_signature_all", dataset="fixture", signatures=signatures, embeddings=embeddings)
    confidence = build_representation(
        ablation="confidence_uncertainty_only",
        dataset="fixture",
        signatures=signatures,
        embeddings=embeddings,
    )
    metadata = build_representation(ablation="metadata_only", dataset="fixture", signatures=signatures, embeddings=embeddings)
    random = build_representation(ablation="random_embedding", dataset="fixture", signatures=signatures, embeddings=embeddings)

    assert full.available and full.matrix.shape == (4, 2)
    assert raw.available and raw.matrix.shape[0] == 4
    assert confidence.available and confidence.matrix.shape[0] == 4
    assert metadata.available and metadata.matrix.shape[0] == 4
    assert random.available and random.matrix.shape == (4, 2)


def test_pairwise_rho_computation() -> None:
    metrics = compute_pairwise_geometry_metrics(np.array([[0.0], [1.0], [2.0]]), [0.0, 0.1, 0.2])

    assert metrics["n_windows"] == pytest.approx(3.0)
    assert metrics["n_pairs"] == pytest.approx(3.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_unavailable_feature_group_is_graceful() -> None:
    signatures = _fake_signatures().drop(columns=["feature_norm_mean"])

    result = build_representation(ablation="feature_only", dataset="fixture", signatures=signatures)

    assert result.available is False
    assert "no matching" in result.notes


def test_metadata_encoding_and_random_embedding_are_deterministic() -> None:
    signatures = _fake_signatures()
    metadata = build_metadata_frame(signatures)
    first = deterministic_random_embedding("fixture", signatures["window_id"], dimension=3, seed=99)
    second = deterministic_random_embedding("fixture", signatures["window_id"], dimension=3, seed=99)
    changed = deterministic_random_embedding("fixture", ["x0", "x1", "x2", "x3"], dimension=3, seed=99)

    assert "parsed_severity" in metadata.columns
    assert np.allclose(first, second)
    assert not np.allclose(first, changed)


def test_embedding_ablation_artifact_generation_with_minimal_run(tmp_path: Path) -> None:
    run_dir = _create_fake_run(tmp_path)
    outputs = [load_run_outputs(run_dir)]
    tables_dir = tmp_path / "artifacts" / "tables"
    figures_dir = tmp_path / "artifacts" / "figures"

    table = make_embedding_ablation_table(outputs)
    written = write_embedding_ablation_artifacts(outputs, tables_dir, figures_dir)
    csv = pd.read_csv(written["embedding_ablations_csv"])

    assert set(table["ablation"]) == set(ABLATION_ORDER)
    assert set(csv["ablation"]) == set(ABLATION_ORDER)
    assert (figures_dir / "embedding_ablation_smoothness.pdf").is_file()
    assert (tables_dir / "embedding_ablations.tex").is_file()
    assert csv[csv["ablation"] == "full_embedding"]["available"].iloc[0]
    assert csv[csv["ablation"] == "random_embedding"]["available"].iloc[0]


def _fake_signatures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "dataset": ["fixture"] * 4,
            "split": ["eval"] * 4,
            "context_id": ["noise_s1", "noise_s2", "blur_s1", "blur_s2"],
            "context_name": ["noise severity 1", "noise severity 2", "blur severity 1", "blur severity 2"],
            "start_pos": [0, 2, 4, 6],
            "end_pos": [2, 4, 6, 8],
            "n_examples": [2, 2, 2, 2],
            "confidence_mean": [0.9, 0.8, 0.5, 0.4],
            "entropy_mean": [0.1, 0.2, 0.5, 0.6],
            "margin_q50": [0.8, 0.7, 0.3, 0.2],
            "prob_max_mean": [0.9, 0.8, 0.5, 0.4],
            "prob_class_0_mean": [0.7, 0.6, 0.3, 0.2],
            "logit_mean": [2.0, 1.8, 0.7, 0.5],
            "feature_norm_mean": [10.0, 9.5, 4.0, 3.5],
            "diag_error_rate": [0.0, 0.1, 0.2, 0.3],
        }
    )


def _fake_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "z0": [0.0, 0.2, 1.0, 1.2],
            "z1": [0.0, 0.1, 0.9, 1.1],
        }
    )


def _create_fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "embedding_fixture"
    results = run_dir / "results"
    intermediate = run_dir / "intermediate"
    results.mkdir(parents=True)
    intermediate.mkdir(parents=True)
    _fake_signatures().to_parquet(intermediate / "signatures.parquet", engine="pyarrow", index=False)
    _fake_embeddings().to_parquet(intermediate / "embeddings.parquet", engine="pyarrow", index=False)
    pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "realized_risk": [0.0, 0.1, 0.2, 0.3],
        }
    ).to_csv(results / "ct_crc_windows.csv", index=False)
    pd.DataFrame({"method": ["ct_crc"], "mean_risk": [0.15], "mean_coverage": [0.5]}).to_csv(
        results / "method_comparison.csv",
        index=False,
    )
    (results / "run_manifest.json").write_text(
        '{"dataset":"fixture","run_name":"embedding_fixture","num_examples":8,"num_windows":4}\\n',
        encoding="utf-8",
    )
    return run_dir
