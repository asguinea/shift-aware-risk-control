from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from contextual_crc.artifacts.leakage_free import write_leakage_free_artifacts
from contextual_crc.embeddings.context_encoder import EncoderConfig, build_context_embedding_frame
from contextual_crc.evaluation.leakage_free import (
    _overlap_counts,
    fit_calibration_only_encoder,
    make_leakage_free_window_split,
    subset_examples_and_remap_windows,
)
from contextual_crc.evaluation.synthetic_experiment import _estimate_ct_l_hat
from contextual_crc.risk_control.crc import RiskControlConfig
from contextual_crc.risk_control.ct_crc import CTCRCConfig, build_validated_contexts
from contextual_crc.risk_control.ct_crc_dkw import (
    build_dkw_validated_contexts,
    evaluate_dkw_target_feasible_windows,
)
from contextual_crc.risk_control.losses import LossConfig


def _examples_and_windows(n_contexts: int = 3, windows_per_context: int = 3):
    examples = []
    windows = []
    position = 0
    for context in range(n_contexts):
        for window in range(windows_per_context):
            indices = []
            for item in range(4):
                examples.append(
                    {
                        "example_id": f"e-{context}-{window}-{item}",
                        "context_id": f"c{context}",
                        "context_name": f"Context {context}",
                        "confidence": 0.2 + 0.1 * item + 0.01 * window,
                        "pred": int(item % 2),
                        "label": int((item + context) % 2),
                    }
                )
                indices.append(position)
                position += 1
            windows.append(
                {
                    "window_id": f"w-{context}-{window}",
                    "context_id": f"c{context}",
                    "context_name": f"Context {context}",
                    "example_indices": indices,
                    "n_examples": len(indices),
                }
            )
    return pd.DataFrame(examples), pd.DataFrame(windows)


def _signature_frame(windows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": windows["window_id"],
            "context_id": windows["context_id"],
            "feature_a": np.arange(len(windows), dtype=float),
            "feature_b": np.arange(len(windows), dtype=float) ** 2,
            "diag_error_rate": np.linspace(0.0, 1.0, len(windows)),
        }
    )


def test_window_split_is_deterministic_and_whole_window():
    examples, windows = _examples_and_windows()
    first = make_leakage_free_window_split(windows, ["c0", "c1"], seed=123)
    second = make_leakage_free_window_split(windows, ["c0", "c1"], seed=123)
    assert first.assignments[["window_id", "oracle_partition"]].equals(
        second.assignments[["window_id", "oracle_partition"]]
    )
    assert len(first.evaluation_windows) == 3
    assert len(first.oracle_calibration_windows) == 6
    assert len(first.deployable_calibration_windows) == 4
    assert all(len(indices) == 4 for indices in first.assignments["example_indices"])


def test_split_has_zero_example_and_window_overlap():
    examples, windows = _examples_and_windows()
    split = make_leakage_free_window_split(windows, ["c0", "c1"], seed=123)
    cal_examples, cal_windows = subset_examples_and_remap_windows(
        examples, split.deployable_calibration_windows
    )
    eval_examples, _ = subset_examples_and_remap_windows(examples, split.evaluation_windows)
    overlap = _overlap_counts(
        cal_examples, cal_windows, eval_examples, split.evaluation_windows
    )
    assert overlap == {"example_overlap": 0, "window_overlap": 0}


def test_minimum_window_constraints_and_too_few_rejected():
    _, windows = _examples_and_windows(windows_per_context=2)
    split = make_leakage_free_window_split(windows, ["c0"], seed=0)
    counts = split.assignments.groupby(["context_id", "oracle_partition"]).size().unstack()
    assert (counts["calibration"] >= 1).all()
    assert (counts["evaluation"] >= 1).all()
    with pytest.raises(ValueError, match="at least two"):
        make_leakage_free_window_split(windows.iloc[[0]], ["c0"], seed=0)


@pytest.mark.parametrize("method", ["standardized", "pca"])
def test_scaler_and_pca_fit_calibration_rows_only(method: str):
    _, windows = _examples_and_windows()
    signatures = _signature_frame(windows)
    fit_ids = windows.iloc[:4]["window_id"].tolist()
    encoder, embedded, audit = fit_calibration_only_encoder(
        signatures,
        fit_ids,
        EncoderConfig(method=method, n_components=2, random_seed=123),
    )
    fit = signatures[signatures["window_id"].isin(fit_ids)]
    assert audit["fit_window_ids"] == sorted(fit_ids)
    assert set(audit["feature_columns"]) == {"feature_a", "feature_b"}
    assert np.allclose(encoder.scaler_.mean_, fit[["feature_a", "feature_b"]].mean())
    assert len(embedded) == len(signatures)
    if method == "pca":
        assert encoder.pca_.n_samples_ == len(fit_ids)


def test_l_hat_depends_only_on_calibration_side_labels():
    examples, windows = _examples_and_windows(n_contexts=2, windows_per_context=3)
    signatures = _signature_frame(windows)
    embeddings = build_context_embedding_frame(
        signatures, signatures[["feature_a", "feature_b"]].to_numpy(dtype=float)
    )
    split = make_leakage_free_window_split(windows, ["c0", "c1"], seed=123)
    cal_examples, _ = subset_examples_and_remap_windows(examples, split.deployable_calibration_windows)
    cal_ids = set(split.deployable_calibration_windows["window_id"])
    cal_embeddings = embeddings[embeddings["window_id"].isin(cal_ids)].reset_index(drop=True)
    config = CTCRCConfig(
        risk_control=RiskControlConfig(
            loss=LossConfig(loss_name="accepted_error"), risk_budget=0.5, delta=0.1
        )
    )
    result = build_validated_contexts(cal_examples, cal_embeddings, config)
    first = _estimate_ct_l_hat(result, {"metric": "euclidean", "l_hat_quantile": 0.95})
    mutated = examples.copy()
    eval_indices = sorted(
        {int(i) for values in split.evaluation_windows["example_indices"] for i in values}
    )
    mutated.loc[eval_indices, "label"] = 1 - mutated.loc[eval_indices, "label"]
    second_result = build_validated_contexts(cal_examples, cal_embeddings, config)
    second = _estimate_ct_l_hat(second_result, {"metric": "euclidean", "l_hat_quantile": 0.95})
    assert first == second


def test_common_evaluation_set_and_oracle_separation():
    examples, windows = _examples_and_windows()
    split = make_leakage_free_window_split(windows, ["c0", "c1"], seed=123)
    common = set(split.evaluation_windows["window_id"])
    method_sets = {method: common.copy() for method in ("global", "metadata", "ct", "cw")}
    assert all(window_ids == common for window_ids in method_sets.values())
    oracle_examples, oracle_windows = subset_examples_and_remap_windows(
        examples, split.oracle_calibration_windows
    )
    eval_examples, _ = subset_examples_and_remap_windows(examples, split.evaluation_windows)
    assert _overlap_counts(
        oracle_examples, oracle_windows, eval_examples, split.evaluation_windows
    ) == {"example_overlap": 0, "window_overlap": 0}
    assert set(oracle_examples["context_id"]) == {"c0", "c1", "c2"}


def test_corrected_ct_is_target_feasible_and_uses_explicit_accept_none():
    examples, windows = _examples_and_windows(n_contexts=2, windows_per_context=3)
    split = make_leakage_free_window_split(windows, ["c0"], seed=123)
    cal_examples, _ = subset_examples_and_remap_windows(examples, split.deployable_calibration_windows)
    signatures = _signature_frame(windows)
    embeddings = build_context_embedding_frame(
        signatures, signatures[["feature_a", "feature_b"]].to_numpy(dtype=float)
    )
    cal_ids = set(split.deployable_calibration_windows["window_id"])
    eval_ids = set(split.evaluation_windows["window_id"])
    cal_embeddings = embeddings[embeddings["window_id"].isin(cal_ids)].reset_index(drop=True)
    eval_embeddings = embeddings[embeddings["window_id"].isin(eval_ids)].reset_index(drop=True)
    config = CTCRCConfig(
        risk_control=RiskControlConfig(
            loss=LossConfig(loss_name="accepted_error"), risk_budget=0.01, delta=0.05
        ),
        l_hat=1.0,
    )
    result, _ = build_dkw_validated_contexts(
        cal_examples, cal_embeddings, config, library_alpha=0.05
    )
    evaluated = evaluate_dkw_target_feasible_windows(
        examples, split.evaluation_windows, eval_embeddings, result
    )
    assert (evaluated["transferred_upper_bound"] <= 0.01 + 1e-12).all()
    assert evaluated["forced_accept_none"].any()
    assert (evaluated.loc[evaluated["forced_accept_none"], "coverage"] == 0.0).all()


def test_artifacts_present_and_absent(tmp_path):
    assert write_leakage_free_artifacts(
        tmp_path,
        main_results_csv=tmp_path / "missing-main.csv",
        matched_frontier_csv=tmp_path / "missing-matched.csv",
    ) == {}
    main = pd.DataFrame(
        [
            {
                "dataset": "cifar10c",
                "method": "global_crc",
                "risk": 0.1,
                "coverage": 0.5,
                "bound_underestimation": 0.0,
                "mean_bound_gap": 0.05,
                "n_calibration_examples": 10,
                "n_calibration_windows": 2,
                "n_evaluation_examples": 5,
                "n_evaluation_windows": 1,
            }
        ]
    )
    matched = pd.DataFrame(
        [
            {
                "dataset": "cifar10c",
                "method": "global_crc",
                "global_coverage_target": 0.5,
                "risk_at_global_coverage": 0.1,
                "coverage_at_tau_0_10": 0.5,
                "post_hoc_evaluation": True,
            }
        ]
    )
    main_path = tmp_path / "main.csv"
    matched_path = tmp_path / "matched.csv"
    main.to_csv(main_path, index=False)
    matched.to_csv(matched_path, index=False)
    paths = write_leakage_free_artifacts(
        tmp_path, main_results_csv=main_path, matched_frontier_csv=matched_path
    )
    assert len(paths) == 4
    assert all(pd.notna(value) for value in paths.values())
    assert all((tmp_path / name).is_file() for name in (
        "leakage_free_main_results.csv",
        "leakage_free_main_results.tex",
        "leakage_free_matched_frontier.csv",
        "leakage_free_matched_frontier.tex",
    ))
