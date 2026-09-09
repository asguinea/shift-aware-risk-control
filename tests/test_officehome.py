from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextual_crc.cache.io import load_prediction_cache, save_prediction_cache
from contextual_crc.cache.validate import validate_prediction_cache_dir
from contextual_crc.datasets.officehome import (
    OfficeHomeConfig,
    build_officehome_index,
    build_officehome_prediction_cache,
    compute_officehome_prediction_frame_and_arrays,
    find_officehome_domain_dirs,
    make_prefit_calibrated_classifier,
    normalize_officehome_domain_name,
    summarize_officehome_confidence_by_domain,
    subsample_index,
    train_officehome_feature_classifier,
)


def test_normalize_officehome_domain_name() -> None:
    assert normalize_officehome_domain_name("Art") == "art"
    assert normalize_officehome_domain_name("artistic") == "art"
    assert normalize_officehome_domain_name("Clip Art") == "clipart"
    assert normalize_officehome_domain_name("Real_World") == "real_world"
    assert normalize_officehome_domain_name("Real-World") == "real_world"

    with pytest.raises(ValueError, match="Unknown OfficeHome domain"):
        normalize_officehome_domain_name("amazon")


def test_find_officehome_domain_dirs(tmp_path: Path) -> None:
    for name in ("Art", "Clip Art", "Product", "Real World", "notes"):
        (tmp_path / name).mkdir()

    found = find_officehome_domain_dirs(tmp_path)

    assert set(found) == {"art", "clipart", "product", "real_world"}
    assert found["real_world"].name == "Real World"


def test_build_officehome_index(tmp_path: Path) -> None:
    _create_fake_officehome_tree(tmp_path, n_per_class=2)
    config = OfficeHomeConfig(data_root=tmp_path, max_train_examples=None)

    index = build_officehome_index(config)

    assert {"image_path", "domain", "class_name", "label"}.issubset(index.columns)
    assert set(index["domain"]) == {"art", "clipart", "product", "real_world"}
    assert index.attrs["class_to_idx"] == {"backpack": 0, "mug": 1}
    assert index.groupby("domain").size().min() > 0

    source = subsample_index(
        index,
        domains=("real world", "product"),
        max_examples=3,
        random_seed=123,
    )
    assert len(source) == 3
    assert set(source["domain"]).issubset({"real_world", "product"})


def test_officehome_config_calibration_validation(tmp_path: Path) -> None:
    config = OfficeHomeConfig(
        data_root=tmp_path,
        calibrate_probabilities=True,
        calibration_fraction=0.25,
        calibration_method="sigmoid",
    )

    assert config.calibrate_probabilities is True
    assert config.calibration_fraction == pytest.approx(0.25)
    assert config.calibration_method == "sigmoid"
    assert OfficeHomeConfig(
        data_root=tmp_path,
        calibrate_probabilities=False,
        calibration_fraction=0.9,
    ).calibrate_probabilities is False

    with pytest.raises(ValueError, match="calibration_fraction"):
        OfficeHomeConfig(data_root=tmp_path, calibrate_probabilities=True, calibration_fraction=0.5)
    with pytest.raises(ValueError, match="calibration_method"):
        OfficeHomeConfig(data_root=tmp_path, calibration_method="temperature")


def test_train_officehome_feature_classifier_uncalibrated() -> None:
    features, labels = _make_classifier_fixture()

    classifier = train_officehome_feature_classifier(
        features,
        labels,
        random_seed=123,
        num_classes=3,
        calibrate_probabilities=False,
    )

    assert hasattr(classifier, "predict_proba") or hasattr(classifier, "decision_function")


def test_make_prefit_calibrated_classifier_constructs() -> None:
    features, labels = _make_classifier_fixture()
    base_classifier = train_officehome_feature_classifier(
        features[:60],
        labels[:60],
        random_seed=123,
        num_classes=3,
        calibrate_probabilities=False,
    )

    calibrator = make_prefit_calibrated_classifier(base_classifier, method="sigmoid")
    calibrator.fit(features[60:], labels[60:])
    probabilities = calibrator.predict_proba(features[:12])

    assert probabilities.shape == (12, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_train_officehome_feature_classifier_calibrated() -> None:
    features, labels = _make_classifier_fixture()

    classifier = train_officehome_feature_classifier(
        features,
        labels,
        random_seed=123,
        num_classes=3,
        calibrate_probabilities=True,
        calibration_fraction=0.25,
        calibration_method="sigmoid",
    )
    probabilities = classifier.predict_proba(features[:12])

    assert classifier.__class__.__name__ == "CalibratedClassifierCV"
    assert probabilities.shape == (12, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_train_officehome_feature_classifier_calibrated_does_not_fallback() -> None:
    features, labels = _make_classifier_fixture()

    classifier = train_officehome_feature_classifier(
        features,
        labels,
        random_seed=123,
        num_classes=3,
        calibrate_probabilities=True,
        calibration_fraction=0.25,
        calibration_method="sigmoid",
    )

    assert classifier.__class__.__name__ == "CalibratedClassifierCV"
    assert hasattr(classifier, "predict_proba")


def test_compute_officehome_prediction_frame_and_arrays_schema(tmp_path: Path) -> None:
    eval_index = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "domain": ["art", "clipart", "product", "real_world"],
            "class_name": ["backpack", "mug", "backpack", "mug"],
            "label": [0, 1, 0, 1],
        }
    )
    probabilities = np.asarray(
        [
            [0.7, 0.3],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.45, 0.55],
        ],
        dtype=float,
    )
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = np.asarray(
        [
            [0.0, 0.1],
            [1.0, 0.2],
            [0.0, 0.3],
            [1.0, 0.4],
        ],
        dtype=float,
    )

    examples, arrays = compute_officehome_prediction_frame_and_arrays(
        eval_index=eval_index,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
    )

    assert examples["dataset"].unique().tolist() == ["officehome"]
    assert examples["context_id"].tolist() == ["art", "clipart", "product", "real_world"]
    assert examples["context_name"].tolist() == ["Art", "Clipart", "Product", "Real World"]
    assert {"logits", "probabilities", "feature_embedding"} == set(arrays)

    metadata = {
        "dataset": "officehome",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "context_fields": ["context_id", "domain_id", "group_id", "domain"],
    }
    save_prediction_cache(tmp_path / "cache", examples, arrays, metadata, overwrite=True)
    validate_prediction_cache_dir(tmp_path / "cache")


def test_summarize_officehome_confidence_by_domain() -> None:
    examples = pd.DataFrame(
        {
            "context_id": ["art", "art", "product", "product"],
            "domain": ["art", "art", "product", "product"],
            "correct": [True, False, True, True],
            "confidence": [0.6, 0.8, 0.9, 0.7],
            "margin": [0.2, 0.4, 0.6, 0.3],
            "entropy": [0.5, 0.3, 0.2, 0.4],
        }
    )

    summary = summarize_officehome_confidence_by_domain(examples)

    assert summary["context_id"].tolist() == ["art", "product"]
    assert summary.loc[summary["context_id"] == "art", "accuracy"].iloc[0] == pytest.approx(0.5)
    assert summary.loc[summary["context_id"] == "product", "mean_confidence"].iloc[0] == pytest.approx(0.8)


def test_build_officehome_prediction_cache_with_monkeypatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_fake_officehome_tree(tmp_path / "data", n_per_class=3)

    monkeypatch.setattr(
        "contextual_crc.datasets.officehome.build_resnet18_feature_extractor",
        lambda *, pretrained, device: ("fake-model", "cpu"),
    )
    monkeypatch.setattr(
        "contextual_crc.datasets.officehome.extract_resnet_features_from_paths",
        _fake_extract_resnet_features_from_paths,
    )

    config = OfficeHomeConfig(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        max_train_examples=None,
        max_eval_examples=None,
        torch_pretrained=False,
        calibrate_probabilities=True,
        calibration_fraction=0.2,
        calibration_method="sigmoid",
    )
    cache_dir = build_officehome_prediction_cache(config.cache_dir, config, overwrite=True)

    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)
    assert metadata["dataset"] == "officehome"
    assert metadata["num_classes"] == 2
    assert metadata["class_to_idx"] == {"backpack": 0, "mug": 1}
    assert set(examples["context_id"]) == {"art", "clipart", "product", "real_world"}
    assert arrays["probabilities"].shape[1] == 2

    with (cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        saved_metadata = json.load(handle)
    assert saved_metadata["officehome_config"]["source_domains"] == ["real_world", "product"]
    assert saved_metadata["officehome_config"]["calibrate_probabilities"] is True
    assert saved_metadata["calibrate_probabilities"] is True
    assert saved_metadata["calibration_method"] == "sigmoid"


def _create_fake_officehome_tree(root: Path, *, n_per_class: int) -> None:
    domains = ("Art", "Clipart", "Product", "Real World")
    classes = ("backpack", "mug")
    for domain in domains:
        for class_name in classes:
            class_dir = root / domain / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(n_per_class):
                (class_dir / f"{class_name}_{index}.jpg").write_bytes(b"not-a-real-image")


def _make_classifier_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    labels = np.tile(np.arange(3), 30)
    features = rng.normal(scale=0.08, size=(len(labels), 5))
    features[:, 0] += labels.astype(float)
    features[:, 1] += (labels == 1).astype(float)
    features[:, 2] += (labels == 2).astype(float)
    return features.astype(np.float32), labels.astype(int)


def _fake_extract_resnet_features_from_paths(
    paths,
    *,
    model,
    device,
    batch_size,
) -> np.ndarray:
    del model, device, batch_size
    rows = []
    domain_codes = {"art": 0.0, "clipart": 1.0, "product": 2.0, "real_world": 3.0}
    class_codes = {"backpack": 0.0, "mug": 1.0}
    for path in paths:
        path = Path(path)
        class_name = path.parent.name
        domain = normalize_officehome_domain_name(path.parent.parent.name)
        label_code = class_codes[class_name]
        rows.append(
            [
                label_code,
                1.0 - label_code,
                domain_codes[domain],
                float(len(path.name) % 5),
            ]
        )
    return np.asarray(rows, dtype=np.float32)
