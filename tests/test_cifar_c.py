from __future__ import annotations

import builtins

import numpy as np
import pytest

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.cache.schema import REQUIRED_EXAMPLE_COLUMNS
from contextual_crc.cache.validate import validate_prediction_cache, validate_prediction_cache_dir
from contextual_crc.config.loader import load_config
from contextual_crc.datasets import cifar_c
from contextual_crc.datasets.cifar_c import (
    CIFARCConfig,
    build_cifar_c_prediction_cache,
    compute_cifar_prediction_frame_and_arrays,
    extract_image_features,
    load_cifar_c_context,
    predict_proba_with_model,
    sanitize_probabilities,
    stable_softmax,
    validate_cifar_c_files,
)


def create_fake_cifar_c(root, dataset: str = "cifar10c", corruptions=("gaussian_noise", "brightness"), n_per_severity: int = 10) -> np.ndarray:
    rng = np.random.default_rng(123)
    base = root / dataset
    base.mkdir(parents=True, exist_ok=True)
    labels = np.arange(5 * n_per_severity) % (10 if dataset == "cifar10c" else 100)
    np.save(base / "labels.npy", labels.astype(np.int64))
    for offset, corruption in enumerate(corruptions):
        images = rng.integers(0, 256, size=(5 * n_per_severity, 32, 32, 3), dtype=np.uint8)
        images = np.clip(images.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        np.save(base / f"{corruption}.npy", images)
    return labels


def make_config(tmp_path, **overrides) -> CIFARCConfig:
    kwargs = {
        "dataset": "cifar10c",
        "data_root": tmp_path,
        "corruptions": ("gaussian_noise", "brightness"),
        "severities": (1, 3),
        "max_examples_per_context": None,
        "feature_mode": "color_stats",
        "train_subset_size": 40,
        "random_seed": 123,
        "download": False,
    }
    kwargs.update(overrides)
    return CIFARCConfig(**kwargs)


def test_cifarc_config_validation(tmp_path) -> None:
    config = make_config(tmp_path)

    assert config.dataset == "cifar10c"
    assert config.severities == (1, 3)
    with pytest.raises(ValueError, match="dataset"):
        make_config(tmp_path, dataset="bad")
    with pytest.raises(ValueError, match="severities"):
        make_config(tmp_path, severities=(0,))
    with pytest.raises(ValueError, match="feature_mode"):
        make_config(tmp_path, feature_mode="bad")


def test_cifarc_config_accepts_resnet_backend(tmp_path) -> None:
    config = make_config(
        tmp_path,
        model_backend="torchvision_resnet18",
        feature_mode="resnet18_features",
        torch_batch_size=32,
        torch_device="auto",
        torch_pretrained=True,
    )

    assert config.model_backend == "torchvision_resnet18"
    assert config.feature_mode == "resnet18_features"
    assert config.torch_batch_size == 32


def test_resnet_config_files_load_without_torch() -> None:
    for config_path in ("configs/cifar10c_resnet18.yaml", "configs/cifar100c_resnet18.yaml"):
        raw = load_config(config_path)
        config = CIFARCConfig(**raw["cifar_c"])

        assert config.model_backend == "torchvision_resnet18"
        assert config.feature_mode == "resnet18_features"


def test_resnet_backend_config_is_lazy_without_torch_import(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision"}:
            raise AssertionError("torch should not be imported while validating CIFARCConfig")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    config = make_config(tmp_path, model_backend="torchvision_resnet18", feature_mode="resnet18_features")

    assert config.model_backend == "torchvision_resnet18"


def test_resnet_backend_missing_torch_raises_clear_error(monkeypatch) -> None:
    original_import = builtins.__import__

    def missing_torch_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision"}:
            raise ImportError("missing optional torch stack")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch_import)

    with pytest.raises(ImportError, match="requires torch and torchvision"):
        cifar_c.build_resnet18_feature_extractor(pretrained=False)


def test_validate_cifar_c_files(tmp_path) -> None:
    create_fake_cifar_c(tmp_path, corruptions=("gaussian_noise",))

    validate_cifar_c_files(make_config(tmp_path, corruptions=("gaussian_noise",)))
    with pytest.raises(FileNotFoundError, match="brightness"):
        validate_cifar_c_files(make_config(tmp_path, corruptions=("brightness",)))


def test_load_cifar_c_context_fake_small(tmp_path) -> None:
    labels = create_fake_cifar_c(tmp_path, corruptions=("gaussian_noise",), n_per_severity=10)
    config = make_config(tmp_path, corruptions=("gaussian_noise",), max_examples_per_context=None)

    images_s1, labels_s1 = load_cifar_c_context(config, "gaussian_noise", 1)
    images_s3, labels_s3 = load_cifar_c_context(config, "gaussian_noise", 3)
    limited_images, limited_labels = load_cifar_c_context(
        make_config(tmp_path, corruptions=("gaussian_noise",), max_examples_per_context=4),
        "gaussian_noise",
        1,
    )

    assert images_s1.shape == (10, 32, 32, 3)
    assert labels_s1.tolist() == labels[:10].tolist()
    assert labels_s3.tolist() == labels[20:30].tolist()
    assert limited_images.shape[0] == 4
    assert limited_labels.shape[0] == 4


def test_extract_image_features_color_stats() -> None:
    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    images[1] = 127
    images[2] = 255

    features = extract_image_features(images, mode="color_stats")

    assert features.shape[0] == 3
    assert features.shape[1] >= 10
    assert np.isfinite(features).all()


def test_stable_softmax_handles_extreme_logits() -> None:
    logits = np.array(
        [
            [1000.0, -1000.0, 0.0],
            [np.nan, np.inf, -np.inf],
            [-1.0e308, -1.0e308, -1.0e308],
        ]
    )

    probabilities = stable_softmax(logits)

    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > 1.0 - 1.0e-9
    assert np.allclose(probabilities[2], np.full(3, 1.0 / 3.0))


def test_sanitize_probabilities_replaces_nan_rows() -> None:
    bad_probabilities = np.array(
        [
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 0.0],
            [np.inf, 1.0, -1.0],
            [0.2, 0.3, 0.5],
        ]
    )

    probabilities = sanitize_probabilities(bad_probabilities, num_classes=3)

    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], np.full(3, 1.0 / 3.0))
    assert np.allclose(probabilities[1], np.full(3, 1.0 / 3.0))
    assert probabilities[2, 1] > 1.0 - 1.0e-9
    with pytest.raises(ValueError, match="shape"):
        sanitize_probabilities(np.ones((2, 2)), num_classes=3)


def test_predict_proba_with_model_prefers_decision_function() -> None:
    class FakeDecisionModel:
        classes_ = np.arange(10)

        def decision_function(self, features):
            logits = np.zeros((features.shape[0], 10), dtype=float)
            logits[:, 0] = 1000.0
            logits[:, 1] = -1000.0
            if len(logits) > 1:
                logits[1, 2] = np.inf
                logits[1, 3] = np.nan
            return logits

        def predict_proba(self, features):
            raise AssertionError("predict_proba should not be called when decision_function exists")

    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)

    probabilities, logits = predict_proba_with_model(FakeDecisionModel(), images, "color_stats", 10)

    assert probabilities.shape == (3, 10)
    assert logits.shape == (3, 10)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(logits).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_proba_with_model_sanitizes_bad_predict_proba() -> None:
    class FakeBadProbabilityModel:
        classes_ = np.array([0, 2, 4])

        def predict_proba(self, features):
            return np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [0.0, 0.0, 0.0],
                    [np.inf, 1.0, -1.0],
                ]
            )

    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)

    probabilities, logits = predict_proba_with_model(FakeBadProbabilityModel(), images, "color_stats", 5)

    assert probabilities.shape == (3, 5)
    assert logits.shape == (3, 5)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(logits).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], np.full(5, 1.0 / 5.0))
    assert np.allclose(probabilities[1], np.full(5, 1.0 / 5.0))


def test_compute_cifar_prediction_frame_and_arrays_schema() -> None:
    rng = np.random.default_rng(123)
    images = rng.integers(0, 256, size=(6, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(6) % 10
    probabilities = np.full((6, 10), 0.05)
    probabilities[np.arange(6), labels] = 0.55
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = extract_image_features(images)

    examples, arrays = compute_cifar_prediction_frame_and_arrays(
        dataset="cifar10c",
        split="eval",
        corruption="gaussian_noise",
        severity=3,
        images=images,
        labels=labels,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        assert column in examples.columns
    assert {"logits", "probabilities", "feature_embedding"} == set(arrays)
    metadata = {
        "dataset": "cifar10c",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 10,
        "created_at": "2026-04-30T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "optional_columns": ["severity", "corruption_type", "domain_id", "group_id", "time_index", "source"],
        "context_fields": ["context_id", "corruption_type", "severity"],
    }
    validate_prediction_cache(examples, arrays, metadata)


def test_build_cifar_c_prediction_cache_with_monkeypatch(tmp_path, monkeypatch) -> None:
    create_fake_cifar_c(tmp_path, corruptions=("gaussian_noise", "brightness"), n_per_severity=8)
    rng = np.random.default_rng(321)
    train_images = rng.integers(0, 256, size=(60, 32, 32, 3), dtype=np.uint8)
    train_labels = np.arange(60) % 10

    def fake_load_clean_cifar_train(dataset, data_root, *, train_subset_size=None, random_seed=123):
        return train_images, train_labels

    monkeypatch.setattr(cifar_c, "load_clean_cifar_train", fake_load_clean_cifar_train)
    config = make_config(
        tmp_path,
        corruptions=("gaussian_noise", "brightness"),
        severities=(1, 2),
        max_examples_per_context=5,
        train_subset_size=60,
    )
    cache_dir = tmp_path / "cache"

    saved = build_cifar_c_prediction_cache(cache_dir, config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert len(set(examples["context_id"])) == 4
    assert arrays["logits"].shape[0] == len(examples)
    assert arrays["logits"].shape[1] == 10
    assert metadata["dataset"] == "cifar10c"
    assert metadata["model_backend"] == "sklearn_logistic"


def test_build_cifar_cache_resnet_backend_with_monkeypatch(tmp_path, monkeypatch) -> None:
    create_fake_cifar_c(tmp_path, corruptions=("gaussian_noise",), n_per_severity=6)
    rng = np.random.default_rng(321)
    train_images = rng.integers(0, 256, size=(40, 32, 32, 3), dtype=np.uint8)
    train_labels = np.arange(40) % 10

    class FakeFeatureClassifier:
        classes_ = np.arange(10)

        def predict_proba(self, features):
            features = np.asarray(features, dtype=float)
            logits = np.zeros((features.shape[0], 10), dtype=float)
            logits[:, 0] = features[:, 0]
            logits[:, 1] = features[:, 1]
            logits[:, 2] = features[:, 2]
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / exp.sum(axis=1, keepdims=True)

    def fake_load_clean_cifar_train(dataset, data_root, *, train_subset_size=None, random_seed=123):
        return train_images, train_labels

    def fake_build_resnet18_feature_extractor(*, pretrained=True, device="auto"):
        return object(), "cpu"

    def fake_extract_resnet18_features(images, *, model, device, batch_size=128):
        normalized = np.asarray(images, dtype=np.float32) / 255.0
        flat = normalized.reshape(normalized.shape[0], -1)
        return np.column_stack(
            [
                flat.mean(axis=1),
                flat.std(axis=1),
                flat.min(axis=1),
                flat.max(axis=1),
            ]
        ).astype(np.float32)

    def fake_train_feature_classifier(train_features, train_labels, *, random_seed, num_classes):
        assert train_features.shape == (40, 4)
        assert num_classes == 10
        return FakeFeatureClassifier()

    monkeypatch.setattr(cifar_c, "load_clean_cifar_train", fake_load_clean_cifar_train)
    monkeypatch.setattr(cifar_c, "build_resnet18_feature_extractor", fake_build_resnet18_feature_extractor)
    monkeypatch.setattr(cifar_c, "extract_resnet18_features", fake_extract_resnet18_features)
    monkeypatch.setattr(cifar_c, "train_feature_classifier", fake_train_feature_classifier)

    config = make_config(
        tmp_path,
        corruptions=("gaussian_noise",),
        severities=(1, 2),
        max_examples_per_context=4,
        model_backend="torchvision_resnet18",
        feature_mode="resnet18_features",
        torch_batch_size=7,
        train_subset_size=40,
    )
    cache_dir = tmp_path / "resnet_cache"

    saved = build_cifar_c_prediction_cache(cache_dir, config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert len(examples) == 8
    assert arrays["logits"].shape == (8, 10)
    assert arrays["feature_embedding"].shape == (8, 4)
    assert metadata["model_backend"] == "torchvision_resnet18"
    assert metadata["feature_mode"] == "resnet18_features"
    assert metadata["torch_device"] == "cpu"
    assert metadata["torch_batch_size"] == 7


def test_cifar100_resnet_fake_backend_cache_has_finite_confidence(tmp_path, monkeypatch) -> None:
    create_fake_cifar_c(tmp_path, dataset="cifar100c", corruptions=("gaussian_noise",), n_per_severity=6)
    rng = np.random.default_rng(321)
    train_images = rng.integers(0, 256, size=(120, 32, 32, 3), dtype=np.uint8)
    train_labels = np.arange(120) % 100

    class FakeBadCifar100Classifier:
        classes_ = np.array([0, 50, 99])

        def predict_proba(self, features):
            probabilities = np.zeros((features.shape[0], 3), dtype=float)
            probabilities[:, 0] = np.nan
            probabilities[:, 1] = 0.0
            probabilities[:, 2] = 1.0
            if len(probabilities) > 0:
                probabilities[0, :] = np.nan
            if len(probabilities) > 1:
                probabilities[1, :] = 0.0
            if len(probabilities) > 2:
                probabilities[2, :] = [np.inf, 0.0, 2.0]
            return probabilities

    def fake_load_clean_cifar_train(dataset, data_root, *, train_subset_size=None, random_seed=123):
        return train_images, train_labels

    def fake_build_resnet18_feature_extractor(*, pretrained=True, device="auto"):
        return object(), "cpu"

    def fake_extract_resnet18_features(images, *, model, device, batch_size=128):
        normalized = np.asarray(images, dtype=np.float32) / 255.0
        flat = normalized.reshape(normalized.shape[0], -1)
        return np.column_stack(
            [
                flat.mean(axis=1),
                flat.std(axis=1),
                flat.min(axis=1),
                flat.max(axis=1),
            ]
        ).astype(np.float32)

    def fake_train_feature_classifier(train_features, train_labels, *, random_seed, num_classes):
        assert train_features.shape == (120, 4)
        assert num_classes == 100
        return FakeBadCifar100Classifier()

    monkeypatch.setattr(cifar_c, "load_clean_cifar_train", fake_load_clean_cifar_train)
    monkeypatch.setattr(cifar_c, "build_resnet18_feature_extractor", fake_build_resnet18_feature_extractor)
    monkeypatch.setattr(cifar_c, "extract_resnet18_features", fake_extract_resnet18_features)
    monkeypatch.setattr(cifar_c, "train_feature_classifier", fake_train_feature_classifier)

    config = make_config(
        tmp_path,
        dataset="cifar100c",
        corruptions=("gaussian_noise",),
        severities=(1, 2),
        max_examples_per_context=4,
        model_backend="torchvision_resnet18",
        feature_mode="resnet18_features",
        train_subset_size=120,
    )
    cache_dir = tmp_path / "cifar100_resnet_cache"

    saved = build_cifar_c_prediction_cache(cache_dir, config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert metadata["dataset"] == "cifar100c"
    assert metadata["model_backend"] == "torchvision_resnet18"
    assert np.isfinite(examples["confidence"].to_numpy(dtype=float)).all()
    assert np.isfinite(arrays["probabilities"]).all()
    assert np.allclose(arrays["probabilities"].sum(axis=1), 1.0)
