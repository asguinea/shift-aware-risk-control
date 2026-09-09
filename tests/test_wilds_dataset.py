from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from contextual_crc.cache.io import load_prediction_cache, save_prediction_cache
from contextual_crc.cache.schema import REQUIRED_EXAMPLE_COLUMNS
from contextual_crc.cache.validate import validate_prediction_cache, validate_prediction_cache_dir
from contextual_crc.datasets.wilds_dataset import (
    WILDSConfig,
    build_wilds_prediction_cache,
    compute_camelyon17_prediction_frame_and_arrays,
    compute_fmow_prediction_frame_and_arrays,
    compute_wilds_prediction_frame_and_arrays,
    extract_text_features,
    load_civilcomments_csv_split,
    load_fmow_wilds_splits,
    _patched_pandas_to_datetime_for_fmow,
    predict_text_proba_and_features,
    train_sklearn_text_classifier,
    validate_wilds_csv_files,
)


def create_fake_civilcomments_csv(root) -> None:
    base = root / "civilcomments"
    base.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "text": [
                "kind and thoughtful comment",
                "you are awful",
                "helpful discussion point",
                "horrible toxic insult",
                "calm reply with context",
                "bad hateful message",
                "constructive and nice",
                "nasty personal attack",
            ],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "comment_text": [
                "a civil and useful note",
                "awful attack here",
                "friendly question?",
                "hateful insult!",
                "measured response",
                "very toxic statement",
            ],
            "y": [0, 1, 0, 1, 0, 1],
            "identity": ["group_a", "group_b", "group_a", "group_b", "group_c", "group_c"],
        }
    )
    train.to_csv(base / "train.csv", index=False)
    eval_frame.to_csv(base / "eval.csv", index=False)


def make_config(tmp_path, **overrides) -> WILDSConfig:
    kwargs = {
        "dataset": "civilcomments",
        "source": "csv",
        "data_root": tmp_path,
        "train_subset_size": None,
        "eval_subset_size": None,
        "random_seed": 123,
        "max_features": 200,
        "ngram_range": (1, 2),
    }
    kwargs.update(overrides)
    return WILDSConfig(**kwargs)


def test_wilds_config_validation(tmp_path) -> None:
    config = make_config(tmp_path)

    assert config.dataset == "civilcomments"
    assert config.source == "csv"
    assert config.ngram_range == (1, 2)
    camelyon = make_config(
        tmp_path,
        dataset="camelyon17",
        source="wilds",
        model_backend="torchvision_resnet18_features",
        context_column="domain_id",
    )
    assert camelyon.dataset == "camelyon17"
    assert camelyon.model_backend == "torchvision_resnet18_features"
    fmow = make_config(
        tmp_path,
        dataset="fmow",
        source="wilds",
        model_backend="torchvision_resnet18_features",
    )
    assert fmow.dataset == "fmow"
    assert fmow.context_column == "domain_id"
    with pytest.raises(ValueError, match="source"):
        make_config(tmp_path, source="bad")
    with pytest.raises(ValueError, match="ngram_range"):
        make_config(tmp_path, ngram_range=(2, 1))


def test_camelyon17_invalid_source_or_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="source='wilds'"):
        make_config(
            tmp_path,
            dataset="camelyon17",
            source="csv",
            model_backend="torchvision_resnet18_features",
        )
    with pytest.raises(ValueError, match="torchvision_resnet18_features"):
        make_config(
            tmp_path,
            dataset="camelyon17",
            source="wilds",
            model_backend="sklearn_tfidf",
        )


def test_wilds_config_accepts_fmow(tmp_path) -> None:
    config = WILDSConfig(
        dataset="fmow",
        source="wilds",
        data_root=tmp_path,
        model_backend="torchvision_resnet18_features",
        max_train_examples=10,
        max_eval_examples=8,
        torch_batch_size=4,
        eval_splits=("val", "test"),
    )

    assert config.dataset == "fmow"
    assert config.source == "wilds"
    assert config.context_column == "domain_id"
    assert config.eval_splits == ("val", "test")


def test_fmow_invalid_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="torchvision_resnet18_features"):
        WILDSConfig(
            dataset="fmow",
            source="wilds",
            data_root=tmp_path,
            model_backend="sklearn_tfidf",
        )


def test_fmow_pandas_to_datetime_patch_restores_original() -> None:
    original = pd.to_datetime

    with _patched_pandas_to_datetime_for_fmow():
        assert pd.to_datetime is not original

    assert pd.to_datetime is original


def test_fmow_pandas_to_datetime_patch_handles_mixed_iso() -> None:
    timestamps = pd.Series(
        [
            "2011-02-07T02:48:56.643Z",
            "2011-02-07T02:48:56Z",
        ]
    )

    with _patched_pandas_to_datetime_for_fmow():
        parsed = pd.to_datetime(timestamps)

    assert len(parsed) == 2
    assert parsed.notna().all()


def test_fmow_loader_uses_pandas_timestamp_patch(monkeypatch, tmp_path) -> None:
    original = pd.to_datetime
    observed = {"patched": False}

    class FakeFMoWDataset:
        split_dict = {"train": 0, "val": 1, "test": 2}

        def get_subset(self, split, transform=None):
            del transform
            return f"{split}_subset"

    def fake_get_dataset(dataset, root_dir=None, download=False):
        del root_dir, download
        assert dataset == "fmow"
        observed["patched"] = pd.to_datetime is not original
        pd.to_datetime(
            pd.Series(
                [
                    "2011-02-07T02:48:56.643Z",
                    "2011-02-07T02:48:56Z",
                ]
            )
        )
        return FakeFMoWDataset()

    monkeypatch.setitem(sys.modules, "wilds", types.SimpleNamespace(get_dataset=fake_get_dataset))
    config = WILDSConfig(
        dataset="fmow",
        source="wilds",
        data_root=tmp_path,
        model_backend="torchvision_resnet18_features",
        eval_splits=("val", "test"),
    )

    train_subset, eval_subsets = load_fmow_wilds_splits(config)

    assert observed["patched"] is True
    assert pd.to_datetime is original
    assert train_subset == "train_subset"
    assert eval_subsets == {"val": "val_subset", "test": "test_subset"}


def test_validate_wilds_csv_files(tmp_path) -> None:
    create_fake_civilcomments_csv(tmp_path)

    validate_wilds_csv_files(make_config(tmp_path))
    (tmp_path / "civilcomments" / "eval.csv").unlink()
    with pytest.raises(FileNotFoundError, match="eval.csv"):
        validate_wilds_csv_files(make_config(tmp_path))


def test_load_civilcomments_csv_split_normalizes_columns(tmp_path) -> None:
    create_fake_civilcomments_csv(tmp_path)
    config = make_config(tmp_path)

    train = load_civilcomments_csv_split(config, "train")
    eval_frame = load_civilcomments_csv_split(config, "eval")

    assert {"text", "label", "split", "group_id", "domain_id", "identity"}.issubset(train.columns)
    assert train["split"].unique().tolist() == ["train"]
    assert train["group_id"].unique().tolist() == ["train"]
    assert eval_frame["text"].str.contains("civil").any()
    assert eval_frame["label"].tolist() == [0, 1, 0, 1, 0, 1]
    assert set(eval_frame["group_id"]) == {"group_a", "group_b", "group_c"}
    assert set(eval_frame["domain_id"]) == {"group_a", "group_b", "group_c"}


def test_train_and_predict_text_classifier(tmp_path) -> None:
    create_fake_civilcomments_csv(tmp_path)
    train_df = load_civilcomments_csv_split(make_config(tmp_path), "train")
    model = train_sklearn_text_classifier(
        train_df,
        max_features=200,
        ngram_range=(1, 2),
        random_seed=123,
    )

    probabilities, logits, features = predict_text_proba_and_features(
        model,
        ["kind useful note", "awful toxic insult"],
    )

    assert probabilities.shape == (2, 2)
    assert logits.shape == (2, 2)
    assert features.shape[0] == 2
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)


def test_extract_text_features() -> None:
    features = extract_text_features(["Hello WORLD!", "why?", ""], probabilities=np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]]))

    assert features.shape[0] == 3
    assert features.shape[1] >= 8
    assert np.isfinite(features).all()


def test_compute_wilds_prediction_frame_and_arrays_schema(tmp_path) -> None:
    eval_df = pd.DataFrame(
        {
            "text": ["ok", "bad", "fine", "toxic"],
            "label": [0, 1, 0, 1],
            "group_id": ["group_a", "group_b", "group_a", "group_b"],
            "domain_id": ["domain_a", "domain_b", "domain_a", "domain_b"],
            "source": ["wilds-csv"] * 4,
        }
    )
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = extract_text_features(eval_df["text"], probabilities)

    examples, arrays = compute_wilds_prediction_frame_and_arrays(
        dataset="civilcomments",
        split="eval",
        eval_df=eval_df,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        assert column in examples.columns
    assert {"logits", "probabilities", "feature_embedding"} == set(arrays)
    metadata = {
        "dataset": "wilds_civilcomments",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "created_at": "2026-04-30T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "optional_columns": ["domain_id", "group_id", "time_index", "source"],
        "context_fields": ["context_id", "domain_id", "group_id"],
    }
    validate_prediction_cache(examples, arrays, metadata)
    saved = save_prediction_cache(tmp_path / "wilds_frame_cache", examples, arrays, metadata, overwrite=True)
    validate_prediction_cache_dir(saved)


def test_compute_camelyon17_prediction_frame_and_arrays_schema(tmp_path) -> None:
    eval_df = pd.DataFrame(
        {
            "original_index": [0, 1, 2, 3],
            "split": ["val", "val", "test", "test"],
            "domain_id": [0, 1, 0, 2],
            "group_id": [0, 1, 0, 2],
            "source": ["wilds-camelyon17"] * 4,
            "time_index": [0, 1, 2, 3],
        }
    )
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75], [0.7, 0.3], [0.1, 0.9]])
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = np.arange(16, dtype=np.float32).reshape(4, 4)

    examples, arrays = compute_camelyon17_prediction_frame_and_arrays(
        eval_df=eval_df,
        labels=labels,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        assert column in examples.columns
    assert examples["dataset"].unique().tolist() == ["wilds_camelyon17"]
    assert set(examples["context_id"]) == {"domain_0", "domain_1", "domain_2"}
    assert {"logits", "probabilities", "feature_embedding"} == set(arrays)
    metadata = {
        "dataset": "wilds_camelyon17",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "created_at": "2026-04-30T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "optional_columns": ["domain_id", "group_id", "time_index", "source"],
        "context_fields": ["context_id", "domain_id", "group_id"],
    }
    validate_prediction_cache(examples, arrays, metadata)
    saved = save_prediction_cache(tmp_path / "camelyon_frame_cache", examples, arrays, metadata, overwrite=True)
    validate_prediction_cache_dir(saved)


def test_compute_fmow_prediction_frame_and_arrays_schema(tmp_path) -> None:
    eval_df = pd.DataFrame(
        {
            "original_index": [0, 1, 2, 3, 4],
            "split": ["val", "val", "test", "test", "test"],
            "domain_id": [10, 11, 10, 12, 12],
            "group_id": [10, 11, 10, 12, 12],
            "region": ["americas", "europe", "americas", "asia", "asia"],
            "year": [2013, 2014, 2015, 2016, 2017],
            "source": ["wilds-fmow"] * 5,
            "time_index": [0, 1, 2, 3, 4],
        }
    )
    labels = np.array([0, 1, 2, 3, 4])
    probabilities = np.array(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],
            [0.05, 0.65, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.60, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.60, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.60],
        ],
        dtype=float,
    )
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = np.arange(20, dtype=np.float32).reshape(5, 4)

    examples, arrays = compute_fmow_prediction_frame_and_arrays(
        eval_df=eval_df,
        labels=labels,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
        context_column="region",
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        assert column in examples.columns
    assert examples["dataset"].unique().tolist() == ["wilds_fmow"]
    assert set(examples["context_id"]) == {"region_americas", "region_europe", "region_asia"}
    assert {"region", "year", "domain_id", "group_id", "source", "time_index"}.issubset(examples.columns)
    assert arrays["probabilities"].shape == (5, 5)
    metadata = {
        "dataset": "wilds_fmow",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 5,
        "created_at": "2026-04-30T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "optional_columns": ["domain_id", "group_id", "time_index", "source"],
        "context_fields": ["context_id", "domain_id", "group_id", "region", "year"],
    }
    validate_prediction_cache(examples, arrays, metadata)
    saved = save_prediction_cache(tmp_path / "fmow_frame_cache", examples, arrays, metadata, overwrite=True)
    validate_prediction_cache_dir(saved)


def test_build_wilds_prediction_cache_csv(tmp_path) -> None:
    data_root = tmp_path / "data" / "wilds"
    create_fake_civilcomments_csv(data_root)
    config = make_config(data_root, train_subset_size=None, eval_subset_size=None)
    cache_dir = tmp_path / "cache"

    saved = build_wilds_prediction_cache(cache_dir, config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert len(set(examples["context_id"])) >= 2
    assert arrays["logits"].shape[0] == len(examples)
    assert arrays["logits"].shape[1] == 2
    assert metadata["dataset"] == "wilds_civilcomments"
    assert metadata["source"] == "csv"


def test_build_wilds_prediction_cache_camelyon17_with_monkeypatch(tmp_path, monkeypatch) -> None:
    config = WILDSConfig(
        dataset="camelyon17",
        source="wilds",
        data_root=tmp_path,
        model_backend="torchvision_resnet18_features",
        max_train_examples=6,
        max_eval_examples=6,
        torch_batch_size=2,
        eval_splits=("val", "test"),
        context_column="domain_id",
    )

    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.load_camelyon17_wilds_splits",
        lambda cfg: ("train_subset", {"val": "val_subset", "test": "test_subset"}),
    )
    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.build_resnet18_feature_extractor",
        lambda pretrained, device: ("fake_model", "cpu"),
    )

    def fake_extract(subset, *, model, device, batch_size, max_examples, random_seed, split):
        del model, device, batch_size, max_examples, random_seed
        if split == "train":
            features = np.array(
                [
                    [0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.1],
                    [1.0, 1.0, 0.2],
                    [1.1, 1.0, 0.2],
                    [0.2, 0.1, 0.1],
                    [1.2, 1.1, 0.2],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 0, 1, 1, 0, 1])
            meta = pd.DataFrame({"original_index": range(6), "split": ["train"] * 6, "domain_id": [0] * 3 + [1] * 3})
            return features, labels, meta
        offset = 0 if split == "val" else 3
        features = np.array(
            [
                [0.05 + offset, 0.0, 0.1],
                [1.05 + offset, 1.0, 0.2],
                [0.15 + offset, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 0])
        meta = pd.DataFrame(
            {
                "original_index": [0, 1, 2],
                "split": [split] * 3,
                "domain_id": [0, 1, 2 if split == "test" else 1],
                "group_id": [0, 1, 2 if split == "test" else 1],
                "source": ["wilds-camelyon17"] * 3,
            }
        )
        return features, labels, meta

    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.extract_resnet_features_from_wilds_subset",
        fake_extract,
    )

    saved = build_wilds_prediction_cache(tmp_path / "camelyon_cache", config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert metadata["dataset"] == "wilds_camelyon17"
    assert metadata["model_backend"] == "torchvision_resnet18_features"
    assert metadata["source"] == "wilds"
    assert len(set(examples["context_id"])) >= 3
    assert arrays["feature_embedding"].shape[0] == len(examples)
    assert arrays["probabilities"].shape == (len(examples), 2)


def test_build_wilds_prediction_cache_fmow_with_monkeypatch(tmp_path, monkeypatch) -> None:
    config = WILDSConfig(
        dataset="fmow",
        source="wilds",
        data_root=tmp_path,
        model_backend="torchvision_resnet18_features",
        max_train_examples=10,
        max_eval_examples=10,
        torch_batch_size=2,
        eval_splits=("val", "test"),
        context_column="region",
    )

    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.load_fmow_wilds_splits",
        lambda cfg: ("train_subset", {"val": "val_subset", "test": "test_subset"}),
    )
    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.build_resnet18_feature_extractor",
        lambda pretrained, device: ("fake_model", "cpu"),
    )

    def fake_extract(subset, *, model, device, batch_size, max_examples, random_seed, split, metadata_parser=None):
        del subset, model, device, batch_size, max_examples, random_seed, metadata_parser
        if split == "train":
            features = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.1, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.1],
                    [1.0, 1.0, 1.0],
                    [1.1, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            meta = pd.DataFrame({"original_index": range(10), "split": ["train"] * 10})
            return features, labels, meta
        regions = ["americas", "europe", "asia", "americas", "europe"]
        offset = 0.0 if split == "val" else 0.2
        features = np.array(
            [
                [0.05 + offset, 0.0, 0.0],
                [1.05 + offset, 0.0, 0.0],
                [0.0, 1.05 + offset, 0.0],
                [0.0, 0.0, 1.05 + offset],
                [1.05 + offset, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 2, 3, 4])
        meta = pd.DataFrame(
            {
                "original_index": range(5),
                "split": [split] * 5,
                "domain_id": [0, 1, 2, 0, 1],
                "group_id": [0, 1, 2, 0, 1],
                "region": regions,
                "year": [2013, 2014, 2015, 2016, 2017],
                "source": ["wilds-fmow"] * 5,
                "time_index": range(5),
            }
        )
        return features, labels, meta

    monkeypatch.setattr(
        "contextual_crc.datasets.wilds_dataset.extract_resnet_features_from_wilds_subset",
        fake_extract,
    )

    saved = build_wilds_prediction_cache(tmp_path / "fmow_cache", config, overwrite=True)

    validate_prediction_cache_dir(saved)
    examples, arrays, metadata = load_prediction_cache(saved)
    assert metadata["dataset"] == "wilds_fmow"
    assert metadata["model_backend"] == "torchvision_resnet18_features"
    assert metadata["source"] == "wilds"
    assert metadata["num_classes"] == 5
    assert len(set(examples["context_id"])) >= 3
    assert {"region", "year"}.issubset(examples.columns)
    assert arrays["feature_embedding"].shape[0] == len(examples)
    assert arrays["probabilities"].shape == (len(examples), 5)
