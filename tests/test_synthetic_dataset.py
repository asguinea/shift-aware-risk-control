from __future__ import annotations

import numpy as np

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.cache.schema import REQUIRED_EXAMPLE_COLUMNS
from contextual_crc.cache.validate import validate_prediction_cache, validate_prediction_cache_dir
from contextual_crc.datasets.synthetic import (
    SyntheticConfig,
    build_synthetic_prediction_cache,
    compute_prediction_frame_and_arrays,
    generate_context_data,
    make_default_contexts,
    train_source_classifier,
)


def test_make_default_contexts() -> None:
    contexts = make_default_contexts(num_classes=3)

    assert len(contexts) == 10
    assert contexts[0].context_id == "source"
    assert {context.shift_type for context in contexts} == {
        "source",
        "covariate",
        "label",
        "mixture",
    }
    context_ids = [context.context_id for context in contexts]
    assert len(context_ids) == len(set(context_ids))
    for context in contexts:
        assert np.isclose(sum(context.class_prior), 1.0)
        assert context.severity >= 0


def test_stronger_shift_contexts_are_more_severe() -> None:
    default = {context.context_id: context for context in make_default_contexts(num_classes=3)}
    stronger = {
        context.context_id: context
        for context in make_default_contexts(
            num_classes=3,
            shift_strength=1.8,
            noise_growth=1.5,
            class_prior_strength=1.4,
        )
    }

    assert stronger["covariate_s3"].severity > default["covariate_s3"].severity
    assert stronger["covariate_s3"].noise_scale > default["covariate_s3"].noise_scale
    assert max(stronger["label_s3"].class_prior) > max(default["label_s3"].class_prior)
    assert np.linalg.norm(stronger["mixture_s3"].mean_shift) > np.linalg.norm(default["mixture_s3"].mean_shift)


def test_generate_context_data_shapes() -> None:
    rng = np.random.default_rng(123)
    context = make_default_contexts(num_classes=3)[1]

    X, y = generate_context_data(context, n=25, num_classes=3, input_dim=2, rng=rng)

    assert X.shape == (25, 2)
    assert y.shape == (25,)
    assert y.min() >= 0
    assert y.max() < 3


def test_train_source_classifier_predicts_probabilities() -> None:
    rng = np.random.default_rng(123)
    source_context = make_default_contexts(num_classes=3)[0]
    X_train, y_train = generate_context_data(
        source_context,
        n=180,
        num_classes=3,
        input_dim=2,
        rng=rng,
    )
    classifier = train_source_classifier(X_train, y_train, random_seed=123)
    X_eval, _ = generate_context_data(
        source_context,
        n=20,
        num_classes=3,
        input_dim=2,
        rng=rng,
    )

    probabilities = classifier.predict_proba(X_eval)

    assert probabilities.shape == (20, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_compute_prediction_frame_and_arrays_schema() -> None:
    rng = np.random.default_rng(123)
    context = make_default_contexts(num_classes=3)[0]
    X, y = generate_context_data(context, n=12, num_classes=3, input_dim=2, rng=rng)
    probabilities = np.full((12, 3), 1.0 / 3.0)
    probabilities[np.arange(12), y] = 0.6
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    examples, arrays = compute_prediction_frame_and_arrays(
        dataset_name="synthetic_contextual_shift",
        split="eval",
        context=context,
        X=X,
        y=y,
        probabilities=probabilities,
    )

    for column in REQUIRED_EXAMPLE_COLUMNS:
        assert column in examples.columns
    for column in ("severity", "domain_id", "group_id", "time_index", "source", "shift_type"):
        assert column in examples.columns
    assert set(arrays) == {"logits", "probabilities", "feature_embedding"}
    assert arrays["logits"].shape == (12, 3)
    assert arrays["feature_embedding"].shape == (12, 2)

    metadata = {
        "dataset": "synthetic_contextual_shift",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 3,
        "created_at": "2026-04-24T00:00:00+00:00",
        "schema_version": "1.0",
        "array_keys": ["logits", "probabilities", "feature_embedding"],
        "required_columns": list(REQUIRED_EXAMPLE_COLUMNS),
        "optional_columns": ["severity", "domain_id", "group_id", "time_index", "source"],
        "context_fields": ["context_id", "context_name", "shift_type", "severity"],
    }
    validate_prediction_cache(examples, arrays, metadata)


def test_build_synthetic_prediction_cache(tmp_path) -> None:
    config = SyntheticConfig(n_train=240, n_per_context=30, random_seed=123)
    cache_dir = tmp_path / "synthetic_cache"

    saved_path = build_synthetic_prediction_cache(cache_dir, config, overwrite=False)

    assert saved_path == cache_dir
    validate_prediction_cache_dir(cache_dir)
    examples, arrays, metadata = load_prediction_cache(cache_dir)

    context_ids = set(examples["context_id"].unique())
    assert len(context_ids) > 1
    assert "source" in context_ids
    assert arrays["logits"].shape[0] == len(examples)
    assert arrays["logits"].shape[1] == config.num_classes
    assert metadata["dataset"] == "synthetic_contextual_shift"
    assert metadata["num_examples"] == len(examples)
