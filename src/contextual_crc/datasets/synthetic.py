from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from contextual_crc.cache.io import save_prediction_cache


@dataclass(frozen=True)
class SyntheticContext:
    context_id: str
    context_name: str
    shift_type: str
    severity: float
    class_prior: tuple[float, ...]
    mean_shift: tuple[float, ...]
    rotation_degrees: float
    noise_scale: float
    time_index: int | None = None


@dataclass(frozen=True)
class SyntheticConfig:
    dataset_name: str = "synthetic_contextual_shift"
    num_classes: int = 3
    input_dim: int = 2
    n_train: int = 3000
    n_per_context: int = 600
    random_seed: int = 123
    source_noise_scale: float = 0.8
    shift_strength: float = 1.0
    noise_growth: float = 1.0
    class_prior_strength: float = 1.0


def make_default_contexts(
    num_classes: int = 3,
    *,
    shift_strength: float = 1.0,
    noise_growth: float = 1.0,
    class_prior_strength: float = 1.0,
) -> list[SyntheticContext]:
    """Create deterministic source, covariate, label, and mixture shift contexts."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if shift_strength <= 0.0:
        raise ValueError("shift_strength must be positive")
    if noise_growth <= 0.0:
        raise ValueError("noise_growth must be positive")
    if class_prior_strength <= 0.0:
        raise ValueError("class_prior_strength must be positive")

    uniform = tuple([1.0 / num_classes] * num_classes)

    contexts = [
        SyntheticContext(
            context_id="source",
            context_name="Source reference",
            shift_type="source",
            severity=0.0,
            class_prior=uniform,
            mean_shift=(0.0, 0.0),
            rotation_degrees=0.0,
            noise_scale=0.8,
            time_index=0,
        )
    ]

    for index, base_severity in enumerate((0.35, 0.7, 1.05), start=1):
        severity = base_severity * shift_strength
        contexts.append(
            SyntheticContext(
                context_id=f"covariate_s{index}",
                context_name=f"Covariate shift severity {index}",
                shift_type="covariate",
                severity=severity,
                class_prior=uniform,
                mean_shift=(0.55 * severity, -0.35 * severity),
                rotation_degrees=12.0 * index,
                noise_scale=0.8 + 0.12 * index * noise_growth,
                time_index=index,
            )
        )

    for index, base_severity in enumerate((0.35, 0.7, 1.05), start=1):
        severity = base_severity * shift_strength
        contexts.append(
            SyntheticContext(
                context_id=f"label_s{index}",
                context_name=f"Label shift severity {index}",
                shift_type="label",
                severity=severity,
                class_prior=_tilted_class_prior(num_classes, severity * class_prior_strength),
                mean_shift=(0.0, 0.0),
                rotation_degrees=0.0,
                noise_scale=0.8 + 0.05 * index * max(noise_growth - 1.0, 0.0),
                time_index=3 + index,
            )
        )

    for index, base_severity in enumerate((0.35, 0.7, 1.05), start=1):
        severity = base_severity * shift_strength
        contexts.append(
            SyntheticContext(
                context_id=f"mixture_s{index}",
                context_name=f"Mixture shift severity {index}",
                shift_type="mixture",
                severity=severity,
                class_prior=_tilted_class_prior(num_classes, 0.7 * severity * class_prior_strength),
                mean_shift=(0.35 * severity, 0.45 * severity),
                rotation_degrees=-10.0 * index,
                noise_scale=0.9 + 0.1 * index * noise_growth,
                time_index=6 + index,
            )
        )

    return contexts


def generate_context_data(
    context: SyntheticContext,
    n: int,
    num_classes: int,
    input_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian class-conditional data for a single context."""
    if n <= 0:
        raise ValueError("n must be positive")
    if input_dim < 2:
        raise ValueError("input_dim must be at least 2")
    if len(context.class_prior) != num_classes:
        raise ValueError("context.class_prior length must match num_classes")

    class_prior = np.asarray(context.class_prior, dtype=float)
    class_prior = class_prior / class_prior.sum()
    y = rng.choice(num_classes, size=n, p=class_prior)

    means = _base_class_means(num_classes, input_dim)
    X = means[y] + rng.normal(scale=context.noise_scale, size=(n, input_dim))

    X = _apply_rotation(X, context.rotation_degrees)
    X = X + _coerce_vector(context.mean_shift, input_dim)
    return X.astype(float), y.astype(int)


def train_source_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 123,
) -> Any:
    """Train a deterministic source-context classifier."""
    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=500, random_state=random_seed),
            ),
        ]
    )
    classifier.fit(X_train, y_train)
    return classifier


def compute_prediction_frame_and_arrays(
    *,
    dataset_name: str,
    split: str,
    context: SyntheticContext,
    X: np.ndarray,
    y: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray | None = None,
    start_example_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build schema-compatible examples and arrays for one synthetic context."""
    probabilities = np.asarray(probabilities, dtype=float)
    if logits is None:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    logits = np.asarray(logits, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if logits.shape != probabilities.shape:
        raise ValueError("logits shape must match probabilities shape")
    if len(X) != len(y) or len(X) != probabilities.shape[0]:
        raise ValueError("X, y, and probabilities must have matching first dimensions")

    pred = probabilities.argmax(axis=1).astype(int)
    correct = pred == y.astype(int)
    confidence = probabilities.max(axis=1)
    entropy = -np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    energy = -_logsumexp(logits, axis=1)

    indices = np.arange(start_example_index, start_example_index + len(y))
    examples = pd.DataFrame(
        {
            "example_id": [
                f"synthetic/{split}/{context.context_id}/{index}" for index in indices
            ],
            "dataset": dataset_name,
            "split": split,
            "context_id": context.context_id,
            "context_name": context.context_name,
            "label": y.astype(int),
            "pred": pred,
            "correct": correct.astype(bool),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "margin": margin.astype(float),
            "energy": energy.astype(float),
            "severity": float(context.severity),
            "domain_id": context.context_id,
            "group_id": context.shift_type,
            "time_index": context.time_index if context.time_index is not None else -1,
            "source": context.shift_type == "source",
            "shift_type": context.shift_type,
        }
    )

    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": np.asarray(X, dtype=float),
    }
    return examples, arrays


def build_synthetic_prediction_cache(
    cache_dir: str | Path,
    config: SyntheticConfig,
    contexts: list[SyntheticContext] | None = None,
    overwrite: bool = False,
) -> Path:
    """Train on source data, evaluate all contexts, and save a prediction cache."""
    contexts = contexts or make_default_contexts(
        config.num_classes,
        shift_strength=config.shift_strength,
        noise_growth=config.noise_growth,
        class_prior_strength=config.class_prior_strength,
    )
    source_context = _source_context_for_training(contexts, config)
    rng = np.random.default_rng(config.random_seed)

    X_train, y_train = generate_context_data(
        source_context,
        n=config.n_train,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        rng=rng,
    )
    classifier = train_source_classifier(X_train, y_train, config.random_seed)

    example_frames: list[pd.DataFrame] = []
    array_chunks: dict[str, list[np.ndarray]] = {
        "logits": [],
        "probabilities": [],
        "feature_embedding": [],
    }
    start_index = 0

    for context in contexts:
        X_eval, y_eval = generate_context_data(
            context,
            n=config.n_per_context,
            num_classes=config.num_classes,
            input_dim=config.input_dim,
            rng=rng,
        )
        probabilities = classifier.predict_proba(X_eval)
        logits = _classifier_logits(classifier, X_eval, probabilities)
        frame, arrays = compute_prediction_frame_and_arrays(
            dataset_name=config.dataset_name,
            split="eval",
            context=context,
            X=X_eval,
            y=y_eval,
            probabilities=probabilities,
            logits=logits,
            start_example_index=start_index,
        )
        example_frames.append(frame)
        for key, value in arrays.items():
            array_chunks[key].append(value)
        start_index += len(frame)

    examples = pd.concat(example_frames, ignore_index=True)
    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in array_chunks.items()}
    metadata = {
        "dataset": config.dataset_name,
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": config.num_classes,
        "context_fields": ["context_id", "context_name", "shift_type", "severity"],
        "synthetic_config": asdict(config),
        "contexts": [asdict(context) for context in contexts],
    }

    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _tilted_class_prior(num_classes: int, severity: float) -> tuple[float, ...]:
    base = np.ones(num_classes, dtype=float)
    base[0] += 1.8 * severity
    if num_classes > 1:
        base[-1] = max(0.2, base[-1] - 0.45 * severity)
    probabilities = base / base.sum()
    return tuple(float(value) for value in probabilities)


def _base_class_means(num_classes: int, input_dim: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_classes, endpoint=False)
    means = np.zeros((num_classes, input_dim), dtype=float)
    means[:, 0] = 2.5 * np.cos(angles)
    means[:, 1] = 2.5 * np.sin(angles)
    for dim in range(2, input_dim):
        means[:, dim] = np.linspace(-1.0, 1.0, num_classes) * (0.6 / dim)
    return means


def _coerce_vector(values: tuple[float, ...], input_dim: int) -> np.ndarray:
    vector = np.zeros(input_dim, dtype=float)
    usable = min(len(values), input_dim)
    vector[:usable] = np.asarray(values[:usable], dtype=float)
    return vector


def _apply_rotation(X: np.ndarray, rotation_degrees: float) -> np.ndarray:
    if X.shape[1] < 2 or rotation_degrees == 0:
        return X
    radians = np.deg2rad(rotation_degrees)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    rotated = X.copy()
    rotated[:, :2] = X[:, :2] @ rotation.T
    return rotated


def _classifier_logits(
    classifier: Any,
    X: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        if decision.ndim == 1:
            return np.column_stack([-0.5 * decision, 0.5 * decision])
        return np.asarray(decision, dtype=float)
    return np.log(np.clip(probabilities, 1e-12, 1.0))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
    return np.squeeze(stable, axis=axis)


def _source_context_for_training(
    contexts: list[SyntheticContext], config: SyntheticConfig
) -> SyntheticContext:
    for context in contexts:
        if context.context_id == "source":
            return SyntheticContext(
                context_id=context.context_id,
                context_name=context.context_name,
                shift_type=context.shift_type,
                severity=context.severity,
                class_prior=context.class_prior,
                mean_shift=context.mean_shift,
                rotation_degrees=context.rotation_degrees,
                noise_scale=config.source_noise_scale,
                time_index=context.time_index,
            )
    uniform = tuple([1.0 / config.num_classes] * config.num_classes)
    return SyntheticContext(
        context_id="source",
        context_name="Source reference",
        shift_type="source",
        severity=0.0,
        class_prior=uniform,
        mean_shift=(0.0, 0.0),
        rotation_degrees=0.0,
        noise_scale=config.source_noise_scale,
        time_index=0,
    )
