from __future__ import annotations

import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.datasets.vision_features import (
    build_resnet18_feature_extractor,
    extract_resnet18_features,
    predict_proba_with_features,
    sanitize_probabilities,
    train_feature_classifier,
)


_ALLOWED_DATASETS = ("civilcomments", "camelyon17", "fmow")
_ALLOWED_SOURCES = ("csv", "wilds")
_ALLOWED_MODEL_BACKENDS = ("sklearn_tfidf", "torchvision_resnet18_features")
_ALLOWED_TORCH_DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class WILDSConfig:
    dataset: str = "civilcomments"
    source: str = "csv"
    data_root: str | Path = "data/wilds"
    download: bool = False
    model_backend: str = "sklearn_tfidf"
    train_subset_size: int | None = 20000
    eval_subset_size: int | None = 5000
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    random_seed: int = 123
    text_column: str = "text"
    label_column: str = "label"
    context_column: str = "group_id"
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    torch_batch_size: int = 64
    torch_device: str = "auto"
    torch_pretrained: bool = True
    eval_splits: tuple[str, ...] = field(default_factory=lambda: ("val", "test"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset).lower())
        object.__setattr__(self, "source", str(self.source).lower())
        object.__setattr__(self, "model_backend", str(self.model_backend).lower())
        object.__setattr__(self, "ngram_range", tuple(int(value) for value in self.ngram_range))
        object.__setattr__(self, "torch_device", str(self.torch_device).lower())
        object.__setattr__(self, "eval_splits", tuple(str(value) for value in self.eval_splits))
        object.__setattr__(self, "text_column", str(self.text_column))
        object.__setattr__(self, "label_column", str(self.label_column))
        object.__setattr__(self, "context_column", str(self.context_column))
        if self.dataset == "fmow" and self.context_column == "group_id":
            object.__setattr__(self, "context_column", "domain_id")

        if self.dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of {_ALLOWED_DATASETS}, got {self.dataset!r}")
        if self.source not in _ALLOWED_SOURCES:
            raise ValueError(f"source must be one of {_ALLOWED_SOURCES}, got {self.source!r}")
        if self.model_backend not in _ALLOWED_MODEL_BACKENDS:
            raise ValueError(
                f"model_backend must be one of {_ALLOWED_MODEL_BACKENDS}, got {self.model_backend!r}"
            )
        if self.dataset == "civilcomments" and self.model_backend != "sklearn_tfidf":
            raise ValueError("CivilComments requires model_backend='sklearn_tfidf'")
        if self.dataset in {"camelyon17", "fmow"}:
            dataset_name = "Camelyon17" if self.dataset == "camelyon17" else "FMoW"
            if self.source != "wilds":
                raise ValueError(f"{dataset_name} requires source='wilds'")
            if self.model_backend != "torchvision_resnet18_features":
                raise ValueError(f"{dataset_name} requires model_backend='torchvision_resnet18_features'")
        if len(self.ngram_range) != 2:
            raise ValueError("ngram_range must contain exactly two integers")
        if self.ngram_range[0] <= 0 or self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n")
        if self.train_subset_size is not None and self.train_subset_size <= 0:
            raise ValueError("train_subset_size must be positive or None")
        if self.eval_subset_size is not None and self.eval_subset_size <= 0:
            raise ValueError("eval_subset_size must be positive or None")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive or None")
        if self.max_eval_examples is not None and self.max_eval_examples <= 0:
            raise ValueError("max_eval_examples must be positive or None")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.torch_batch_size <= 0:
            raise ValueError("torch_batch_size must be positive")
        if self.torch_device not in _ALLOWED_TORCH_DEVICES:
            raise ValueError(f"torch_device must be one of {_ALLOWED_TORCH_DEVICES}, got {self.torch_device!r}")
        if not self.eval_splits:
            raise ValueError("eval_splits must be non-empty")
        for field_name in ("text_column", "label_column", "context_column"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")


def wilds_dataset_dir(config: WILDSConfig) -> Path:
    return Path(config.data_root) / config.dataset


def validate_wilds_csv_files(config: WILDSConfig) -> None:
    if config.source != "csv":
        return

    base = wilds_dataset_dir(config)
    missing = [str(base / filename) for filename in ("train.csv", "eval.csv") if not (base / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing CivilComments CSV files. Expected train.csv and eval.csv under "
            f"{base}. Missing: {missing}"
        )


def load_civilcomments_csv_split(
    config: WILDSConfig,
    split: str,
) -> pd.DataFrame:
    split = str(split).lower()
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")

    path = wilds_dataset_dir(config) / f"{split}.csv"
    frame = pd.read_csv(path)
    frame = _normalize_civilcomments_frame(
        frame,
        config=config,
        default_split=split,
        source="wilds-csv",
    )
    subset_size = config.train_subset_size if split == "train" else config.eval_subset_size
    return _apply_subset(frame, subset_size, config.random_seed)


def load_civilcomments_wilds(
    config: WILDSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise ImportError(
            "Real WILDS loading requires the optional 'wilds' package. Install it separately "
            "or use source='csv' with train.csv/eval.csv files."
        ) from exc

    dataset = get_dataset(
        dataset="civilcomments",
        root_dir=str(Path(config.data_root)),
        download=bool(config.download),
    )
    train_df = _civilcomments_subset_to_frame(dataset, "train", config)
    eval_split = _first_available_wilds_split(dataset, ("test", "val", "id_val"))
    eval_df = _civilcomments_subset_to_frame(dataset, eval_split, config)

    train_df = _apply_subset(train_df, config.train_subset_size, config.random_seed)
    eval_df = _apply_subset(eval_df, config.eval_subset_size, config.random_seed)
    return train_df, eval_df


def load_camelyon17_wilds_splits(config: WILDSConfig) -> tuple[Any, dict[str, Any]]:
    """Load Camelyon17-WILDS train and evaluation subsets with lazy WILDS import."""
    return _load_image_wilds_splits(config, dataset_name="camelyon17", display_name="Camelyon17")


def load_fmow_wilds_splits(config: WILDSConfig) -> tuple[Any, dict[str, Any]]:
    """Load FMoW-WILDS train and evaluation subsets with lazy WILDS import."""
    return _load_image_wilds_splits(config, dataset_name="fmow", display_name="FMoW")


def _load_image_wilds_splits(
    config: WILDSConfig,
    *,
    dataset_name: str,
    display_name: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise ImportError(
            f"{display_name} loading requires the optional 'wilds' package. Install it with "
            "`python -m pip install wilds` and rerun with source='wilds'."
        ) from exc

    datetime_context = (
        _patched_pandas_to_datetime_for_fmow()
        if dataset_name == "fmow"
        else nullcontext()
    )
    try:
        with datetime_context:
            try:
                dataset = get_dataset(
                    dataset=dataset_name,
                    root_dir=str(Path(config.data_root)),
                    download=bool(config.download),
                )
            except TypeError:
                dataset = get_dataset(
                    dataset=dataset_name,
                    download=bool(config.download),
                )
    except Exception as exc:
        raise RuntimeError(f"Could not load WILDS {display_name} dataset: {exc}") from exc

    train_subset = _get_wilds_subset(dataset, "train")
    eval_subsets: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for split in config.eval_splits:
        split_name = str(split)
        try:
            eval_subsets[split_name] = _get_wilds_subset(dataset, split_name)
        except RuntimeError as exc:
            errors[split_name] = str(exc)
    if errors:
        available = _available_wilds_split_names(dataset)
        raise RuntimeError(
            f"Could not load requested WILDS {display_name} eval split(s): {sorted(errors)}. "
            f"Attempted splits: {list(config.eval_splits)}. Available splits: {available}. "
            f"Errors: {errors}"
        )
    if not eval_subsets:
        raise RuntimeError(f"No WILDS {display_name} evaluation splits were loaded")
    return train_subset, eval_subsets


def load_wilds_train_eval(config: WILDSConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.dataset != "civilcomments":
        raise ValueError("load_wilds_train_eval is only available for CivilComments")
    if config.source == "csv":
        validate_wilds_csv_files(config)
        return (
            load_civilcomments_csv_split(config, "train"),
            load_civilcomments_csv_split(config, "eval"),
        )
    if config.source == "wilds":
        return load_civilcomments_wilds(config)
    raise ValueError(f"source must be one of {_ALLOWED_SOURCES}, got {config.source!r}")


@contextmanager
def _patched_pandas_to_datetime_for_fmow() -> Any:
    """Temporarily tolerate mixed ISO timestamps in WILDS FMoW metadata."""
    original_to_datetime = pd.to_datetime

    def safe_to_datetime(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_to_datetime(*args, **kwargs)
        except ValueError as original_error:
            if kwargs.get("format") is not None:
                raise
            retry_errors: list[str] = []
            for timestamp_format in ("mixed", "ISO8601"):
                retry_kwargs = dict(kwargs)
                retry_kwargs["format"] = timestamp_format
                try:
                    return original_to_datetime(*args, **retry_kwargs)
                except (TypeError, ValueError) as retry_error:
                    retry_errors.append(f"{timestamp_format}: {retry_error}")
            raise RuntimeError(
                "Could not parse FMoW-WILDS metadata timestamps with pandas.to_datetime. "
                f"Original error: {original_error}. Retry errors: {retry_errors}"
            ) from original_error

    pd.to_datetime = safe_to_datetime
    try:
        yield
    finally:
        pd.to_datetime = original_to_datetime


def train_sklearn_text_classifier(
    train_df: pd.DataFrame,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    random_seed: int,
) -> Any:
    if "text" not in train_df or "label" not in train_df:
        raise ValueError("train_df must contain text and label columns")

    labels = pd.to_numeric(train_df["label"], errors="raise").astype(int).to_numpy()
    unique_labels = sorted(np.unique(labels).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"CivilComments training labels must contain both binary classes 0 and 1, got {unique_labels}")

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=int(max_features),
                    ngram_range=tuple(ngram_range),
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=random_seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(train_df["text"].astype(str).tolist(), labels)
    return model


def predict_text_proba_and_features(
    model: Any,
    texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_texts = ["" if text is None else str(text) for text in texts]
    if hasattr(model, "predict_proba"):
        raw_probabilities = np.asarray(model.predict_proba(normalized_texts), dtype=float)
    else:
        raw_probabilities = None

    classes = np.asarray(getattr(model, "classes_", np.arange(raw_probabilities.shape[1] if raw_probabilities is not None else 2)), dtype=int)
    probabilities = np.zeros((len(normalized_texts), 2), dtype=float)
    if raw_probabilities is not None:
        for source_col, class_id in enumerate(classes):
            if int(class_id) in (0, 1):
                probabilities[:, int(class_id)] = raw_probabilities[:, source_col]
    elif hasattr(model, "decision_function"):
        logits = _ensure_binary_logits(np.asarray(model.decision_function(normalized_texts), dtype=float))
        probabilities = _softmax(logits)
    else:
        raise ValueError("model must implement predict_proba or decision_function")

    row_sums = probabilities.sum(axis=1, keepdims=True)
    missing_mask = row_sums.squeeze(axis=1) <= 0.0
    if missing_mask.any():
        probabilities[missing_mask, :] = 0.5
        row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / row_sums

    if hasattr(model, "decision_function"):
        logits = _ensure_binary_logits(np.asarray(model.decision_function(normalized_texts), dtype=float))
    else:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    features = extract_text_features(normalized_texts, probabilities=probabilities)
    return probabilities.astype(float), logits.astype(float), features


def extract_text_features(
    texts: Sequence[str],
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        value = "" if text is None else str(text)
        characters = len(value)
        words = re.findall(r"\S+", value)
        word_count = len(words)
        alphabetic_count = sum(char.isalpha() for char in value)
        uppercase_count = sum(char.isupper() for char in value)
        digit_count = sum(char.isdigit() for char in value)
        punctuation_count = sum(char in "!?.,;:" for char in value)
        row = [
            float(characters),
            float(word_count),
            float(characters / max(word_count, 1)),
            float(value.count("!")),
            float(value.count("?")),
            float(uppercase_count / max(alphabetic_count, 1)),
            float(digit_count / max(characters, 1)),
            float(punctuation_count / max(characters, 1)),
        ]
        rows.append(row)

    features = np.asarray(rows, dtype=np.float32)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(texts) or probabilities.shape[1] < 2:
            raise ValueError("probabilities must have shape [n, 2+] when provided")
        features = np.column_stack([features, probabilities[:, 1].astype(np.float32)]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_resnet_features_from_wilds_subset(
    subset: Any,
    *,
    model: Any,
    device: str,
    batch_size: int,
    max_examples: int | None,
    random_seed: int,
    split: str | None = None,
    split_name: str | None = None,
    metadata_parser: Callable[..., dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract ResNet features, labels, and metadata from a WILDS image subset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_split = str(split_name if split_name is not None else split if split is not None else "eval")
    indices = _selected_indices(len(subset), max_examples, random_seed)
    metadata_fields = _metadata_fields_for_subset(subset)

    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    image_batch: list[np.ndarray] = []
    pending_labels: list[int] = []
    pending_metadata: list[dict[str, Any]] = []

    for original_index in indices:
        image, label, metadata = _unpack_wilds_sample(subset[int(original_index)])
        row = _metadata_to_row(metadata, metadata_fields)
        row["original_index"] = int(original_index)
        row["split"] = resolved_split
        if metadata_parser is None:
            row = _normalize_camelyon_metadata_row(row, split=resolved_split)
        else:
            row = metadata_parser(row, split=resolved_split)
        image_batch.append(_image_to_hwc_array(image))
        pending_labels.append(int(_scalar_value(label)))
        pending_metadata.append(row)
        if len(image_batch) >= batch_size:
            batch_features = extract_resnet18_features(
                np.stack(image_batch, axis=0),
                model=model,
                device=device,
                batch_size=batch_size,
            )
            features.append(batch_features)
            labels.extend(pending_labels)
            metadata_rows.extend(pending_metadata)
            image_batch = []
            pending_labels = []
            pending_metadata = []

    if image_batch:
        batch_features = extract_resnet18_features(
            np.stack(image_batch, axis=0),
            model=model,
            device=device,
            batch_size=batch_size,
        )
        features.append(batch_features)
        labels.extend(pending_labels)
        metadata_rows.extend(pending_metadata)

    feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False) if features else np.empty((0, 0), dtype=np.float32)
    label_array = np.asarray(labels, dtype=int)
    metadata_df = pd.DataFrame(metadata_rows)
    if len(metadata_df) == 0:
        metadata_df = pd.DataFrame(columns=["original_index", "split", "domain_id", "group_id", "source"])
    return feature_array, label_array, metadata_df.reset_index(drop=True)


def compute_camelyon17_prediction_frame_and_arrays(
    *,
    eval_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    feature_embedding: np.ndarray,
    dataset: str = "wilds_camelyon17",
    split: str = "eval",
    start_example_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build cache-compatible examples/arrays for Camelyon17-WILDS predictions."""
    labels = np.asarray(labels, dtype=int)
    probabilities = sanitize_probabilities(probabilities, num_classes=2)
    logits = np.asarray(logits, dtype=float)
    feature_embedding = np.asarray(feature_embedding, dtype=float)
    if logits.shape != probabilities.shape:
        raise ValueError("logits shape must match probabilities shape")
    if len(eval_df) != len(labels) or probabilities.shape[0] != len(labels):
        raise ValueError("eval_df, labels, probabilities, and logits must have matching first dimensions")
    if feature_embedding.shape[0] != len(labels):
        raise ValueError("feature_embedding rows must match labels")

    pred = probabilities.argmax(axis=1).astype(int)
    correct = pred == labels
    confidence = probabilities.max(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    energy = -_logsumexp(logits, axis=1)
    indices = np.arange(start_example_index, start_example_index + len(labels))

    normalized = eval_df.copy().reset_index(drop=True)
    if "domain_id" not in normalized.columns:
        normalized["domain_id"] = normalized["split"] if "split" in normalized.columns else split
    if "group_id" not in normalized.columns:
        normalized["group_id"] = normalized["domain_id"]
    if "source" not in normalized.columns:
        normalized["source"] = "wilds-camelyon17"
    if "time_index" not in normalized.columns:
        normalized["time_index"] = np.arange(len(labels), dtype=int)
    time_values = pd.to_numeric(normalized["time_index"], errors="coerce").to_numpy(dtype=float)
    missing_time = ~np.isfinite(time_values)
    if missing_time.any():
        time_values[missing_time] = np.arange(len(labels), dtype=float)[missing_time]

    domain_values = [_sanitize_context_value(value) for value in normalized["domain_id"].tolist()]
    group_values = [_sanitize_context_value(value) for value in normalized["group_id"].tolist()]
    source_split_values = normalized["split"].astype(str).tolist() if "split" in normalized.columns else [split] * len(labels)
    cache_split_values = [str(split)] * len(labels)
    context_values = [f"domain_{domain}" if not str(domain).startswith("domain_") else str(domain) for domain in domain_values]
    context_names = [f"Domain {str(domain).replace('domain_', '')}" for domain in context_values]

    examples = pd.DataFrame(
        {
            "example_id": [
                f"{dataset}/{split_value}/{context_id}/{index}"
                for split_value, context_id, index in zip(source_split_values, context_values, indices, strict=True)
            ],
            "dataset": dataset,
            "split": cache_split_values,
            "context_id": context_values,
            "context_name": context_names,
            "label": labels.astype(int),
            "pred": pred,
            "correct": correct.astype(bool),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "margin": margin.astype(float),
            "energy": energy.astype(float),
            "domain_id": domain_values,
            "group_id": group_values,
            "wilds_split": source_split_values,
            "source": normalized["source"].astype(str).tolist(),
            "time_index": time_values.astype(int),
        }
    )
    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": feature_embedding.astype(float),
    }
    return examples, arrays


def compute_fmow_prediction_frame_and_arrays(
    *,
    eval_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    feature_embedding: np.ndarray,
    dataset: str = "wilds_fmow",
    split: str = "eval",
    start_example_index: int = 0,
    context_column: str = "domain_id",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build cache-compatible examples/arrays for FMoW-WILDS predictions."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [n, num_classes>=2] for FMoW")
    probabilities = sanitize_probabilities(probabilities, num_classes=int(probabilities.shape[1]))
    logits = np.asarray(logits, dtype=float)
    feature_embedding = np.asarray(feature_embedding, dtype=float)
    if logits.shape != probabilities.shape:
        raise ValueError("logits shape must match probabilities shape")
    if len(eval_df) != len(labels) or probabilities.shape[0] != len(labels):
        raise ValueError("eval_df, labels, probabilities, and logits must have matching first dimensions")
    if feature_embedding.shape[0] != len(labels):
        raise ValueError("feature_embedding rows must match labels")

    pred = probabilities.argmax(axis=1).astype(int)
    correct = pred == labels
    confidence = probabilities.max(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    energy = -_logsumexp(logits, axis=1)
    indices = np.arange(start_example_index, start_example_index + len(labels))

    normalized = eval_df.copy().reset_index(drop=True)
    if "split" not in normalized.columns:
        normalized["split"] = split
    if "source" not in normalized.columns:
        normalized["source"] = "wilds-fmow"
    if "time_index" not in normalized.columns:
        normalized["time_index"] = np.arange(start_example_index, start_example_index + len(labels), dtype=int)
    time_values = pd.to_numeric(normalized["time_index"], errors="coerce").to_numpy(dtype=float)
    missing_time = ~np.isfinite(time_values)
    if missing_time.any():
        time_values[missing_time] = np.arange(start_example_index, start_example_index + len(labels), dtype=float)[missing_time]

    context_source = _fmow_context_source(normalized, context_column=context_column)
    context_values = [
        _prefixed_context_value(prefix, value)
        for prefix, value in zip(context_source["prefix"], context_source["value"], strict=True)
    ]
    context_names = [
        _context_name(prefix, value)
        for prefix, value in zip(context_source["prefix"], context_source["value"], strict=True)
    ]
    domain_values = (
        normalized["domain_id"].map(_sanitize_context_value).tolist()
        if "domain_id" in normalized.columns
        else context_values
    )
    group_values = (
        normalized["group_id"].map(_sanitize_context_value).tolist()
        if "group_id" in normalized.columns
        else domain_values
    )
    source_split_values = normalized["split"].astype(str).tolist()

    payload: dict[str, Any] = {
        "example_id": [
            f"{dataset}/{split_value}/{context_id}/{index}"
            for split_value, context_id, index in zip(source_split_values, context_values, indices, strict=True)
        ],
        "dataset": dataset,
        "split": [str(split)] * len(labels),
        "context_id": context_values,
        "context_name": context_names,
        "label": labels.astype(int),
        "pred": pred,
        "correct": correct.astype(bool),
        "confidence": confidence.astype(float),
        "entropy": entropy.astype(float),
        "margin": margin.astype(float),
        "energy": energy.astype(float),
        "domain_id": domain_values,
        "group_id": group_values,
        "source": normalized["source"].astype(str).tolist(),
        "time_index": time_values.astype(int),
    }
    for optional_column in ("region", "year"):
        if optional_column in normalized.columns:
            payload[optional_column] = normalized[optional_column].map(_sanitize_context_value).tolist()
    if "split" in normalized.columns:
        payload["wilds_split"] = source_split_values

    examples = pd.DataFrame(payload)
    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": feature_embedding.astype(float),
    }
    return examples, arrays


def compute_wilds_prediction_frame_and_arrays(
    *,
    dataset: str,
    split: str,
    eval_df: pd.DataFrame,
    probabilities: np.ndarray,
    logits: np.ndarray,
    feature_embedding: np.ndarray,
    context_column: str = "group_id",
    start_example_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if "label" not in eval_df:
        raise ValueError("eval_df must contain a label column")
    if context_column not in eval_df:
        raise ValueError(f"eval_df is missing context column {context_column!r}")

    labels = pd.to_numeric(eval_df["label"], errors="raise").astype(int).to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    logits = np.asarray(logits, dtype=float)
    feature_embedding = np.asarray(feature_embedding, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape [n, 2] for CivilComments")
    if logits.shape != probabilities.shape:
        raise ValueError("logits shape must match probabilities shape")
    if len(labels) != probabilities.shape[0]:
        raise ValueError("eval_df, labels, probabilities, and logits must have matching first dimensions")
    if feature_embedding.shape[0] != len(labels):
        raise ValueError("feature_embedding rows must match labels")

    pred = probabilities.argmax(axis=1).astype(int)
    correct = pred == labels
    confidence = probabilities.max(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    energy = -_logsumexp(logits, axis=1)
    indices = np.arange(start_example_index, start_example_index + len(labels))
    dataset_name = _cache_dataset_name(dataset)

    context_values = eval_df[context_column].map(_sanitize_context_value).tolist()
    domain_values = eval_df["domain_id"].map(_sanitize_context_value).tolist() if "domain_id" in eval_df else context_values
    group_values = eval_df["group_id"].map(_sanitize_context_value).tolist() if "group_id" in eval_df else context_values
    source_values = eval_df["source"].astype(str).tolist() if "source" in eval_df else ["wilds"] * len(labels)

    examples = pd.DataFrame(
        {
            "example_id": [
                f"{dataset_name}/{split}/{context_id}/{index}"
                for context_id, index in zip(context_values, indices, strict=True)
            ],
            "dataset": dataset_name,
            "split": split,
            "context_id": context_values,
            "context_name": context_values,
            "label": labels.astype(int),
            "pred": pred,
            "correct": correct.astype(bool),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "margin": margin.astype(float),
            "energy": energy.astype(float),
            "domain_id": domain_values,
            "group_id": group_values,
            "source": source_values,
            "time_index": np.arange(len(labels), dtype=int),
        }
    )
    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": feature_embedding.astype(float),
    }
    return examples, arrays


def build_wilds_prediction_cache(
    cache_dir: str | Path,
    config: WILDSConfig,
    *,
    overwrite: bool = False,
) -> Path:
    if config.dataset == "camelyon17":
        return _build_camelyon17_prediction_cache(cache_dir, config, overwrite=overwrite)
    if config.dataset == "fmow":
        return _build_fmow_prediction_cache(cache_dir, config, overwrite=overwrite)

    train_df, eval_df = load_wilds_train_eval(config)
    model = train_sklearn_text_classifier(
        train_df,
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        random_seed=config.random_seed,
    )
    probabilities, logits, features = predict_text_proba_and_features(
        model,
        eval_df["text"].astype(str).tolist(),
    )
    examples, arrays = compute_wilds_prediction_frame_and_arrays(
        dataset=config.dataset,
        split="eval",
        eval_df=eval_df,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=features,
        context_column=config.context_column,
    )
    metadata = {
        "dataset": _cache_dataset_name(config.dataset),
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "context_fields": ["context_id", "domain_id", "group_id"],
        "wilds_config": _config_to_jsonable(config),
        "model_backend": config.model_backend,
        "source": config.source,
    }
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _build_camelyon17_prediction_cache(
    cache_dir: str | Path,
    config: WILDSConfig,
    *,
    overwrite: bool = False,
) -> Path:
    train_subset, eval_subsets = load_camelyon17_wilds_splits(config)
    feature_extractor, resolved_device = build_resnet18_feature_extractor(
        pretrained=config.torch_pretrained,
        device=config.torch_device,
    )
    train_features, train_labels, _ = extract_resnet_features_from_wilds_subset(
        train_subset,
        model=feature_extractor,
        device=resolved_device,
        batch_size=config.torch_batch_size,
        max_examples=_max_train_examples(config),
        random_seed=config.random_seed,
        split="train",
    )
    classifier = train_feature_classifier(
        train_features,
        train_labels,
        random_seed=config.random_seed,
        num_classes=2,
    )

    frames: list[pd.DataFrame] = []
    array_chunks: dict[str, list[np.ndarray]] = {"logits": [], "probabilities": [], "feature_embedding": []}
    start_index = 0
    per_split_limit = _per_split_eval_limit(config, len(eval_subsets))
    for split, subset in eval_subsets.items():
        features, labels, eval_df = extract_resnet_features_from_wilds_subset(
            subset,
            model=feature_extractor,
            device=resolved_device,
            batch_size=config.torch_batch_size,
            max_examples=per_split_limit,
            random_seed=config.random_seed,
            split=split,
        )
        probabilities, logits = predict_proba_with_features(classifier, features, 2)
        frame, arrays = compute_camelyon17_prediction_frame_and_arrays(
            eval_df=eval_df,
            labels=labels,
            probabilities=probabilities,
            logits=logits,
            feature_embedding=features,
            split="eval",
            start_example_index=start_index,
        )
        frames.append(frame)
        for key, value in arrays.items():
            array_chunks[key].append(value)
        start_index += len(frame)

    if not frames:
        raise ValueError("No Camelyon17 evaluation examples were extracted")
    examples = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate(values, axis=0) for key, values in array_chunks.items()}
    metadata = {
        "dataset": "wilds_camelyon17",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": 2,
        "context_fields": ["context_id", "domain_id", "group_id"],
        "wilds_config": _config_to_jsonable(config),
        "model_backend": config.model_backend,
        "source": config.source,
        "eval_splits": list(config.eval_splits),
        "max_train_examples": _max_train_examples(config),
        "max_eval_examples": _max_eval_examples(config),
        "torch_device": resolved_device,
        "torch_batch_size": config.torch_batch_size,
        "torch_pretrained": config.torch_pretrained,
    }
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _build_fmow_prediction_cache(
    cache_dir: str | Path,
    config: WILDSConfig,
    *,
    overwrite: bool = False,
) -> Path:
    train_subset, eval_subsets = load_fmow_wilds_splits(config)
    feature_extractor, resolved_device = build_resnet18_feature_extractor(
        pretrained=config.torch_pretrained,
        device=config.torch_device,
    )

    def metadata_parser(row: dict[str, Any], *, split: str) -> dict[str, Any]:
        return _normalize_fmow_metadata_row(
            row,
            split=split,
            context_column=config.context_column,
        )

    train_features, train_labels, _ = extract_resnet_features_from_wilds_subset(
        train_subset,
        model=feature_extractor,
        device=resolved_device,
        batch_size=config.torch_batch_size,
        max_examples=_max_train_examples(config),
        random_seed=config.random_seed,
        split="train",
        metadata_parser=metadata_parser,
    )
    num_classes = _infer_num_classes(train_labels, train_subset)
    classifier = train_feature_classifier(
        train_features,
        train_labels,
        random_seed=config.random_seed,
        num_classes=num_classes,
    )

    frames: list[pd.DataFrame] = []
    array_chunks: dict[str, list[np.ndarray]] = {"logits": [], "probabilities": [], "feature_embedding": []}
    start_index = 0
    per_split_limit = _per_split_eval_limit(config, len(eval_subsets))
    for split, subset in eval_subsets.items():
        features, labels, eval_df = extract_resnet_features_from_wilds_subset(
            subset,
            model=feature_extractor,
            device=resolved_device,
            batch_size=config.torch_batch_size,
            max_examples=per_split_limit,
            random_seed=config.random_seed,
            split=split,
            metadata_parser=metadata_parser,
        )
        probabilities, logits = predict_proba_with_features(classifier, features, num_classes)
        frame, arrays = compute_fmow_prediction_frame_and_arrays(
            eval_df=eval_df,
            labels=labels,
            probabilities=probabilities,
            logits=logits,
            feature_embedding=features,
            split="eval",
            start_example_index=start_index,
            context_column=config.context_column,
        )
        frames.append(frame)
        for key, value in arrays.items():
            array_chunks[key].append(value)
        start_index += len(frame)

    if not frames:
        raise ValueError("No FMoW evaluation examples were extracted")
    examples = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate(values, axis=0) for key, values in array_chunks.items()}
    metadata = {
        "dataset": "wilds_fmow",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": int(arrays["logits"].shape[1]),
        "context_fields": _context_fields_present(examples, ("context_id", "domain_id", "group_id", "region", "year")),
        "wilds_config": _config_to_jsonable(config),
        "model_backend": config.model_backend,
        "source": config.source,
        "eval_splits": list(config.eval_splits),
        "max_train_examples": _max_train_examples(config),
        "max_eval_examples": _max_eval_examples(config),
        "torch_device": resolved_device,
        "torch_batch_size": config.torch_batch_size,
        "torch_pretrained": config.torch_pretrained,
    }
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _normalize_civilcomments_frame(
    frame: pd.DataFrame,
    *,
    config: WILDSConfig,
    default_split: str,
    source: str,
) -> pd.DataFrame:
    normalized = frame.copy()
    text_source = config.text_column if config.text_column in normalized.columns else "comment_text"
    if text_source not in normalized.columns:
        raise ValueError(
            f"CivilComments CSV must contain {config.text_column!r} or 'comment_text' text column"
        )
    label_source = _first_existing_column(normalized, (config.label_column, "y", "toxic"))
    if label_source is None:
        raise ValueError(
            f"CivilComments CSV must contain {config.label_column!r}, 'y', or 'toxic' label column"
        )

    normalized["text"] = normalized[text_source].fillna("").astype(str)
    normalized["label"] = pd.to_numeric(normalized[label_source], errors="raise").astype(int)
    invalid_labels = sorted(set(normalized["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"CivilComments labels must be binary 0/1, got invalid values {invalid_labels}")

    if "split" not in normalized:
        normalized["split"] = default_split
    normalized["split"] = normalized["split"].fillna(default_split).astype(str)

    has_identity = "identity" in normalized.columns
    if has_identity:
        normalized["identity"] = normalized["identity"].fillna("").astype(str)
    else:
        normalized["identity"] = ""

    if "group_id" not in normalized.columns:
        normalized["group_id"] = normalized["identity"] if has_identity else normalized["split"]
    normalized["group_id"] = normalized["group_id"].fillna(normalized["split"]).astype(str)

    if "domain_id" not in normalized.columns:
        normalized["domain_id"] = normalized["identity"] if has_identity else normalized["split"]
    normalized["domain_id"] = normalized["domain_id"].fillna(normalized["split"]).astype(str)

    if not has_identity:
        normalized["identity"] = normalized["group_id"].astype(str)

    if config.context_column not in normalized.columns:
        normalized[config.context_column] = normalized["group_id"]
    normalized[config.context_column] = normalized[config.context_column].fillna(normalized["group_id"]).astype(str)
    normalized["source"] = source
    return normalized.reset_index(drop=True)


def _get_wilds_subset(dataset: Any, split: str) -> Any:
    try:
        return dataset.get_subset(split, transform=None)
    except TypeError:
        return dataset.get_subset(split)
    except Exception as exc:
        raise RuntimeError(f"Could not load WILDS split {split!r}: {exc}") from exc


def _selected_indices(length: int, max_examples: int | None, random_seed: int) -> np.ndarray:
    indices = np.arange(int(length), dtype=int)
    if max_examples is not None and int(max_examples) < len(indices):
        rng = np.random.default_rng(int(random_seed))
        indices = np.sort(rng.choice(indices, size=int(max_examples), replace=False))
    return indices


def _max_train_examples(config: WILDSConfig) -> int | None:
    return config.max_train_examples if config.max_train_examples is not None else config.train_subset_size


def _max_eval_examples(config: WILDSConfig) -> int | None:
    return config.max_eval_examples if config.max_eval_examples is not None else config.eval_subset_size


def _per_split_eval_limit(config: WILDSConfig, num_splits: int) -> int | None:
    max_eval = _max_eval_examples(config)
    if max_eval is None:
        return None
    return max(1, int(np.ceil(int(max_eval) / max(int(num_splits), 1))))


def _infer_num_classes(labels: np.ndarray, *sources: Any) -> int:
    label_array = np.asarray(labels, dtype=int)
    inferred = int(np.max(label_array)) + 1 if label_array.size else 0
    for source in sources:
        for candidate in (source, getattr(source, "dataset", None)):
            if candidate is None:
                continue
            for attribute in ("n_classes", "num_classes"):
                value = getattr(candidate, attribute, None)
                if value is None:
                    continue
                try:
                    inferred = max(inferred, int(value))
                except (TypeError, ValueError):
                    pass
    if inferred < 2:
        raise ValueError("At least two classes are required to train a prediction cache classifier")
    return inferred


def _context_fields_present(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _metadata_fields_for_subset(subset: Any) -> list[str]:
    fields = getattr(subset, "metadata_fields", None)
    if fields is None and hasattr(subset, "dataset"):
        fields = getattr(subset.dataset, "metadata_fields", None)
    if fields is None:
        return []
    return [str(field) for field in list(fields)]


def _available_wilds_split_names(dataset: Any) -> list[str]:
    split_dict = getattr(dataset, "split_dict", None)
    if isinstance(split_dict, dict):
        return sorted(str(key) for key in split_dict)
    split_names = getattr(dataset, "split_names", None)
    if split_names is not None:
        try:
            return sorted(str(value) for value in split_names)
        except TypeError:
            pass
    return []


def _unpack_wilds_sample(item: Any) -> tuple[Any, Any, Any | None]:
    if isinstance(item, dict):
        image = _first_mapping_value(item, ("image", "x", "input", "features"))
        label = _first_mapping_value(item, ("label", "y", "target"))
        if image is None or label is None:
            raise RuntimeError("Unexpected WILDS mapping sample; expected image/x and label/y keys")
        metadata = _first_mapping_value(item, ("metadata", "meta", "metadata_array"), default=None)
        return image, label, metadata
    if not isinstance(item, tuple) or len(item) < 2:
        raise RuntimeError("Unexpected WILDS sample format; expected at least (image, label)")
    image = item[0]
    label = item[1]
    metadata = item[2] if len(item) >= 3 else None
    return image, label, metadata


def _first_mapping_value(mapping: dict[str, Any], keys: Sequence[str], *, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _metadata_to_row(metadata: Any, metadata_fields: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if metadata is None:
        return row
    if isinstance(metadata, dict):
        return {str(key): _scalar_value(value) for key, value in metadata.items()}
    try:
        values = metadata.tolist() if hasattr(metadata, "tolist") else list(metadata)
    except TypeError:
        values = [metadata]
    for metadata_field, value in zip(metadata_fields, values, strict=False):
        row[str(metadata_field)] = _scalar_value(value)
    if not row and values:
        row["metadata_0"] = _scalar_value(values[0])
    return row


def _normalize_camelyon_metadata_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    normalized = dict(row)
    domain = _first_metadata_value(
        normalized,
        ("hospital", "hospital_id", "center", "center_id", "domain_id", "slide", "slide_id", "metadata_0"),
        default=split,
    )
    group = _first_metadata_value(
        normalized,
        ("group_id", "domain_id", "hospital", "hospital_id", "center", "center_id", "metadata_0"),
        default=domain,
    )
    normalized["domain_id"] = _sanitize_context_value(domain)
    normalized["group_id"] = _sanitize_context_value(group)
    normalized["source"] = "wilds-camelyon17"
    return normalized


def _normalize_fmow_metadata_row(
    row: dict[str, Any],
    *,
    split: str,
    context_column: str = "domain_id",
) -> dict[str, Any]:
    normalized = dict(row)
    region = _first_metadata_value(
        normalized,
        ("region", "region_id", "geographic_region"),
        default=None,
    )
    year = _first_metadata_value(
        normalized,
        ("year", "capture_year", "timestamp_year"),
        default=None,
    )
    domain = _first_metadata_value(
        normalized,
        ("domain_id", "domain", "metadata_0"),
        default=region if region is not None else year if year is not None else split,
    )
    group = _first_metadata_value(
        normalized,
        ("group_id", "group", "domain_id", "domain", "metadata_0"),
        default=domain,
    )

    if region is not None:
        normalized["region"] = _sanitize_context_value(region)
    if year is not None:
        normalized["year"] = _sanitize_context_value(year)
    normalized["domain_id"] = _sanitize_context_value(domain)
    normalized["group_id"] = _sanitize_context_value(group)
    if context_column not in normalized and context_column not in {"region", "year"}:
        if "region" in normalized:
            normalized[context_column] = normalized["region"]
        elif "year" in normalized:
            normalized[context_column] = normalized["year"]
        else:
            normalized[context_column] = normalized["domain_id"]
    normalized["source"] = "wilds-fmow"
    if "time_index" not in normalized:
        if "year" in normalized:
            normalized["time_index"] = normalized["year"]
        else:
            normalized["time_index"] = normalized.get("original_index", 0)
    return normalized


def _first_metadata_value(row: dict[str, Any], candidates: Sequence[str], *, default: Any) -> Any:
    for candidate in candidates:
        if candidate in row and not _is_missing_scalar(row[candidate]):
            return row[candidate]
    return default


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (np.ndarray, list)):
        return False
    return bool(missing)


def _image_to_hwc_array(image: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected image array with shape [h, w, c], got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.asarray(array, dtype=np.float32)


def _scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _civilcomments_subset_to_frame(
    dataset: Any,
    split: str,
    config: WILDSConfig,
) -> pd.DataFrame:
    try:
        subset = dataset.get_subset(split, transform=None)
    except TypeError:
        subset = dataset.get_subset(split)
    except Exception as exc:
        raise RuntimeError(f"Could not load WILDS CivilComments split {split!r}: {exc}") from exc

    metadata_fields = list(getattr(dataset, "metadata_fields", getattr(subset, "metadata_fields", [])))
    rows: list[dict[str, Any]] = []
    for index in range(len(subset)):
        item = subset[index]
        if not isinstance(item, tuple) or len(item) < 2:
            raise RuntimeError("Unexpected WILDS sample format; expected at least (text, label)")
        text = item[0]
        label = item[1]
        metadata = item[2] if len(item) >= 3 else None
        row: dict[str, Any] = {
            "text": text.decode("utf-8") if isinstance(text, bytes) else str(text),
            "label": int(label.item() if hasattr(label, "item") else label),
            "split": "train" if split == "train" else "eval",
            "source": "wilds",
        }
        if metadata is not None:
            values = metadata.tolist() if hasattr(metadata, "tolist") else list(metadata)
            for field, value in zip(metadata_fields, values, strict=False):
                row[str(field)] = value.item() if hasattr(value, "item") else value
        rows.append(row)

    raw = pd.DataFrame(rows)
    if "identity" not in raw and "black" in raw:
        identity_columns = [
            column
            for column in ("black", "white", "christian", "muslim", "other_religions", "male", "female", "LGBTQ")
            if column in raw
        ]
        if identity_columns:
            raw["identity"] = raw[identity_columns].astype(float).idxmax(axis=1)
    return _normalize_civilcomments_frame(raw, config=config, default_split="train" if split == "train" else "eval", source="wilds")


def _first_available_wilds_split(dataset: Any, candidates: tuple[str, ...]) -> str:
    split_dict = getattr(dataset, "split_dict", {})
    if isinstance(split_dict, dict):
        for candidate in candidates:
            if candidate in split_dict:
                return candidate
    for candidate in candidates:
        try:
            dataset.get_subset(candidate)
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"Could not find an eval/test WILDS split among {candidates}")


def _apply_subset(frame: pd.DataFrame, subset_size: int | None, random_seed: int) -> pd.DataFrame:
    if subset_size is not None and subset_size < len(frame):
        frame = frame.sample(n=int(subset_size), random_state=int(random_seed)).sort_index()
    return frame.reset_index(drop=True)


def _first_existing_column(frame: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _sanitize_context_value(value: Any) -> str:
    text = "unknown" if pd.isna(value) else str(value)
    text = text.strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _fmow_context_source(frame: pd.DataFrame, *, context_column: str) -> pd.DataFrame:
    for column in (context_column, "region", "year", "domain_id", "split"):
        if column in frame.columns:
            return pd.DataFrame(
                {
                    "prefix": [_context_prefix_for_column(column)] * len(frame),
                    "value": frame[column].tolist(),
                }
            )
    return pd.DataFrame({"prefix": ["split"] * len(frame), "value": ["eval"] * len(frame)})


def _context_prefix_for_column(column: str) -> str:
    if column == "region":
        return "region"
    if column == "year":
        return "year"
    if column in {"domain_id", "domain"}:
        return "domain"
    if column in {"group_id", "group"}:
        return "group"
    if column == "context_id":
        return "context"
    if column == "split":
        return "split"
    return "context"


def _prefixed_context_value(prefix: str, value: Any) -> str:
    cleaned = _sanitize_context_value(value)
    for known_prefix in ("region", "year", "domain", "group", "split", "context"):
        if cleaned.startswith(f"{known_prefix}_"):
            return cleaned
    return f"{prefix}_{cleaned}"


def _context_name(prefix: str, value: Any) -> str:
    cleaned = _sanitize_context_value(value).replace("_", " ")
    return f"{prefix.replace('_', ' ').title()} {cleaned}"


def _cache_dataset_name(dataset: str) -> str:
    dataset = str(dataset).lower()
    if dataset == "civilcomments":
        return "wilds_civilcomments"
    if dataset == "camelyon17":
        return "wilds_camelyon17"
    if dataset == "fmow":
        return "wilds_fmow"
    return dataset


def _ensure_binary_logits(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        return np.column_stack([-0.5 * decision, 0.5 * decision])
    if decision.ndim == 2 and decision.shape[1] == 2:
        return decision
    raise ValueError(f"Expected binary decision scores, got shape {decision.shape}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
    return np.squeeze(stable, axis=axis)


def _config_to_jsonable(config: WILDSConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(payload["data_root"])
    payload["ngram_range"] = list(payload["ngram_range"])
    payload["eval_splits"] = list(payload["eval_splits"])
    return payload
