from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.datasets.vision_features import (
    build_resnet18_feature_extractor,
    extract_resnet18_features,
    predict_proba_with_features,
    sanitize_probabilities,
    train_feature_classifier,
)

OFFICEHOME_DOMAINS = ("art", "clipart", "product", "real_world")
OFFICEHOME_DOMAIN_DISPLAY = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}
OFFICEHOME_TIME_INDEX = {domain: index for index, domain in enumerate(OFFICEHOME_DOMAINS)}
_ALLOWED_MODEL_BACKENDS = ("torchvision_resnet18_features",)
_ALLOWED_TORCH_DEVICES = ("auto", "cpu", "cuda")
_ALLOWED_CALIBRATION_METHODS = ("sigmoid", "isotonic")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfficeHomeConfig:
    data_root: str | Path = "data/officehome"
    cache_dir: str | Path = "outputs/caches/officehome_resnet18_subset"
    model_backend: str = "torchvision_resnet18_features"
    source_domains: tuple[str, ...] = ("real_world", "product")
    eval_domains: tuple[str, ...] = ("art", "clipart", "product", "real_world")
    max_train_examples: int | None = 12000
    max_eval_examples: int | None = None
    torch_batch_size: int = 64
    torch_device: str = "auto"
    torch_pretrained: bool = True
    calibrate_probabilities: bool = True
    calibration_fraction: float = 0.2
    calibration_method: str = "sigmoid"
    calibration_cv: str = "prefit"
    random_seed: int = 123

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "model_backend", str(self.model_backend).lower())
        object.__setattr__(
            self,
            "source_domains",
            tuple(normalize_officehome_domain_name(domain) for domain in self.source_domains),
        )
        object.__setattr__(
            self,
            "eval_domains",
            tuple(normalize_officehome_domain_name(domain) for domain in self.eval_domains),
        )
        object.__setattr__(self, "torch_device", str(self.torch_device).lower())
        object.__setattr__(self, "calibrate_probabilities", bool(self.calibrate_probabilities))
        object.__setattr__(self, "calibration_fraction", float(self.calibration_fraction))
        object.__setattr__(self, "calibration_method", str(self.calibration_method).lower())
        object.__setattr__(self, "calibration_cv", str(self.calibration_cv).lower())

        if self.model_backend not in _ALLOWED_MODEL_BACKENDS:
            raise ValueError(
                f"model_backend must be one of {_ALLOWED_MODEL_BACKENDS}, got {self.model_backend!r}"
            )
        if not self.source_domains:
            raise ValueError("source_domains must be non-empty")
        if not self.eval_domains:
            raise ValueError("eval_domains must be non-empty")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive or None")
        if self.max_eval_examples is not None and self.max_eval_examples <= 0:
            raise ValueError("max_eval_examples must be positive or None")
        if self.torch_batch_size <= 0:
            raise ValueError("torch_batch_size must be positive")
        if self.torch_device not in _ALLOWED_TORCH_DEVICES:
            raise ValueError(f"torch_device must be one of {_ALLOWED_TORCH_DEVICES}, got {self.torch_device!r}")
        if self.calibration_method not in _ALLOWED_CALIBRATION_METHODS:
            raise ValueError(
                f"calibration_method must be one of {_ALLOWED_CALIBRATION_METHODS}, "
                f"got {self.calibration_method!r}"
            )
        if self.calibrate_probabilities and not (0.0 < self.calibration_fraction < 0.5):
            raise ValueError("calibration_fraction must be in (0, 0.5) when calibration is enabled")


def normalize_officehome_domain_name(name: str) -> str:
    value = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    value = "_".join(part for part in value.split("_") if part)
    aliases = {
        "art": "art",
        "artistic": "art",
        "clipart": "clipart",
        "clip_art": "clipart",
        "product": "product",
        "real_world": "real_world",
        "realworld": "real_world",
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(f"Unknown OfficeHome domain name: {name!r}")


def find_officehome_domain_dirs(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"OfficeHome data root not found: {root}")

    domains: dict[str, Path] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        try:
            domain = normalize_officehome_domain_name(child.name)
        except ValueError:
            continue
        domains.setdefault(domain, child)
    if not domains:
        raise FileNotFoundError(f"No OfficeHome domain directories found under {root}")
    return domains


def build_officehome_index(config: OfficeHomeConfig) -> pd.DataFrame:
    domain_dirs = find_officehome_domain_dirs(config.data_root)
    requested_domains = _ordered_requested_domains(config)
    missing_domains = [domain for domain in requested_domains if domain not in domain_dirs]
    if missing_domains:
        raise FileNotFoundError(
            f"OfficeHome domain directories missing for {missing_domains}. "
            f"Found domains: {sorted(domain_dirs)}"
        )

    class_names = sorted(
        {
            class_dir.name
            for domain in requested_domains
            for class_dir in domain_dirs[domain].iterdir()
            if class_dir.is_dir()
        }
    )
    if len(class_names) < 2:
        raise ValueError(f"OfficeHome requires at least two classes, found {class_names}")
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

    rows: list[dict[str, Any]] = []
    for domain in requested_domains:
        domain_dir = domain_dirs[domain]
        domain_rows = []
        for class_name in class_names:
            class_dir = domain_dir / class_name
            if not class_dir.is_dir():
                continue
            for image_path in _iter_image_paths(class_dir):
                domain_rows.append(
                    {
                        "image_path": str(image_path),
                        "domain": domain,
                        "class_name": class_name,
                        "label": int(class_to_idx[class_name]),
                    }
                )
        if not domain_rows:
            raise ValueError(f"OfficeHome domain {domain!r} has no indexed image files")
        rows.extend(domain_rows)

    frame = pd.DataFrame(rows, columns=["image_path", "domain", "class_name", "label"])
    frame.attrs["class_to_idx"] = class_to_idx
    return frame.reset_index(drop=True)


def subsample_index(
    index: pd.DataFrame,
    *,
    domains: Sequence[str],
    max_examples: int | None,
    random_seed: int,
) -> pd.DataFrame:
    normalized_domains = [normalize_officehome_domain_name(domain) for domain in domains]
    frame = index[index["domain"].isin(normalized_domains)].copy()
    if frame.empty:
        raise ValueError(f"No OfficeHome examples found for domains {normalized_domains}")
    if max_examples is not None and int(max_examples) < len(frame):
        frame = frame.sample(n=int(max_examples), random_state=int(random_seed)).sort_index()
    return frame.reset_index(drop=True)


def load_images_from_paths(paths: Sequence[str | Path]) -> np.ndarray:
    images: list[np.ndarray] = []
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("OfficeHome image loading requires Pillow/PIL to be installed") from exc

    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((224, 224))
            images.append(np.asarray(image, dtype=np.float32))
    if not images:
        return np.empty((0, 224, 224, 3), dtype=np.float32)
    return np.stack(images, axis=0).astype(np.float32, copy=False)


def extract_resnet_features_from_paths(
    paths: Sequence[str | Path],
    *,
    model: Any,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    feature_chunks: list[np.ndarray] = []
    path_list = [Path(path) for path in paths]
    for start in range(0, len(path_list), int(batch_size)):
        batch_paths = path_list[start : start + int(batch_size)]
        images = load_images_from_paths(batch_paths)
        features = extract_resnet18_features(
            images,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        feature_chunks.append(features)
    if not feature_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.nan_to_num(
        np.concatenate(feature_chunks, axis=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def train_officehome_feature_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    random_seed: int,
    num_classes: int,
    calibrate_probabilities: bool = True,
    calibration_fraction: float = 0.2,
    calibration_method: str = "sigmoid",
) -> Any:
    """Train the OfficeHome sklearn classifier, optionally with held-out calibration."""
    features = np.asarray(train_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=int)
    calibration_method = str(calibration_method).lower()
    if calibration_method not in _ALLOWED_CALIBRATION_METHODS:
        raise ValueError(
            f"calibration_method must be one of {_ALLOWED_CALIBRATION_METHODS}, got {calibration_method!r}"
        )
    if not calibrate_probabilities:
        return train_feature_classifier(
            features,
            labels,
            random_seed=random_seed,
            num_classes=num_classes,
        )
    if not (0.0 < float(calibration_fraction) < 0.5):
        raise ValueError("calibration_fraction must be in (0, 0.5) when calibration is enabled")

    split = _officehome_calibration_split(
        features,
        labels,
        calibration_fraction=float(calibration_fraction),
        random_seed=random_seed,
    )
    if split is None:
        LOGGER.warning("Falling back to uncalibrated OfficeHome classifier because calibration split is too small")
        return train_feature_classifier(
            features,
            labels,
            random_seed=random_seed,
            num_classes=num_classes,
        )
    fit_features, calibration_features, fit_labels, calibration_labels = split
    if len(np.unique(fit_labels)) < 2 or len(np.unique(calibration_labels)) < 2:
        LOGGER.warning("Falling back to uncalibrated OfficeHome classifier because a split has fewer than two classes")
        return train_feature_classifier(
            features,
            labels,
            random_seed=random_seed,
            num_classes=num_classes,
        )

    base_classifier = train_feature_classifier(
        fit_features,
        fit_labels,
        random_seed=random_seed,
        num_classes=num_classes,
    )
    try:
        calibrator = make_prefit_calibrated_classifier(base_classifier, method=calibration_method)
        calibrator.fit(calibration_features, calibration_labels)
    except (RuntimeError, TypeError, ValueError) as exc:
        LOGGER.warning("Falling back to uncalibrated OfficeHome classifier after calibration failed: %s", exc)
        return train_feature_classifier(
            features,
            labels,
            random_seed=random_seed,
            num_classes=num_classes,
    )
    return calibrator


def make_prefit_calibrated_classifier(base_classifier: Any, *, method: str) -> Any:
    """Wrap a fitted sklearn classifier for probability calibration across sklearn APIs."""
    method = str(method).lower()
    if method not in _ALLOWED_CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {_ALLOWED_CALIBRATION_METHODS}, got {method!r}")

    construction_errors: list[str] = []
    frozen_estimator = _resolve_frozen_estimator()
    if frozen_estimator is not None:
        try:
            return CalibratedClassifierCV(frozen_estimator(base_classifier), method=method)
        except TypeError as exc:
            construction_errors.append(f"FrozenEstimator API failed: {exc}")

    try:
        return CalibratedClassifierCV(estimator=base_classifier, method=method, cv="prefit")
    except TypeError as exc:
        construction_errors.append(f"estimator/cv='prefit' API failed: {exc}")

    try:
        return CalibratedClassifierCV(base_estimator=base_classifier, method=method, cv="prefit")
    except TypeError as exc:
        construction_errors.append(f"base_estimator/cv='prefit' API failed: {exc}")

    details = "; ".join(construction_errors) if construction_errors else "no compatible sklearn API found"
    raise RuntimeError(f"Could not construct a prefit calibrated classifier: {details}")


def compute_officehome_prediction_frame_and_arrays(
    *,
    eval_index: pd.DataFrame,
    probabilities: np.ndarray,
    logits: np.ndarray,
    feature_embedding: np.ndarray,
    dataset: str = "officehome",
    split: str = "eval",
    start_example_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    required_columns = {"domain", "class_name", "label"}
    missing = sorted(required_columns.difference(eval_index.columns))
    if missing:
        raise ValueError(f"eval_index is missing required columns: {missing}")

    labels = pd.to_numeric(eval_index["label"], errors="raise").astype(int).to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [n, num_classes>=2]")
    probabilities = sanitize_probabilities(probabilities, num_classes=int(probabilities.shape[1]))
    logits = np.asarray(logits, dtype=float)
    feature_embedding = np.asarray(feature_embedding, dtype=float)
    if logits.shape != probabilities.shape:
        raise ValueError("logits shape must match probabilities shape")
    if len(eval_index) != len(labels) or probabilities.shape[0] != len(labels):
        raise ValueError("eval_index, labels, probabilities, and logits must have matching first dimensions")
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

    normalized = eval_index.copy().reset_index(drop=True)
    normalized["domain"] = normalized["domain"].map(normalize_officehome_domain_name)
    domain_values = normalized["domain"].astype(str).tolist()
    time_values = [OFFICEHOME_TIME_INDEX[domain] for domain in domain_values]

    examples = pd.DataFrame(
        {
            "example_id": [
                f"{dataset}/{split}/{domain}/{index}"
                for domain, index in zip(domain_values, indices, strict=True)
            ],
            "dataset": dataset,
            "split": split,
            "context_id": domain_values,
            "context_name": [OFFICEHOME_DOMAIN_DISPLAY[domain] for domain in domain_values],
            "label": labels.astype(int),
            "pred": pred,
            "correct": correct.astype(bool),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "margin": margin.astype(float),
            "energy": energy.astype(float),
            "domain_id": domain_values,
            "group_id": domain_values,
            "source": ["officehome"] * len(labels),
            "time_index": np.asarray(time_values, dtype=int),
            "domain": domain_values,
            "class_name": normalized["class_name"].astype(str).tolist(),
        }
    )
    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": feature_embedding.astype(float),
    }
    return examples, arrays


def summarize_officehome_confidence_by_domain(examples: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"context_id", "correct", "confidence", "margin", "entropy"}
    missing = sorted(required_columns.difference(examples.columns))
    if missing:
        raise ValueError(f"examples is missing required columns: {missing}")
    summary = (
        examples.groupby("context_id", sort=True, dropna=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("margin", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .reset_index()
    )
    if "domain" in examples.columns:
        domains = examples.groupby("context_id", sort=True, dropna=False)["domain"].first().reset_index()
        summary = summary.merge(domains, on="context_id", how="left")
    else:
        summary["domain"] = summary["context_id"].astype(str)
    return summary[["context_id", "domain", "n", "accuracy", "mean_confidence", "mean_margin", "mean_entropy"]]


def build_officehome_prediction_cache(
    cache_dir: str | Path,
    config: OfficeHomeConfig,
    *,
    overwrite: bool = False,
) -> Path:
    index = build_officehome_index(config)
    class_to_idx = dict(index.attrs.get("class_to_idx") or _class_to_idx_from_index(index))
    train_index = subsample_index(
        index,
        domains=config.source_domains,
        max_examples=config.max_train_examples,
        random_seed=config.random_seed,
    )
    eval_index = subsample_index(
        index,
        domains=config.eval_domains,
        max_examples=config.max_eval_examples,
        random_seed=config.random_seed,
    )

    feature_extractor, resolved_device = build_resnet18_feature_extractor(
        pretrained=config.torch_pretrained,
        device=config.torch_device,
    )
    train_features = extract_resnet_features_from_paths(
        train_index["image_path"].tolist(),
        model=feature_extractor,
        device=resolved_device,
        batch_size=config.torch_batch_size,
    )
    train_labels = train_index["label"].astype(int).to_numpy()
    classifier = train_officehome_feature_classifier(
        train_features,
        train_labels,
        random_seed=config.random_seed,
        num_classes=len(class_to_idx),
        calibrate_probabilities=config.calibrate_probabilities,
        calibration_fraction=config.calibration_fraction,
        calibration_method=config.calibration_method,
    )
    eval_features = extract_resnet_features_from_paths(
        eval_index["image_path"].tolist(),
        model=feature_extractor,
        device=resolved_device,
        batch_size=config.torch_batch_size,
    )
    probabilities, logits = predict_proba_with_features(classifier, eval_features, len(class_to_idx))
    examples, arrays = compute_officehome_prediction_frame_and_arrays(
        eval_index=eval_index,
        probabilities=probabilities,
        logits=logits,
        feature_embedding=eval_features,
    )
    metadata = {
        "dataset": "officehome",
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": len(class_to_idx),
        "context_fields": ["context_id", "domain_id", "group_id", "domain"],
        "officehome_config": _config_to_jsonable(config),
        "source_domains": list(config.source_domains),
        "eval_domains": list(config.eval_domains),
        "model_backend": config.model_backend,
        "class_to_idx": class_to_idx,
        "torch_device": resolved_device,
        "torch_batch_size": config.torch_batch_size,
        "torch_pretrained": config.torch_pretrained,
        "calibrate_probabilities": config.calibrate_probabilities,
        "calibration_fraction": config.calibration_fraction,
        "calibration_method": config.calibration_method,
    }
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _officehome_calibration_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return None
    try:
        return train_test_split(
            features,
            labels,
            test_size=calibration_fraction,
            random_state=random_seed,
            stratify=labels,
        )
    except ValueError as exc:
        LOGGER.warning(
            "Stratified OfficeHome calibration split failed; falling back to deterministic unstratified split: %s",
            exc,
        )
    try:
        return train_test_split(
            features,
            labels,
            test_size=calibration_fraction,
            random_state=random_seed,
            stratify=None,
        )
    except ValueError as exc:
        LOGGER.warning("OfficeHome calibration split failed: %s", exc)
        return None


def _resolve_frozen_estimator() -> Any | None:
    try:
        from sklearn.calibration import FrozenEstimator
    except ImportError:
        try:
            from sklearn.frozen import FrozenEstimator
        except ImportError:
            return None
    return FrozenEstimator


def _ordered_requested_domains(config: OfficeHomeConfig) -> list[str]:
    requested = set(config.source_domains) | set(config.eval_domains)
    return [domain for domain in OFFICEHOME_DOMAINS if domain in requested]


def _iter_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


def _class_to_idx_from_index(index: pd.DataFrame) -> dict[str, int]:
    class_names = sorted(str(value) for value in index["class_name"].dropna().unique())
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
    return np.squeeze(stable, axis=axis)


def _config_to_jsonable(config: OfficeHomeConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(payload["data_root"])
    payload["cache_dir"] = str(payload["cache_dir"])
    payload["source_domains"] = list(payload["source_domains"])
    payload["eval_domains"] = list(payload["eval_domains"])
    return payload
