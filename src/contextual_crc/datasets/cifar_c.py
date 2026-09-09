from __future__ import annotations

import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from contextual_crc.cache.io import save_prediction_cache
from contextual_crc.utils.paths import ensure_dir


CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"

DEFAULT_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

_ALLOWED_DATASETS = ("cifar10c", "cifar100c")
_ALLOWED_MODEL_BACKENDS = ("sklearn_logistic", "torchvision_resnet18")
_ALLOWED_FEATURE_MODES = ("color_stats", "flatten", "resnet18_features")
_ALLOWED_TORCH_DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class CIFARCConfig:
    dataset: str = "cifar10c"
    data_root: str | Path = "data"
    corruptions: tuple[str, ...] = ("gaussian_noise", "brightness", "contrast")
    severities: tuple[int, ...] = (1, 3, 5)
    max_examples_per_context: int | None = 1000
    model_backend: str = "sklearn_logistic"
    feature_mode: str = "color_stats"
    train_subset_size: int | None = 10000
    random_seed: int = 123
    download: bool = False
    torch_batch_size: int = 128
    torch_device: str = "auto"
    torch_pretrained: bool = True
    feature_cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset).lower())
        object.__setattr__(self, "model_backend", str(self.model_backend).lower())
        object.__setattr__(self, "feature_mode", str(self.feature_mode).lower())
        object.__setattr__(self, "torch_device", str(self.torch_device).lower())
        object.__setattr__(self, "corruptions", tuple(self.corruptions))
        object.__setattr__(self, "severities", tuple(int(severity) for severity in self.severities))

        if self.dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of {_ALLOWED_DATASETS}, got {self.dataset!r}")
        if not self.corruptions:
            raise ValueError("corruptions must be non-empty")
        invalid_corruptions = [corruption for corruption in self.corruptions if not corruption]
        if invalid_corruptions:
            raise ValueError(f"corruptions must be non-empty strings: {invalid_corruptions}")
        invalid_severities = [severity for severity in self.severities if severity < 1 or severity > 5]
        if invalid_severities:
            raise ValueError(f"severities must be integers in 1..5: {invalid_severities}")
        if self.model_backend not in _ALLOWED_MODEL_BACKENDS:
            raise ValueError(
                f"model_backend must be one of {_ALLOWED_MODEL_BACKENDS}, got {self.model_backend!r}"
            )
        if self.feature_mode not in _ALLOWED_FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {_ALLOWED_FEATURE_MODES}, got {self.feature_mode!r}")
        if self.model_backend == "sklearn_logistic" and self.feature_mode == "resnet18_features":
            raise ValueError("feature_mode='resnet18_features' requires model_backend='torchvision_resnet18'")
        if self.max_examples_per_context is not None and self.max_examples_per_context <= 0:
            raise ValueError("max_examples_per_context must be positive or None")
        if self.train_subset_size is not None and self.train_subset_size <= 0:
            raise ValueError("train_subset_size must be positive or None")
        if self.torch_batch_size <= 0:
            raise ValueError("torch_batch_size must be positive")
        if self.torch_device not in _ALLOWED_TORCH_DEVICES:
            raise ValueError(f"torch_device must be one of {_ALLOWED_TORCH_DEVICES}, got {self.torch_device!r}")


def dataset_dir(config: CIFARCConfig) -> Path:
    return Path(config.data_root) / config.dataset


def expected_num_classes(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset == "cifar10c":
        return 10
    if dataset == "cifar100c":
        return 100
    raise ValueError(f"dataset must be one of {_ALLOWED_DATASETS}, got {dataset!r}")


def archive_url(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "cifar10c":
        return CIFAR10C_URL
    if dataset == "cifar100c":
        return CIFAR100C_URL
    raise ValueError(f"dataset must be one of {_ALLOWED_DATASETS}, got {dataset!r}")


def download_and_extract_cifar_c(config: CIFARCConfig) -> Path:
    """Download and safely extract official CIFAR-C archives when explicitly requested."""
    target_dir = dataset_dir(config)
    if target_dir.exists() and (target_dir / "labels.npy").is_file():
        return target_dir

    ensure_dir(Path(config.data_root))
    archive_path = Path(config.data_root) / f"{config.dataset}.tar"
    if not archive_path.exists():
        try:
            _download_file(archive_url(config.dataset), archive_path)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                "Failed to download CIFAR-C archive. Either retry with network access or manually place "
                f"labels.npy and corruption .npy files under {target_dir}. Original error: {exc}"
            ) from exc

    with tarfile.open(archive_path, "r") as handle:
        _safe_extract(handle, Path(config.data_root))

    extracted_candidates = [
        Path(config.data_root) / "CIFAR-10-C",
        Path(config.data_root) / "CIFAR-100-C",
        Path(config.data_root) / "CIFAR-10-C-bar",
        Path(config.data_root) / "CIFAR-100-C-bar",
    ]
    for candidate in extracted_candidates:
        if candidate.exists() and (candidate / "labels.npy").is_file():
            if candidate != target_dir:
                if target_dir.exists():
                    return target_dir
                shutil.move(str(candidate), str(target_dir))
            return target_dir

    if not target_dir.exists():
        raise FileNotFoundError(
            f"Could not locate extracted CIFAR-C files. Expected manual layout under {target_dir}."
        )
    return target_dir


def validate_cifar_c_files(config: CIFARCConfig) -> None:
    base = dataset_dir(config)
    missing = []
    if not (base / "labels.npy").is_file():
        missing.append(str(base / "labels.npy"))
    for corruption in config.corruptions:
        if not (base / f"{corruption}.npy").is_file():
            missing.append(str(base / f"{corruption}.npy"))
    if missing:
        raise FileNotFoundError(
            "Missing CIFAR-C files. Place files manually under "
            f"{base} or rerun with download=true/--download. Missing: {missing}"
        )


def load_cifar_c_context(
    config: CIFARCConfig,
    corruption: str,
    severity: int,
) -> tuple[np.ndarray, np.ndarray]:
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in 1..5")
    if corruption not in config.corruptions:
        # Allow direct use for any available corruption while still catching empty names.
        if not corruption:
            raise ValueError("corruption must be a non-empty string")

    base = dataset_dir(config)
    images = np.load(base / f"{corruption}.npy")
    labels = np.load(base / "labels.npy")
    if len(images) != len(labels):
        raise ValueError(f"Images and labels must have matching lengths: {len(images)} != {len(labels)}")
    if len(labels) % 5 != 0:
        raise ValueError("CIFAR-C labels length must be divisible by 5 severities")

    n_per_severity = len(labels) // 5
    start = (severity - 1) * n_per_severity
    end = severity * n_per_severity
    context_images = images[start:end]
    context_labels = labels[start:end]
    if config.max_examples_per_context is not None:
        context_images = context_images[: config.max_examples_per_context]
        context_labels = context_labels[: config.max_examples_per_context]
    return context_images, context_labels.astype(int)


def extract_image_features(images: np.ndarray, mode: str = "color_stats") -> np.ndarray:
    if mode not in _ALLOWED_FEATURE_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_FEATURE_MODES}, got {mode!r}")
    if mode == "resnet18_features":
        raise ValueError("resnet18_features are produced by the torchvision_resnet18 backend")

    normalized = np.asarray(images, dtype=np.float32)
    if normalized.ndim != 4 or normalized.shape[-1] != 3:
        raise ValueError(f"images must have shape [n, h, w, 3], got {normalized.shape}")
    if normalized.size and np.nanmax(normalized) > 1.0:
        normalized = normalized / 255.0
    normalized = np.clip(normalized, 0.0, 1.0)

    if mode == "flatten":
        return normalized.reshape(normalized.shape[0], -1).astype(np.float32)

    per_channel_mean = normalized.mean(axis=(1, 2))
    per_channel_std = normalized.std(axis=(1, 2))
    flat = normalized.reshape(normalized.shape[0], -1)
    global_stats = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
        ]
    )
    quantiles = np.quantile(flat, [0.25, 0.5, 0.75], axis=1).T
    features = np.concatenate([per_channel_mean, per_channel_std, global_stats, quantiles], axis=1)
    return features.astype(np.float32)


def load_clean_cifar_train(
    dataset: str,
    data_root: str | Path,
    *,
    train_subset_size: int | None = 10000,
    random_seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Load clean CIFAR train data via optional torchvision."""
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "Clean CIFAR training requires torchvision. Install torchvision or provide a future "
            "precomputed model backend. CIFAR-C tests monkeypatch this function and do not require it."
        ) from exc

    dataset = dataset.lower()
    clean_root = Path(data_root) / "clean"
    if dataset == "cifar10c":
        train_dataset = datasets.CIFAR10(root=clean_root, train=True, download=True)
    elif dataset == "cifar100c":
        train_dataset = datasets.CIFAR100(root=clean_root, train=True, download=True)
    else:
        raise ValueError(f"dataset must be one of {_ALLOWED_DATASETS}, got {dataset!r}")

    images = np.asarray(train_dataset.data)
    labels = np.asarray(train_dataset.targets, dtype=int)
    if train_subset_size is not None and train_subset_size < len(labels):
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(labels), size=train_subset_size, replace=False)
        indices.sort()
        images = images[indices]
        labels = labels[indices]
    return images, labels


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve an optional torch device name for the ResNet backend."""
    if device not in _ALLOWED_TORCH_DEVICES:
        raise ValueError(f"device must be one of {_ALLOWED_TORCH_DEVICES}, got {device!r}")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The torchvision_resnet18 CIFAR backend requires torch and torchvision. "
            "Install compatible torch/torchvision packages, or use model_backend='sklearn_logistic'."
        ) from exc

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch_device='cuda' was requested, but torch.cuda.is_available() is false")
    return device


def build_resnet18_feature_extractor(
    *,
    pretrained: bool = True,
    device: str = "auto",
) -> tuple[Any, str]:
    """Build a torchvision ResNet-18 penultimate-layer feature extractor."""
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "The torchvision_resnet18 CIFAR backend requires torch and torchvision. "
            "Install compatible torch/torchvision packages, or use model_backend='sklearn_logistic'."
        ) from exc

    resolved_device = resolve_torch_device(device)
    try:
        if pretrained:
            weights_enum = getattr(models, "ResNet18_Weights", None)
            if weights_enum is not None:
                model = models.resnet18(weights=weights_enum.DEFAULT)
            else:
                model = models.resnet18(pretrained=True)
        else:
            try:
                model = models.resnet18(weights=None)
            except TypeError:
                model = models.resnet18(pretrained=False)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to load pretrained torchvision ResNet-18 weights. "
                "Ensure the weights are already cached or network access is available, "
                "or set torch_pretrained=false in the CIFAR config."
            ) from exc
        raise

    model.fc = torch.nn.Identity()
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def extract_resnet18_features(
    images: np.ndarray,
    *,
    model: Any,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract ImageNet-normalized ResNet-18 features from CIFAR-shaped images."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "The torchvision_resnet18 CIFAR backend requires torch. "
            "Install compatible torch/torchvision packages, or use model_backend='sklearn_logistic'."
        ) from exc

    image_array = np.asarray(images)
    if image_array.ndim != 4 or image_array.shape[-1] != 3:
        raise ValueError(f"images must have shape [n, h, w, 3], got {image_array.shape}")

    feature_chunks: list[np.ndarray] = []
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for start in range(0, len(image_array), batch_size):
            batch = np.asarray(image_array[start : start + batch_size], dtype=np.float32)
            if batch.size and np.nanmax(batch) > 1.0:
                batch = batch / 255.0
            batch = np.clip(batch, 0.0, 1.0)

            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
            if tuple(tensor.shape[-2:]) != (224, 224):
                tensor = functional.interpolate(
                    tensor,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            tensor = (tensor - mean) / std
            features = model(tensor)
            if isinstance(features, tuple):
                features = features[0]
            features_array = features.detach().cpu().numpy().reshape(tensor.shape[0], -1)
            feature_chunks.append(features_array.astype(np.float32, copy=False))

    if not feature_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(feature_chunks, axis=0)


def _extract_resnet18_features_maybe_cached(
    images: np.ndarray,
    *,
    config: CIFARCConfig,
    model: Any,
    device: str,
    cache_key: str,
) -> np.ndarray:
    if config.feature_cache_dir is None:
        return extract_resnet18_features(
            images,
            model=model,
            device=device,
            batch_size=config.torch_batch_size,
        )

    feature_cache_dir = ensure_dir(Path(config.feature_cache_dir))
    cache_path = feature_cache_dir / f"{_safe_cache_key(cache_key)}.npy"
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape[0] == len(images):
            return cached.astype(np.float32, copy=False)

    features = extract_resnet18_features(
        images,
        model=model,
        device=device,
        batch_size=config.torch_batch_size,
    )
    np.save(cache_path, features.astype(np.float32, copy=False))
    return features


def train_sklearn_cifar_classifier(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    *,
    feature_mode: str,
    random_seed: int,
) -> Any:
    features = extract_image_features(train_images, feature_mode)
    labels = np.asarray(train_labels, dtype=int)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("train_images and train_labels must have matching lengths")

    if feature_mode == "flatten":
        classifier = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_seed)
    else:
        classifier = LogisticRegression(max_iter=500, random_state=random_seed)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    model.fit(features, labels)
    return model


def train_feature_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    random_seed: int,
    num_classes: int,
) -> Any:
    """Train a lightweight sklearn classifier on fixed image features."""
    features = np.asarray(train_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=int)
    if features.ndim != 2:
        raise ValueError(f"train_features must be 2D, got {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("train_features and train_labels must have matching lengths")
    if labels.size == 0:
        raise ValueError("train_labels must be non-empty")
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("train_labels contain class ids outside the expected CIFAR class range")

    if num_classes > 20 or features.shape[0] > 20_000:
        classifier = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_seed)
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=random_seed)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    model.fit(features, labels)
    return model


def stable_softmax(logits: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Compute a row-wise softmax that remains finite for pathological logits."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"logits must be 2D, got {values.shape}")
    axis = int(axis)
    if axis < 0:
        axis += values.ndim
    if axis not in (0, 1):
        raise ValueError("axis must be valid for a 2D logits array")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    values = _finite_logits(values)
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.clip(values - max_values, -745.0, 0.0)
    exp_values = np.exp(shifted)
    sums = np.sum(exp_values, axis=axis, keepdims=True)

    valid = np.isfinite(sums) & (sums > 0.0)
    probabilities = np.divide(
        exp_values,
        sums,
        out=np.zeros_like(exp_values, dtype=np.float64),
        where=valid,
    )
    if axis == 1:
        invalid_rows = ~valid.reshape(-1)
        if invalid_rows.any():
            probabilities[invalid_rows, :] = 1.0 / values.shape[1]
    else:
        invalid_columns = ~valid.reshape(-1)
        if invalid_columns.any():
            probabilities[:, invalid_columns] = 1.0 / values.shape[0]

    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, eps, 1.0)
    renormalizer = np.sum(probabilities, axis=axis, keepdims=True)
    probabilities = np.divide(
        probabilities,
        renormalizer,
        out=np.full_like(probabilities, 1.0 / probabilities.shape[axis], dtype=np.float64),
        where=np.isfinite(renormalizer) & (renormalizer > 0.0),
    )
    return probabilities


def sanitize_probabilities(
    probabilities: np.ndarray,
    *,
    num_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return finite, normalized probability rows with the requested CIFAR class width."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != int(num_classes):
        raise ValueError(f"probabilities must have shape [n, {num_classes}], got {values.shape}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if values.shape[0] == 0:
        return values.astype(np.float64, copy=False)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, np.inf)
    row_sums = values.sum(axis=1, keepdims=True)
    valid_rows = np.isfinite(row_sums) & (row_sums > 0.0)
    sanitized = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=np.float64),
        where=valid_rows,
    )
    invalid_rows = ~valid_rows.reshape(-1)
    if invalid_rows.any():
        sanitized[invalid_rows, :] = 1.0 / int(num_classes)

    sanitized = np.clip(sanitized, eps, 1.0)
    sanitized = sanitized / sanitized.sum(axis=1, keepdims=True)
    if not np.isfinite(sanitized).all():
        sanitized = np.full(values.shape, 1.0 / int(num_classes), dtype=np.float64)
    return sanitized


def predict_proba_with_features(
    model: Any,
    features: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict normalized class probabilities from an arbitrary sklearn feature model."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features), dtype=float)
        raw_logits = _finite_logits(_ensure_2d_logits(decision))
        classes = _classes_for_output(model, raw_logits.shape[1])
        if raw_logits.shape[1] == num_classes and _classes_cover_num_classes(classes, num_classes):
            logits = _align_class_columns(raw_logits, classes, num_classes, fill_value=0.0)
            probabilities = stable_softmax(logits, axis=1)
        elif raw_logits.shape[1] == num_classes and classes is None:
            logits = raw_logits
            probabilities = stable_softmax(logits, axis=1)
        else:
            raw_probabilities = stable_softmax(raw_logits, axis=1)
            aligned_probabilities = _align_class_columns(raw_probabilities, classes, num_classes, fill_value=0.0)
            probabilities = sanitize_probabilities(aligned_probabilities, num_classes=num_classes)
            logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    elif hasattr(model, "predict_proba"):
        raw_probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = _classes_for_output(model, raw_probabilities.shape[1])
        aligned_probabilities = _align_class_columns(raw_probabilities, classes, num_classes, fill_value=0.0)
        probabilities = sanitize_probabilities(aligned_probabilities, num_classes=num_classes)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    else:
        raise ValueError("model must implement predict_proba or decision_function")

    probabilities = sanitize_probabilities(probabilities, num_classes=num_classes)
    logits = _finite_logits(logits)
    if logits.shape != probabilities.shape:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    return probabilities.astype(float), logits.astype(float)


def predict_proba_with_model(
    model: Any,
    images: np.ndarray,
    feature_mode: str,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = extract_image_features(images, feature_mode)
    return predict_proba_with_features(model, features, num_classes)


def compute_cifar_prediction_frame_and_arrays(
    *,
    dataset: str,
    split: str,
    corruption: str,
    severity: int,
    images: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    feature_embedding: np.ndarray,
    start_example_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    logits = np.asarray(logits, dtype=float)
    feature_embedding = np.asarray(feature_embedding, dtype=float)
    if probabilities.ndim != 2 or logits.shape != probabilities.shape:
        raise ValueError("probabilities and logits must be 2D arrays with matching shapes")
    probabilities = sanitize_probabilities(probabilities, num_classes=probabilities.shape[1])
    logits = _finite_logits(logits)
    if len(labels) != probabilities.shape[0] or len(images) != len(labels):
        raise ValueError("images, labels, probabilities, and logits must have matching first dimensions")
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
    context_id = f"{corruption}_s{int(severity)}"
    context_name = f"{corruption} severity {int(severity)}"
    indices = np.arange(start_example_index, start_example_index + len(labels))

    examples = pd.DataFrame(
        {
            "example_id": [f"{dataset}/{split}/{context_id}/{index}" for index in indices],
            "dataset": dataset,
            "split": split,
            "context_id": context_id,
            "context_name": context_name,
            "label": labels.astype(int),
            "pred": pred,
            "correct": correct.astype(bool),
            "confidence": confidence.astype(float),
            "entropy": entropy.astype(float),
            "margin": margin.astype(float),
            "energy": energy.astype(float),
            "severity": int(severity),
            "corruption_type": corruption,
            "domain_id": corruption,
            "group_id": context_id,
            "time_index": int(severity),
            "source": "cifar-c",
        }
    )
    arrays = {
        "logits": logits.astype(float),
        "probabilities": probabilities.astype(float),
        "feature_embedding": feature_embedding.astype(float),
    }
    return examples, arrays


def build_cifar_c_prediction_cache(
    cache_dir: str | Path,
    config: CIFARCConfig,
    *,
    overwrite: bool = False,
) -> Path:
    if config.download:
        download_and_extract_cifar_c(config)
    validate_cifar_c_files(config)

    if config.model_backend == "sklearn_logistic":
        return _build_sklearn_cifar_c_prediction_cache(cache_dir, config, overwrite=overwrite)
    if config.model_backend == "torchvision_resnet18":
        return _build_resnet18_cifar_c_prediction_cache(cache_dir, config, overwrite=overwrite)
    raise ValueError(f"Unsupported CIFAR model_backend: {config.model_backend!r}")


def _build_sklearn_cifar_c_prediction_cache(
    cache_dir: str | Path,
    config: CIFARCConfig,
    *,
    overwrite: bool = False,
) -> Path:
    num_classes = expected_num_classes(config.dataset)
    train_images, train_labels = load_clean_cifar_train(
        config.dataset,
        config.data_root,
        train_subset_size=config.train_subset_size,
        random_seed=config.random_seed,
    )
    model = train_sklearn_cifar_classifier(
        train_images,
        train_labels,
        feature_mode=config.feature_mode,
        random_seed=config.random_seed,
    )

    frames: list[pd.DataFrame] = []
    array_chunks: dict[str, list[np.ndarray]] = {
        "logits": [],
        "probabilities": [],
        "feature_embedding": [],
    }
    start_index = 0
    for corruption in config.corruptions:
        for severity in config.severities:
            images, labels = load_cifar_c_context(config, corruption, severity)
            probabilities, logits = predict_proba_with_model(model, images, config.feature_mode, num_classes)
            features = extract_image_features(images, config.feature_mode)
            frame, arrays = compute_cifar_prediction_frame_and_arrays(
                dataset=config.dataset,
                split="eval",
                corruption=corruption,
                severity=severity,
                images=images,
                labels=labels,
                probabilities=probabilities,
                logits=logits,
                feature_embedding=features,
                start_example_index=start_index,
            )
            frames.append(frame)
            for key, value in arrays.items():
                array_chunks[key].append(value)
            start_index += len(frame)

    examples = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate(values, axis=0) for key, values in array_chunks.items()}
    metadata = _build_cache_metadata(config, examples, num_classes)
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _build_resnet18_cifar_c_prediction_cache(
    cache_dir: str | Path,
    config: CIFARCConfig,
    *,
    overwrite: bool = False,
) -> Path:
    num_classes = expected_num_classes(config.dataset)
    feature_extractor, resolved_device = build_resnet18_feature_extractor(
        pretrained=config.torch_pretrained,
        device=config.torch_device,
    )
    train_images, train_labels = load_clean_cifar_train(
        config.dataset,
        config.data_root,
        train_subset_size=config.train_subset_size,
        random_seed=config.random_seed,
    )
    train_features = _extract_resnet18_features_maybe_cached(
        train_images,
        config=config,
        model=feature_extractor,
        device=resolved_device,
        cache_key=(
            f"{config.dataset}_train_resnet18_pretrained{int(config.torch_pretrained)}_"
            f"seed{config.random_seed}_n{len(train_images)}"
        ),
    )
    feature_classifier = train_feature_classifier(
        train_features,
        train_labels,
        random_seed=config.random_seed,
        num_classes=num_classes,
    )

    frames: list[pd.DataFrame] = []
    array_chunks: dict[str, list[np.ndarray]] = {
        "logits": [],
        "probabilities": [],
        "feature_embedding": [],
    }
    start_index = 0
    for corruption in config.corruptions:
        for severity in config.severities:
            images, labels = load_cifar_c_context(config, corruption, severity)
            features = _extract_resnet18_features_maybe_cached(
                images,
                config=config,
                model=feature_extractor,
                device=resolved_device,
                cache_key=(
                    f"{config.dataset}_{corruption}_s{severity}_resnet18_"
                    f"pretrained{int(config.torch_pretrained)}_n{len(images)}"
                ),
            )
            probabilities, logits = predict_proba_with_features(feature_classifier, features, num_classes)
            frame, arrays = compute_cifar_prediction_frame_and_arrays(
                dataset=config.dataset,
                split="eval",
                corruption=corruption,
                severity=severity,
                images=images,
                labels=labels,
                probabilities=probabilities,
                logits=logits,
                feature_embedding=features,
                start_example_index=start_index,
            )
            frames.append(frame)
            for key, value in arrays.items():
                array_chunks[key].append(value)
            start_index += len(frame)

    examples = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate(values, axis=0) for key, values in array_chunks.items()}
    metadata = _build_cache_metadata(
        config,
        examples,
        num_classes,
        resolved_torch_device=resolved_device,
    )
    return save_prediction_cache(cache_dir, examples, arrays, metadata, overwrite=overwrite)


def _build_cache_metadata(
    config: CIFARCConfig,
    examples: pd.DataFrame,
    num_classes: int,
    *,
    resolved_torch_device: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "dataset": config.dataset,
        "split": "eval",
        "num_examples": len(examples),
        "num_classes": num_classes,
        "context_fields": ["context_id", "corruption_type", "severity"],
        "cifar_c_config": _config_to_jsonable(config),
        "corruptions": list(config.corruptions),
        "severities": list(config.severities),
        "model_backend": config.model_backend,
        "feature_mode": config.feature_mode,
        "torch_batch_size": config.torch_batch_size,
        "torch_device": resolved_torch_device or config.torch_device,
        "torch_device_requested": config.torch_device,
        "torch_pretrained": config.torch_pretrained,
    }
    if config.feature_cache_dir is not None:
        metadata["feature_cache_dir"] = str(config.feature_cache_dir)
    return metadata


def _download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = 100.0 * downloaded / total_bytes
                print(f"Downloading {destination.name}: {percent:5.1f}%", end="\r")
        if total_bytes:
            print(f"Downloading {destination.name}: 100.0%")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    archive.extractall(destination)


def _finite_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return np.clip(values, -1.0e6, 1.0e6)


def _ensure_2d_logits(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        return np.column_stack([-decision, decision])
    return decision


def _softmax(logits: np.ndarray) -> np.ndarray:
    return stable_softmax(logits, axis=1)


def _classes_for_output(model: Any, output_width: int) -> np.ndarray | None:
    if not hasattr(model, "classes_"):
        return None
    try:
        classes = np.asarray(getattr(model, "classes_"), dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(classes) != int(output_width):
        return None
    return classes


def _classes_cover_num_classes(classes: np.ndarray | None, num_classes: int) -> bool:
    if classes is None:
        return False
    expected = np.arange(int(num_classes), dtype=int)
    return len(classes) == int(num_classes) and np.array_equal(np.sort(classes), expected)


def _align_class_columns(
    values: np.ndarray,
    classes: np.ndarray | None,
    num_classes: int,
    *,
    fill_value: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got {values.shape}")
    if values.shape[1] == int(num_classes) and classes is None:
        return values

    aligned = np.full((values.shape[0], int(num_classes)), float(fill_value), dtype=np.float64)
    source_classes = classes if classes is not None else np.arange(values.shape[1], dtype=int)
    for source_col, class_id in enumerate(source_classes):
        if source_col >= values.shape[1]:
            break
        class_index = int(class_id)
        if 0 <= class_index < int(num_classes):
            aligned[:, class_index] = values[:, source_col]
    return aligned


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
    return np.squeeze(stable, axis=axis)


def _safe_cache_key(value: str) -> str:
    return "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)


def _config_to_jsonable(config: CIFARCConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_root"] = str(payload["data_root"])
    if payload["feature_cache_dir"] is not None:
        payload["feature_cache_dir"] = str(payload["feature_cache_dir"])
    payload["corruptions"] = list(payload["corruptions"])
    payload["severities"] = list(payload["severities"])
    return payload
