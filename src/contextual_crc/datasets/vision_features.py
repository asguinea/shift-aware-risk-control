from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ALLOWED_TORCH_DEVICES = ("auto", "cpu", "cuda")


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve an optional torch device name for vision feature extraction."""
    device = str(device).lower()
    if device not in _ALLOWED_TORCH_DEVICES:
        raise ValueError(f"device must be one of {_ALLOWED_TORCH_DEVICES}, got {device!r}")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The torchvision ResNet-18 vision backend requires torch and torchvision. "
            "Install compatible torch/torchvision packages, or use a non-vision backend."
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
            "The torchvision ResNet-18 vision backend requires torch and torchvision. "
            "Install compatible torch/torchvision packages."
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
                "Ensure the weights are cached or network access is available, "
                "or set torch_pretrained=false in the dataset config."
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
    batch_size: int = 64,
) -> np.ndarray:
    """Extract ImageNet-normalized ResNet-18 features from HWC RGB image arrays."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "The torchvision ResNet-18 vision backend requires torch. "
            "Install compatible torch/torchvision packages."
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
            feature_chunks.append(features.detach().cpu().numpy().astype(np.float32, copy=False))

    if not feature_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)


def train_feature_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    random_seed: int,
    num_classes: int,
) -> Any:
    """Train a lightweight sklearn classifier on fixed vision features."""
    features = np.asarray(train_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=int)
    if features.ndim != 2:
        raise ValueError(f"train_features must be 2D, got {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("train_features and train_labels must have matching lengths")
    if labels.size == 0:
        raise ValueError("train_labels must be non-empty")
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("train_labels contain class ids outside the expected class range")

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
    """Compute a softmax that remains finite for pathological logits."""
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

    values = finite_logits(values)
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
    probabilities = np.divide(probabilities, renormalizer, out=np.zeros_like(probabilities), where=renormalizer > 0.0)
    return probabilities.astype(np.float64, copy=False)


def sanitize_probabilities(
    probabilities: np.ndarray,
    *,
    num_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize classifier probabilities into finite valid class distributions."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != int(num_classes):
        raise ValueError(f"probabilities must have shape [n, {num_classes}], got {values.shape}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    row_sums = np.sum(values, axis=1, keepdims=True)
    invalid_rows = ~np.isfinite(row_sums).reshape(-1) | (row_sums.reshape(-1) <= 0.0)
    if invalid_rows.any():
        values[invalid_rows, :] = 1.0 / int(num_classes)
        row_sums = np.sum(values, axis=1, keepdims=True)
    values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0.0)
    values = np.clip(values, eps, 1.0)
    values = values / np.sum(values, axis=1, keepdims=True)
    return values.astype(np.float64, copy=False)


def predict_proba_with_features(
    model: Any,
    features: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict finite probabilities/logits from a sklearn feature classifier."""
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError(f"features must be 2D, got {feature_array.shape}")

    if hasattr(model, "decision_function"):
        raw_logits = np.asarray(model.decision_function(feature_array), dtype=np.float64)
        if raw_logits.ndim == 1 and int(num_classes) == 2:
            raw_logits = np.column_stack([-0.5 * raw_logits, 0.5 * raw_logits])
        classes = classes_for_output(model, raw_logits.shape[1] if raw_logits.ndim == 2 else int(num_classes))
        logits = align_class_columns(raw_logits, classes, int(num_classes), fill_value=-1e9)
        probabilities = stable_softmax(logits, axis=1)
    elif hasattr(model, "predict_proba"):
        raw_probabilities = np.asarray(model.predict_proba(feature_array), dtype=np.float64)
        classes = classes_for_output(model, raw_probabilities.shape[1])
        probabilities = align_class_columns(raw_probabilities, classes, int(num_classes), fill_value=0.0)
        probabilities = sanitize_probabilities(probabilities, num_classes=int(num_classes))
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    else:
        raise ValueError("model must implement decision_function or predict_proba")

    probabilities = sanitize_probabilities(probabilities, num_classes=int(num_classes))
    logits = finite_logits(logits)
    if logits.shape != probabilities.shape:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    return probabilities.astype(float), logits.astype(float)


def finite_logits(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e9, neginf=-1e9)


def classes_for_output(model: Any, output_width: int) -> np.ndarray | None:
    if not hasattr(model, "classes_"):
        return None
    try:
        classes = np.asarray(getattr(model, "classes_"), dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(classes) != int(output_width):
        return None
    return classes


def align_class_columns(
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
