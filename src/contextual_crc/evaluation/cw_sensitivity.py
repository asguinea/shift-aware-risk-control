from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from contextual_crc.cache.io import load_prediction_cache
from contextual_crc.config.loader import load_config
from contextual_crc.evaluation.metrics import (
    bound_underestimation_rate,
    mean_bound_gap,
    violation_rate,
)
from contextual_crc.evaluation.synthetic_experiment import parse_cwcrc_config, parse_risk_control_config
from contextual_crc.risk_control.cw_crc import evaluate_cwcrc_windows
from contextual_crc.signatures.io import load_windows
from contextual_crc.utils.paths import ensure_dir


DEFAULT_ESS_GRID = (3.0, 6.0, 12.0, 24.0)
DEFAULT_MARGIN_GRID = (0.0, 0.03, 0.06, 0.10)

SENSITIVITY_COLUMNS = (
    "dataset",
    "dataset_display",
    "run_dir",
    "sweep_type",
    "sweep_value",
    "target_effective_sample_size",
    "bound_margin",
    "bound_multiplier",
    "is_current_setting",
    "risk",
    "coverage",
    "bound_underestimation",
    "mean_bound_gap",
    "violation_rate",
    "mean_effective_sample_size",
    "median_effective_sample_size",
    "mean_selected_threshold",
    "n_windows",
    "notes",
)

DATASET_DISPLAY_NAMES = {
    "cifar10c": "CIFAR-10-C",
    "cifar100c": "CIFAR-100-C",
    "synthetic_contextual_shift": "Synthetic",
    "synthetic_stronger_contextual_shift": "Synthetic",
    "officehome": "OfficeHome",
    "wilds_camelyon17": "Camelyon17-WILDS",
    "wilds_fmow": "FMoW-WILDS",
}


@dataclass(frozen=True)
class CWSensitivitySetting:
    sweep_type: str
    sweep_value: float
    target_effective_sample_size: float | None
    bound_margin: float
    bound_multiplier: float
    is_current_setting: bool = False


@dataclass(frozen=True)
class CWSensitivityRunData:
    dataset: str
    run_dir: Path
    config: dict[str, Any]
    examples: pd.DataFrame
    calibration_examples: pd.DataFrame
    calibration_windows: pd.DataFrame
    calibration_embedding_frame: pd.DataFrame
    evaluation_windows: pd.DataFrame
    evaluation_embedding_frame: pd.DataFrame


def build_compact_cw_sensitivity_grid(
    current_cw_config: Mapping[str, Any],
    *,
    ess_grid: Sequence[float] = DEFAULT_ESS_GRID,
    margin_grid: Sequence[float] = DEFAULT_MARGIN_GRID,
) -> list[CWSensitivitySetting]:
    """Build the compact appendix grid: ESS sweep plus margin sweep."""
    current_ess = _optional_float(current_cw_config.get("target_effective_sample_size", 6.0))
    current_margin = float(current_cw_config.get("bound_margin", 0.0))
    current_multiplier = float(current_cw_config.get("bound_multiplier", 1.0))
    settings: list[CWSensitivitySetting] = []
    for ess in ess_grid:
        ess_value = float(ess)
        settings.append(
            CWSensitivitySetting(
                sweep_type="target_effective_sample_size",
                sweep_value=ess_value,
                target_effective_sample_size=ess_value,
                bound_margin=current_margin,
                bound_multiplier=current_multiplier,
                is_current_setting=_same_optional_float(ess_value, current_ess),
            )
        )
    for margin in margin_grid:
        margin_value = float(margin)
        settings.append(
            CWSensitivitySetting(
                sweep_type="bound_margin",
                sweep_value=margin_value,
                target_effective_sample_size=current_ess,
                bound_margin=margin_value,
                bound_multiplier=current_multiplier,
                is_current_setting=np.isclose(margin_value, current_margin, atol=1e-12, rtol=0.0),
            )
        )
    return settings


def summarize_cw_sensitivity_windows(
    windows: pd.DataFrame,
    *,
    risk_budget: float = 0.10,
) -> dict[str, float]:
    """Aggregate per-window CW sensitivity outputs into appendix metrics."""
    if windows.empty:
        return {
            "risk": float("nan"),
            "coverage": float("nan"),
            "bound_underestimation": float("nan"),
            "mean_bound_gap": float("nan"),
            "violation_rate": float("nan"),
            "mean_effective_sample_size": float("nan"),
            "median_effective_sample_size": float("nan"),
            "mean_selected_threshold": float("nan"),
            "n_windows": 0.0,
        }
    risk_col = "realized_risk" if "realized_risk" in windows.columns else "mean_loss"
    bound_col = "weighted_risk_upper_bound"
    risk = pd.to_numeric(windows.get(risk_col), errors="coerce")
    coverage = pd.to_numeric(windows.get("coverage"), errors="coerce")
    bounds = pd.to_numeric(windows.get(bound_col), errors="coerce") if bound_col in windows.columns else None
    ess = pd.to_numeric(windows.get("effective_sample_size"), errors="coerce") if "effective_sample_size" in windows.columns else None
    thresholds = pd.to_numeric(windows.get("threshold"), errors="coerce") if "threshold" in windows.columns else None

    return {
        "risk": _safe_mean(risk),
        "coverage": _safe_mean(coverage),
        "bound_underestimation": (
            bound_underestimation_rate(risk, bounds)
            if bounds is not None
            else float("nan")
        ),
        "mean_bound_gap": mean_bound_gap(risk, bounds) if bounds is not None else float("nan"),
        "violation_rate": violation_rate(risk, risk_budget),
        "mean_effective_sample_size": _safe_mean(ess) if ess is not None else float("nan"),
        "median_effective_sample_size": _safe_median(ess) if ess is not None else float("nan"),
        "mean_selected_threshold": _safe_mean(thresholds) if thresholds is not None else float("nan"),
        "n_windows": float(len(windows)),
    }


def load_cw_sensitivity_run_data(run_dir: str | Path) -> CWSensitivityRunData:
    """Load cached examples and saved intermediates needed to rerun only CW-CRC windows."""
    run_path = Path(run_dir)
    config = load_config(run_path / "configs" / "resolved_config.yaml")
    manifest = _read_json(run_path / "results" / "run_manifest.json")
    dataset = str(manifest.get("dataset") or manifest.get("schema_dataset") or run_path.name)
    examples, _, _ = load_prediction_cache(Path(config["paths"]["cache_dir"]))
    windows = load_windows(run_path / "intermediate" / "windows.parquet")
    embeddings = pd.read_parquet(run_path / "intermediate" / "embeddings.parquet", engine="pyarrow")

    calibration_contexts = _contexts_from_manifest_or_config(
        manifest,
        config,
        key="calibration_contexts",
        examples=examples,
    )
    evaluation_contexts = _contexts_from_manifest_or_config(
        manifest,
        config,
        key="evaluation_contexts",
        examples=examples,
    )

    calibration_examples, calibration_windows = _subset_examples_and_remap_windows(
        examples,
        windows,
        calibration_contexts,
    )
    evaluation_windows = windows[windows["context_id"].astype(str).isin(evaluation_contexts)].reset_index(drop=True)
    calibration_ids = set(calibration_windows["window_id"])
    evaluation_ids = set(evaluation_windows["window_id"])
    calibration_embedding_frame = embeddings[embeddings["window_id"].isin(calibration_ids)].reset_index(drop=True)
    evaluation_embedding_frame = embeddings[embeddings["window_id"].isin(evaluation_ids)].reset_index(drop=True)
    return CWSensitivityRunData(
        dataset=dataset,
        run_dir=run_path,
        config=config,
        examples=examples,
        calibration_examples=calibration_examples,
        calibration_windows=calibration_windows,
        calibration_embedding_frame=calibration_embedding_frame,
        evaluation_windows=evaluation_windows,
        evaluation_embedding_frame=evaluation_embedding_frame,
    )


def evaluate_cw_sensitivity_setting(
    run_data: CWSensitivityRunData,
    setting: CWSensitivitySetting,
) -> dict[str, Any]:
    """Evaluate one CW-CRC sensitivity setting without modifying saved run outputs."""
    config = _config_with_setting(run_data.config, setting)
    risk_config = parse_risk_control_config(config.get("risk_control", {}))
    cw_config = parse_cwcrc_config(config.get("cw_crc", {}), risk_config)
    windows = evaluate_cwcrc_windows(
        run_data.examples,
        run_data.calibration_examples,
        run_data.calibration_windows,
        run_data.calibration_embedding_frame,
        run_data.evaluation_windows,
        run_data.evaluation_embedding_frame,
        cw_config,
    )
    summary = summarize_cw_sensitivity_windows(windows, risk_budget=risk_config.risk_budget)
    return {
        "dataset": run_data.dataset,
        "dataset_display": display_dataset(run_data.dataset),
        "run_dir": str(run_data.run_dir),
        "sweep_type": setting.sweep_type,
        "sweep_value": setting.sweep_value,
        "target_effective_sample_size": (
            float(setting.target_effective_sample_size)
            if setting.target_effective_sample_size is not None
            else np.nan
        ),
        "bound_margin": setting.bound_margin,
        "bound_multiplier": setting.bound_multiplier,
        "is_current_setting": setting.is_current_setting,
        **summary,
        "notes": "appendix sensitivity; base model/cache/intermediates reused",
    }


def run_cw_sensitivity(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    ess_grid: Sequence[float] = DEFAULT_ESS_GRID,
    margin_grid: Sequence[float] = DEFAULT_MARGIN_GRID,
) -> Path:
    """Run compact CW-CRC sensitivity over existing saved experiment runs."""
    output_path = ensure_dir(output_dir)
    csv_path = output_path / "cw_sensitivity.csv"
    rows = _load_existing_sensitivity_rows(csv_path)
    existing_keys = {_sensitivity_key(row) for row in rows}
    manifest_runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_data = load_cw_sensitivity_run_data(run_dir)
        settings = build_compact_cw_sensitivity_grid(
            run_data.config.get("cw_crc", {}),
            ess_grid=ess_grid,
            margin_grid=margin_grid,
        )
        manifest_runs.append(
            {
                "dataset": run_data.dataset,
                "run_dir": str(run_data.run_dir),
                "num_settings": len(settings),
                "n_evaluation_windows": int(len(run_data.evaluation_windows)),
            }
        )
        for setting in settings:
            key = _sensitivity_key(
                {
                    "run_dir": str(run_data.run_dir),
                    "sweep_type": setting.sweep_type,
                    "sweep_value": setting.sweep_value,
                    "target_effective_sample_size": setting.target_effective_sample_size,
                    "bound_margin": setting.bound_margin,
                    "bound_multiplier": setting.bound_multiplier,
                }
            )
            if key in existing_keys:
                continue
            print(
                "CW sensitivity:",
                run_data.dataset,
                setting.sweep_type,
                setting.sweep_value,
                flush=True,
            )
            rows.append(evaluate_cw_sensitivity_setting(run_data, setting))
            existing_keys.add(key)
            _write_sensitivity_csv(rows, csv_path)
    _write_sensitivity_csv(rows, csv_path)
    _write_json(
        {
            "run_dirs": [str(path) for path in run_dirs],
            "output_csv": str(csv_path),
            "ess_grid": [float(value) for value in ess_grid],
            "margin_grid": [float(value) for value in margin_grid],
            "runs": manifest_runs,
            "notes": "CW-CRC appendix sensitivity; no main run outputs modified.",
        },
        output_path / "cw_sensitivity_manifest.json",
    )
    return output_path


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset).replace("_", " "))


def _config_with_setting(config: Mapping[str, Any], setting: CWSensitivitySetting) -> dict[str, Any]:
    copied = yaml.safe_load(yaml.safe_dump(dict(config), sort_keys=False))
    cw_raw = copied.setdefault("cw_crc", {})
    cw_raw["target_effective_sample_size"] = setting.target_effective_sample_size
    cw_raw["bandwidth"] = None
    cw_raw["bound_margin"] = float(setting.bound_margin)
    cw_raw["bound_multiplier"] = float(setting.bound_multiplier)
    return copied


def _subset_examples_and_remap_windows(
    examples: pd.DataFrame,
    windows: pd.DataFrame,
    contexts: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_set = {str(value) for value in contexts}
    mask = examples["context_id"].astype(str).isin(context_set)
    original_positions = np.flatnonzero(mask.to_numpy())
    original_to_local = {int(original): local for local, original in enumerate(original_positions)}
    subset_examples = examples.iloc[original_positions].reset_index(drop=True)
    subset_windows = windows[windows["context_id"].astype(str).isin(context_set)].copy().reset_index(drop=True)
    remapped_indices = []
    for indices in subset_windows["example_indices"]:
        remapped_indices.append([original_to_local[int(index)] for index in indices])
    subset_windows["example_indices"] = remapped_indices
    return subset_examples, subset_windows


def _contexts_from_manifest_or_config(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    key: str,
    examples: pd.DataFrame,
) -> list[str]:
    manifest_value = manifest.get(key)
    if isinstance(manifest_value, list) and manifest_value:
        return [str(value) for value in manifest_value]
    raw = config.get("splits", {}).get(key, "all")
    if raw == "all" or raw == ["all"]:
        return [str(value) for value in examples["context_id"].drop_duplicates().tolist()]
    if raw == "auto":
        available = sorted(str(value) for value in examples["context_id"].drop_duplicates().tolist())
        requested = int(config.get("splits", {}).get("num_auto_calibration_contexts", 2))
        return available[: min(requested, len(available))]
    return [str(value) for value in raw]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_existing_sensitivity_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def _write_sensitivity_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    for column in SENSITIVITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.loc[:, list(SENSITIVITY_COLUMNS)]
    frame.to_csv(path, index=False)


def _sensitivity_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("run_dir")),
        str(row.get("sweep_type")),
        _key_float(row.get("sweep_value")),
        _key_float(row.get("target_effective_sample_size")),
        _key_float(row.get("bound_margin")),
        _key_float(row.get("bound_multiplier")),
    )


def _safe_mean(values: Any) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.mean(numeric)) if numeric.size else float("nan")


def _safe_median(values: Any) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if numeric.size else float("nan")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _key_float(value: Any) -> str:
    numeric = _optional_float(value)
    return "nan" if numeric is None else f"{numeric:.12g}"


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0.0))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
