from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from contextual_crc.risk_control.policies import candidate_thresholds
from contextual_crc.risk_control.source_certificates import (
    accepted_error_risks,
    coverage_at_thresholds,
)


@dataclass(frozen=True)
class TransferPenalty:
    family: str
    intercept: float = 0.0
    slope: float = 0.0
    bin_edges: tuple[float, ...] = ()
    bin_values: tuple[float, ...] = ()
    fit_status: str = "fitted"
    interpretation: str = "diagnostic"

    def evaluate(self, distance: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(distance, dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("distance must be finite and nonnegative")
        if self.family == "G4_piecewise_bins":
            if not self.bin_values:
                raise ValueError("piecewise penalty has no bin values")
            edges = np.asarray(self.bin_edges, dtype=float)
            penalties = np.asarray(self.bin_values, dtype=float)
            indices = np.searchsorted(edges, values, side="right")
            indices = np.clip(indices, 0, len(penalties) - 1)
            output = penalties[indices]
        else:
            output = self.intercept + self.slope * values
        output = np.clip(output, 0.0, 1.0)
        return float(output) if values.ndim == 0 else output


def observable_transfer_residual_upper(
    source_upper: Sequence[float],
    target_upper: Sequence[float],
) -> np.ndarray:
    """Positive observable upper bound [U_b(theta)-U_a(theta)]_+."""
    source = np.asarray(source_upper, dtype=float)
    target = np.asarray(target_upper, dtype=float)
    if source.shape != target.shape:
        raise ValueError("source_upper and target_upper must have matching shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("risk envelopes must be finite")
    return np.maximum(0.0, target - source)


def predeclared_penalty_grids(
    distances: Sequence[float],
    *,
    n_slopes: int = 201,
    n_intercepts: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct geometry-only grids before residual/risk values are inspected."""
    values = np.asarray(distances, dtype=float).reshape(-1)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        slope_grid = np.asarray([0.0])
    else:
        slope_cap = 1.0 / float(np.min(positive))
        slope_grid = np.linspace(0.0, slope_cap, int(n_slopes))
    intercept_grid = np.linspace(0.0, 1.0, int(n_intercepts))
    return slope_grid, intercept_grid


def fit_transfer_penalty(
    pair_constraints: pd.DataFrame,
    family: str,
    *,
    slope_grid: Sequence[float] | None = None,
    intercept_grid: Sequence[float] | None = None,
    bin_edges: Sequence[float] | None = None,
) -> TransferPenalty:
    required = {"distance", "max_residual_upper"}
    missing = sorted(required - set(pair_constraints.columns))
    if missing:
        raise ValueError(f"pair constraints missing columns: {missing}")
    pairs = pair_constraints.copy()
    distances = pd.to_numeric(pairs["distance"], errors="coerce").to_numpy(dtype=float)
    residuals = pd.to_numeric(
        pairs["max_residual_upper"], errors="coerce"
    ).to_numpy(dtype=float)
    if len(pairs) == 0 or not np.isfinite(distances).all() or not np.isfinite(residuals).all():
        return TransferPenalty(family=family, fit_status="insufficient_pairs")

    interpretation = "simultaneous observed-pair residual envelope"
    if family == "G0_no_penalty":
        return TransferPenalty(family=family, interpretation="diagnostic zero penalty")
    if family == "G1_global_residual":
        return TransferPenalty(
            family=family,
            intercept=float(np.max(residuals)),
            interpretation=interpretation,
        )
    if family == "G2_pure_lipschitz":
        positive = distances > 0.0
        if bool(((~positive) & (residuals > 1e-12)).any()):
            return TransferPenalty(family=family, fit_status="infeasible_zero_distance")
        slope = float(np.max(residuals[positive] / distances[positive])) if positive.any() else 0.0
        return TransferPenalty(family=family, slope=slope, interpretation=interpretation)
    if family == "G3_lipschitz_intercept":
        if slope_grid is None or intercept_grid is None:
            raise ValueError("G3 requires predeclared slope and intercept grids")
        slopes = np.asarray(slope_grid, dtype=float)
        intercepts = np.asarray(intercept_grid, dtype=float)
        feasible_candidates: list[tuple[float, float, float]] = []
        mean_distance = float(np.mean(distances))
        for intercept in intercepts:
            required_slopes = np.where(
                distances > 0.0,
                np.maximum(0.0, residuals - intercept) / np.maximum(distances, 1e-300),
                np.where(residuals <= intercept + 1e-12, 0.0, np.inf),
            )
            required = float(np.max(required_slopes))
            valid = slopes[slopes + 1e-12 >= required]
            if valid.size:
                slope = float(valid[0])
                mean_penalty = float(intercept + slope * mean_distance)
                feasible_candidates.append((mean_penalty, float(intercept), slope))
        if not feasible_candidates:
            return TransferPenalty(family=family, fit_status="grid_infeasible")
        _, intercept, slope = min(feasible_candidates, key=lambda row: (row[0], row[1], row[2]))
        return TransferPenalty(
            family=family,
            intercept=intercept,
            slope=slope,
            interpretation=interpretation + "; finite geometry-predeclared grid",
        )
    if family == "G4_piecewise_bins":
        if bin_edges is None:
            raise ValueError("G4 requires predeclared distance-bin edges")
        edges = np.asarray(bin_edges, dtype=float).reshape(-1)
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0):
            raise ValueError("bin edges must be finite and strictly increasing")
        indices = np.searchsorted(edges, distances, side="right")
        raw_values = []
        for bin_index in range(len(edges) + 1):
            in_bin = residuals[indices == bin_index]
            raw_values.append(float(np.max(in_bin)) if len(in_bin) else 0.0)
        monotone_values = np.maximum.accumulate(raw_values)
        return TransferPenalty(
            family=family,
            bin_edges=tuple(float(value) for value in edges),
            bin_values=tuple(float(value) for value in monotone_values),
            interpretation=interpretation + "; predeclared monotone distance bins",
        )
    raise ValueError(f"unknown transfer-penalty family: {family}")


def transfer_penalty_dominates_constraints(
    penalty: TransferPenalty,
    pair_constraints: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if pair_constraints.empty or penalty.fit_status != "fitted":
        return False
    predicted = np.asarray(penalty.evaluate(pair_constraints["distance"].to_numpy()), dtype=float)
    required = pair_constraints["max_residual_upper"].to_numpy(dtype=float)
    return bool((predicted + tolerance >= required).all())


def select_gamma_feasible_threshold(
    deployment_scores: np.ndarray,
    source_scores: np.ndarray,
    source_error_scores: np.ndarray,
    *,
    source_epsilon: float,
    gamma: float,
    risk_budget: float,
) -> dict[str, Any]:
    """Select using calibration labels and deployment scores, never deployment labels."""
    deployment = np.asarray(deployment_scores, dtype=float).reshape(-1)
    source = np.asarray(source_scores, dtype=float).reshape(-1)
    if deployment.size == 0 or source.size == 0:
        raise ValueError("deployment and source scores must be non-empty")
    if not np.isfinite(deployment).all() or not np.isfinite(source).all():
        raise ValueError("deployment and source scores must be finite")
    thresholds = candidate_thresholds(
        np.concatenate([deployment, source]), accept_if="greater_equal", include_extremes=True
    )
    source_risks = accepted_error_risks(source_error_scores, thresholds)
    source_ucbs = np.clip(source_risks + float(source_epsilon), 0.0, 1.0)
    final_ucbs = np.clip(source_ucbs + float(gamma), 0.0, 1.0)
    coverages = coverage_at_thresholds(deployment, thresholds)
    accepted_counts = np.rint(coverages * len(deployment)).astype(int)
    feasible = (accepted_counts > 0) & (final_ucbs <= float(risk_budget) + 1e-12)
    if feasible.any():
        choices = np.flatnonzero(feasible)
        selected = min(
            choices,
            key=lambda index: (
                -accepted_counts[index],
                final_ucbs[index],
                thresholds[index],
            ),
        )
        forced_none = False
        operational_bound = float(final_ucbs[selected])
    else:
        none = np.flatnonzero(accepted_counts == 0)
        if not len(none):
            raise RuntimeError("candidate construction omitted accept-none")
        selected = int(none[0])
        forced_none = True
        operational_bound = 0.0
    return {
        "threshold": float(thresholds[selected]),
        "source_risk_estimate": float(source_risks[selected]),
        "source_ucb": float(source_ucbs[selected]),
        "gamma": float(gamma),
        "raw_final_ucb": float(final_ucbs[selected]),
        "operational_bound": operational_bound,
        "coverage_from_scores": float(coverages[selected]),
        "n_accepted_from_scores": int(accepted_counts[selected]),
        "nontrivial_feasible": not forced_none,
        "forced_accept_none": forced_none,
        "num_candidates": len(thresholds),
    }
