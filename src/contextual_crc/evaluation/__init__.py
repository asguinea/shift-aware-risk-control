from contextual_crc.evaluation.diagnostics import (
    bound_calibration_diagnostics,
    calibration_gap_by_group,
    context_smoothness_diagnostics,
    context_smoothness_pairwise_frame,
    knn_risk_prediction_diagnostics,
    label_efficiency_summary,
    method_comparison_table,
    window_diagnostics_summary,
)
from contextual_crc.evaluation.metrics import (
    aggregate_by_group,
    bound_underestimation_rate,
    bound_violation_rate,
    coverage_risk_correlation,
    max_violation_gap,
    mean_bound_gap,
    mean_excess_risk_over_bound,
    safe_mean,
    safe_std,
    summarize_window_results,
    temporal_variability,
    violation_gap,
    violation_rate,
)

__all__ = [
    "run_cifar_c_experiment",
    "run_cifar_online_trace",
    "run_officehome_experiment",
    "run_synthetic_experiment",
    "run_wilds_experiment",
    "aggregate_by_group",
    "bound_calibration_diagnostics",
    "bound_underestimation_rate",
    "bound_violation_rate",
    "calibration_gap_by_group",
    "context_smoothness_diagnostics",
    "context_smoothness_pairwise_frame",
    "coverage_risk_correlation",
    "knn_risk_prediction_diagnostics",
    "label_efficiency_summary",
    "max_violation_gap",
    "mean_bound_gap",
    "mean_excess_risk_over_bound",
    "method_comparison_table",
    "safe_mean",
    "safe_std",
    "summarize_window_results",
    "temporal_variability",
    "violation_gap",
    "violation_rate",
    "window_diagnostics_summary",
]

from contextual_crc.evaluation.synthetic_experiment import run_synthetic_experiment

from contextual_crc.evaluation.cifar_c_experiment import run_cifar_c_experiment
from contextual_crc.evaluation.online_trace import run_cifar_online_trace
from contextual_crc.evaluation.officehome_experiment import run_officehome_experiment
from contextual_crc.evaluation.wilds_experiment import run_wilds_experiment
