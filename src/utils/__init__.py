"""Utilities package."""
from .conformal_prediction import (
    compute_qhat,
    evaluate_conformal_prediction,
    get_probs,
    compute_qhat_aps,
    evaluate_aps,
    compute_qhat_rmcp,
    evaluate_rmcp,
    compute_qhat_rcs,
    evaluate_rcs,
)
from .training import train_round, eval_acc
from .visualization import (
    plot_accuracy_vs_samples,
    plot_accuracy_gap,
    plot_cp_coverage,
    plot_cp_set_size,
    plot_all_metrics,
    load_results,
    print_summary_table,
    compute_aulc,
    compute_accuracy_gap,
    COLORS,
    LABELS
)

__all__ = [
    'compute_qhat',
    'evaluate_conformal_prediction',
    'get_probs',
    'compute_qhat_aps',
    'evaluate_aps',
    'compute_qhat_rmcp',
    'evaluate_rmcp',
    'compute_qhat_rcs',
    'evaluate_rcs',
    'train_round',
    'eval_acc',
    'plot_accuracy_vs_samples',
    'plot_accuracy_gap',
    'plot_cp_coverage',
    'plot_cp_set_size',
    'plot_all_metrics',
    'load_results',
    'print_summary_table',
    'compute_aulc',
    'compute_accuracy_gap',
    'COLORS',
    'LABELS',
]
