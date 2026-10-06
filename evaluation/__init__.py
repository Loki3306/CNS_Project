"""
Evaluation package for computing metrics, confusion matrices, ROC/PR curves, and error analysis.
"""

from .metrics import calculate_classification_metrics
from .confusion import plot_confusion_matrix
from .curves import plot_roc_comparison, plot_pr_comparison
from .error_analysis import perform_error_analysis

__all__ = [
    "calculate_classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_comparison",
    "plot_pr_comparison",
    "perform_error_analysis",
]
