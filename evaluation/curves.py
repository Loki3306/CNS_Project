"""
ROC and Precision-Recall Curves Module.
Generates multi-model overlaid evaluation curves with AUC annotations.
"""

from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from config import PLOT_DPI, PLOTS_DIR


def plot_roc_comparison(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path = PLOTS_DIR / "roc_curve_comparison.png"
) -> Path:
    """
    Plot overlaid ROC curves for all models.
    
    models_predictions format:
      {
         "Model Name": (y_true, y_prob)
      }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(
            fpr,
            tpr,
            lw=2.0,
            color=color,
            label=f"{model_name} (AUC = {roc_auc:.4f})"
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pr_comparison(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path = PLOTS_DIR / "pr_curve_comparison.png"
) -> Path:
    """
    Plot overlaid Precision-Recall curves for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Baseline positive prevalence line from first model
    first_y_true = next(iter(models_predictions.values()))[0]
    prevalence = np.mean(first_y_true)
    ax.plot([0, 1], [prevalence, prevalence], "k--", lw=1.2,
            label=f"Baseline Prevalence (AP = {prevalence:.3f})")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        color = colors[idx % len(colors)]
        ax.plot(
            recall,
            precision,
            lw=2.0,
            color=color,
            label=f"{model_name} (PR-AUC = {pr_auc:.4f})"
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Phishing Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    return output_path
