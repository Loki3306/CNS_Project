"""
Confusion Matrix Visualization Module.
Plots and persists publication-ready annotated confusion matrices.
"""

from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import PLOT_DPI, PLOTS_DIR


def plot_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    model_name: str,
    output_dir: Path = PLOTS_DIR,
    normalize: bool = False
) -> Path:
    """
    Generate and save an annotated confusion matrix plot.
    
    Labels:
      0 = Legitimate
      1 = Phishing
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_names = ["Legitimate", "Phishing"]

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Text annotations with count and cell percentage
    total = np.sum(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            annot[i, j] = f"{count:,}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )

    clean_title_name = model_name.replace("_", " ").title()
    ax.set_title(f"Confusion Matrix: {clean_title_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Ground Truth Label", fontsize=11, fontweight="bold", labelpad=8)

    clean_file_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"confusion_matrix_{clean_file_name}.png"
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    return output_path
