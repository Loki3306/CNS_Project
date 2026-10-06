"""
Mean Decrease in Impurity (MDI) Feature Importance Module.
Extracts tree-based Gini/impurity feature importance, generates ranked tables and plots.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from config import METRICS_DIR, PLOT_DPI, PLOTS_DIR


def analyze_feature_importance(
    model: GradientBoostingClassifier,
    feature_names: List[str],
    output_dir: Path = METRICS_DIR,
    plots_dir: Path = PLOTS_DIR,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Extract and visualize tree-based Gini impurity importance (model.feature_importances_).
    
    Generates:
      - results/metrics/gb_feature_importance.csv
      - plots/gb_feature_importance.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    df_importance["cumulative_importance"] = df_importance["importance"].cumsum()
    df_importance["rank"] = df_importance.index + 1

    csv_path = output_dir / "gb_feature_importance.csv"
    df_importance.to_csv(csv_path, index=False)

    # Plot top_n features
    top_df = df_importance.head(top_n).iloc[::-1]  # reverse for horizontal bar chart

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_df["feature"], top_df["importance"], color="#2b5c8f", edgecolor="black", linewidth=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_title(f"Gradient Boosting: Top {top_n} Features by Gini Impurity (MDI)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Impurity (Relative Feature Importance)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(top_df["importance"]) * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plot_path = plots_dir / "gb_feature_importance.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    return df_importance
