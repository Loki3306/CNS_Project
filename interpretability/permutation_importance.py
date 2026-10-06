"""
Permutation Feature Importance Module.
Calculates test-set permutation importance with multiple reshuffles (n_repeats=20).
Contrasts model-agnostic permutation importance against tree-internal impurity importance.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import METRICS_DIR, PLOT_DPI, PLOTS_DIR, RANDOM_STATE


def analyze_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = METRICS_DIR,
    plots_dir: Path = PLOTS_DIR,
    scoring: str = "f1",
    n_repeats: int = 20,
    random_state: int = RANDOM_STATE,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Calculate and plot permutation feature importance on the final test partition.
    
    Generates:
      - results/metrics/permutation_importance.csv
      - plots/permutation_importance.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    feature_names = list(X_test.columns)
    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    df_perm["rank"] = df_perm.index + 1

    csv_path = output_dir / "permutation_importance.csv"
    df_perm.to_csv(csv_path, index=False)

    # Plot top_n features
    top_df = df_perm.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_df["feature"],
        top_df["importance_mean"],
        xerr=top_df["importance_std"],
        capsize=3,
        color="#31a354",
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title(f"Permutation Feature Importance (Test Set, {scoring.upper()} Score, {n_repeats} Repeats)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"Mean Decrease in {scoring.upper()} Score", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plot_path = plots_dir / "permutation_importance.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    return df_perm
