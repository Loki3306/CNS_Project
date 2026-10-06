"""
Baseline Experiment Module.
Executes 5-fold cross-validation on the development set for baseline classifiers
and generates comparative cross-validation performance summaries and plots.
"""

from pathlib import Path
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from config import METRICS_DIR, PLOT_DPI, PLOTS_DIR, RANDOM_STATE
from models import (
    build_decision_tree_model,
    build_gradient_boosting_model,
    build_logistic_regression_model,
    build_random_forest_model,
)
from preprocessing.split import get_cv_splitter


def run_baseline_experiment(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    output_dir: Path = METRICS_DIR,
    plots_dir: Path = PLOTS_DIR
) -> pd.DataFrame:
    """
    Evaluate candidate baseline models using 5-fold stratified cross-validation on X_dev, y_dev.
    
    Protects test set: X_test, y_test must never be passed to this function.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cv = get_cv_splitter()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    models = {
        "Logistic Regression": build_logistic_regression_model(random_state=RANDOM_STATE),
        "Decision Tree": build_decision_tree_model(random_state=RANDOM_STATE),
        "Random Forest": build_random_forest_model(random_state=RANDOM_STATE),
        "Gradient Boosting (Initial)": build_gradient_boosting_model(random_state=RANDOM_STATE)
    }

    results = []

    for name, model in models.items():
        start_time = time.time()
        cv_res = cross_validate(
            model,
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        fit_time_total = time.time() - start_time

        record = {
            "model": name,
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "std_accuracy": float(np.std(cv_res["test_accuracy"])),
            "mean_precision": float(np.mean(cv_res["test_precision"])),
            "std_precision": float(np.std(cv_res["test_precision"])),
            "mean_recall": float(np.mean(cv_res["test_recall"])),
            "std_recall": float(np.std(cv_res["test_recall"])),
            "mean_f1": float(np.mean(cv_res["test_f1"])),
            "std_f1": float(np.std(cv_res["test_f1"])),
            "mean_roc_auc": float(np.mean(cv_res["test_roc_auc"])),
            "std_roc_auc": float(np.std(cv_res["test_roc_auc"])),
            "fit_time": round(float(np.mean(cv_res["fit_time"])), 4),
            "total_elapsed_sec": round(fit_time_total, 2)
        }
        results.append(record)

    df_results = pd.DataFrame(results)
    csv_path = output_dir / "baseline_cv_results.csv"
    df_results.to_csv(csv_path, index=False)

    # Generate Baseline Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ["mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "mean_roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    x = np.arange(len(df_results))
    width = 0.16

    colors = ["#2b5c8f", "#41b6c4", "#7fcdbb", "#fdae61", "#d7191c"]

    for idx, (m_col, m_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        means = df_results[m_col]
        stds = df_results[m_col.replace("mean_", "std_")]
        ax.bar(
            x + (idx - 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=m_label,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6
        )

    ax.set_ylabel("5-Fold Cross-Validation Score (Mean ± Std)", fontsize=11, fontweight="bold")
    ax.set_title("Baseline Model Performance Comparison (Development Set 5-Fold CV)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["model"], fontsize=10, fontweight="bold")
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plot_path = plots_dir / "baseline_model_comparison.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    return df_results


if __name__ == "__main__":
    from config import ensure_directories, PROCESSED_CSV_PATH
    from preprocessing.split import create_train_test_split
    ensure_directories()
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, _, y_dev, _ = create_train_test_split(df)
    res = run_baseline_experiment(X_dev, y_dev)
    print(res[["model", "mean_f1", "std_f1", "mean_roc_auc", "std_roc_auc"]])
