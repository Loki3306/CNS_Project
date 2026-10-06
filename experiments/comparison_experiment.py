"""
Final Model Comparison and Test Set Evaluation Module.
Retrains all baseline and tuned models on the complete development set,
evaluates on the untouched test partition, records final comparative metrics,
and performs seed robustness sensitivity checks.
"""

from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Any, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier

from config import (
    FINAL_MODEL_PATH,
    METRICS_DIR,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PLOT_DPI,
    PLOTS_DIR,
    RANDOM_STATE,
    RAW_LABEL_MAPPING,
    ROBUSTNESS_SEEDS,
)
from evaluation.confusion import plot_confusion_matrix
from evaluation.curves import plot_pr_comparison, plot_roc_comparison
from evaluation.metrics import calculate_classification_metrics
from models import (
    build_decision_tree_model,
    build_gradient_boosting_model,
    build_logistic_regression_model,
    build_random_forest_model,
)
from preprocessing.split import create_train_test_split


def run_comparison_experiment(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuned_gb_model: GradientBoostingClassifier,
    best_params: dict,
    output_dir: Path = METRICS_DIR,
    plots_dir: Path = PLOTS_DIR,
    models_dir: Path = MODELS_DIR,
    run_seed_robustness: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train candidate models on full X_dev, y_dev and evaluate on final X_test, y_test.
    
    Generates:
      - results/metrics/final_model_comparison.csv
      - results/metrics/final_test_metrics.json
      - plots/final_model_comparison.png
      - plots/roc_curve_comparison.png
      - plots/pr_curve_comparison.png
      - plots/confusion_matrix_<model>.png (for all models)
      - artifacts/models/gradient_boosting_final.joblib
      - artifacts/models/model_metadata.json
      - results/metrics/seed_sensitivity.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "Logistic Regression": build_logistic_regression_model(random_state=RANDOM_STATE),
        "Decision Tree": build_decision_tree_model(random_state=RANDOM_STATE),
        "Random Forest": build_random_forest_model(random_state=RANDOM_STATE),
        "Gradient Boosting (Initial)": build_gradient_boosting_model(random_state=RANDOM_STATE),
        "Gradient Boosting (Tuned)": tuned_gb_model
    }

    comparison_records = []
    roc_pr_predictions = {}

    y_test_arr = np.asarray(y_test)

    for name, model in models.items():
        t0 = time.time()
        # Train on full development partition
        model.fit(X_dev, y_dev)
        fit_time = time.time() - t0

        # Predict on final test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = calculate_classification_metrics(y_test_arr, y_pred, y_prob)
        metrics["model"] = name
        metrics["fit_time"] = round(fit_time, 4)
        comparison_records.append(metrics)

        # Store for ROC/PR comparison plots
        roc_pr_predictions[name] = (y_test_arr, y_prob)

        # Confusion Matrix
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(y_test_arr, y_pred, model_name=clean_name, output_dir=plots_dir)

    df_comparison = pd.DataFrame(comparison_records)
    # Reorder columns for readability
    cols_order = [
        "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "mcc", "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn", "fit_time"
    ]
    df_comparison = df_comparison[cols_order]
    comp_csv_path = output_dir / "final_model_comparison.csv"
    df_comparison.to_csv(comp_csv_path, index=False)

    # Multi-model ROC and PR Curves
    plot_roc_comparison(roc_pr_predictions, output_path=plots_dir / "roc_curve_comparison.png")
    plot_pr_comparison(roc_pr_predictions, output_path=plots_dir / "pr_curve_comparison.png")

    # Final Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#2b5c8f", "#41b6c4", "#7fcdbb", "#fdae61", "#d7191c"]

    x = np.arange(len(df_comparison))
    width = 0.15

    for idx, (m_col, m_label) in enumerate(zip(metrics_plot, labels_plot)):
        vals = df_comparison[m_col]
        ax.bar(
            x + (idx - 2) * width,
            vals,
            width,
            label=m_label,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.6
        )

    ax.set_ylabel("Final Test Score", fontsize=11, fontweight="bold")
    ax.set_title("Final Model Evaluation on Untouched Test Set (20% Holdout)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison["model"], fontsize=9, fontweight="bold", rotation=10)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.savefig(plots_dir / "final_model_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # Serialize Final Tuned Gradient Boosting Model & Metadata
    final_gb = models["Gradient Boosting (Tuned)"]
    joblib.dump(final_gb, FINAL_MODEL_PATH)

    final_test_metrics = df_comparison[df_comparison["model"] == "Gradient Boosting (Tuned)"].iloc[0].to_dict()
    metrics_json_path = output_dir / "final_test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    metadata = {
        "model_type": "GradientBoostingClassifier",
        "parameters": best_params,
        "feature_names": list(X_dev.columns),
        "target_mapping": {str(k): int(v) for k, v in RAW_LABEL_MAPPING.items()},
        "development_samples": len(X_dev),
        "test_samples": len(X_test),
        "random_seed": RANDOM_STATE,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": "1.8.0"
        },
        "created_at_utc": datetime.utcnow().isoformat() + "Z"
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Optional Seed Robustness Check across seeds
    if run_seed_robustness:
        seed_records = []
        raw_df = pd.concat([X_dev, X_test])
        raw_y = pd.concat([y_dev, y_test])
        full_df = raw_df.copy()
        full_df["Result"] = raw_y

        for s in ROBUSTNESS_SEEDS:
            X_d_s, X_t_s, y_d_s, y_t_s = create_train_test_split(full_df, random_state=s)
            s_model = clone(final_gb)
            s_model.random_state = s
            s_model.fit(X_d_s, y_d_s)

            y_p_s = s_model.predict(X_t_s)
            y_prob_s = s_model.predict_proba(X_t_s)[:, 1]
            s_metrics = calculate_classification_metrics(y_t_s, y_p_s, y_prob_s)
            seed_records.append({
                "seed": s,
                "f1": s_metrics["f1"],
                "roc_auc": s_metrics["roc_auc"],
                "recall": s_metrics["recall"],
                "precision": s_metrics["precision"],
                "accuracy": s_metrics["accuracy"]
            })

        df_seeds = pd.DataFrame(seed_records)
        df_seeds.to_csv(output_dir / "seed_sensitivity.csv", index=False)

    return df_comparison, final_test_metrics
