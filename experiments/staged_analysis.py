"""
Boosting-Stage Learning Dynamics Analysis Module.
Evaluates training/validation loss convergence, error progression, F1/ROC-AUC trajectory,
and pseudo-residual dynamics across boosting iterations strictly using development data.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from config import (
    LEARNING_DYNAMICS_DIR,
    PLOT_DPI,
    PLOTS_DIR,
    RANDOM_STATE,
)


def run_staged_analysis(
    model: GradientBoostingClassifier,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    output_dir: Path = LEARNING_DYNAMICS_DIR,
    plots_dir: Path = PLOTS_DIR,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Perform stage-by-stage boosting analysis using an internal development validation split.
    
    CRITICAL: Never uses X_test, y_test.
    
    Generates:
      - results/learning_dynamics/staged_metrics.csv
      - plots/gb_training_loss.png
      - plots/gb_train_validation_loss.png
      - plots/gb_error_vs_estimators.png
      - plots/gb_f1_vs_estimators.png
      - plots/gb_roc_auc_vs_estimators.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Internal development split for honest validation loss tracking
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=random_state
    )

    # Fit a clone of the model configuration on the sub-training partition
    staged_model = clone(model)
    staged_model.fit(X_train_sub, y_train_sub)

    n_estimators = staged_model.n_estimators_
    iterations = np.arange(1, n_estimators + 1)

    # 2. Extract Training Deviance / Loss from train_score_
    train_loss = staged_model.train_score_

    # 3. Calculate Staged Metrics across Iterations
    val_loss = []
    train_error = []
    val_error = []
    val_f1 = []
    val_roc_auc = []
    mean_abs_pseudo_residual = []

    y_train_arr = np.asarray(y_train_sub)
    y_val_arr = np.asarray(y_val_sub)

    # Collect staged predictions for validation set
    for i, (val_pred, val_prob) in enumerate(zip(
        staged_model.staged_predict(X_val_sub),
        staged_model.staged_predict_proba(X_val_sub)
    )):
        p_phish = val_prob[:, 1]
        
        # Validation Log Loss
        v_loss = log_loss(y_val_arr, val_prob)
        val_loss.append(v_loss)

        # Classification Error (1 - Accuracy)
        v_err = 1.0 - accuracy_score(y_val_arr, val_pred)
        val_error.append(v_err)

        # Validation F1
        v_f1 = f1_score(y_val_arr, val_pred, zero_division=0)
        val_f1.append(v_f1)

        # Validation ROC-AUC
        v_auc = roc_auc_score(y_val_arr, p_phish)
        val_roc_auc.append(v_auc)

        # Mean Absolute Pseudo-Residual |y - p|
        # In binary logistic loss, the negative gradient (pseudo-residual) is exactly y - p(x)
        pseudo_res = np.mean(np.abs(y_val_arr - p_phish))
        mean_abs_pseudo_residual.append(pseudo_res)

    # Collect staged training error
    for train_pred in staged_model.staged_predict(X_train_sub):
        t_err = 1.0 - accuracy_score(y_train_arr, train_pred)
        train_error.append(t_err)

    val_loss = np.array(val_loss)
    train_error = np.array(train_error)
    val_error = np.array(val_error)
    val_f1 = np.array(val_f1)
    val_roc_auc = np.array(val_roc_auc)
    mean_abs_pseudo_residual = np.array(mean_abs_pseudo_residual)

    # Identify optimal iteration based on validation loss
    best_loss_iter = int(iterations[np.argmin(val_loss)])
    best_loss_val = float(np.min(val_loss))
    best_f1_iter = int(iterations[np.argmax(val_f1)])
    best_f1_val = float(np.max(val_f1))

    # 4. Save Staged Metrics Table
    staged_df = pd.DataFrame({
        "iteration": iterations,
        "training_deviance": train_loss,
        "validation_log_loss": val_loss,
        "training_error": train_error,
        "validation_error": val_error,
        "validation_f1": val_f1,
        "validation_roc_auc": val_roc_auc,
        "mean_abs_pseudo_residual": mean_abs_pseudo_residual
    })
    staged_csv_path = output_dir / "staged_metrics.csv"
    staged_df.to_csv(staged_csv_path, index=False)

    # 5. Plot 1: Training Loss (train_score_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, train_loss, color="#2b5c8f", lw=2.0, label="Training Loss (Deviance)")
    ax.set_title("Gradient Boosting Training Loss vs. Boosting Iterations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Boosting Iteration (Number of Estimators)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Deviance / Loss", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.savefig(plots_dir / "gb_training_loss.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 6. Plot 2: Training vs. Validation Loss Overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, train_loss, color="#2b5c8f", lw=2.0, label="Sub-Training Loss")
    ax.plot(iterations, val_loss, color="#d95f02", lw=2.0, label="Validation Log Loss")
    ax.axvline(best_loss_iter, color="black", linestyle=":", lw=1.5,
               label=f"Min Val Loss (Iter {best_loss_iter}: {best_loss_val:.4f})")
    ax.set_title("Training vs. Validation Loss Progression", fontsize=12, fontweight="bold")
    ax.set_xlabel("Boosting Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.savefig(plots_dir / "gb_train_validation_loss.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 7. Plot 3: Classification Error vs. Boosting Iterations
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, train_error, color="#1f77b4", lw=2.0, label="Sub-Training Error (1 - Acc)")
    ax.plot(iterations, val_error, color="#d62728", lw=2.0, label="Validation Error (1 - Acc)")
    ax.set_title("Classification Error vs. Boosting Iterations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Boosting Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classification Error", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.savefig(plots_dir / "gb_error_vs_estimators.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 8. Plot 4: Validation F1 vs. Boosting Iterations
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, val_f1, color="#2ca02c", lw=2.0, label="Validation F1 Score")
    ax.axvline(best_f1_iter, color="black", linestyle=":", lw=1.5,
               label=f"Max F1 (Iter {best_f1_iter}: {best_f1_val:.4f})")
    ax.set_title("Validation F1 Score vs. Boosting Iterations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Boosting Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score (Positive: Phishing)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.savefig(plots_dir / "gb_f1_vs_estimators.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 9. Plot 5: Validation ROC-AUC vs. Boosting Iterations
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, val_roc_auc, color="#9467bd", lw=2.0, label="Validation ROC-AUC")
    ax.set_title("Validation ROC-AUC vs. Boosting Iterations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Boosting Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.savefig(plots_dir / "gb_roc_auc_vs_estimators.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    summary = {
        "n_estimators_evaluated": int(n_estimators),
        "best_val_loss_iteration": best_loss_iter,
        "best_val_loss": best_loss_val,
        "best_val_f1_iteration": best_f1_iter,
        "best_val_f1": best_f1_val,
        "initial_pseudo_residual": float(mean_abs_pseudo_residual[0]),
        "final_pseudo_residual": float(mean_abs_pseudo_residual[-1]),
        "pseudo_residual_reduction_pct": float(
            ((mean_abs_pseudo_residual[0] - mean_abs_pseudo_residual[-1]) / mean_abs_pseudo_residual[0]) * 100.0
        )
    }

    return summary
