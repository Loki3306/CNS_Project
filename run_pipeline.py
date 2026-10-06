"""
Master Pipeline Orchestration Script.
Executes the end-to-end phishing URL classification research workflow from dataset audit
to final model evaluation, error analysis, artifact validation, and report generation.
"""

import json
from pathlib import Path
import shutil
import sys
import time

from config import (
    AUDITS_DIR,
    BASE_DIR,
    FINAL_MODEL_PATH,
    KAGGLE_DATASET_ID,
    LEARNING_DYNAMICS_DIR,
    METRICS_DIR,
    MODEL_METADATA_PATH,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_CSV_PATH,
    RAW_CSV_PATH,
    RAW_DATA_DIR,
    RAW_TXT_PATH,
    REPORTS_DIR,
    SPLIT_METADATA_PATH,
    TARGET_COLUMN,
    TUNING_DIR,
    ensure_directories,
)
from evaluation.error_analysis import perform_error_analysis
from experiments.baseline_experiment import run_baseline_experiment
from experiments.comparison_experiment import run_comparison_experiment
from experiments.hyperparameter_search import run_hyperparameter_search
from experiments.staged_analysis import run_staged_analysis
from interpretability.feature_importance import analyze_feature_importance
from interpretability.permutation_importance import analyze_permutation_importance
from preprocessing.audit import audit_dataset
from preprocessing.clean import clean_dataset
from preprocessing.leakage import analyze_leakage
from preprocessing.split import create_train_test_split
from reports.generate_reports import generate_all_reports


def acquire_dataset():
    """Ensure dataset is present in data/raw/, downloading via kagglehub if necessary."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV_PATH.exists():
        return RAW_CSV_PATH

    print(f"[CNS] Attempting automated download of '{KAGGLE_DATASET_ID}' via kagglehub...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        src_dir = Path(downloaded_path)
        for f in src_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, RAW_DATA_DIR / f.name)
        print(f"[CNS] Successfully acquired dataset to {RAW_DATA_DIR}")
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset acquisition failed: {e}\n"
            f"Please manually download '{KAGGLE_DATASET_ID}' from Kaggle and place 'phishing.csv' into: {RAW_DATA_DIR}"
        ) from e

    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"phishing.csv was not found in {RAW_DATA_DIR} after acquisition attempt.")

    return RAW_CSV_PATH


def validate_artifacts():
    """Programmatically verify that all required outputs, plots, and reports were produced."""
    required_files = [
        # Data & metadata
        PROCESSED_CSV_PATH,
        SPLIT_METADATA_PATH,
        # Audits
        AUDITS_DIR / "missing_values.csv",
        AUDITS_DIR / "class_distribution.csv",
        AUDITS_DIR / "duplicate_analysis.json",
        AUDITS_DIR / "leakage_report.json",
        # Metrics & Tuning
        METRICS_DIR / "baseline_cv_results.csv",
        TUNING_DIR / "randomized_search_results.csv",
        TUNING_DIR / "best_parameters.json",
        METRICS_DIR / "final_test_metrics.json",
        METRICS_DIR / "final_model_comparison.csv",
        METRICS_DIR / "gb_feature_importance.csv",
        METRICS_DIR / "permutation_importance.csv",
        # Learning dynamics
        LEARNING_DYNAMICS_DIR / "staged_metrics.csv",
        # Predictions & Error analysis
        PREDICTIONS_DIR / "final_test_predictions.csv",
        PREDICTIONS_DIR / "false_negatives.csv",
        PREDICTIONS_DIR / "false_positives.csv",
        PREDICTIONS_DIR / "low_confidence_predictions.csv",
        # Models
        FINAL_MODEL_PATH,
        MODEL_METADATA_PATH,
        # Reports
        REPORTS_DIR / "dataset_audit.md",
        REPORTS_DIR / "leakage_analysis.md",
        REPORTS_DIR / "methodology.md",
        REPORTS_DIR / "experimental_results.md",
        REPORTS_DIR / "final_summary.md",
        # Plots
        PLOTS_DIR / "class_distribution.png",
        PLOTS_DIR / "correlation_heatmap.png",
        PLOTS_DIR / "baseline_model_comparison.png",
        PLOTS_DIR / "confusion_matrix_logistic_regression.png",
        PLOTS_DIR / "confusion_matrix_decision_tree.png",
        PLOTS_DIR / "confusion_matrix_random_forest.png",
        PLOTS_DIR / "confusion_matrix_gradient_boosting_initial.png",
        PLOTS_DIR / "confusion_matrix_gradient_boosting_tuned.png",
        PLOTS_DIR / "roc_curve_comparison.png",
        PLOTS_DIR / "pr_curve_comparison.png",
        PLOTS_DIR / "gb_training_loss.png",
        PLOTS_DIR / "gb_train_validation_loss.png",
        PLOTS_DIR / "gb_error_vs_estimators.png",
        PLOTS_DIR / "gb_f1_vs_estimators.png",
        PLOTS_DIR / "gb_roc_auc_vs_estimators.png",
        PLOTS_DIR / "gb_feature_importance.png",
        PLOTS_DIR / "permutation_importance.png",
        PLOTS_DIR / "final_model_comparison.png",
    ]

    missing = []
    for req in required_files:
        if not req.exists() or req.stat().st_size == 0:
            missing.append(str(req))

    if missing:
        raise RuntimeError(f"Artifact Validation Failed! The following expected outputs are missing or empty:\n" + "\n".join(missing))

    print(f"[CNS] Artifact Validation: All {len(required_files)} mandatory files verified successfully.")


def run_pipeline():
    """Execute complete research pipeline."""
    total_start = time.time()
    ensure_directories()

    print("=" * 70)
    print("CNS PROJECT: A Gradient Boosting Tree Framework for Intelligent Phishing URL Classification")
    print("=" * 70)

    # Phase 1: Acquisition
    print("\n[CNS] Phase 1/10: Dataset Acquisition")
    csv_file = acquire_dataset()
    print(f"[CNS] Dataset located at: {csv_file}")

    # Phase 2: Audit
    print("\n[CNS] Phase 2/10: Dataset Audit & Quality Inspection")
    audit_info = audit_dataset(csv_file)
    print(f"[CNS] Dataset Shape: {audit_info['rows']:,} rows × {audit_info['cols']} columns")
    print(f"[CNS] Missing Values: {audit_info['missing_total']} cells")
    print(f"[CNS] Exact Duplicate Rows: {audit_info['duplicate_rows_raw']}")
    print(f"[CNS] Duplicate Feature Vectors: {audit_info['duplicate_X_no_id']:,}")
    print(f"[CNS] Conflicting Duplicate Vectors: {audit_info['conflicting_vectors']}")

    # Phase 3: Cleaning
    print("\n[CNS] Phase 3/10: Data Cleaning & Label Encoding")
    df_clean = clean_dataset()
    print(f"[CNS] Cleaned Shape: {df_clean.shape}")
    print(f"[CNS] Binary Target Distribution: {dict(df_clean[TARGET_COLUMN].value_counts())}")

    # Phase 4: Splitting
    print("\n[CNS] Phase 4/10: Stratified Development / Test Splitting (80/20)")
    X_dev, X_test, y_dev, y_test = create_train_test_split(df_clean)
    print(f"[CNS] Development Set: {X_dev.shape[0]:,} samples | Holdout Test Set: {X_test.shape[0]:,} samples")

    # Phase 5: Leakage Analysis
    print("\n[CNS] Phase 5/10: Leakage & Contamination Analysis")
    leakage_res = analyze_leakage(df_clean, X_dev=X_dev, X_test=X_test, y_dev=y_dev, y_test=y_test)
    print(f"[CNS] Cross-Split Index Intersection: {leakage_res['split_contamination']['index_intersection']}")
    print(f"[CNS] Suspicious Predictors (AUC >= 0.90): {len(leakage_res['suspicious_predictors'])}")

    # Phase 6: Baselines
    print("\n[CNS] Phase 6/10: Baseline Models & 5-Fold Cross-Validation")
    baseline_df = run_baseline_experiment(X_dev, y_dev)
    for _, row in baseline_df.iterrows():
        print(f"  - {row['model']:<28}: F1={row['mean_f1']:.4f}±{row['std_f1']:.4f} | ROC-AUC={row['mean_roc_auc']:.4f}±{row['std_roc_auc']:.4f}")

    # Phase 7: Hyperparameter Tuning
    print("\n[CNS] Phase 7/10: Gradient Boosting Hyperparameter Optimization (50 Configurations)")
    best_gb_model, tuning_summary = run_hyperparameter_search(X_dev, y_dev)
    print(f"[CNS] Best CV F1: {tuning_summary['best_cv_f1']:.4f} ± {tuning_summary['best_cv_f1_std']:.4f}")
    print(f"[CNS] Best Parameters: {tuning_summary['best_parameters']}")

    # Phase 8: Staged Learning Dynamics
    print("\n[CNS] Phase 8/10: Boosting-Stage Learning Dynamics Analysis")
    staged_summary = run_staged_analysis(best_gb_model, X_dev, y_dev)
    print(f"[CNS] Evaluated {staged_summary['n_estimators_evaluated']} estimators.")
    print(f"[CNS] Min Validation Loss at Iteration {staged_summary['best_val_loss_iteration']}: {staged_summary['best_val_loss']:.4f}")
    print(f"[CNS] Pseudo-Residual Reduction: {staged_summary['pseudo_residual_reduction_pct']:.1f}%")

    # Phase 9: Final Model Evaluation & Comparison
    print("\n[CNS] Phase 9/10: Final Model Retraining & Untouched Holdout Test Set Evaluation")
    comp_df, final_metrics = run_comparison_experiment(
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
        y_test=y_test,
        tuned_gb_model=best_gb_model,
        best_params=tuning_summary['best_parameters']
    )
    print("\n[CNS] Final Holdout Test Performance Summary:")
    for _, row in comp_df.iterrows():
        print(f"  - {row['model']:<28}: Acc={row['accuracy']*100:.2f}% | F1={row['f1']:.4f} | ROC-AUC={row['roc_auc']:.4f} | Recall={row['recall']*100:.2f}%")

    # Phase 10: Interpretability & Error Analysis
    print("\n[CNS] Phase 10/10: Interpretability, Error Analysis & Report Compilation")
    analyze_feature_importance(best_gb_model, list(X_dev.columns))
    analyze_permutation_importance(best_gb_model, X_test, y_test)
    error_summary = perform_error_analysis(best_gb_model, X_test, y_test)
    print(f"[CNS] Test False Negatives: {error_summary['fn_count']} | False Positives: {error_summary['fp_count']}")
    print(f"[CNS] Low-Confidence Predictions (0.40 - 0.60): {error_summary['low_confidence_count']}")

    # Generate Reports
    print("[CNS] Compiling Markdown Technical Reports...")
    generate_all_reports()

    # Validate Artifacts
    print("\n[CNS] Final Artifact Integrity Verification...")
    validate_artifacts()

    elapsed = time.time() - total_start
    print(f"\n[CNS] Pipeline completed successfully in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    run_pipeline()
