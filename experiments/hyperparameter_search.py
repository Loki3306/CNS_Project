"""
Gradient Boosting Hyperparameter Optimization Module.
Executes RandomizedSearchCV strictly over the development partition using 5-fold Stratified CV.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from config import (
    PARAM_DISTRIBUTIONS,
    RANDOM_SEARCH_N_ITER,
    RANDOM_STATE,
    TUNING_DIR,
)
from preprocessing.split import get_cv_splitter


def run_hyperparameter_search(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    param_distributions: dict = PARAM_DISTRIBUTIONS,
    n_iter: int = RANDOM_SEARCH_N_ITER,
    output_dir: Path = TUNING_DIR,
    random_state: int = RANDOM_STATE
) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    Tune GradientBoostingClassifier hyperparameters via RandomizedSearchCV on development data.
    
    Protects test set: X_test, y_test must never participate in hyperparameter tuning.
    
    Saves:
      - results/tuning/randomized_search_results.csv
      - results/tuning/best_parameters.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = get_cv_splitter(random_state=random_state)

    base_estimator = GradientBoostingClassifier(random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    search.fit(X_dev, y_dev)

    # 1. Export comprehensive CV results
    cv_results_df = pd.DataFrame(search.cv_results_)
    # Sort by rank_test_f1
    cv_results_df = cv_results_df.sort_values(by="rank_test_f1", ascending=True)
    results_csv_path = output_dir / "randomized_search_results.csv"
    cv_results_df.to_csv(results_csv_path, index=False)

    # 2. Extract best parameters and best CV scores
    best_params = search.best_params_
    best_idx = search.best_index_

    best_summary = {
        "best_parameters": best_params,
        "refit_metric": "f1",
        "best_cv_f1": float(search.cv_results_["mean_test_f1"][best_idx]),
        "best_cv_f1_std": float(search.cv_results_["std_test_f1"][best_idx]),
        "best_cv_accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
        "best_cv_accuracy_std": float(search.cv_results_["std_test_accuracy"][best_idx]),
        "best_cv_precision": float(search.cv_results_["mean_test_precision"][best_idx]),
        "best_cv_precision_std": float(search.cv_results_["std_test_precision"][best_idx]),
        "best_cv_recall": float(search.cv_results_["mean_test_recall"][best_idx]),
        "best_cv_recall_std": float(search.cv_results_["std_test_recall"][best_idx]),
        "best_cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_idx]),
        "best_cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best_idx]),
        "total_configurations_tested": n_iter,
        "random_state": random_state
    }

    best_json_path = output_dir / "best_parameters.json"
    with open(best_json_path, "w") as f:
        json.dump(best_summary, f, indent=2)

    return search.best_estimator_, best_summary


if __name__ == "__main__":
    from config import ensure_directories, PROCESSED_CSV_PATH
    from preprocessing.split import create_train_test_split
    ensure_directories()
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, _, y_dev, _ = create_train_test_split(df)
    best_model, summary = run_hyperparameter_search(X_dev, y_dev, n_iter=10)
    print("Search complete. Best CV F1:", summary["best_cv_f1"])
