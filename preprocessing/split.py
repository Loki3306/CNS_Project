"""
Dataset Splitting Module.
Implements reproducible, stratified train/test partitioning and records partition metadata.
"""

import json
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    CV_FOLDS,
    PROCESSED_CSV_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_METADATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into development (80%) and final test (20%) partitions with stratification.
    
    Protects test set integrity by recording split metadata without leaking test information.
    Saves split metadata to data/processed/split_metadata.json.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Verify and record split metadata
    total_len = len(df)
    dev_len = len(X_dev)
    test_len = len(X_test)

    dev_class_counts = y_dev.value_counts().to_dict()
    test_class_counts = y_test.value_counts().to_dict()

    dev_proportions = {str(k): round(v / dev_len, 4) for k, v in dev_class_counts.items()}
    test_proportions = {str(k): round(v / test_len, 4) for k, v in test_class_counts.items()}

    # Assert exact disjointness
    assert len(set(X_dev.index).intersection(set(X_test.index))) == 0, (
        "Fatal leakage: Indices overlap between development and test sets!"
    )

    # Check stratification consistency
    for label in [0, 1]:
        dev_prop = dev_proportions.get(str(label), 0.0)
        test_prop = test_proportions.get(str(label), 0.0)
        diff = abs(dev_prop - test_prop)
        assert diff < 0.01, f"Stratification discrepancy exceeds 1% for label {label}: {diff}"

    metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": CV_FOLDS,
        "total_samples": total_len,
        "development_samples": dev_len,
        "test_samples": test_len,
        "development_class_counts": {str(k): int(v) for k, v in dev_class_counts.items()},
        "test_class_counts": {str(k): int(v) for k, v in test_class_counts.items()},
        "development_class_proportions": dev_proportions,
        "test_class_proportions": test_proportions,
        "stratification_verified": True
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return X_dev, X_test, y_dev, y_test


def get_cv_splitter(cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Return the canonical reusable StratifiedKFold cross-validator."""
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    from config import ensure_directories
    ensure_directories()
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, X_test, y_dev, y_test = create_train_test_split(df)
    print("Split created successfully:")
    print(f"Dev: {X_dev.shape}, Test: {X_test.shape}")
