"""
Unit Tests for Data Ingestion, Cleaning, and Partitioning.
"""

import pandas as pd
import pytest

from config import PROCESSED_CSV_PATH, SPLIT_METADATA_PATH, TARGET_COLUMN
from preprocessing.clean import clean_dataset
from preprocessing.split import create_train_test_split


def test_target_exists_and_domain():
    """Verify target column exists and contains strictly binary domain {0, 1}."""
    assert PROCESSED_CSV_PATH.exists(), "Processed dataset must exist."
    df = pd.read_csv(PROCESSED_CSV_PATH)
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing from processed data."
    unique_vals = set(df[TARGET_COLUMN].unique())
    assert unique_vals == {0, 1}, f"Target values must be exactly {{0, 1}}, got {unique_vals}"


def test_no_target_in_features():
    """Verify feature matrix X does not include target column or identifier columns."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    assert TARGET_COLUMN not in X.columns
    assert "Index" not in X.columns
    assert "Unnamed: 0" not in X.columns


def test_split_proportions_and_stratification():
    """Verify 80/20 train/test split proportions and stratification consistency."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, X_test, y_dev, y_test = create_train_test_split(df)

    n_total = len(df)
    n_dev = len(X_dev)
    n_test = len(X_test)

    assert n_dev + n_test == n_total
    assert pytest.approx(n_test / n_total, abs=0.01) == 0.20
    assert pytest.approx(n_dev / n_total, abs=0.01) == 0.80

    # Stratification check
    dev_pos_ratio = y_dev.mean()
    test_pos_ratio = y_test.mean()
    assert abs(dev_pos_ratio - test_pos_ratio) < 0.01, (
        f"Stratification mismatch: dev {dev_pos_ratio:.4f} vs test {test_pos_ratio:.4f}"
    )
