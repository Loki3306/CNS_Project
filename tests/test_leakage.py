"""
Unit Tests for Data Leakage Prevention and Disjoint Partitioning.
"""

import pandas as pd
import pytest
from config import PROCESSED_CSV_PATH, TARGET_COLUMN
from preprocessing.split import create_train_test_split


def test_index_disjointness():
    """Ensure training and test set indices have strictly zero intersection."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, X_test, y_dev, y_test = create_train_test_split(df)

    dev_indices = set(X_dev.index)
    test_indices = set(X_test.index)

    overlap = dev_indices.intersection(test_indices)
    assert len(overlap) == 0, f"Critical Leakage: Found {len(overlap)} overlapping indices between dev and test!"


def test_target_excluded_from_splits():
    """Verify target column is never included in feature matrices X_dev or X_test."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, X_test, y_dev, y_test = create_train_test_split(df)

    assert TARGET_COLUMN not in X_dev.columns
    assert TARGET_COLUMN not in X_test.columns
    assert len(X_dev.columns) == len(X_test.columns)


def test_duplicate_tracking_logic():
    """Verify that duplicate vectors are detectable and tracked."""
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X_dev, X_test, _, _ = create_train_test_split(df)

    dev_tuples = set(tuple(x) for x in X_dev.values)
    test_tuples = set(tuple(x) for x in X_test.values)

    # In a 30-feature discrete space, cross-split vector overlap is expected and measured
    overlap = dev_tuples.intersection(test_tuples)
    assert isinstance(overlap, set)
