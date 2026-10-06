"""
Unit Tests for Common Evaluation Metrics.
Validates metric outputs against deterministic handcrafted reference matrices.
"""

import numpy as np
import pytest
from evaluation.metrics import calculate_classification_metrics


def test_metrics_perfect_predictions():
    """Test metrics when predictions match ground truth perfectly."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["roc_auc"] == 1.0


def test_metrics_known_confusion_matrix():
    """
    Test metrics with a known confusion matrix:
    TP = 2, TN = 3, FP = 1, FN = 2
    Total = 8
    Accuracy = 5/8 = 0.625
    Precision = 2 / (2 + 1) = 2/3 ≈ 0.6667
    Recall = 2 / (2 + 2) = 0.5
    Specificity = 3 / (3 + 1) = 0.75
    FPR = 1 / 4 = 0.25
    FNR = 2 / 4 = 0.5
    F1 = 2 * (2/3 * 0.5) / (2/3 + 0.5) = 2 * (1/3) / (7/6) = (2/3) / (7/6) = 4/7 ≈ 0.5714
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.625
    assert pytest.approx(metrics["precision"], abs=1e-4) == 2.0 / 3.0
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.75
    assert metrics["fpr"] == 0.25
    assert metrics["fnr"] == 0.5
    assert pytest.approx(metrics["f1"], abs=1e-4) == 4.0 / 7.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
