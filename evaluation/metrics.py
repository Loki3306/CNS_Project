"""
Common Evaluation Metrics Module.
Calculates standardized classification metrics and confusion matrix counts.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    confusion_matrix,
)


def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series] = None
) -> Dict[str, Any]:
    """
    Calculate a comprehensive suite of binary classification metrics.
    
    Positive Class (1): Phishing
    Negative Class (0): Legitimate
    
    Returns:
      accuracy, precision, recall, f1, roc_auc, pr_auc, mcc,
      specificity, fpr, fnr, tp, tn, fp, fn
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # Verify probabilities rather than hard class predictions
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
        try:
            prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(rec_vals, prec_vals))
        except Exception:
            pr_auc = None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
