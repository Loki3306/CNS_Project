"""
Error and Prediction Analysis Module.
Conducts fine-grained inspection of False Negatives, False Positives, and Borderline/Low-Confidence Predictions.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from config import PREDICTIONS_DIR


def perform_error_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = PREDICTIONS_DIR,
    low_conf_range: tuple = (0.40, 0.60)
) -> Dict[str, Any]:
    """
    Generate detailed error analysis dataframes and summary statistics.
    
    Positive Class (1): Phishing
    Negative Class (0): Legitimate
    
    Generates:
      - results/predictions/final_test_predictions.csv
      - results/predictions/false_negatives.csv
      - results/predictions/false_positives.csv
      - results/predictions/low_confidence_predictions.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    # Confidence: distance from decision threshold 0.5 (or max prob)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct = (y_true == y_pred)

    error_types = []
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            error_types.append("TP")
        elif actual == 0 and pred == 0:
            error_types.append("TN")
        elif actual == 0 and pred == 1:
            error_types.append("FP")
        elif actual == 1 and pred == 0:
            error_types.append("FN")
        else:
            error_types.append("UNKNOWN")

    predictions_df = X_test.copy()
    predictions_df.insert(0, "actual_label", y_true)
    predictions_df.insert(1, "predicted_label", y_pred)
    predictions_df.insert(2, "phishing_probability", np.round(y_prob, 4))
    predictions_df.insert(3, "prediction_confidence", np.round(confidence, 4))
    predictions_df.insert(4, "correct", correct)
    predictions_df.insert(5, "error_type", error_types)

    # 1. Full Predictions File
    pred_path = output_dir / "final_test_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    # 2. False Negatives (Actual Phishing -> Predicted Legitimate)
    # High-severity failure mode in cybersecurity
    fn_df = predictions_df[predictions_df["error_type"] == "FN"].copy()
    fn_path = output_dir / "false_negatives.csv"
    fn_df.to_csv(fn_path, index=False)

    # 3. False Positives (Actual Legitimate -> Predicted Phishing)
    # Causes user friction/alert fatigue
    fp_df = predictions_df[predictions_df["error_type"] == "FP"].copy()
    fp_path = output_dir / "false_positives.csv"
    fp_df.to_csv(fp_path, index=False)

    # 4. Low-Confidence Predictions (Exploratory Uncertainty Band)
    low_band_min, low_band_max = low_conf_range
    low_conf_mask = (y_prob >= low_band_min) & (y_prob <= low_band_max)
    low_conf_df = predictions_df[low_conf_mask].copy()
    low_conf_path = output_dir / "low_confidence_predictions.csv"
    low_conf_df.to_csv(low_conf_path, index=False)

    summary = {
        "total_test_samples": len(predictions_df),
        "total_correct": int(correct.sum()),
        "total_errors": int((~correct).sum()),
        "tp_count": int((predictions_df["error_type"] == "TP").sum()),
        "tn_count": int((predictions_df["error_type"] == "TN").sum()),
        "fp_count": int(len(fp_df)),
        "fn_count": int(len(fn_df)),
        "low_confidence_count": int(len(low_conf_df)),
        "low_confidence_accuracy": float(low_conf_df["correct"].mean()) if len(low_conf_df) > 0 else 0.0,
        "fn_mean_phishing_prob": float(fn_df["phishing_probability"].mean()) if len(fn_df) > 0 else 0.0,
        "fp_mean_phishing_prob": float(fp_df["phishing_probability"].mean()) if len(fp_df) > 0 else 0.0,
    }

    return summary
