"""
Baseline Model 1: Logistic Regression Pipeline.
Uses StandardScaler + LogisticRegression within a scikit-learn Pipeline.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE


def build_logistic_regression_model(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
    **kwargs
) -> Pipeline:
    """
    Construct Logistic Regression baseline pipeline with standardization.
    
    StandardScaler is required here for numerical stability and regularizer fairness,
    unlike tree-based models where monotonic feature invariance applies.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        ))
    ])
    return pipeline
