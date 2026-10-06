"""
Comparison Model: Random Forest Classifier.
Implements bagging and randomized split selection (parallel tree ensemble).
"""

from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE


def build_random_forest_model(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features="sqrt",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestClassifier:
    """
    Construct Random Forest comparison classifier.
    
    Provides parallel bagging ensemble baseline for contrasting against sequential boosting.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
