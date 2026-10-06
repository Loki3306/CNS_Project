"""
Primary Model: Gradient Boosting Classifier.
Implements sequential pseudo-residual boosting using sklearn.ensemble.GradientBoostingClassifier.
"""

from sklearn.ensemble import GradientBoostingClassifier
from config import RANDOM_STATE


def build_gradient_boosting_model(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    max_features=None,
    random_state: int = RANDOM_STATE,
    **kwargs
) -> GradientBoostingClassifier:
    """
    Construct Gradient Boosting classifier.
    
    Default parameters represent the baseline un-tuned configuration.
    Operates directly on unscaled discrete categorical/ordinal features.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state,
        **kwargs
    )
