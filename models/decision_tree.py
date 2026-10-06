"""
Baseline Model 2: Single Decision Tree.
Provides an interpretable, non-linear single-tree baseline without ensembling.
"""

from sklearn.tree import DecisionTreeClassifier
from config import RANDOM_STATE


def build_decision_tree_model(
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_STATE,
    **kwargs
) -> DecisionTreeClassifier:
    """
    Construct Decision Tree baseline classifier.
    
    Operates directly on unscaled discrete categorical/ordinal features.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        **kwargs
    )
