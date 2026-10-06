"""
Interpretability package for MDI and Permutation feature importances.
"""

from .feature_importance import analyze_feature_importance
from .permutation_importance import analyze_permutation_importance

__all__ = ["analyze_feature_importance", "analyze_permutation_importance"]
