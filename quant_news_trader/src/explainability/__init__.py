"""
Explainability Module
====================

SHAP-based model interpretation for trading signals.
"""

from .shap_analyzer import SHAPAnalyzer, compare_model_behaviors

__all__ = ['SHAPAnalyzer', 'compare_model_behaviors']
