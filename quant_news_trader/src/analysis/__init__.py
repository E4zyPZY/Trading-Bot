"""
Analysis Module
===============

Market analysis tools: regime detection, signal decomposition.
"""

from .regime_detection import (
    HurstAnalyzer,
    RegimeDetector,
    MarketRegime,
    RegimeSnapshot,
    VolatilityRegime,
    adaptive_signal_by_regime
)

__all__ = [
    'HurstAnalyzer',
    'RegimeDetector',
    'MarketRegime',
    'RegimeSnapshot',
    'VolatilityRegime',
    'adaptive_signal_by_regime',
]
