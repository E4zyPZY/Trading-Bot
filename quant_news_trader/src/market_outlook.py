"""
Market Outlook Assessment
=========================

Assesses current market conditions to determine trading aggressiveness.
Uses regime detection to classify market state and select appropriate profile.

Profiles:
- aggro: Trending low vol or mean-revert low vol
- balanced: Trending high vol or random walk
- conservative: Mean-revert high vol
"""

import yaml
from pathlib import Path
from typing import Optional
import pandas as pd
from src.analysis.regime_detection import RegimeDetector, MarketRegime
from src.data.market_data import fetch_market_data


def assess_market_outlook(root: Path) -> str:
    """
    Assess market outlook and return the appropriate profile.

    Returns:
        str: 'aggro', 'balanced', or 'conservative'
    """
    # Fetch recent market data for SPY (S&P 500 ETF)
    try:
        market_data = fetch_market_data(
            ticker="SPY",
            lookback_days=100,  # Use 100 days for regime detection
            interval="1d"
        )
        prices = market_data["Close"]
        returns = prices.pct_change().dropna()

        # Detect regime
        lookback = min(60, len(prices) - 10)  # Adjust lookback to available data
        detector = RegimeDetector(prices.values, returns.values, lookback=lookback)
        regime_snapshot = detector.get_regime()
        regime = regime_snapshot.state
        h_exp = regime_snapshot.hurst_exponent
        vol = regime_snapshot.volatility
        score = regime_snapshot.regime_score

        # Map regime to profile
        if regime in [MarketRegime.TRENDING_LOW_VOL, MarketRegime.MEAN_REVERT_LOW_VOL]:
            return "aggro"
        elif regime in [MarketRegime.TRENDING_HIGH_VOL, MarketRegime.RANDOM_WALK]:
            return "balanced"
        elif regime == MarketRegime.MEAN_REVERT_HIGH_VOL:
            return "conservative"
        else:
            return "balanced"  # default

    except Exception as e:
        print(f"Error assessing market outlook: {e}")
        return "balanced"  # default to balanced on error


def get_dynamic_profile(root: Path) -> str:
    """
    Get the dynamic profile based on market outlook.
    """
    return assess_market_outlook(root)