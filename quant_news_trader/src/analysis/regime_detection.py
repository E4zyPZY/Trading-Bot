"""
Hurst Exponent & Regime Detection
==================================

Uses Hurst exponent to identify market regimes:
- H > 0.5: Trending (momentum dominates, mean-reversion fails)
- H = 0.5: Random walk (no pattern, avoid trading)
- H < 0.5: Mean-reverting (spreads widen predictably, arbitrage opportunity)

Combines with volatility regimes for adaptive position sizing:
- Low vol + high H: Strong trend → momentum signal
- Low vol + low H: Choppy mean-reversion → pairs trading
- High vol + low H: Panic selling → mean-reversion bounce
- High vol + high H: Trend + fear → momentum with exit plan

Example:
    from src.analysis.regime_detection import HurstAnalyzer, RegimeDetector
    
    # Simple Hurst calculation
    hurst = HurstAnalyzer(lookback=252)
    h_exp = hurst.calculate(prices)  # ~0.5 = random, <0.5 = revert, >0.5 = trend
    
    # Full regime detection
    detector = RegimeDetector(prices, returns)
    regime = detector.get_regime()
    print(f"Market regime: {regime}")  # Returns (state, h_exponent, vol, score)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regimes based on Hurst exponent and volatility."""
    TRENDING_LOW_VOL = "Trending Low Vol"      # H > 0.55, vol < median
    TRENDING_HIGH_VOL = "Trending High Vol"    # H > 0.55, vol > median
    MEAN_REVERT_LOW_VOL = "Mean-Revert Low"    # H < 0.45, vol < median
    MEAN_REVERT_HIGH_VOL = "Mean-Revert High"  # H < 0.45, vol > median
    RANDOM_WALK = "Random Walk"                 # 0.45 <= H <= 0.55


@dataclass
class RegimeSnapshot:
    """Current market regime snapshot."""
    state: MarketRegime
    hurst_exponent: float
    volatility: float
    regime_score: float  # Confidence 0-1
    suggested_strategy: str


class HurstAnalyzer:
    """
    Calculate Hurst exponent using rescaled range analysis (R/S).
    
    Hurst exponent H characterizes time-series behavior:
    - H ≈ 0.5: Random walk (Brownian motion)
    - H > 0.5: Persistent (trending, autocorrelated)
    - H < 0.5: Anti-persistent (mean-reverting, oscillates)
    
    Formula: log(R/S) = H * log(N) + constant
    """
    
    def __init__(self, lookback: int = 252, lags: int = 50):
        """
        Initialize Hurst analyzer.
        
        Args:
            lookback: Window size for calculation
            lags: Number of lag scales to test (more = better accuracy)
        """
        self.lookback = lookback
        self.lags = lags
        logger.info(f"Initialized HurstAnalyzer (lookback={lookback}, lags={lags})")
    
    def calculate(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent using rescaled range analysis.
        
        Args:
            prices: Price series
        
        Returns:
            Hurst exponent (0-1, typically 0.3-0.7)
        """
        # Convert prices to log returns
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.lookback:
            logger.warning(f"Series too short ({len(returns)} < {self.lookback})")
            return 0.5  # Default to random walk
        
        # Use last lookback period
        returns = returns[-self.lookback:]
        
        # Calculate R/S for multiple lag scales
        lags = np.logspace(0.5, np.log10(len(returns) // 2), self.lags, dtype=int)
        rs_values = []
        
        for lag in lags:
            # Mean-centered returns
            mean_returns = returns - np.mean(returns)
            
            # Cumulative sum (price path)
            cumsum = np.cumsum(mean_returns)
            
            # Range: max - min over chunks of size lag
            chunks = len(cumsum) // lag
            ranges = []
            
            for i in range(chunks):
                chunk = cumsum[i*lag:(i+1)*lag]
                range_chunk = np.max(chunk) - np.min(chunk)
                ranges.append(range_chunk)
            
            if ranges:
                R = np.mean(ranges)
                
                # Standard deviation of returns
                S = np.std(returns)
                
                # Avoid division by zero
                if S > 0 and R > 0:
                    rs_values.append((lag, R / S))
        
        if not rs_values:
            return 0.5
        
        # Fit log(R/S) = H * log(lag) + constant
        lags_array = np.array([x[0] for x in rs_values])
        rs_array = np.array([x[1] for x in rs_values])
        
        # Linear regression in log-log space
        log_lags = np.log(lags_array)
        log_rs = np.log(rs_array)
        
        # y = H * x + c
        H = np.polyfit(log_lags, log_rs, 1)[0]
        
        # Clamp to reasonable range
        H = np.clip(H, 0.0, 1.0)
        
        logger.debug(f"Hurst exponent calculated: {H:.4f}")
        return float(H)
    
    def calculate_windowed(self, prices: np.ndarray, window: int) \
            -> pd.Series:
        """
        Calculate rolling Hurst exponent.
        
        Args:
            prices: Price series
            window: Rolling window size
        
        Returns:
            Series of Hurst exponents (one per window)
        """
        hurst_values = []
        
        for i in range(window, len(prices)):
            h = self.calculate(prices[i-window:i])
            hurst_values.append(h)
        
        return pd.Series(hurst_values, 
                        index=pd.RangeIndex(window, len(prices)))


class VolatilityRegime:
    """Classify volatility into regimes."""
    
    @staticmethod
    def calculate_realized_vol(returns: np.ndarray, window: int = 20) -> float:
        """
        Calculate realized volatility (rolling std of returns).
        
        Args:
            returns: Return series
            window: Rolling window
        
        Returns:
            Most recent realized volatility
        """
        if len(returns) < window:
            return np.std(returns)
        return np.std(returns[-window:])
    
    @staticmethod
    def get_vol_regime(vol: float, vol_history: np.ndarray) -> str:
        """
        Classify vol into low/normal/high vs historical median.
        
        Args:
            vol: Current volatility
            vol_history: Historical volatility series
        
        Returns:
            "low", "normal", or "high"
        """
        median_vol = np.median(vol_history)
        
        if vol < 0.75 * median_vol:
            return "low"
        elif vol > 1.25 * median_vol:
            return "high"
        else:
            return "normal"


class RegimeDetector:
    """
    Combined regime detection using Hurst + Volatility.
    
    Detects 5 market regimes and suggests appropriate strategies.
    """
    
    def __init__(self, prices: np.ndarray, returns: np.ndarray,
                 lookback: int = 252, vol_window: int = 20):
        """
        Initialize regime detector.
        
        Args:
            prices: Price series
            returns: Return series
            lookback: Window for Hurst calculation
            vol_window: Window for volatility calculation
        """
        self.prices = prices
        self.returns = returns
        self.lookback = lookback
        self.vol_window = vol_window
        
        self.hurst_analyzer = HurstAnalyzer(lookback=lookback)
        
        # Calculate historical volatility for comparison
        self.vol_history = []
        for i in range(vol_window, len(returns)):
            vol = np.std(returns[i-vol_window:i])
            self.vol_history.append(vol)
        
        logger.info(f"Initialized RegimeDetector")
    
    def get_regime(self) -> RegimeSnapshot:
        """
        Detect current market regime.
        
        Returns:
            RegimeSnapshot with state, Hurst exponent, vol, and strategy
        """
        # Calculate Hurst exponent
        h_exp = self.hurst_analyzer.calculate(self.prices)
        
        # Calculate current volatility
        current_vol = VolatilityRegime.calculate_realized_vol(
            self.returns, self.vol_window
        )
        
        # Classify regime
        if h_exp > 0.55:
            trend = "trending"
        elif h_exp < 0.45:
            trend = "mean_reverting"
        else:
            trend = "random_walk"
        
        vol_regime = VolatilityRegime.get_vol_regime(
            current_vol, np.array(self.vol_history)
        )
        
        # Determine state
        if trend == "random_walk":
            state = MarketRegime.RANDOM_WALK
            strategy = "AVOID: No alpha opportunity"
            score = 0.3
        
        elif trend == "trending":
            if vol_regime == "low":
                state = MarketRegime.TRENDING_LOW_VOL
                strategy = "MOMENTUM: Follow trend, wide stops"
                score = 0.8
            else:
                state = MarketRegime.TRENDING_HIGH_VOL
                strategy = "MOMENTUM: Trend with tight risk, watch exits"
                score = 0.7
        
        else:  # mean_reverting
            if vol_regime == "low":
                state = MarketRegime.MEAN_REVERT_LOW_VOL
                strategy = "PAIRS/REVERT: Comfortable edges, normal sizing"
                score = 0.85
            else:
                state = MarketRegime.MEAN_REVERT_HIGH_VOL
                strategy = "REVERT: Dislocation opportunity, check tail risk"
                score = 0.75
        
        return RegimeSnapshot(
            state=state,
            hurst_exponent=h_exp,
            volatility=current_vol,
            regime_score=score,
            suggested_strategy=strategy
        )
    
    def get_regime_history(self, window: int = 252) -> pd.DataFrame:
        """
        Get rolling regime classification over time.
        
        Args:
            window: Rolling window for Hurst calculation
        
        Returns:
            DataFrame with regime history
        """
        hurst_values = self.hurst_analyzer.calculate_windowed(self.prices, window)
        
        # Calculate rolling vol
        vol_values = pd.Series(
            [np.std(self.returns[i-self.vol_window:i])
             for i in range(self.vol_window, len(self.returns))],
            index=pd.RangeIndex(self.vol_window, len(self.returns))
        )
        
        # Classify each point
        regimes = []
        for idx in hurst_values.index:
            if idx < window or idx >= len(self.returns):
                continue
            
            h = hurst_values[idx]
            vol = vol_values[idx] if idx in vol_values.index else np.nan
            
            if np.isnan(h) or np.isnan(vol):
                continue
            
            if h > 0.55:
                regime = "TREND_LOW" if vol < np.median(self.vol_history) else "TREND_HIGH"
            elif h < 0.45:
                regime = "REVERT_LOW" if vol < np.median(self.vol_history) else "REVERT_HIGH"
            else:
                regime = "RANDOM"
            
            regimes.append({'hurst': h, 'vol': vol, 'regime': regime})
        
        return pd.DataFrame(regimes)
    
    def get_position_sizing_multiplier(self) -> float:
        """
        Suggest position size multiplier based on regime confidence.
        
        Returns:
            Multiplier 0-1 (0=avoid, 1=normal size)
        """
        regime = self.get_regime()
        
        # Scale by regime confidence
        base_multiplier = regime.regime_score
        
        # Reduce far out-of-sample predictions
        h_exp = regime.hurst_exponent
        if abs(h_exp - 0.5) < 0.1:
            base_multiplier *= 0.5  # Very uncertain
        
        return base_multiplier


def adaptive_signal_by_regime(base_signal: float, prices: np.ndarray,
                              returns: np.ndarray) -> float:
    """
    Adjust signal strength based on market regime.
    
    Args:
        base_signal: Original signal (-1 to 1)
        prices: Price series
        returns: Return series
    
    Returns:
        Regime-adjusted signal
    """
    detector = RegimeDetector(prices, returns)
    regime = detector.get_regime()
    multiplier = detector.get_position_sizing_multiplier()
    
    # Different signal interpretation by regime
    if regime.state == MarketRegime.TRENDING_LOW_VOL:
        # Strengthen buy signal (momentum)
        adjusted = np.sign(base_signal) * min(1.0, abs(base_signal) * 1.2)
    
    elif regime.state == MarketRegime.MEAN_REVERT_LOW_VOL:
        # Reverse signal (mean-reversion)
        adjusted = -base_signal * 0.8
    
    elif regime.state == MarketRegime.TRENDING_HIGH_VOL:
        # Weaken signal (risky conditions)
        adjusted = base_signal * 0.6
    
    elif regime.state == MarketRegime.MEAN_REVERT_HIGH_VOL:
        # Opportunity but cautious
        adjusted = -base_signal * 0.5
    
    else:  # RANDOM_WALK
        # Suppress signal (no alpha)
        adjusted = 0.0
    
    # Apply confidence multiplier
    adjusted *= multiplier
    
    return np.clip(adjusted, -1.0, 1.0)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate price series with mixed regimes
    prices = [100]
    for _ in range(1000):
        # Mix of trending and mean-reverting segments
        if len(prices) < 250:
            # Trending regime
            ret = np.random.normal(0.001, 0.02)
        elif len(prices) < 500:
            # Mean-reverting regime
            ret = np.random.normal(0, 0.012)
        else:
            # High vol trending
            ret = np.random.normal(0.002, 0.03)
        
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    returns = np.diff(np.log(prices))
    
    # Analyze
    detector = RegimeDetector(prices, returns)
    current_regime = detector.get_regime()
    
    print(f"Current regime: {current_regime.state.value}")
    print(f"Hurst exponent: {current_regime.hurst_exponent:.4f}")
    print(f"Volatility: {current_regime.volatility:.4f}")
    print(f"Strategy: {current_regime.suggested_strategy}")
    print(f"Position sizing multiplier: {detector.get_position_sizing_multiplier():.2f}")
    
    # Adjust signal by regime
    base_signal = 0.7  # Strong buy signal
    adjusted_signal = adaptive_signal_by_regime(base_signal, prices, returns)
    print(f"Base signal: {base_signal:.2f} → Adjusted: {adjusted_signal:.2f}")
    
    logger.info("Regime detection example complete!")
