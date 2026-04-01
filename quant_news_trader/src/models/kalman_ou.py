"""
Kalman Filter + Ornstein-Uhlenbeck (O-U) Mean-Reversion Model
==============================================================

Identifies mean-reversion trading opportunities using:
1. Kalman Filter: estimates true price vs noise
2. O-U Process: models mean-reversion dynamics
3. Spread analysis: detects deviation from equilibrium
4. Signal generation: entry/exit for pairs trading

The O-U process: dx(t) = θ(μ - x(t))dt + σ dW(t)
- θ: mean-reversion speed (faster = quicker recovery)
- μ: long-term equilibrium
- σ: volatility

Example:
    from src.models.kalman_ou import KalmanOUFilter, PairsTrader
    
    # Track mean-reversion on single stock
    kou = KalmanOUFilter(lookback=252)
    kou.fit(price_series)
    signal = kou.predict(current_price)  # 1.0=max buy, -1.0=max sell
    
    # Pairs trading (two correlated stocks)
    pairs = PairsTrader('STOCK_A', 'STOCK_B')
    pairs.fit(price_a, price_b)
    hedge_ratio, spread = pairs.get_spread()  # Spread from equilibrium
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    """Ornstein-Uhlenbeck process parameters."""
    theta: float  # Mean-reversion speed
    mu: float     # Long-term equilibrium
    sigma: float  # Volatility
    
    def __str__(self):
        return f"O-U(θ={self.theta:.4f}, μ={self.mu:.4f}, σ={self.sigma:.4f})"


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for price tracking.
    
    State: true price
    Measurement: observed price (noisy)
    """
    
    def __init__(self, process_variance: float = 1e-4, 
                 measurement_variance: float = 1e-2):
        """
        Initialize Kalman filter.
        
        Args:
            process_variance: σ_process² (how much state can change)
            measurement_variance: σ_measurement² (sensor noise)
        """
        self.q = process_variance      # Process noise
        self.r = measurement_variance  # Measurement noise
        self.x = None  # State estimate
        self.p = 1.0   # Error covariance
    
    def update(self, z: float) -> float:
        """
        Update filter with new measurement.
        
        Args:
            z: Observed price
        
        Returns:
            Filtered (denoised) price estimate
        """
        if self.x is None:
            self.x = z
            return z
        
        # Prediction step
        x_pred = self.x
        p_pred = self.p + self.q
        
        # Update step
        K = p_pred / (p_pred + self.r)  # Kalman gain
        self.x = x_pred + K * (z - x_pred)
        self.p = (1 - K) * p_pred
        
        return self.x
    
    def reset(self):
        """Reset filter to initial state."""
        self.x = None
        self.p = 1.0


class KalmanOUFilter:
    """
    Combined Kalman Filter + Ornstein-Uhlenbeck model for mean-reversion trading.
    
    Workflow:
    1. Filter noisy prices with Kalman
    2. Fit O-U process to filtered prices
    3. Compute spread (price - equilibrium)
    4. Generate signal based on spread deviation
    """
    
    def __init__(self, lookback: int = 252, z_threshold: float = 2.0):
        """
        Initialize Kalman+O-U system.
        
        Args:
            lookback: Historical window for fitting
            z_threshold: Z-score threshold for signals (2=2σ deviation)
        """
        self.lookback = lookback
        self.z_threshold = z_threshold
        
        self.kalman = KalmanFilter1D()
        self.ou_params = None
        
        self.price_history = []
        self.filtered_history = []
        
        logger.info(f"Initialized KalmanOUFilter (lookback={lookback}, z={z_threshold}σ)")
    
    def fit(self, prices: np.ndarray) -> OUParams:
        """
        Fit O-U process to price series.
        
        Args:
            prices: Historical price array
        
        Returns:
            Fitted OUParams
        """
        # Pass through Kalman filter first
        filtered_prices = []
        for p in prices:
            filtered = self.kalman.update(p)
            filtered_prices.append(filtered)
        
        self.filtered_history = filtered_prices
        self.price_history = list(prices)
        
        # Fit O-U parameters using MLE
        self.ou_params = self._fit_ou_mle(np.array(filtered_prices))
        
        logger.info(f"Fitted O-U model: {self.ou_params}")
        return self.ou_params
    
    def _fit_ou_mle(self, prices: np.ndarray, dt: float = 1.0) -> OUParams:
        """
        Estimate O-U parameters using maximum likelihood.
        
        dx = θ(μ - x)dt + σ dW
        
        Args:
            prices: Price series
            dt: Time step (1 = daily)
        
        Returns:
            OUParams with fitted θ, μ, σ
        """
        # Simple estimators
        mu = np.mean(prices)
        dx = np.diff(prices)
        
        # θ: mean-reversion speed (regression of dx on (mu - x))
        x_deviations = prices[:-1] - mu
        
        if np.sum(x_deviations**2) > 0:
            theta = -np.sum(dx * x_deviations) / np.sum(x_deviations**2) / dt
            theta = max(0.001, theta)  # Ensure positive
        else:
            theta = 0.05
        
        # σ: residual volatility
        predicted_dx = -theta * x_deviations * dt
        residuals = dx - predicted_dx
        sigma = np.std(residuals) / np.sqrt(dt)
        sigma = max(0.001, sigma)
        
        return OUParams(theta=theta, mu=mu, sigma=sigma)
    
    def predict(self, price: float) -> Tuple[float, Dict[str, float]]:
        """
        Generate mean-reversion signal.
        
        Args:
            price: Current price
        
        Returns:
            Tuple of (signal, diagnostics)
            - signal: -1.0 (strong sell), 0.0 (neutral), 1.0 (strong buy)
            - diagnostics: spread, z_score, mean, etc.
        """
        if self.ou_params is None:
            raise ValueError("Call fit() first")
        
        # Filter current price
        filtered_price = self.kalman.update(price)
        
        # Compute spread from equilibrium
        spread = filtered_price - self.ou_params.mu
        
        # Z-score (std = sigma / sqrt(2*theta) for O-U equilibrium)
        ou_std = self.ou_params.sigma / np.sqrt(2 * self.ou_params.theta)
        z_score = spread / ou_std if ou_std > 0 else 0
        
        # Signal:
        # z > +threshold: price too high → sell
        # z < -threshold: price too low → buy
        # between: neutral
        
        if z_score > self.z_threshold:
            signal = -1.0  # Strong sell
        elif z_score > 0.5 * self.z_threshold:
            signal = -0.5  # Weak sell
        elif z_score < -self.z_threshold:
            signal = 1.0  # Strong buy
        elif z_score < -0.5 * self.z_threshold:
            signal = 0.5  # Weak buy
        else:
            signal = 0.0  # Neutral
        
        diagnostics = {
            'filtered_price': filtered_price,
            'equilibrium': self.ou_params.mu,
            'spread': spread,
            'z_score': z_score,
            'theta': self.ou_params.theta,
            'sigma': self.ou_params.sigma,
            'ou_std': ou_std,
        }
        
        return signal, diagnostics
    
    def half_life_reversion(self) -> float:
        """
        Compute half-life of mean reversion.
        
        Time for price to revert halfway to equilibrium.
        
        Returns:
            Half-life in periods (e.g., days if daily data)
        """
        if self.ou_params is None:
            return float('inf')
        
        # Half-life = ln(2) / θ
        return np.log(2) / self.ou_params.theta if self.ou_params.theta > 0 else float('inf')


class PairsTrader:
    """
    Statistical arbitrage using pairs trading (cointegration).
    
    Identifies two correlated assets, trades their spread when it deviates
    from equilibrium (mean-reversion setup).
    
    Strategy:
    - Buy undervalued leg, short overvalued leg → delta-neutral
    - Exit when spread reverts
    """
    
    def __init__(self, ticker_a: str, ticker_b: str, lookback: int = 252):
        """
        Initialize pairs trader.
        
        Args:
            ticker_a: First stock (e.g., 'AAPL')
            ticker_b: Second stock (e.g., 'MSFT')
            lookback: Historical window
        """
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.lookback = lookback
        
        self.kalman_ou_a = KalmanOUFilter(lookback=lookback)
        self.kalman_ou_b = KalmanOUFilter(lookback=lookback)
        
        self.hedge_ratio = 1.0  # How many B to short per 1 A long
        self.spread_ou = None
        
        logger.info(f"Initialized PairsTrader({ticker_a}, {ticker_b})")
    
    def fit(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """
        Fit hedge ratio using linear regression.
        
        prices_b = β * prices_a + α + ε
        
        Args:
            prices_a: Price series for ticker A
            prices_b: Price series for ticker B
        
        Returns:
            Hedge ratio (β)
        """
        # Fit each series through O-U filter
        self.kalman_ou_a.fit(prices_a)
        self.kalman_ou_b.fit(prices_b)
        
        # Hedge ratio: regress B on A
        A = np.column_stack([prices_a, np.ones_like(prices_a)])
        coeffs = np.linalg.lstsq(A, prices_b, rcond=None)[0]
        self.hedge_ratio = coeffs[0]
        
        # Spread: residuals from regression
        spread = prices_b - self.hedge_ratio * prices_a
        
        # Fit O-U to spread
        self.spread_ou = KalmanOUFilter(lookback=self.lookback)
        self.spread_ou.fit(spread)
        
        logger.info(f"Fitted hedge ratio β={self.hedge_ratio:.4f}")
        logger.info(f"Spread O-U: {self.spread_ou.ou_params}")
        
        return self.hedge_ratio
    
    def get_spread(self, price_a: float, price_b: float) -> Tuple[float, Dict]:
        """
        Compute current spread from equilibrium.
        
        Args:
            price_a: Current price of ticker A
            price_b: Current price of ticker B
        
        Returns:
            Tuple of (hedge_ratio, spread_info)
        """
        if self.spread_ou is None:
            raise ValueError("Call fit() first")
        
        spread = price_b - self.hedge_ratio * price_a
        signal, diagnostics = self.spread_ou.predict(spread)
        
        return self.hedge_ratio, {
            'spread': spread,
            'signal': signal,
            'diagnostics': diagnostics,
        }
    
    def get_position_size(self, price_a: float, price_b: float,
                         notional_capital: float) -> Dict[str, float]:
        """
        Compute position sizing for pairs trade.
        
        Args:
            price_a: Price of A
            price_b: Price of B
            notional_capital: Capital to allocate
        
        Returns:
            Dict with quantity_a, quantity_b (positive=long, negative=short)
        """
        hedge_ratio, spread_info = self.get_spread(price_a, price_b)
        signal = spread_info['signal']
        
        if abs(signal) < 0.5:
            return {'quantity_a': 0, 'quantity_b': 0, 'notional': 0}
        
        # If spread is too high: short B, long A
        # If spread is too low: long B, short A
        
        if signal > 0:  # Spread too low → buy A, short B
            qty_a = notional_capital / (2 * price_a)
            qty_b = -hedge_ratio * qty_a
        else:  # Spread too high → short A, buy B
            qty_a = -notional_capital / (2 * price_a)
            qty_b = -hedge_ratio * qty_a
        
        return {
            'quantity_a': qty_a,
            'quantity_b': qty_b,
            'notional': abs(notional_capital),
            'signal': signal,
            'spread': spread_info['spread'],
        }


def portfolio_ou_filter(returns_df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Apply O-U filtering to entire portfolio of assets.
    
    Args:
        returns_df: DataFrame with prices by column
        lookback: Fitting window
    
    Returns:
        DataFrame with mean-reversion signals for each asset
    """
    signals = {}
    
    for col in returns_df.columns:
        try:
            kou = KalmanOUFilter(lookback=lookback)
            prices = returns_df[col].dropna().values
            
            if len(prices) > lookback:
                kou.fit(prices[-lookback:])
                signal, _ = kou.predict(prices[-1])
                signals[col] = signal
        except Exception as e:
            logger.error(f"Error processing {col}: {e}")
            signals[col] = 0.0
    
    return pd.DataFrame([signals])


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate mean-reverting price series
    ou_process = OUParams(theta=0.05, mu=100, sigma=2)
    prices = [100]
    for _ in range(500):
        dt = 1.0
        dx = ou_process.theta * (ou_process.mu - prices[-1]) * dt + \
             ou_process.sigma * np.random.normal(0, np.sqrt(dt))
        prices.append(prices[-1] + dx)
    
    prices = np.array(prices)
    
    # Fit and predict
    kou = KalmanOUFilter(lookback=252)
    kou.fit(prices[:250])
    
    signal, diag = kou.predict(prices[250])
    print(f"Signal: {signal:.2f}")
    print(f"Diagnostics: {diag}")
    print(f"Half-life of reversion: {kou.half_life_reversion():.1f} periods")
    
    logger.info("Kalman+O-U example complete!")
