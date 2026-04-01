# src/risk/monte_carlo_var.py
"""
Monte Carlo Value-at-Risk (VaR) and Expected Shortfall (ES) calculator.
Used for tail-risk hedging and position sizing.

VaR: Loss threshold at 99% confidence level (1% worst case)
ES (CVaR): Average loss conditional on exceeding VaR
"""

import numpy as np
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


class MonteCarloVaR:
    """Monte Carlo VaR/ES for equity portfolios with correlated assets."""

    @staticmethod
    def simulate_price_paths(
        spot: float,
        mu: float,
        sigma: float,
        T: float,
        dt: float,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate geometric Brownian motion price paths.
        
        dS = mu * S * dt + sigma * S * dW
        
        spot: initial price
        mu: drift (annual return)
        sigma: volatility (annual)
        T: horizon (years)
        dt: time step (years)
        num_paths: number of Monte Carlo simulations
        seed: random seed for reproducibility
        
        Returns: array of shape (num_steps, num_paths) with simulated prices
        """
        np.random.seed(seed)
        
        num_steps = int(T / dt) + 1
        paths = np.zeros((num_steps, num_paths))
        paths[0] = spot
        
        for t in range(1, num_steps):
            Z = np.random.standard_normal(num_paths)
            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
        
        return paths

    @staticmethod
    def simulate_portfolio_paths(
        spots: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        corr_matrix: np.ndarray,
        T: float,
        dt: float,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate correlated price paths for a portfolio.
        
        spots: array of initial prices
        mus: array of drifts
        sigmas: array of volatilities
        corr_matrix: correlation matrix
        T, dt, num_paths, seed: as in simulate_price_paths
        
        Returns: array of shape (num_steps, num_assets, num_paths)
        """
        np.random.seed(seed)
        num_assets = len(spots)
        num_steps = int(T / dt) + 1
        
        paths = np.zeros((num_steps, num_assets, num_paths))
        paths[0, :, :] = spots[:, np.newaxis]
        
        # Cholesky decomposition of correlation matrix for correlated shocks
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite; using identity")
            L = np.eye(num_assets)
        
        for t in range(1, num_steps):
            Z_indep = np.random.standard_normal((num_assets, num_paths))
            Z_corr = L @ Z_indep
            
            for i in range(num_assets):
                mu_i = mus[i]
                sigma_i = sigmas[i]
                paths[t, i, :] = paths[t - 1, i, :] * np.exp(
                    (mu_i - 0.5 * sigma_i**2) * dt + sigma_i * np.sqrt(dt) * Z_corr[i, :]
                )
        
        return paths

    @staticmethod
    def calculate_var_es(
        returns: np.ndarray,
        confidence: float = 0.99,
    ) -> Dict[str, float]:
        """
        Calculate VaR and ES from return distribution.
        
        returns: array of simulated P&L or returns (negative for losses)
        confidence: confidence level (0.99 = 99%)
        
        Returns: dict with 'var' (value-at-risk) and 'es' (expected shortfall)
        """
        alpha = 1.0 - confidence  # tail probability
        var = np.percentile(returns, alpha * 100)
        
        # ES = average of returns worse than VaR
        tail_returns = returns[returns <= var]
        if len(tail_returns) > 0:
            es = np.mean(tail_returns)
        else:
            es = var
        
        return {
            "var": float(var),
            "es": float(es),
            "var_pct": float(var / np.abs(np.mean(returns)) * 100) if np.mean(returns) != 0 else 0.0,
            "es_pct": float(es / np.abs(np.mean(returns)) * 100) if np.mean(returns) != 0 else 0.0,
        }

    @staticmethod
    def one_day_var(
        current_price: float,
        position_size: float,
        expected_return: float = 0.0,
        volatility: float = 0.15,
        confidence: float = 0.99,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Calculate 1-day ahead VaR for a single position.
        
        current_price: spot price
        position_size: number of shares held
        expected_return: daily drift (default 0)
        volatility: daily volatility
        confidence: confidence level (0.99)
        num_paths: MC simulations
        seed: random seed
        
        Returns: dict with 'var_loss', 'es_loss', 'var_loss_pct', etc.
        """
        T = 1.0 / 252.0  # 1 trading day
        dt = T
        
        paths = MonteCarloVaR.simulate_price_paths(
            spot=current_price,
            mu=expected_return * 252,  # annualize
            sigma=volatility * np.sqrt(252),  # annualize
            T=T,
            dt=dt,
            num_paths=num_paths,
            seed=seed,
        )
        
        final_prices = paths[-1, :]
        pnl = (final_prices - current_price) * position_size
        
        var_dict = MonteCarloVaR.calculate_var_es(pnl, confidence)
        
        # Make losses positive (negative abs value)
        var_dict["var_loss"] = -var_dict["var"]
        var_dict["es_loss"] = -var_dict["es"]
        var_dict["var_loss_pct"] = -var_dict.get("var_pct", 0)
        var_dict["es_loss_pct"] = -var_dict.get("es_pct", 0)
        
        return var_dict

    @staticmethod
    def portfolio_var(
        holdings: Dict[str, Dict],
        confidence: float = 0.99,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> Dict:
        """
        Calculate portfolio VaR across multiple positions.
        
        holdings: dict like {
            'AAPL': {'price': 150, 'shares': 100, 'daily_return': 0.001, 'daily_vol': 0.018},
            'MSFT': {'price': 300, 'shares': 50, 'daily_return': 0.0008, 'daily_vol': 0.016},
            ...
        }
        
        confidence: confidence level
        num_paths: MC simulations
        seed: random seed
        
        Returns: dict with portfolio-level VaR/ES + per-asset breakdown
        """
        tickers = list(holdings.keys())
        num_assets = len(tickers)
        
        spots = np.array([holdings[t]["price"] for t in tickers])
        mus = np.array([holdings[t].get("daily_return", 0.0) for t in tickers])
        sigmas = np.array([holdings[t].get("daily_vol", 0.02) for t in tickers])
        shares = np.array([holdings[t]["shares"] for t in tickers])
        
        # Default: assume 50% correlation between assets (can customize)
        corr_matrix = np.eye(num_assets) * 0.5
        np.fill_diagonal(corr_matrix, 1.0)
        
        T = 1.0 / 252.0
        dt = T
        
        # Simulate correlated paths
        paths = MonteCarloVaR.simulate_portfolio_paths(
            spots=spots,
            mus=mus * 252,  # annualize
            sigmas=sigmas * np.sqrt(252),  # annualize
            corr_matrix=corr_matrix,
            T=T,
            dt=dt,
            num_paths=num_paths,
            seed=seed,
        )
        
        # Calculate final P&L across all scenarios
        final_prices = paths[-1]  # shape (num_assets, num_paths)
        portfolio_values = np.sum(
            (final_prices - spots[:, np.newaxis]) * shares[:, np.newaxis],
            axis=0,
        )
        
        portfolio_var_dict = MonteCarloVaR.calculate_var_es(portfolio_values, confidence)
        
        # Per-asset VaR
        per_asset_var = {}
        for i, ticker in enumerate(tickers):
            asset_pnl = (final_prices[i, :] - spots[i]) * shares[i]
            per_asset_var[ticker] = MonteCarloVaR.calculate_var_es(asset_pnl, confidence)
        
        return {
            "portfolio_var": -portfolio_var_dict["var"],
            "portfolio_es": -portfolio_var_dict["es"],
            "portfolio_var_pct": -portfolio_var_dict.get("var_pct", 0),
            "portfolio_es_pct": -portfolio_var_dict.get("es_pct", 0),
            "per_asset": per_asset_var,
            "confidence_level": confidence,
            "num_simulations": num_paths,
        }


def stress_test_portfolio(
    holdings: Dict[str, Dict],
    shock_scenarios: Dict[str, float],
) -> Dict:
    """
    Apply historical/hypothetical shocks to portfolio and calculate P&L impact.
    
    holdings: as in portfolio_var
    shock_scenarios: dict like {
        'market_crash_10pct': -0.10,
        'tech_spike_5pct': 0.05,
        'vol_spike': 0.20,
    }
    
    Returns: P&L under each scenario
    """
    results = {}
    
    for scenario_name, shock in shock_scenarios.items():
        portfolio_pnl = 0.0
        for ticker, holding in holdings.items():
            price = holding["price"]
            shares = holding["shares"]
            
            # Apply shock to price
            new_price = price * (1 + shock)
            position_pnl = (new_price - price) * shares
            
            portfolio_pnl += position_pnl
        
        results[scenario_name] = {
            "total_pnl": portfolio_pnl,
            "pnl_pct": portfolio_pnl / sum(h["price"] * h["shares"] for h in holdings.values()) * 100,
        }
    
    return results


def kelly_criterion_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    kelly_fraction = (bp - q) / b
    where p = win_rate, q = 1 - win_rate
          b = avg_win / avg_loss
    
    Returns fraction of capital to allocate (typically use kelly / 4 for safety).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss
    
    kelly_frac = (b * p - q) / b
    
    return max(0.0, kelly_frac)  # never negative
