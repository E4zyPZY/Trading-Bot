# src/models/greeks.py
"""
Greeks calculator for options positions: Delta, Gamma, Vega.
Used for Delta-Gamma-Vega neutral hedging strategies.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


class GreeksCalculator:
    """Compute Black-Scholes Greeks for European options."""

    @staticmethod
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1, d2 for Black-Scholes.
        
        S: current spot price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility (annualized)
        """
        if T <= 0 or sigma <= 0:
            return np.nan, np.nan
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Delta: ∂V/∂S = rate of change of option value w.r.t. spot price.
        
        For calls: N(d1)
        For puts: -N(-d1) = N(d1) - 1
        """
        d1, _ = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if np.isnan(d1):
            return 0.0
        
        if option_type.lower() == "call":
            return float(norm.cdf(d1))
        else:  # put
            return float(norm.cdf(d1) - 1.0)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Gamma: ∂²V/∂S² = curvature of option value.
        
        Same for calls and puts (always positive).
        Highest at-the-money.
        """
        d1, _ = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if np.isnan(d1) or S <= 0 or sigma <= 0 or T <= 0:
            return 0.0
        
        numerator = norm.pdf(d1)
        denominator = S * sigma * np.sqrt(T)
        return float(numerator / denominator)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Vega: ∂V/∂σ = sensitivity to 1% change in volatility.
        
        Same for calls and puts.
        Per 1% change in vol (divide by 100 for 0.01 vol change).
        """
        d1, _ = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if np.isnan(d1) or S <= 0 or T <= 0:
            return 0.0
        
        return float(S * norm.pdf(d1) * np.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Theta: ∂V/∂t = time decay per day.
        
        Negative for long calls/puts (time decay works against long)
        Positive for short calls/puts.
        Return is daily theta (divided by 365).
        """
        d1, d2 = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if np.isnan(d1) or T <= 0 or S <= 0:
            return 0.0
        
        sqrt_T = np.sqrt(T)
        
        if option_type.lower() == "call":
            term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = term1 - term2
        else:  # put
            term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = term1 + term2
        
        return float(theta / 365.0)  # daily theta

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Rho: ∂V/∂r = sensitivity to 1% change in interest rate.
        """
        d1, d2 = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if np.isnan(d2) or T <= 0 or K <= 0:
            return 0.0
        
        if option_type.lower() == "call":
            return float(K * T * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))


def portfolio_greeks(positions: list) -> Dict[str, float]:
    """
    Aggregate Greeks across a portfolio of options.
    
    positions: list of dicts with keys:
        - quantity: shares/contracts held (positive=long, negative=short)
        - S: spot price
        - K: strike price
        - T: time to maturity
        - r: risk-free rate
        - sigma: volatility
        - type: 'call' or 'put'
    
    Returns: aggregate Delta, Gamma, Vega
    """
    calc = GreeksCalculator()
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    
    for pos in positions:
        qty = pos.get("quantity", 0)
        S = pos.get("S", 0)
        K = pos.get("K", 0)
        T = pos.get("T", 0)
        r = pos.get("r", 0.03)
        sigma = pos.get("sigma", 0)
        opt_type = pos.get("type", "call")
        
        delta_i = calc.delta(S, K, T, r, sigma, opt_type)
        gamma_i = calc.gamma(S, K, T, r, sigma)
        vega_i = calc.vega(S, K, T, r, sigma)
        
        # Scale by quantity
        total_delta += qty * delta_i
        total_gamma += qty * gamma_i
        total_vega += qty * vega_i
    
    return {
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
    }


def delta_neutral_hedge(
    spot_price: float,
    position_delta: float,
    hedge_instrument_delta: float = 1.0,
) -> float:
    """
    Calculate number of shares/contracts needed to delta-hedge a position.
    
    If position_delta = 50 (50 delta exposure, e.g., 50 call options),
    and we hedge with stock (delta_per_share = 1.0),
    we need to sell 50 shares to be delta-neutral.
    
    Returns: number of units to buy (positive) or sell (negative).
    """
    if hedge_instrument_delta == 0:
        return 0.0
    
    hedge_size = -position_delta / hedge_instrument_delta
    return hedge_size


def gamma_exposure_metric(gamma_position: float, spot_price: float, move_pct: float = 0.01) -> float:
    """
    Estimate daily P&L from gamma exposure under a hypothetical spot move.
    
    Approximation: gamma_pnl ≈ 0.5 * gamma * (ΔS)²
    
    gamma_position: aggregate portfolio gamma
    spot_price: current price
    move_pct: expected spot move as % (e.g., 0.01 for 1%)
    
    Returns: estimated P&L from gamma
    """
    spot_move = spot_price * move_pct
    gamma_pnl = 0.5 * gamma_position * (spot_move ** 2)
    return gamma_pnl


def vega_exposure_metric(vega_position: float, vol_move_pct: float = 0.01) -> float:
    """
    Estimate daily P&L from vega exposure under a volatility move.
    
    vega_pnl = vega * Δvol
    
    vega_position: aggregate portfolio vega
    vol_move_pct: expected vol move as absolute % (e.g., 0.01 for +1%)
    
    Returns: estimated P&L from vega
    """
    vega_pnl = vega_position * vol_move_pct
    return vega_pnl


def jump_risk_adjustment(implied_vol: float, jump_intensity: float = 0.05) -> float:
    """
    Adjust Greeks for jump risk (sudden volatility spikes).
    
    Multiplier applied to vega hedging demands when jump risk is elevated.
    
    jump_intensity: estimated probability of jump event (0-1)
    
    Returns: vega hedge scaling factor
    """
    # Merton jump-diffusion style: increase vega hedge when jump prob rises
    base_vega_hedge = 1.0
    jump_adjustment = 1.0 + (jump_intensity * 2.0)  # linear scaling
    
    return base_vega_hedge * jump_adjustment
