# src/risk/risk_manager.py
"""
Risk manager for position sizing, hedging, and tail-risk control.
Integrates Greeks, Monte Carlo VaR, and position limits.
"""

import logging
from typing import Dict, Tuple
import numpy as np
from src.models.greeks import GreeksCalculator, portfolio_greeks, delta_neutral_hedge
from src.risk.monte_carlo_var import MonteCarloVaR


logger = logging.getLogger(__name__)


class RiskManager:
    """
    Unified risk management system.
    Monitors Greeks, VaR, and enforces position limits.
    """

    def __init__(
        self,
        initial_capital: float,
        max_var_loss_pct: float = 2.0,  # 2% of capital
        max_delta_notional_pct: float = 1.0,  # 100% of capital
        max_gamma_per_pct_move: float = 0.1,  # max gamma exposure
        enable_delta_hedge: bool = False,
        enable_var_limit: bool = True,
        confidence_level: float = 0.99,
    ):
        """
        Initialize risk manager.
        
        initial_capital: starting capital
        max_var_loss_pct: max acceptable loss at confidence level
        max_delta_notional_pct: max notional delta exposure
        max_gamma_per_pct_move: max gamma exposure per 1% move
        enable_delta_hedge: auto-hedge delta with stock
        enable_var_limit: enforce VaR limits
        confidence_level: VaR confidence (0.99 = 99%)
        """
        self.initial_capital = initial_capital
        self.max_var_loss_pct = max_var_loss_pct
        self.max_delta_notional_pct = max_delta_notional_pct
        self.max_gamma_per_pct_move = max_gamma_per_pct_move
        self.enable_delta_hedge = enable_delta_hedge
        self.enable_var_limit = enable_var_limit
        self.confidence_level = confidence_level
        
        self.current_equity = initial_capital
        self.options_positions = []  # list of dicts with option details
        self.stock_positions = {}  # ticker -> shares

    def add_option_position(
        self,
        position: Dict,
    ):
        """
        Register an options position.
        
        position should have keys:
            - quantity, S, K, T, r, sigma, type
        """
        self.options_positions.append(position)

    def add_stock_position(self, ticker: str, shares: float):
        """Register a stock position."""
        self.stock_positions[ticker] = shares

    def compute_greeks(self) -> Dict:
        """Calculate portfolio Greeks."""
        if not self.options_positions:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
        
        return portfolio_greeks(self.options_positions)

    def check_delta_limit(
        self,
        spot_price: float,
        model_signal: float,
    ) -> Tuple[bool, float]:
        """
        Check if proposed trade exceeds delta limit.
        
        spot_price: current price
        model_signal: target position fraction (-1 to +1)
        
        Returns: (is_allowed, risk_score)
        """
        greeks = self.compute_greeks()
        current_delta = greeks["delta"]
        
        target_notional = model_signal * self.initial_capital
        max_notional = (self.max_delta_notional_pct / 100.0) * self.initial_capital
        
        if abs(target_notional) > max_notional:
            risk_score = abs(target_notional) / max_notional
            logger.warning(
                f"Delta limit exceeded: target_notional={target_notional:.0f}, "
                f"max={max_notional:.0f}, score={risk_score:.2f}"
            )
            return False, risk_score
        
        return True, 1.0

    def compute_var_limit(
        self,
        current_price: float,
        position_size: float,
        volatility: float = 0.15,
        num_paths: int = 10000,
    ) -> Tuple[bool, float, float]:
        """
        Check if position violates VaR limit.
        
        Returns: (is_allowed, var_loss, var_loss_pct)
        """
        if not self.enable_var_limit:
            return True, 0.0, 0.0
        
        var_result = MonteCarloVaR.one_day_var(
            current_price=current_price,
            position_size=position_size,
            volatility=volatility,
            confidence=self.confidence_level,
            num_paths=num_paths,
        )
        
        var_loss = var_result["var_loss"]
        max_allowed_loss = (self.max_var_loss_pct / 100.0) * self.initial_capital
        
        if var_loss > max_allowed_loss:
            logger.warning(
                f"VaR limit exceeded: var_loss={var_loss:.0f}, "
                f"max_allowed={max_allowed_loss:.0f}"
            )
            return False, var_loss, var_result.get("var_loss_pct", 0)
        
        return True, var_loss, var_result.get("var_loss_pct", 0)

    def suggest_delta_hedge_size(
        self,
        hedge_price: float,
    ) -> float:
        """
        Suggest number of shares to buy/sell for delta hedge.
        
        hedge_price: price of hedge instrument (typically stock)
        
        Returns: shares to buy (positive) or sell (negative)
        """
        greeks = self.compute_greeks()
        delta = greeks["delta"]
        
        hedge_size = delta_neutral_hedge(
            spot_price=hedge_price,
            position_delta=delta,
            hedge_instrument_delta=1.0,
        )
        
        return hedge_size

    def compute_gamma_risk(self) -> Dict:
        """
        Estimate gamma risk under various market moves.
        
        Returns: dict with P&L estimate per 1% move
        """
        greeks = self.compute_greeks()
        gamma = greeks["gamma"]
        
        results = {}
        for move_pct in [0.01, 0.02, 0.05, 0.10]:  # 1%, 2%, 5%, 10% moves
            pnl = 0.5 * gamma * (move_pct ** 2) * self.initial_capital
            results[f"{move_pct*100:.0f}pct_move"] = pnl
        
        return results

    def compute_vega_risk(self) -> Dict:
        """
        Estimate vega risk under various vol moves.
        
        Returns: dict with P&L estimate per 1% vol move
        """
        greeks = self.compute_greeks()
        vega = greeks["vega"]
        
        results = {}
        for vol_move in [0.01, 0.02, 0.05, 0.10]:
            pnl = vega * vol_move * self.initial_capital
            results[f"{vol_move*100:.0f}pct_vol_move"] = pnl
        
        return results

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk snapshot."""
        greeks = self.compute_greeks()
        gamma_risk = self.compute_gamma_risk()
        vega_risk = self.compute_vega_risk()
        
        return {
            "greeks": greeks,
            "gamma_risk_pnl": gamma_risk,
            "vega_risk_pnl": vega_risk,
            "max_var_loss_pct": self.max_var_loss_pct,
            "max_delta_notional_pct": self.max_delta_notional_pct,
            "enable_delta_hedge": self.enable_delta_hedge,
            "enable_var_limit": self.enable_var_limit,
        }


def apply_jump_risk_hedge(
    vega: float,
    implied_vol: float,
    jump_indicator: float = 0.0,  # 0-1 scale
) -> float:
    """
    Adjust vega hedge for jump risk.
    
    When market conditions suggest high jump risk (volatility clustering,
    upcoming earnings, etc.), increase vega hedging.
    
    jump_indicator: 0-1 scale for jump risk intensity
    
    Returns: adjusted vega hedge multiplier
    """
    base_multiplier = 1.0
    jump_adjustment = 1.0 + (jump_indicator * 2.0)  # up to 3x more hedging
    
    return base_multiplier * jump_adjustment


def rebalance_greeks(
    current_greeks: Dict,
    target_greeks: Dict = None,
    rebalance_threshold: float = 0.1,
) -> Dict:
    """
    Determine if rebalancing is needed based on Greeks drift.
    
    current_greeks: current portfolio greeks
    target_greeks: desired greeks (default: all zeros)
    rebalance_threshold: drift tolerance (e.g., 0.1 = 10%)
    
    Returns: rebalancing recommendation
    """
    if target_greeks is None:
        target_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    
    recommendation = {}
    for greek_name in ["delta", "gamma", "vega"]:
        current = current_greeks.get(greek_name, 0.0)
        target = target_greeks.get(greek_name, 0.0)
        
        drift = abs(current - target) / (abs(target) + 1e-9)
        
        if drift > rebalance_threshold:
            recommendation[greek_name] = {
                "current": current,
                "target": target,
                "drift_pct": drift * 100,
                "action": "rebalance_needed",
            }
        else:
            recommendation[greek_name] = {
                "current": current,
                "target": target,
                "drift_pct": drift * 100,
                "action": "no_action",
            }
    
    return recommendation
