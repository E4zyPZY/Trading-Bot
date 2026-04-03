import math
from typing import List, Dict

import pandas as pd


class SimBroker:
    """
    Simple risk-limited broker simulator.

    - Fixed starting capital (initial_cash)
    - No leverage (cannot have negative cash from buying)
    - max_loss_fraction: e.g. 1.0 = can lose up to 100% of initial_cash, but never go below 0.
                          0.3  = can lose up to 30% (equity floor at 70% of initial_cash)
    - step_to_target_fraction(dt, price, target_fraction):
        target_fraction in [-1, 1] = fraction of initial_cash to be exposed to the asset
        (long or short).
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        commission_rate: float = 0.0,
        slippage_bp: float = 0.0,
        max_loss_fraction: float = 1.0,
        max_position_fraction: float = 0.2,  # NEW
    ):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0

        self.commission_per_trade = float(commission_per_trade)  # minimum fee per trade
        self.commission_rate = float(commission_rate)  # fraction of trade value (e.g. 0.00003)
        self.slippage_bp = float(slippage_bp)
        self.max_loss_fraction = float(max_loss_fraction)
        self.max_position_fraction = float(max_position_fraction)

        self.active = True
        self.trade_log = []
        self.max_equity = float(initial_cash)

    # ---------- Core metrics ----------

    def equity(self, price: float) -> float:
        """Current account equity at given price."""
        return self.cash + self.position * float(price)

    def _calc_commission(self, gross_trade_value: float) -> float:
        """
        Commission = max(commission_per_trade, commission_rate * abs(trade_value)).
        If both are zero, commission is zero.
        """
        rate_fee = self.commission_rate * abs(gross_trade_value)
        return max(self.commission_per_trade, rate_fee)

    def _equity_floor(self) -> float:
        """
        Minimum allowed equity based on max_loss_fraction.
        max_loss_fraction = 1.0 → floor = 0
        max_loss_fraction = 0.3 → floor = 0.7 * initial_cash
        """
        max_loss = self.initial_cash * self.max_loss_fraction
        return self.initial_cash - max_loss

    def _max_loss_reached(self, price: float) -> bool:
        """Check if equity has breached allowed loss."""
        eq = self.equity(price)
        floor = self._equity_floor()
        return eq <= floor + 1e-8

    # ---------- Trading logic ----------

    def step_to_target_fraction(self, dt, price: float, target_fraction: float):
        """
        Move position so that notional exposure ≈ target_fraction * initial_cash,
        subject to:
          - no leverage (cash cannot go negative on a buy)
          - max-loss hard stop
        """
        price = float(price)
        if price <= 0:
            # Can't do anything sane with non-positive prices
            self._log_state(dt, price, target_fraction, event="invalid_price")
            return

        # If we've already hit max loss, stay flat forever.
        if not self.active or self._max_loss_reached(price):
            if self.active:
                self._force_liquidation(dt, price, reason="max_loss_reached_entry")
            else:
                # Already inactive
                self._log_state(dt, price, target_fraction, event="inactive")
            return

        # Clamp target_fraction to [-1, 1] (max 100% long or short of initial capital)
        target_fraction = max(min(target_fraction, 1.0), -1.0)

        # Use initial_cash as risk capital baseline
        # Equity-based exposure limit
        equity_now = self.equity(price)
        max_notional = self.max_position_fraction * equity_now

        # Desired notional based on model
        desired_notional = target_fraction * self.initial_cash

        # Clamp exposure so we never exceed 20% of equity
        target_notional = max(-max_notional, min(max_notional, desired_notional))

        current_notional = self.position * price
        delta_notional = target_notional - current_notional

        desired_shares_delta = delta_notional / price
        # Round to nearest integer shares
        if desired_shares_delta > 0:
            shares_to_trade = int(math.floor(desired_shares_delta))
        else:
            shares_to_trade = int(math.ceil(desired_shares_delta))

        if shares_to_trade == 0:
            # No change needed
            self._log_state(dt, price, target_fraction, event="hold")
            return

        # Determine trade direction & slippage-adjusted price
        slip_mult = 1.0 + (self.slippage_bp / 10000.0)
        if shares_to_trade > 0:
            # Buying
            trade_price = price * slip_mult
        else:
            # Selling / shorting
            trade_price = price / slip_mult

        # Pre-trade state for risk checks
        pre_cash = self.cash
        pre_pos = self.position
        pre_eq = self.equity(price)

        # Gross trade value (positive for buys, negative for sells)
        gross_trade_value = shares_to_trade * trade_price
        commission = self._calc_commission(gross_trade_value) if shares_to_trade != 0 else 0.0

        # Enforce "no leverage" when BUYING
        if shares_to_trade > 0:
            total_cost = gross_trade_value + commission
            if total_cost > pre_cash + 1e-8:
                # Scale down buy size
                max_affordable_shares = int((pre_cash - commission) // trade_price)
                max_affordable_shares = max(max_affordable_shares, 0)
                if max_affordable_shares == 0:
                    # Can't afford any shares
                    self._log_state(dt, price, target_fraction, event="insufficient_cash")
                    return
                shares_to_trade = max_affordable_shares
                gross_trade_value = shares_to_trade * trade_price
                commission = self._calc_commission(gross_trade_value) if shares_to_trade != 0 else 0.0

        # Simulate post-trade state for risk check
        post_pos = pre_pos + shares_to_trade
        # cash change: for buys, gross_trade_value > 0 → cash -= gross_trade_value + commission
        #              for sells, gross_trade_value < 0 → cash -= gross_trade_value + commission
        # (commission always reduces cash)
        post_cash = pre_cash - gross_trade_value - commission
        post_eq = post_cash + post_pos * price

        # Hard risk check: don't allow equity below floor
        floor = self._equity_floor()
        if post_eq < floor - 1e-8:
            self._log_state(dt, price, target_fraction, event="trade_rejected_risk")
            return

        # Apply trade
        self.cash = post_cash
        self.position = post_pos

        # Log trade + new state
        self._log_state(
            dt,
            price,
            target_fraction,
            event="trade",
            shares_delta=shares_to_trade,
            trade_price=trade_price,
        )

        # If after the trade we somehow hit max loss (e.g. due to price move),
        # force liquidation and deactivate.
        if self._max_loss_reached(price):
            self._force_liquidation(dt, price, reason="max_loss_reached_after_trade")

    def _force_liquidation(self, dt, price: float, reason: str = "max_loss_reached"):
        """
        Close all positions at current price (with slippage & commission),
        enforce equity floor, and deactivate broker.
        """
        price = float(price)

        if self.position != 0:
            slip_mult = 1.0 + (self.slippage_bp / 10000.0)
            # To close, trade in opposite direction
            if self.position > 0:
                # Sell longs
                trade_price = price / slip_mult
            else:
                # Buy to cover shorts
                trade_price = price * slip_mult

            shares_to_trade = -self.position
            gross_trade_value = shares_to_trade * trade_price
            commission = self._calc_commission(gross_trade_value) if shares_to_trade != 0 else 0.0

            self.cash = self.cash - gross_trade_value - commission
            self.position = 0

        # Enforce equity floor
        eq = self.equity(price)
        floor = self._equity_floor()
        if eq < floor:
            # Adjust cash upward so equity == floor (this is a simulation safety clamp)
            delta = floor - eq
            self.cash += delta

        self.active = False
        self._log_state(dt, price, target_fraction=0.0, event=reason, shares_delta=0, trade_price=None)

    def _log_state(
        self,
        dt,
        price: float,
        target_fraction: float,
        event: str,
        shares_delta: int = 0,
        trade_price: float = None,
    ):
        """Record current broker state + event into trade_log."""
        eq = self.equity(price)
        self.max_equity = max(self.max_equity, eq)

        row = {
            "datetime": dt,
            "price": float(price),
            "cash": float(self.cash),
            "position": int(self.position),
            "equity": float(eq),
            "target_fraction": float(target_fraction),
            "event": event,
            "shares_delta": int(shares_delta),
            "trade_price": float(trade_price) if trade_price is not None else None,
        }
        self.trade_log.append(row)

    # ---------- Backtest helper ----------

    def run_backtest(self, price_series: pd.Series, signals: pd.Series) -> pd.DataFrame:
        """
        Run backtest over price_series with precomputed target FRACTION signals.

        price_series: pd.Series indexed by datetime, values = prices
        signals:      pd.Series aligned with price_series; values = desired fraction [-1,1]

        Returns DataFrame with:
          - equity: strategy equity over time
          - buy_hold_cum: buy & hold equity over time (starting with initial_cash)
        """
        # Reset state
        self.cash = self.initial_cash
        self.position = 0
        self.active = True
        self.trade_log = []
        self.max_equity = self.initial_cash

        if not price_series.index.equals(signals.index):
            signals = signals.reindex(price_series.index).fillna(0.0)

        first_price = float(price_series.iloc[0])
        buy_hold_shares = self.initial_cash / first_price

        records = []

        for dt, price in price_series.items():
            sig = float(signals.loc[dt])
            self.step_to_target_fraction(dt, price, sig)
            eq = self.equity(price)
            bh_eq = buy_hold_shares * float(price)

            records.append(
                {
                    "datetime": dt,
                    "price": float(price),
                    "signal": sig,
                    "equity": eq,
                    "buy_hold_cum": bh_eq,
                }
            )

        df = pd.DataFrame(records).set_index("datetime")
        return df
