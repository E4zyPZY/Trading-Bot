# src/trading/moomoo_broker.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from moomoo import (
    OpenQuoteContext,
    OpenSecTradeContext,
    TrdEnv,
    TrdMarket,
    SecurityFirm,
    TrdSide,
    RET_OK,
)


def _to_moomoo_code(market: str, ticker: str) -> str:
    """
    Map plain ticker to moomoo code, e.g. 'US.AAPL'.
    """
    market = market.upper()
    if market == "US":
        return f"US.{ticker}"
    elif market == "HK":
        return f"HK.{ticker}"
    elif market == "CN":
        # you can refine SH/SZ mapping later
        return f"SH.{ticker}"
    else:
        raise ValueError(f"Unsupported market: {market}")


@dataclass
class AccountState:
    equity: float
    cash: float
    sec_market_value: float


class MoomooPaperBroker:
    """
    Thin wrapper around moomoo paper trading (TrdEnv.SIMULATE).

    IMPORTANT: this implementation is **LONG-ONLY** for new trades.
    It will:
      - open / resize long positions,
      - close existing shorts by BUYing to cover,
      - never open new short positions.
    """

    def __init__(
        self,
        host: str,
        port: int,
        market: str = "US",
        security_firm: str = "FUTUSECURITIES",
        logger=None,
    ):
        self.logger = logger
        self.market = market.upper()

        # Map config strings to SDK enums
        self._trd_market = TrdMarket.US if self.market == "US" else TrdMarket.HK
        self._security_firm = getattr(SecurityFirm, security_firm)

        # Create contexts
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        self.trd_ctx = OpenSecTradeContext(
            filter_trdmarket=self._trd_market,
            host=host,
            port=port,
            security_firm=self._security_firm,
        )

        # Get paper-trading account id (SIMULATE)
        ret, acc_df = self.trd_ctx.get_acc_list()
        if ret != RET_OK:
            raise RuntimeError(f"get_acc_list failed: {acc_df}")

        sim_acc = acc_df[acc_df["trd_env"] == TrdEnv.SIMULATE]
        if sim_acc.empty:
            raise RuntimeError(
                "No SIMULATE account found. Check moomoo OpenAPI / account settings."
            )

        self.acc_id = int(sim_acc["acc_id"].iloc[0])

        if self.logger:
            self.logger.info(
                "MoomooPaperBroker initialized with SIMULATE acc_id=%s", self.acc_id
            )

    # ---------- Account / positions ----------

    def get_account_state(self, currency: str) -> AccountState:
        ret, df = self.trd_ctx.accinfo_query(
            trd_env=TrdEnv.SIMULATE,
            acc_id=self.acc_id,
            currency=currency,
        )
        if ret != RET_OK:
            raise RuntimeError(f"accinfo_query failed: {df}")

        row = df.iloc[0]
        equity = float(row["total_assets"])
        cash = float(row["cash"])
        sec_mv = float(row["market_val"])

        return AccountState(equity=equity, cash=cash, sec_market_value=sec_mv)

    def get_positions(self) -> pd.DataFrame:
        ret, df = self.trd_ctx.position_list_query(
            trd_env=TrdEnv.SIMULATE,
            acc_id=self.acc_id,
        )
        if ret != RET_OK:
            raise RuntimeError(f"position_list_query failed: {df}")
        return df

    # ---------- Quotes (still here if you ever need it) ----------

    def get_last_price(self, ticker: str) -> float:
        """
        Fallback quote method (uses moomoo snapshot). In your live loop
        you normally use free yfinance quotes instead.
        """
        code = _to_moomoo_code(self.market, ticker)
        ret, snap_df = self.quote_ctx.get_market_snapshot([code])
        if ret != RET_OK:
            raise RuntimeError(f"get_market_snapshot failed: {snap_df}")
        price = float(snap_df["last_price"].iloc[0])
        return price

    # ---------- Trading primitives ----------

    def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a simple limit order in SIMULATE env.
        side: 'BUY' or 'SELL'

        NOTE: We always send a price (your caller passes free-quote price),
        so this is effectively a limit order near market.
        """
        code = _to_moomoo_code(self.market, ticker)
        side_up = side.upper()

        if side_up == "BUY":
            trd_side = TrdSide.BUY
        elif side_up == "SELL":
            trd_side = TrdSide.SELL
        else:
            raise ValueError(f"Unknown side: {side}")

        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        if price is None:
            raise ValueError("price must be provided for place_order()")

        kwargs = dict(
            code=code,
            qty=qty,
            price=price,
            trd_side=trd_side,
            trd_env=TrdEnv.SIMULATE,
            acc_id=self.acc_id,
        )

        ret, data = self.trd_ctx.place_order(**kwargs)
        if ret != RET_OK:
            raise RuntimeError(f"place_order failed: {data}")
        return {"raw": data}

    def place_order_market_buy(
        self, ticker: str, qty: int, price: float
    ) -> Dict[str, Any]:
        return self.place_order(ticker=ticker, qty=qty, side="BUY", price=price)

    def place_order_market_sell(
        self, ticker: str, qty: int, price: float
    ) -> Dict[str, Any]:
        return self.place_order(ticker=ticker, qty=qty, side="SELL", price=price)

    # ---------- High-level rebalance (LONG-ONLY) ----------

    def rebalance_to_fraction(
        self,
        ticker: str,
        target_fraction: float,
        max_fraction: float,
        currency: str,
        logger=None,
        price: float | None = None,
    ):
        """
        Rebalance position in `ticker` so that its market value is
        `target_fraction` of account equity, but never exceeding `max_fraction`.

        This implementation is **long-only**:
          - If target_fraction < 0, it is treated as 0 (flat).
          - It will not open new short positions.
          - If there is an existing short (negative qty), it will BUY to cover
            and then stop for this loop (no instant flip to long in one call).

        Parameters
        ----------
        ticker : str
        target_fraction : float
            Desired fraction of equity in this name (0..max_fraction).
        max_fraction : float
            Max allowed fraction in this name.
        currency : str
        price : float | None
            If provided, external price (e.g. from yfinance). Required here.
        """

        # clamp to [0, max_fraction]  (no shorts LIVE)
        if target_fraction < 0:
            if logger:
                logger.info(
                    "[REB] %s – negative target_fraction %.3f clamped to 0 (long-only).",
                    ticker,
                    target_fraction,
                )
            target_fraction = 0.0

        target_fraction = min(max_fraction, max(0.0, target_fraction))

        acct = self.get_account_state(currency)
        equity = acct.equity

        if equity <= 0:
            if logger:
                logger.warning(
                    "[REB] equity<=0 (%.2f) – skip rebalance for %s", equity, ticker
                )
            return None

        if price is None:
            # In your live loop you always pass price from free_quotes.get_last_price.
            last_price = self.get_last_price(ticker)
        else:
            last_price = float(price)

        if last_price <= 0:
            if logger:
                logger.warning(
                    "[REB] last_price<=0 (%.4f) – skip rebalance for %s",
                    last_price,
                    ticker,
                )
            return None

        # --- current position ---
        positions_df = self.get_positions()
        code = _to_moomoo_code(self.market, ticker)
        pos_row = positions_df[positions_df["code"] == code]

        if pos_row.empty:
            current_qty = 0
        else:
            current_qty = int(pos_row["qty"].iloc[0])

        current_mv = current_qty * last_price
        current_fraction = current_mv / equity if equity > 0 else 0.0

        # --- handle existing shorts cleanly (if any) ---
        if current_qty < 0:
            # we are short; we do NOT open/resize shorts in this wrapper.
            # Just BUY to cover and get flat, then exit. Next loop can open long.
            cover_qty = -current_qty  # positive
            if logger:
                logger.info(
                    "[REB] %s – existing short qty=%d; BUY %d to cover (price=%.2f).",
                    ticker,
                    current_qty,
                    cover_qty,
                    last_price,
                )
            self.place_order_market_buy(ticker, cover_qty, last_price)
            return {
                "side": "BUY",
                "qty": cover_qty,
                "price_used": last_price,
                "equity_before": equity,
                "target_fraction": 0.0,
                "note": "cover_short_only",
            }

        # from here on, current_qty >= 0 (long or flat), and target_fraction >= 0

        target_mv = target_fraction * equity
        desired_qty = int(target_mv / last_price)
        if desired_qty < 0:
            desired_qty = 0  # safety

        delta_qty = desired_qty - current_qty

        # ignore tiny resizes (e.g. 1 share on large book if you want)
        if delta_qty == 0:
            if logger:
                logger.info(
                    "[REB] %s – desired_qty == current_qty == %d, no trade.",
                    ticker,
                    current_qty,
                )
            return None

        # if delta_qty > 0 ⇒ need to BUY more
        if delta_qty > 0:
            trade_qty = delta_qty
            if logger:
                logger.info(
                    "[REB] %s – equity=%.2f, target_frac=%.3f, cur_frac=%.3f, "
                    "last_price=%.2f → BUY %d",
                    ticker,
                    equity,
                    target_fraction,
                    current_fraction,
                    last_price,
                    trade_qty,
                )
            self.place_order_market_buy(ticker, trade_qty, last_price)
            return {
                "side": "BUY",
                "qty": trade_qty,
                "price_used": last_price,
                "equity_before": equity,
                "target_fraction": target_fraction,
            }

        # delta_qty < 0 ⇒ need to SELL some (reduce long)
        trade_qty = -delta_qty  # positive amount to SELL

        # Never sell more than you own
        trade_qty = min(trade_qty, current_qty)

        if trade_qty <= 0:
            if logger:
                logger.info(
                    "[REB] %s – computed SELL qty<=0 after clamp, no trade.", ticker
                )
            return None

        if logger:
            logger.info(
                "[REB] %s – equity=%.2f, target_frac=%.3f, cur_frac=%.3f, "
                "last_price=%.2f → SELL %d",
                ticker,
                equity,
                target_fraction,
                current_fraction,
                last_price,
                trade_qty,
            )

        self.place_order_market_sell(ticker, trade_qty, last_price)
        return {
            "side": "SELL",
            "qty": trade_qty,
            "price_used": last_price,
            "equity_before": equity,
            "target_fraction": target_fraction,
        }

    # ---------- EOD snapshot ----------

    def snapshot_portfolio(self, currency: str) -> Dict[str, Any]:
        acct = self.get_account_state(currency)
        positions = self.get_positions()
        return {
            "equity": acct.equity,
            "cash": acct.cash,
            "sec_market_value": acct.sec_market_value,
            "positions": positions,
        }

    # ---------- Cleanup ----------

    def close(self):
        self.quote_ctx.close()
        self.trd_ctx.close()
