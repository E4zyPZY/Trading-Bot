from datetime import datetime, timedelta
from typing import Literal

import yfinance as yf
import pandas as pd


def fetch_market_data(
    ticker: str,
    lookback_days: int,
    interval: Literal["1d", "1h", "30m", "15m", "5m"] = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and normalise it so we always
    have a single-level column index and an 'Adj Close' column.

    For intraday intervals (<= 60d history), we clamp lookback_days
    to avoid Yahoo's range limitations.
    """
    end = datetime.utcnow()

    # Clamp lookback for intraday data
    intraday_max_days = 59  # Yahoo limit ~60 days
    if interval in ("1h", "30m", "15m", "5m"):
        if lookback_days > intraday_max_days:
            lookback_days = intraday_max_days

    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No market data returned for {ticker}")

    # Handle MultiIndex columns (multi-ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except Exception:
            try:
                df = df.xs(ticker, axis=1, level=0)
            except Exception as e:
                raise RuntimeError(
                    f"Could not extract single-ticker data for {ticker} from MultiIndex columns"
                ) from e

    # Ensure 'Adj Close' exists
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            raise RuntimeError(
                f"'Adj Close' and 'Close' both missing from market data for {ticker}. "
                f"Columns present: {list(df.columns)}"
            )

    df = df.dropna()
    df.index.name = "datetime"
    df.attrs["price_source"] = "yahoo_bar"
    return df
