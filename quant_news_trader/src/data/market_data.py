from datetime import datetime, timedelta
from typing import Literal
import os

import yfinance as yf
import pandas as pd
import requests


_NEWSAPI_PRICE_WARNING_EMITTED = False


def _try_newsapi_price_source(
    ticker: str,
    lookback_days: int,
    interval: Literal["1d", "1h", "30m", "15m", "5m"] = "1d",
) -> pd.DataFrame | None:
    """
    Connect to NewsAPI first when key is provided.

    NewsAPI does not expose OHLCV/quote endpoints, so this function always
    returns None after a connectivity check, allowing fallback providers.
    """
    api_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
    if not api_key:
        return None

    # Connectivity check only; NewsAPI cannot provide market bars.
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker,
                "language": "en",
                "pageSize": 1,
                "apiKey": api_key,
            },
            timeout=8,
        )
        # If the key is invalid, let fallback continue silently.
        if resp.status_code >= 400:
            return None
    except Exception:
        return None

    global _NEWSAPI_PRICE_WARNING_EMITTED
    if not _NEWSAPI_PRICE_WARNING_EMITTED:
        _NEWSAPI_PRICE_WARNING_EMITTED = True
        print(
            "[market_data] NewsAPI connected, but it does not provide live OHLCV prices. "
            "Falling back to Yahoo for market bars."
        )
    return None


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
    # Try NewsAPI first if requested by user policy; fallback to Yahoo bars.
    provider = (os.getenv("MARKET_DATA_PROVIDER") or "").strip().lower()
    if provider in ("newsapi", "news_api", ""):
        try:
            newsapi_df = _try_newsapi_price_source(
                ticker=ticker,
                lookback_days=lookback_days,
                interval=interval,
            )
            if newsapi_df is not None and not newsapi_df.empty:
                return newsapi_df
        except Exception:
            # Fall through to Yahoo for resilience.
            pass

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
