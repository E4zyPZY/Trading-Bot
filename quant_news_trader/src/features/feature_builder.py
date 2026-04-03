# src/features/feature_builder.py

import pandas as pd
import numpy as np


def build_features(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine price data with aggregated daily news sentiment.
    Create features + multi-horizon forward returns/labels and
    simple "stop-run" style features to detect possible liquidity grabs.

    Returns a DataFrame indexed by date with:
      - price-based features
      - volatility features
      - multi-horizon forward returns + direction labels
      - daily sentiment
      - stop-run / support / volume-spike style features
    """
    df = price_df.copy()

    # ---------------- BASIC PRICE FEATURES ----------------
    df["return"] = df["Adj Close"].pct_change()
    df["ret_1d"] = df["return"].shift(1)
    df["ret_5d"] = df["Adj Close"].pct_change(5)
    df["rolling_vol_5"] = df["return"].rolling(5).std()
    df["rolling_vol_20"] = df["return"].rolling(20).std()

    # ---------------- MULTI-HORIZON LABELS ----------------
    # forward returns / directions for 1d, 3d, 5d
    for h in [1, 3, 5]:
        fwd_ret_col = f"ret_fwd_{h}d"
        dir_col = f"direction_{h}d"

        df[fwd_ret_col] = df["Adj Close"].shift(-h) / df["Adj Close"] - 1.0
        df[dir_col] = (df[fwd_ret_col] > 0).astype(int)

    # backward compat: 'direction' and 'target_up' = 1-day label
    df["direction"] = df["direction_1d"]
    df["target_up"] = df["direction_1d"]

    # ---------------- NEWS / SENTIMENT FEATURES ----------------
    if not news_df.empty and "sentiment" in news_df.columns:
        ndf = news_df.copy()
        # ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(ndf["published_at"]):
            ndf["published_at"] = pd.to_datetime(ndf["published_at"], errors="coerce", utc=True)

        ndf = ndf.dropna(subset=["published_at"])
        # Convert news dates to UTC, then to naive for matching with price dates
        ndf_ts = pd.to_datetime(ndf["published_at"], utc=True)
        if ndf_ts.dt.tz is not None:
            ndf_ts = ndf_ts.dt.tz_convert(None)
        ndf["date"] = ndf_ts.dt.normalize()
        
        daily_sent = (
            ndf.groupby("date")["sentiment"]
            .mean()
            .rename("daily_sentiment")
        )

        # Convert price dates to naive datetime for matching
        if isinstance(df.index, pd.DatetimeIndex):
            price_index = df.index.to_series().reset_index(drop=True)
        else:
            price_index = pd.to_datetime(df.index)
        
        # Ensure price_index is naive (no timezone)
        if hasattr(price_index, 'dt'):
            if price_index.dt.tz is not None:
                price_index_naive = price_index.dt.tz_convert(None)
            else:
                price_index_naive = price_index
        else:
            price_index_naive = price_index
        
        price_dates = pd.to_datetime(price_index_naive).dt.normalize()
        df["daily_sentiment"] = price_dates.map(daily_sent)
        # Propagate latest known daily sentiment across intraday bars until new news arrives.
        df["daily_sentiment"] = df["daily_sentiment"].ffill()
    else:
        df["daily_sentiment"] = 0.0

    df["daily_sentiment"] = df["daily_sentiment"].fillna(0.0)

    # ---------------- TECHNICAL + SENTIMENT FACTORS (expanded) ----------------
    # exponential moving averages
    df["ema_10"] = df["Adj Close"].ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["Adj Close"].ewm(span=200, adjust=False).mean()
    df["ema_10_50_diff"] = df["ema_10"] - df["ema_50"]
    df["ema_50_200_diff"] = df["ema_50"] - df["ema_200"]

    # moving averages
    df["ma_20"] = df["Adj Close"].rolling(20).mean()
    df["ma_50"] = df["Adj Close"].rolling(50).mean()

    # Bollinger Band %B
    df["bb_upper"] = df["ma_20"] + 2 * df["return"].rolling(20).std()
    df["bb_lower"] = df["ma_20"] - 2 * df["return"].rolling(20).std()
    df["bb_pct"] = (df["Adj Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # MACD
    df["macd_12_26"] = df["Adj Close"].ewm(span=12, adjust=False).mean() - df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["macd_signal_9"] = df["macd_12_26"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_12_26"] - df["macd_signal_9"]

    # 14-day RSI
    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-9))
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = compute_rsi(df["Adj Close"], 14)

    # ATR
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Adj Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Adj Close"].shift(1)).abs()
    df["tr"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # On-Balance Volume (OBV)
    df["obv"] = ((df["Adj Close"].diff().fillna(0) >= 0).replace({True: 1, False: -1}) * df["Volume"]).cumsum()

    # average daily sentiment and momentum sentiment
    df["daily_sentiment_ema_5"] = df["daily_sentiment"].ewm(span=5, adjust=False).mean()
    df["daily_sentiment_ema_20"] = df["daily_sentiment"].ewm(span=20, adjust=False).mean()

    # calendar factors
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # ---------------- STOP-LOSS / LIQUIDITY GRAB PROXIES ----------------
    # Idea:
    #   - many stops cluster just below obvious swing lows / support
    #   - "stop run" day: price trades below recent swing low then closes back above,
    #     ideally on higher volume → possible smart-money absorption.

    # recent swing low (e.g. prior 20-day low, excluding today)
    lookback = 20
    df["swing_low_20"] = df["Low"].rolling(lookback, min_periods=10).min().shift(1)

    # distance of today's close to that swing low
    df["dist_to_swing_low_20"] = df["Adj Close"] / df["swing_low_20"] - 1.0

    # volume regime: relative to 20-day mean
    vol_ma = df["Volume"].rolling(lookback, min_periods=10).mean()
    df["volume_rel_20"] = df["Volume"] / vol_ma
    df["volume_rel_20"] = df["volume_rel_20"].replace([np.inf, -np.inf], np.nan)

    # simple stop-run condition:
    #   1) today's LOW flushes slightly below prior swing low (e.g. 0.2% below)
    #   2) but CLOSE recovers back above that swing low
    eps = 0.002  # 0.2% "poke through" buffer

    cond_have_low = df["swing_low_20"].notna()
    cond_under = df["Low"] < df["swing_low_20"] * (1.0 - eps)
    cond_reclaim = df["Adj Close"] > df["swing_low_20"]

    df["stop_run_20"] = ((cond_have_low) & cond_under & cond_reclaim).astype(int)

    # optional: stronger "liquidity grab" when volume is elevated
    # (e.g. > 1.5x 20-day average)
    df["stop_run_hi_vol_20"] = (
        (df["stop_run_20"] == 1) & (df["volume_rel_20"] > 1.5)
    ).astype(int)

    # ---------------- CLEANUP ----------------
    # drop rows where we don't have full horizons / indicators
    df = df.dropna()

    return df
