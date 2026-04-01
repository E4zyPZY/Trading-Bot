# src/data/free_quotes.py

import yfinance as yf


def get_last_price(ticker: str) -> float:
    try:
        df = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        # Use .iloc[-1] then .item() to avoid the FutureWarning
        last_close = df["Close"].iloc[-1]
        price = float(last_close.item() if hasattr(last_close, "item") else last_close)
        return price

    except Exception as e:
        raise RuntimeError(f"Free quote failed for {ticker}: {e}")
