from pathlib import Path

import pandas as pd

import sys
from pathlib import Path

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_free
from src.features.sentiment import add_sentiment
from src.features.feature_builder import build_features
from src.models.option_models import choose_best_action_from_models
from src.trading.broker import SimBroker
from src.models.backtest import compute_performance_stats


def backtest_single_ticker(ticker: str, cfg: dict, logger):
    """
    Run the full option-model + broker backtest for a single ticker.
    Returns stats dict and number of trades.
    """
    logger.info("=== Ticker %s ===", ticker)

    mcfg = cfg["market"]
    ncfg = cfg["news"]
    opt_cfg = cfg.get("option_models", {})
    bcfg = cfg["broker"]

    # 1) Market data
    price_df = fetch_market_data(
        ticker=ticker,
        lookback_days=mcfg["lookback_days"],
        interval=mcfg["interval"],
    )
    logger.info("Got %d rows of price data for %s", len(price_df), ticker)

    # 2) News + sentiment (shared query or override later per ticker)
    news_df = fetch_news_free(
        query=ncfg["query"],
        language=ncfg["language"],
        lookback_days=ncfg["lookback_days"],
        max_articles=ncfg["max_articles"],
    )
    logger.info("Got %d news articles for %s", len(news_df), ticker)
    news_df = add_sentiment(news_df)

    # 3) Features
    feat_df = build_features(price_df, news_df)
    if feat_df.empty:
        logger.warning("Empty feature DF for %s, skipping.", ticker)
        return None, 0

    feat_df["sigma_annual"] = feat_df["rolling_vol_20"] * (252 ** 0.5)
    feat_df = feat_df[feat_df["sigma_annual"] > 0].copy()
    if feat_df.empty:
        logger.warning("No valid sigma_annual for %s, skipping.", ticker)
        return None, 0

    # 4) Option-model signals
    signals = []
    for dt, row in feat_df.iterrows():
        S = float(row["Adj Close"])
        sigma_ann = float(row["sigma_annual"])
        signal, details = choose_best_action_from_models(S, sigma_ann, opt_cfg)
        signals.append(signal)

    feat_df["signal"] = signals

    # 5) Broker backtest
    broker = SimBroker(
        initial_cash=bcfg["initial_cash"],
        commission_per_trade=bcfg["commission_per_trade"],
        commission_rate=bcfg.get("commission_rate", 0.0),
        slippage_bp=bcfg["slippage_bp"],
    )

    backtest_df = broker.run_backtest(
        price_series=feat_df["Adj Close"],
        signals=feat_df["signal"],
    )

    stats = compute_performance_stats(backtest_df["equity"])
    n_trades = len(broker.trade_log)

    logger.info("Stats for %s: %s", ticker, stats)
    logger.info("Number of trades: %d", n_trades)

    return stats, n_trades


def main():
    logger = setup_logging()
    root = Path(__file__).resolve().parents[1]

    # --- parse --profile=xxx from command line ---
    profile = None
    for arg in sys.argv[1:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]

    cfg = load_config(root, profile)
    root = Path(__file__).resolve().parents[1]

    mcfg = cfg["market"]
    tickers = mcfg.get("tickers")
    if not tickers:
        logger.error("No 'market.tickers' list in config.yaml. Add some tickers first.")
        return

    results = []

    for ticker in tickers:
        try:
            stats, n_trades = backtest_single_ticker(ticker, cfg, logger)
            if stats is None:
                continue
            results.append(
                {
                    "ticker": ticker,
                    "total_return": stats["total_return"],
                    "cagr": stats["cagr"],
                    "sharpe": stats["sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    "trades": n_trades,
                }
            )
        except Exception as e:
            logger.exception("Error backtesting %s: %s", ticker, e)

    if not results:
        print("No valid results.")
        return

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="sharpe", ascending=False)

    print("\n=== Multi-ticker backtest (Option-model signals) ===")
    print(df_sorted.to_string(index=False))

    # Define what "confident" means for now
    confident = df_sorted[
        (df_sorted["total_return"] > 0.05) &   # > +5% total return
        (df_sorted["sharpe"] > 0.8) &          # Sharpe > 0.8
        (df_sorted["trades"] > 5)             # enough trades
    ]

    print("\nTickers where the strategy seems 'confident' (Sharpe > 0.8, return > 5%, trades > 5):")
    if confident.empty:
        print("None (with current parameters).")
    else:
        print(confident.to_string(index=False))

    # Save to CSV so the live trader can use it
    root = Path(__file__).resolve().parents[1]
    approved_path = root / "approved_universe.csv"

    if confident.empty:
        # Still write an empty file to make behaviour explicit
        confident.to_csv(approved_path, index=False)
        print(f"\n[INFO] No confident tickers. Wrote empty {approved_path.name}.")
    else:
        confident.to_csv(approved_path, index=False)
        print(f"\n[INFO] Saved approved tickers to {approved_path.name}.")


if __name__ == "__main__":
    main()
