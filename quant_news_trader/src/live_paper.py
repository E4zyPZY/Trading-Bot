import time
from pathlib import Path
import sys
import json
from zoneinfo import ZoneInfo

import pandas as pd

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_free
from src.features.sentiment import add_sentiment
from src.features.feature_builder import build_features
from src.trading.broker import SimBroker
from src.models.model import train_pca_direction_model, predict_proba_pca

try:
    from src.trading.moomoo_broker import MoomooPaperBroker
    HAS_MOOMOO_BROKER = True
except Exception:
    HAS_MOOMOO_BROKER = False


def _as_bool(value, default: bool = False) -> bool:
    """Parse config/env values into a strict boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _to_local_time_str(ts, timezone_name: str) -> str:
    """Convert a timestamp-like value to local timezone string."""
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        return ""
    return dt.tz_convert(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M:%S%z")


def _inject_live_price_into_last_bar(price_df: pd.DataFrame, live_price: float) -> pd.DataFrame:
    """Inject a fresher live price into the latest OHLC row."""
    if price_df.empty:
        return price_df

    df = price_df.copy()
    idx = df.index[-1]
    px = float(live_price)

    if "Adj Close" in df.columns:
        df.at[idx, "Adj Close"] = px
    if "Close" in df.columns:
        df.at[idx, "Close"] = px
    if "High" in df.columns:
        try:
            df.at[idx, "High"] = max(float(df.at[idx, "High"]), px)
        except Exception:
            df.at[idx, "High"] = px
    if "Low" in df.columns:
        try:
            df.at[idx, "Low"] = min(float(df.at[idx, "Low"]), px)
        except Exception:
            df.at[idx, "Low"] = px

    return df


def _compute_bar_age_seconds(ts) -> float | None:
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    return float((pd.Timestamp.now("UTC") - dt).total_seconds())


def _restore_state_from_log(log_path: Path, state: dict, logger) -> None:
    """Restore per-ticker broker cash/position/price from the latest logged rows."""
    if not log_path.exists():
        return

    try:
        df = pd.read_csv(log_path)
        if df.empty or "ticker" not in df.columns:
            return

        # Prefer execution_time ordering if available; otherwise keep file order.
        if "execution_time" in df.columns:
            df["_order_ts"] = pd.to_datetime(df["execution_time"], errors="coerce", utc=True)
            df = df.sort_values("_order_ts", kind="stable")

        restored = 0
        for ticker, st in state.items():
            ticker_rows = df[df["ticker"].astype(str) == str(ticker)]
            if ticker_rows.empty:
                continue

            last = ticker_rows.iloc[-1]
            broker = st["broker"]

            # Restore broker state from last known row.
            if "cash" in last and pd.notna(last["cash"]):
                broker.cash = float(last["cash"])
            if "position" in last and pd.notna(last["position"]):
                broker.position = int(float(last["position"]))
            if "equity" in last and pd.notna(last["equity"]):
                broker.max_equity = max(float(last["equity"]), broker.max_equity)

            if "price" in last and pd.notna(last["price"]):
                st["last_price"] = float(last["price"])

            # Keep broker active unless explicitly marked as inactive in logs.
            event = str(last.get("event", "")) if hasattr(last, "get") else str(last["event"])
            broker.active = not event.startswith("max_loss_reached")

            restored += 1

        if restored:
            logger.info("Restored live paper state for %d tickers from %s", restored, log_path)
    except Exception as e:
        logger.warning("Could not restore live paper state from %s: %s", log_path, e)


def _append_transaction_rows(transaction_log_path: Path, rows: list[dict], logger) -> None:
    """Append trade-like events to a dedicated transaction ledger."""
    if not rows:
        return

    tx_rows = []
    for row in rows:
        try:
            shares_delta = int(float(row.get("shares_delta", 0) or 0))
        except Exception:
            shares_delta = 0

        event = str(row.get("event", ""))
        is_trade_event = (shares_delta != 0) or event.startswith("max_loss_reached")
        if not is_trade_event:
            continue

        tx_rows.append(
            {
                "execution_time": row.get("execution_time"),
                "execution_time_local": row.get("execution_time_local"),
                "bar_datetime": row.get("datetime"),
                "ticker": row.get("ticker"),
                "event": event,
                "shares_delta": shares_delta,
                "position": row.get("position"),
                "trade_price": row.get("trade_price"),
                "price": row.get("price"),
                "cash": row.get("cash"),
                "equity": row.get("equity"),
                "portfolio_equity": row.get("portfolio_equity"),
                "signal": row.get("signal"),
                "prob_up": row.get("prob_up"),
                "prob_down": row.get("prob_down"),
            }
        )

    if not tx_rows:
        return

    tx_df = pd.DataFrame(tx_rows)
    write_header = not transaction_log_path.exists()
    tx_df.to_csv(transaction_log_path, mode="a", header=write_header, index=False)
    logger.info("Appended %d transaction rows to %s", len(tx_rows), transaction_log_path)


def _build_portfolio_snapshot(state: dict, timestamp_utc: pd.Timestamp, timezone_name: str) -> dict:
    """Build a live portfolio snapshot with total and per-ticker state."""
    total_cash = 0.0
    total_position_value = 0.0
    total_equity = 0.0
    positions = []

    for ticker, st in state.items():
        broker = st["broker"]
        last_price = st.get("last_price")
        cash = float(broker.cash)
        position = int(broker.position)

        if last_price is None:
            position_value = 0.0
            equity = cash
        else:
            px = float(last_price)
            position_value = float(position * px)
            equity = float(broker.equity(px))

        total_cash += cash
        total_position_value += position_value
        total_equity += equity

        positions.append(
            {
                "ticker": ticker,
                "position": position,
                "last_price": float(last_price) if last_price is not None else None,
                "position_value": position_value,
                "cash": cash,
                "equity": equity,
                "active": bool(broker.active),
            }
        )

    return {
        "as_of_utc": timestamp_utc.isoformat(),
        "as_of_local": _to_local_time_str(timestamp_utc, timezone_name),
        "display_timezone": timezone_name,
        "total_cash": total_cash,
        "total_position_value": total_position_value,
        "total_equity": total_equity,
        "tickers": len(state),
        "positions": positions,
    }


def _write_portfolio_snapshot(snapshot_path: Path, snapshot: dict, logger) -> None:
    """Persist the latest portfolio snapshot JSON (overwrite each cycle)."""
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.debug("Updated portfolio snapshot: %s", snapshot_path)


def train_initial_model_for_ticker(cfg, logger, ticker: str):
    """
    One-off NN training for a single ticker before the live loop.
    """
    mcfg = cfg["market"]
    ncfg = cfg["news"]
    m_model = cfg.get("model", {})

    logger.info("Initial data load for live NN session on %s...", ticker)
    price_df = fetch_market_data(
        ticker=ticker,
        lookback_days=mcfg["lookback_days"],
        interval=mcfg["interval"],
    )
    logger.info("Got %d rows of price data for %s.", len(price_df), ticker)

    logger.info("Fetching news for initial window for %s...", ticker)
    news_df = fetch_news_free(
        query=ncfg["query"],
        language=ncfg["language"],
        lookback_days=ncfg["lookback_days"],
        max_articles=ncfg["max_articles"],
    )
    logger.info("Got %d news articles for %s.", len(news_df), ticker)
    news_df = add_sentiment(news_df)

    logger.info("Building features for training on %s...", ticker)
    feat_df = build_features(price_df, news_df)
    logger.info("Feature DF shape for %s: %s", ticker, feat_df.shape)

    if feat_df.empty:
        raise RuntimeError(f"No features for training on {ticker}. Check data sources.")

    model_pipeline = train_pca_direction_model(
        feat_df,
        n_components=int(m_model.get("pca_components", 8)),
        test_size=float(m_model.get("test_size", 0.2)),
        random_state=int(m_model.get("random_state", 42)),
        n_jobs=int(m_model.get("n_jobs", 1)),
    )

    report = model_pipeline.get("report", {})
    feature_cols = model_pipeline.get("feature_cols", [])
    context = {"model_type": "pca_rf"}

    logger.info("Initial PCA+RF model classification report for %s: %s", ticker, report)
    logger.info("Feature columns for %s: %s", ticker, feature_cols)

    return model_pipeline, feature_cols, context, news_df


def main(profile: str | None = None):
    logger = setup_logging()
    root = Path(__file__).resolve().parents[1]

    # CLI fallback: --profile=balanced / --profile=aggro / --profile=conservative
    if profile is None:
        for arg in sys.argv[1:]:
            if arg.startswith("--profile="):
                profile = arg.split("=", 1)[1]

    cfg = load_config(root, profile)

    live_cfg = cfg.get("live", {})
    interval_sec = int(live_cfg.get("interval_seconds", 60))
    display_timezone = str(live_cfg.get("display_timezone", "Asia/Shanghai"))
    use_moomoo_quotes = _as_bool(live_cfg.get("use_moomoo", False), default=False)

    moomoo_quote_broker = None
    if use_moomoo_quotes:
        if not HAS_MOOMOO_BROKER:
            logger.warning("use_moomoo=true but moomoo broker import failed; using Yahoo bars only.")
        else:
            mcfg_moo = cfg.get("moomoo", {})
            try:
                moomoo_quote_broker = MoomooPaperBroker(
                    host=mcfg_moo.get("host", "127.0.0.1"),
                    port=int(mcfg_moo.get("port", 11111)),
                    market=mcfg_moo.get("market", "US"),
                    security_firm=mcfg_moo.get("security_firm", "FUTUSECURITIES"),
                    logger=logger,
                )
                logger.info("Live quotes source: moomoo snapshot (with Yahoo fallback)")
            except Exception as e:
                logger.warning("Could not initialize moomoo quote broker; using Yahoo bars only: %s", e)

    # === 1) Load approved universe of tickers ===
    approved_path = root / "data" / "approved_universe.csv"
    tickers = []

    if approved_path.exists():
        try:
            approved_df = pd.read_csv(approved_path)
            if not approved_df.empty and "ticker" in approved_df.columns:
                tickers = approved_df["ticker"].dropna().astype(str).tolist()
                logger.info("Loaded approved universe: %s", tickers)
            else:
                logger.info("approved_universe.csv is empty or missing 'ticker' column.")
        except Exception as e:
            logger.exception("Failed to read approved_universe.csv: %s", e)

    if not tickers:
        # Fallback to single ticker from config
        mcfg = cfg["market"]
        fallback_ticker = mcfg["ticker"]
        tickers = [fallback_ticker]
        logger.info("No approved universe; falling back to config ticker: %s", fallback_ticker)

    num_tickers = len(tickers)
    logger.info("Live NN paper-trading will use tickers: %s", tickers)

    # === 2) Split total capital across tickers (equal slices) ===
    bcfg = cfg["broker"]
    total_initial_cash = float(bcfg["initial_cash"])
    per_ticker_cash = total_initial_cash / num_tickers
    logger.info(
        "Total initial cash: %.2f, split into %.2f per ticker (N=%d).",
        total_initial_cash,
        per_ticker_cash,
        num_tickers,
    )

    # thresholds for signals
    long_thresh = float(live_cfg.get("long_threshold", 0.60))
    short_thresh = float(live_cfg.get("short_threshold", 0.80))

    # Log path (also used to resume state/history)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "live_log_multi_nn.csv"
    transaction_log_path = logs_dir / "transactions_live_multi_nn.csv"
    snapshot_path = logs_dir / "portfolio_snapshot_live_nn.json"
    legacy_log_path = root / "live_log_multi_nn.csv"

    # One-time migration from older location to keep project root clean.
    if legacy_log_path.exists() and not log_path.exists():
        try:
            legacy_log_path.rename(log_path)
            logger.info("Migrated legacy live log to %s", log_path)
        except Exception as e:
            logger.warning("Could not migrate legacy live log to %s: %s", log_path, e)

    # === 3) Per-ticker state: model, broker, news_df, etc. ===
    # state[ticker] = {
    #   "model": ...,
    #   "feature_cols": [...],
    #   "context": {...},
    #   "news_df": DataFrame,
    #   "broker": SimBroker(...),
    #   "last_price": float or None,
    # }
    state = {}

    for ticker in tickers:
        try:
            model, feature_cols, context, news_df = train_initial_model_for_ticker(cfg, logger, ticker)

            broker = SimBroker(
                initial_cash=per_ticker_cash,
                commission_per_trade=bcfg["commission_per_trade"],
                slippage_bp=bcfg["slippage_bp"],
                max_loss_fraction=bcfg.get("max_loss_fraction", 1.0),
            )

            state[ticker] = {
                "model": model,
                "feature_cols": feature_cols,
                "context": context,
                "news_df": news_df,
                "broker": broker,
                "last_price": None,
            }
        except Exception as e:
            logger.exception("Error setting up ticker %s: %s", ticker, e)

    if not state:
        logger.error("No tickers successfully initialized for live trading. Exiting.")
        return

    # Optional resume across restarts: restore cash/positions from prior log.
    resume_state = _as_bool(live_cfg.get("resume_state", True), default=True)
    if resume_state:
        _restore_state_from_log(log_path, state, logger)

    # === 4) Pre-market rebalance for each ticker ===
    if log_path.exists():
        try:
            history_rows = pd.read_csv(log_path).to_dict(orient="records")
        except Exception:
            history_rows = []
    else:
        history_rows = []

    mcfg = cfg["market"]
    premarket_rows = []

    for ticker, st in state.items():
        try:
            logger.info("Pre-market analysis for %s...", ticker)

            price_df = fetch_market_data(
                ticker=ticker,
                lookback_days=mcfg["lookback_days"],
                interval=mcfg["interval"],
            )
            price_source = "yahoo_bar"
            if moomoo_quote_broker is not None:
                try:
                    live_px = float(moomoo_quote_broker.get_last_price(ticker))
                    if live_px > 0:
                        price_df = _inject_live_price_into_last_bar(price_df, live_px)
                        price_source = "moomoo_snapshot"
                except Exception as ex_q:
                    logger.warning("Moomoo quote failed for %s, using Yahoo bar: %s", ticker, ex_q)

            feat_df = build_features(price_df, st["news_df"])

            if feat_df.empty:
                logger.warning("No features in pre-market for %s; starting flat.", ticker)
                continue

            latest = feat_df.iloc[-1]
            dt = latest.name
            price = float(latest["Adj Close"])
            st["last_price"] = price

            bar_age_sec = _compute_bar_age_seconds(dt)
            if bar_age_sec is not None and bar_age_sec > max(180.0, float(interval_sec) * 2.0):
                logger.warning(
                    "Stale market bar for %s: age=%.1fs source=%s interval=%ss",
                    ticker,
                    bar_age_sec,
                    price_source,
                    interval_sec,
                )

            x_row = latest[st["feature_cols"]]
            prob_down, prob_up = predict_proba_pca(st["model"], x_row)

            if prob_up >= long_thresh:
                pre_signal = 1.0
                rec = "LONG"
            elif prob_down >= short_thresh:
                pre_signal = -1.0
                rec = "SHORT"
            else:
                pre_signal = 0.0
                rec = "FLAT"

            logger.info(
                "PRE-MARKET %s | %s price=%.2f prob_up=%.3f prob_down=%.3f recommended=%s (signal=%+.1f)",
                dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "strftime") else str(dt),
                ticker,
                price,
                prob_up,
                prob_down,
                rec,
                pre_signal,
            )

            broker = st["broker"]
            broker.step_to_target_fraction(dt, price, pre_signal)
            equity = broker.equity(price)

            last_event = broker.trade_log[-1] if broker.trade_log else None

            drawdown = 0.0
            if broker.max_equity > 0:
                drawdown = 1.0 - (equity / broker.max_equity)

            # portfolio equity = sum of all brokers' equity at their last prices
            portfolio_equity = 0.0
            for tk, st2 in state.items():
                lp = st2["last_price"]
                if lp is not None:
                    portfolio_equity += st2["broker"].equity(lp)

            row = {
                "execution_time": pd.Timestamp.now("UTC"),
                "execution_time_local": _to_local_time_str(pd.Timestamp.now("UTC"), display_timezone),
                "datetime": dt,
                "ticker": ticker,
                "price": price,
                "price_source": price_source,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "signal": pre_signal,
                "equity": equity,
                "cash": last_event["cash"] if last_event else broker.cash,
                "position": last_event["position"] if last_event else broker.position,
                "event": last_event["event"] if last_event else "pre_market_no_trade",
                "shares_delta": last_event["shares_delta"] if last_event else 0,
                "trade_price": last_event["trade_price"] if last_event else None,
                "drawdown": drawdown,
                "model_confidence": max(prob_up, prob_down),
                "portfolio_equity": portfolio_equity,
            }
            history_rows.append(row)
            premarket_rows.append(row)

        except Exception as e:
            logger.exception("Error in pre-market for %s: %s", ticker, e)

    if history_rows:
        pd.DataFrame(history_rows).to_csv(log_path, index=False)

    if premarket_rows:
        _append_transaction_rows(transaction_log_path, premarket_rows, logger)

    snapshot = _build_portfolio_snapshot(state, pd.Timestamp.now("UTC"), display_timezone)
    _write_portfolio_snapshot(snapshot_path, snapshot, logger)

    logger.info(
        "Starting live NN multi-ticker paper-trading loop for %s (interval=%ds). Ctrl+C to stop.",
        tickers,
        interval_sec,
    )

    # === 5) LIVE LOOP: iterate over all tickers each cycle ===
    while True:
        try:
            # We’ll compute portfolio equity *after* updating all tickers in this cycle
            cycle_rows = []

            for ticker, st in state.items():
                try:
                    price_df = fetch_market_data(
                        ticker=ticker,
                        lookback_days=mcfg["lookback_days"],
                        interval=mcfg["interval"],
                    )
                    price_source = "yahoo_bar"
                    if moomoo_quote_broker is not None:
                        try:
                            live_px = float(moomoo_quote_broker.get_last_price(ticker))
                            if live_px > 0:
                                price_df = _inject_live_price_into_last_bar(price_df, live_px)
                                price_source = "moomoo_snapshot"
                        except Exception as ex_q:
                            logger.warning("Moomoo quote failed for %s, using Yahoo bar: %s", ticker, ex_q)

                    feat_df = build_features(price_df, st["news_df"])
                    if feat_df.empty:
                        logger.warning("No features in live step for %s; skipping.", ticker)
                        continue

                    latest = feat_df.iloc[-1]
                    dt = latest.name
                    price = float(latest["Adj Close"])
                    st["last_price"] = price

                    bar_age_sec = _compute_bar_age_seconds(dt)
                    if bar_age_sec is not None and bar_age_sec > max(180.0, float(interval_sec) * 2.0):
                        logger.warning(
                            "Stale market bar for %s: age=%.1fs source=%s interval=%ss",
                            ticker,
                            bar_age_sec,
                            price_source,
                            interval_sec,
                        )

                    x_row = latest[st["feature_cols"]]
                    prob_down, prob_up = predict_proba_pca(st["model"], x_row)

                    if prob_up >= long_thresh:
                        signal = 1.0
                    elif prob_down >= short_thresh:
                        signal = -1.0
                    else:
                        signal = 0.0

                    broker = st["broker"]
                    broker.step_to_target_fraction(dt, price, signal)
                    equity = broker.equity(price)

                    last_event = broker.trade_log[-1] if broker.trade_log else None

                    drawdown = 0.0
                    if broker.max_equity > 0:
                        drawdown = 1.0 - (equity / broker.max_equity)

                    # we'll fill portfolio_equity after looping all tickers
                    row = {
                        "execution_time": pd.Timestamp.now("UTC"),
                        "execution_time_local": _to_local_time_str(pd.Timestamp.now("UTC"), display_timezone),
                        "datetime": dt,
                        "ticker": ticker,
                        "price": price,
                        "price_source": price_source,
                        "prob_up": prob_up,
                        "prob_down": prob_down,
                        "signal": signal,
                        "equity": equity,
                        "cash": last_event["cash"] if last_event else broker.cash,
                        "position": last_event["position"] if last_event else broker.position,
                        "event": last_event["event"] if last_event else "no_event",
                        "shares_delta": last_event["shares_delta"] if last_event else 0,
                        "trade_price": last_event["trade_price"] if last_event else None,
                        "drawdown": drawdown,
                        "model_confidence": max(prob_up, prob_down),
                        "portfolio_equity": None,  # to be set after all tickers processed
                    }
                    cycle_rows.append(row)

                except Exception as ex_t:
                    logger.exception("Error in live step for %s: %s", ticker, ex_t)

            # Now compute portfolio equity for this cycle using latest prices
            if cycle_rows:
                portfolio_equity = 0.0
                for tk, st2 in state.items():
                    lp = st2["last_price"]
                    if lp is not None:
                        portfolio_equity += st2["broker"].equity(lp)

                for r in cycle_rows:
                    r["portfolio_equity"] = portfolio_equity
                    history_rows.append(r)

                # persist logs
                pd.DataFrame(history_rows).to_csv(log_path, index=False)
                _append_transaction_rows(transaction_log_path, cycle_rows, logger)

                snapshot = _build_portfolio_snapshot(state, pd.Timestamp.now("UTC"), display_timezone)
                _write_portfolio_snapshot(snapshot_path, snapshot, logger)

                # Simple console summary
                logger.info(
                    "LIVE CYCLE | portfolio_equity=%.2f across %d tickers",
                    portfolio_equity,
                    len(state),
                )

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping live multi-ticker session.")
            break
        except Exception as e:
            logger.exception("Error in live multi-ticker NN loop: %s", e)

        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
