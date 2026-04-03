import time
from pathlib import Path
import sys
import json
from collections import Counter
from zoneinfo import ZoneInfo
import yaml

import pandas as pd

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_for_watchlist
from src.features.sentiment import add_sentiment
from src.features.feature_builder import build_features
from src.trading.broker import SimBroker
from src.models.model import train_pca_direction_model, predict_proba_pca
from src.analysis.regime_detection import HurstAnalyzer
from src.utils.market_hours import (
    is_us_market_open,
    is_us_market_premarket,
    is_us_market_afterhours,
    minutes_until_market_open,
    minutes_until_market_close,
    get_us_market_status,
)

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


def _news_query_for_ticker(base_query: str, ticker: str, multi_ticker: bool) -> str:
    """Use ticker-specific query in multi-ticker mode for better sentiment relevance."""
    if multi_ticker:
        return f"{ticker} stock"
    q = str(base_query or "").strip()
    return q if q else f"{ticker} stock"


def _load_universe_yaml(root: Path) -> list[str]:
    """Load optional broader universe from data/nasdaq_universe.yaml."""
    yaml_path = root / "data" / "nasdaq_universe.yaml"
    if not yaml_path.exists():
        return []
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        tickers = payload.get("tickers", [])
        return [str(t).strip().upper() for t in tickers if str(t).strip()]
    except Exception:
        return []


def _merge_tickers(primary: list[str], secondary: list[str], cap: int) -> list[str]:
    """Keep primary order, then append unique secondary symbols up to cap."""
    out = []
    seen = set()
    for t in (primary + secondary):
        t2 = str(t).strip().upper()
        if not t2 or t2 in seen:
            continue
        seen.add(t2)
        out.append(t2)
        if cap > 0 and len(out) >= cap:
            break
    return out


def _rank_approved_universe(df: pd.DataFrame) -> list[str]:
    """
    Rank approved symbols by available performance columns, then return unique tickers.
    Preferred sort columns: sharpe, total_return, cagr, trades (descending).
    """
    if df is None or df.empty or "ticker" not in df.columns:
        return []

    work = df.copy()
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    work = work[work["ticker"] != ""]

    sort_cols = [c for c in ["sharpe", "total_return", "cagr", "trades"] if c in work.columns]
    for col in sort_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    if sort_cols:
        work = work.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")

    return work["ticker"].drop_duplicates(keep="first").tolist()


def _adaptive_thresholds_from_hurst(
    prices: pd.Series,
    base_long: float,
    base_short: float,
) -> tuple[float, float, float | None]:
    """Adjust thresholds using Hurst trendiness to produce more actionable live signals."""
    try:
        px = pd.to_numeric(prices, errors="coerce").dropna().values
        if len(px) < 80:
            return base_long, base_short, None

        analyzer = HurstAnalyzer(lookback=min(252, len(px) - 1))
        h = float(analyzer.calculate(px))

        long_t = base_long
        short_t = base_short

        # Trending regime: easier longs, harder shorts.
        if h >= 0.60:
            long_t = max(0.52, base_long - 0.05)
            short_t = min(0.85, base_short + 0.08)
        # Mean-reverting/choppy regime: allow both sides a bit more often.
        elif h <= 0.45:
            long_t = max(0.53, base_long - 0.03)
            short_t = max(0.53, base_short - 0.03)

        return long_t, short_t, h
    except Exception:
        return base_long, base_short, None


def _apply_signal_confirmation(state_entry: dict, raw_signal: float, confirm_bars: int) -> tuple[float, int, bool]:
    """
    Require N consecutive identical long/short raw signals before committing.
    - raw_signal in {-1, 0, +1}
    - 0 (flat) is applied immediately for risk-off behavior.
    Returns: (execution_signal, streak_count, confirmed_now)
    """
    n = max(1, int(confirm_bars))
    raw = float(raw_signal)

    if raw == 0.0:
        state_entry["last_raw_signal"] = 0.0
        state_entry["signal_streak"] = 0
        state_entry["committed_signal"] = 0.0
        return 0.0, 0, True

    prev_raw = float(state_entry.get("last_raw_signal", 0.0))
    streak = int(state_entry.get("signal_streak", 0))

    if raw == prev_raw:
        streak += 1
    else:
        streak = 1

    state_entry["last_raw_signal"] = raw
    state_entry["signal_streak"] = streak

    if streak >= n:
        state_entry["committed_signal"] = raw
        return raw, streak, True

    return float(state_entry.get("committed_signal", 0.0)), streak, False


def _refresh_news_if_due(
    cfg: dict,
    state_entry: dict,
    ticker: str,
    logger,
    force: bool = False,
) -> None:
    """Refresh per-ticker news on interval so sentiment/alpha remains live."""
    ncfg = cfg.get("news", {})
    refresh_minutes = int(cfg.get("live", {}).get("news_refresh_minutes", 15))
    now = pd.Timestamp.now("UTC")

    last_refresh = state_entry.get("last_news_refresh")
    due = (
        force
        or last_refresh is None
        or (now - last_refresh).total_seconds() >= max(1, refresh_minutes) * 60
    )
    if not due:
        return

    try:
        news_df = fetch_news_for_watchlist(
            base_query=ncfg.get("query"),
            tickers=[ticker],
            language=ncfg.get("language", "en"),
            lookback_days=int(ncfg.get("lookback_days", 30)),
            max_articles=int(ncfg.get("max_articles", 100)),
            include_macro_news=bool(ncfg.get("include_macro_news", True)),
            macro_queries=ncfg.get("macro_queries"),
            per_query_article_cap=ncfg.get("per_query_article_cap"),
        )
        news_df = add_sentiment(news_df)
        state_entry["news_df"] = news_df
        state_entry["last_news_refresh"] = now
        news_meta = _summarize_news_source(news_df)
        state_entry["news_source"] = news_meta["news_source"]
        state_entry["news_source_breakdown"] = news_meta["news_source_breakdown"]
        logger.info("Refreshed news for %s (%d articles)", ticker, len(news_df))
    except Exception as e:
        logger.warning("News refresh failed for %s: %s", ticker, e)


def _summarize_news_source(news_df: pd.DataFrame) -> dict:
    """Return dominant provider and compact provider-count breakdown."""
    if news_df is None or news_df.empty:
        return {
            "news_source": "no_news",
            "news_source_breakdown": "",
        }

    providers = []
    if "provider" in news_df.columns:
        providers = [str(v).strip() for v in news_df["provider"].dropna().tolist() if str(v).strip()]

    if not providers:
        return {
            "news_source": "unknown",
            "news_source_breakdown": "",
        }

    counts = Counter(providers)
    dominant = counts.most_common(1)[0][0]
    breakdown = ",".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    return {
        "news_source": dominant,
        "news_source_breakdown": breakdown,
    }


def _restore_state_from_log(
    log_path: Path,
    state: dict,
    logger,
    expected_total_equity: float,
    max_equity_deviation_fraction: float = 0.25,
    min_restore_fraction: float = 0.5,
) -> None:
    """Restore per-ticker broker state safely, rejecting discontinuous restores."""
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
        proposals = {}
        proposed_total_equity = 0.0

        for ticker, st in state.items():
            ticker_rows = df[df["ticker"].astype(str) == str(ticker)]
            if ticker_rows.empty:
                continue

            last = ticker_rows.iloc[-1]
            cash = float(last["cash"]) if ("cash" in last and pd.notna(last["cash"])) else float(st["broker"].cash)
            position = int(float(last["position"])) if ("position" in last and pd.notna(last["position"])) else int(st["broker"].position)
            eq = float(last["equity"]) if ("equity" in last and pd.notna(last["equity"])) else cash
            price = float(last["price"]) if ("price" in last and pd.notna(last["price"])) else st.get("last_price")
            event = str(last.get("event", "")) if hasattr(last, "get") else str(last["event"])
            active = not event.startswith("max_loss_reached")

            proposals[ticker] = {
                "cash": cash,
                "position": position,
                "equity": eq,
                "price": price,
                "active": active,
            }
            proposed_total_equity += eq

            restored += 1

        if not restored:
            return

        restore_fraction = restored / max(1, len(state))
        deviation = abs(proposed_total_equity - expected_total_equity) / max(1.0, expected_total_equity)

        if restore_fraction < float(min_restore_fraction):
            logger.warning(
                "Skipping resume: only %d/%d tickers found in log (%.1f%% < %.1f%%).",
                restored,
                len(state),
                restore_fraction * 100.0,
                float(min_restore_fraction) * 100.0,
            )
            return

        if deviation > float(max_equity_deviation_fraction):
            logger.warning(
                "Skipping resume: restored equity %.2f deviates from configured initial cash %.2f by %.1f%% (limit %.1f%%).",
                proposed_total_equity,
                expected_total_equity,
                deviation * 100.0,
                float(max_equity_deviation_fraction) * 100.0,
            )
            return

        for ticker, st in state.items():
            if ticker not in proposals:
                continue
            p = proposals[ticker]
            broker = st["broker"]
            broker.cash = p["cash"]
            broker.position = p["position"]
            broker.max_equity = max(float(p["equity"]), broker.max_equity)
            st["last_price"] = p["price"]
            broker.active = p["active"]

        logger.info(
            "Restored live paper state for %d tickers from %s (total_equity=%.2f, expected=%.2f)",
            restored,
            log_path,
            proposed_total_equity,
            expected_total_equity,
        )
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
    portfolio_news_sources: dict[str, str] = {}

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

        news_source = str(st.get("news_source", "unknown"))
        news_source_breakdown = str(st.get("news_source_breakdown", ""))
        portfolio_news_sources[ticker] = news_source

        positions.append(
            {
                "ticker": ticker,
                "position": position,
                "last_price": float(last_price) if last_price is not None else None,
                "position_value": position_value,
                "cash": cash,
                "equity": equity,
                "active": bool(broker.active),
                "news_source": news_source,
                "news_source_breakdown": news_source_breakdown,
                "news_articles": int(len(st.get("news_df", pd.DataFrame()))),
                "last_news_refresh": _to_local_time_str(st.get("last_news_refresh"), timezone_name)
                if st.get("last_news_refresh") is not None else "",
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
        "news_sources": portfolio_news_sources,
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
    news_df = fetch_news_for_watchlist(
        base_query=ncfg.get("query"),
        tickers=[ticker],
        language=ncfg["language"],
        lookback_days=ncfg["lookback_days"],
        max_articles=ncfg["max_articles"],
        include_macro_news=bool(ncfg.get("include_macro_news", True)),
        macro_queries=ncfg.get("macro_queries"),
        per_query_article_cap=ncfg.get("per_query_article_cap"),
    )
    logger.info("Got %d news articles for %s.", len(news_df), ticker)
    news_df = add_sentiment(news_df)
    news_meta = _summarize_news_source(news_df)

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

    return model_pipeline, feature_cols, context, news_df, news_meta


def _get_stale_bar_threshold(live_cfg: dict) -> float:
    """
    Get stale bar warning threshold based on market hours.
    
    Returns threshold in seconds. Higher during after-hours.
    """
    threshold_market = float(live_cfg.get("stale_bar_threshold_market_open", 180))
    threshold_afterhours = float(live_cfg.get("stale_bar_threshold_afterhours", 3600))
    
    if is_us_market_open():
        return threshold_market
    else:
        return threshold_afterhours


def _market_interval_to_seconds(interval: str) -> float:
    mapping = {
        "1d": 86400.0,
        "1h": 3600.0,
        "30m": 1800.0,
        "15m": 900.0,
        "5m": 300.0,
    }
    return float(mapping.get(str(interval), 60.0))


def _get_adaptive_stale_threshold(live_cfg: dict, market_interval: str, price_source: str) -> float:
    """Compute stale threshold using market session, bar interval, and source latency."""
    if not is_us_market_open():
        return float(live_cfg.get("stale_bar_threshold_afterhours", 3600))

    configured = float(live_cfg.get("stale_bar_threshold_market_open", 180))
    cadence = _market_interval_to_seconds(market_interval)
    base = max(configured, cadence * 2.0)

    # Yahoo free intraday data is commonly delayed; avoid false stale alerts.
    if str(price_source) == "yahoo_bar":
        yahoo_delay_allowance = float(live_cfg.get("yahoo_delay_allowance_seconds", 1800))
        base = max(base, cadence + yahoo_delay_allowance)

    return base


def _should_block_stale_bar(
    live_cfg: dict,
    market_interval: str,
    price_source: str,
    bar_age_sec: float | None,
) -> tuple[bool, float]:
    """Return whether trading should be blocked because the latest signal bar is stale."""
    stale_threshold = _get_adaptive_stale_threshold(live_cfg, market_interval, price_source)
    hard_block = _as_bool(live_cfg.get("hard_stale_bar_block", True), default=True)
    is_stale = bar_age_sec is not None and bar_age_sec > stale_threshold
    return hard_block and is_stale, stale_threshold


def _check_market_hours_and_wait(logger, live_cfg: dict) -> None:
    """
    Check if market is open. If not, optionally wait for market open.
    
    Args:
        logger: Logger instance
        live_cfg: Live trading config dict with 'wait_for_market' flag
    """
    # Check if user wants to wait for market open
    wait_for_market = _as_bool(live_cfg.get("wait_for_market_open", True), default=True)
    
    # Log market hours configuration
    market_status = get_us_market_status()
    logger.info(f"[MARKET HOURS CHECK] Current status: {market_status}")
    logger.info(f"[MARKET HOURS CHECK] wait_for_market_open setting: {wait_for_market}")
    
    if not is_us_market_open():
        logger.warning("⚠️  US Stock Market is CLOSED. Status: %s", market_status)
        
        if wait_for_market:
            minutes = minutes_until_market_open()
            if minutes > 0:
                logger.warning(
                    "⏳ Market is closed. Will wait for market to open in %.1f minutes (9:30 AM ET)...",
                    minutes,
                )
                start_time = time.time()
                check_count = 0
                while not is_us_market_open():
                    time.sleep(60)
                    check_count += 1
                    remaining = minutes_until_market_open()
                    if remaining > 0:
                        logger.info("⏳ [%d min checks] Still waiting... %.1f minutes until market open", 
                                  check_count, remaining)
                    else:
                        break
                
                elapsed = time.time() - start_time
                logger.warning("✅ Market is now OPEN! Starting live trading (waited %.0f seconds)", elapsed)
            else:
                logger.warning(
                    "Market is closed and calculated time is non-positive. "
                    "Proceeding anyway, but expect stale data."
                )
        else:
            logger.warning(
                "Market is closed. Proceeding with after-hours data. "
                "Expect stale market bars and potentially no fills. "
                "Set 'wait_for_market_open: true' in config to wait for market open."
            )
    else:
        logger.info("US Stock Market is OPEN. Ready to start trading.")
        minutes_to_close = minutes_until_market_close()
        logger.info("Market will close in %.0f minutes", minutes_to_close)


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
    max_live_tickers = int(live_cfg.get("max_live_tickers", 30))
    universe_candidate_cap = int(live_cfg.get("universe_candidate_cap", 2000))
    universe_candidate_cap = max(universe_candidate_cap, max_live_tickers)
    approved_ranked = []

    if approved_path.exists():
        try:
            approved_df = pd.read_csv(approved_path)
            if not approved_df.empty and "ticker" in approved_df.columns:
                approved_ranked = _rank_approved_universe(approved_df)
                logger.info("Loaded approved universe (%d symbols)", len(approved_ranked))
            else:
                logger.info("approved_universe.csv is empty or missing 'ticker' column.")
        except Exception as e:
            logger.exception("Failed to read approved_universe.csv: %s", e)

    # Supplement with broader YAML universe to create a large candidate pool,
    # then trade only the top max_live_tickers names.
    yaml_universe = _load_universe_yaml(root)
    candidate_tickers = _merge_tickers(approved_ranked, yaml_universe, universe_candidate_cap)

    if len(candidate_tickers) < universe_candidate_cap:
        logger.warning(
            "Candidate universe smaller than requested cap (%d < %d).",
            len(candidate_tickers),
            universe_candidate_cap,
        )

    tickers = candidate_tickers[:max_live_tickers]

    if not tickers:
        # Fallback to single ticker from config
        mcfg = cfg["market"]
        fallback_ticker = mcfg["ticker"]
        tickers = [fallback_ticker]
        logger.info("No approved universe; falling back to config ticker: %s", fallback_ticker)

    num_tickers = len(tickers)
    logger.info(
        "Live candidate universe size=%d (cap=%d), trading top %d symbols",
        len(candidate_tickers) if 'candidate_tickers' in locals() else num_tickers,
        universe_candidate_cap,
        num_tickers,
    )

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
    signal_confirm_bars = int(live_cfg.get("signal_confirm_bars", 2))
    signal_confirm_bars = max(1, signal_confirm_bars)

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
            model, feature_cols, context, news_df, news_meta = train_initial_model_for_ticker(cfg, logger, ticker)

            broker = SimBroker(
                initial_cash=per_ticker_cash,
                commission_per_trade=bcfg["commission_per_trade"],
                commission_rate=bcfg.get("commission_rate", 0.0),
                slippage_bp=bcfg["slippage_bp"],
                max_loss_fraction=bcfg.get("max_loss_fraction", 1.0),
                max_position_fraction=bcfg.get("max_position_fraction", 0.2),
            )

            state[ticker] = {
                "model": model,
                "feature_cols": feature_cols,
                "context": context,
                "news_df": news_df,
                "news_source": news_meta["news_source"],
                "news_source_breakdown": news_meta["news_source_breakdown"],
                "last_news_refresh": pd.Timestamp.now("UTC"),
                "broker": broker,
                "last_price": None,
                "last_raw_signal": 0.0,
                "signal_streak": 0,
                "committed_signal": 0.0,
            }
        except Exception as e:
            logger.exception("Error setting up ticker %s: %s", ticker, e)

    if not state:
        logger.error("No tickers successfully initialized for live trading. Exiting.")
        return

    # Optional resume across restarts: restore cash/positions from prior log.
    resume_state = _as_bool(live_cfg.get("resume_state", True), default=True)
    if resume_state:
        _restore_state_from_log(
            log_path,
            state,
            logger,
            expected_total_equity=total_initial_cash,
            max_equity_deviation_fraction=float(live_cfg.get("resume_equity_deviation_fraction", 0.25)),
            min_restore_fraction=float(live_cfg.get("resume_min_restore_fraction", 0.5)),
        )

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

            _refresh_news_if_due(cfg, st, ticker, logger, force=False)

            price_df = fetch_market_data(
                ticker=ticker,
                lookback_days=mcfg["lookback_days"],
                interval=mcfg["interval"],
            )
            price_source = str(price_df.attrs.get("price_source", "yahoo_bar"))
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
            block_stale_bar, stale_threshold = _should_block_stale_bar(
                live_cfg,
                str(mcfg.get("interval", "5m")),
                price_source,
                bar_age_sec,
            )
            if bar_age_sec is not None and bar_age_sec > stale_threshold:
                logger.warning(
                    "Stale market bar for %s: age=%.1fs source=%s interval=%ss (threshold=%.0fs)",
                    ticker,
                    bar_age_sec,
                    price_source,
                    interval_sec,
                    stale_threshold,
                )

            if block_stale_bar:
                broker = st["broker"]
                equity = broker.equity(price)
                drawdown = 0.0
                if broker.max_equity > 0:
                    drawdown = 1.0 - (equity / broker.max_equity)

                logger.warning(
                    "Blocking pre-market trade for %s because the latest bar is stale.",
                    ticker,
                )

                row = {
                    "execution_time": pd.Timestamp.now("UTC"),
                    "execution_time_local": _to_local_time_str(pd.Timestamp.now("UTC"), display_timezone),
                    "datetime": dt,
                    "ticker": ticker,
                    "price": price,
                    "price_source": price_source,
                    "prob_up": None,
                    "prob_down": None,
                    "signal": 0.0,
                    "equity": equity,
                    "cash": broker.cash,
                    "position": broker.position,
                    "event": "stale_bar_block",
                    "shares_delta": 0,
                    "trade_price": None,
                    "drawdown": drawdown,
                    "model_confidence": None,
                    "portfolio_equity": None,
                    "news_source": st.get("news_source", "unknown"),
                    "news_source_breakdown": st.get("news_source_breakdown", ""),
                }
                history_rows.append(row)
                premarket_rows.append(row)
                continue

            x_row = latest[st["feature_cols"]]
            prob_down, prob_up = predict_proba_pca(st["model"], x_row)

            local_long, local_short, h = _adaptive_thresholds_from_hurst(
                feat_df["Adj Close"],
                long_thresh,
                short_thresh,
            )

            if prob_up >= local_long:
                raw_pre_signal = 1.0
                rec = "LONG"
            elif prob_down >= local_short:
                raw_pre_signal = -1.0
                rec = "SHORT"
            else:
                raw_pre_signal = 0.0
                rec = "FLAT"

            pre_signal, streak, confirmed_now = _apply_signal_confirmation(
                st,
                raw_pre_signal,
                signal_confirm_bars,
            )

            logger.info(
                "PRE-MARKET %s | %s price=%.2f prob_up=%.3f prob_down=%.3f recommended=%s raw=%+.1f exec=%+.1f streak=%d/%d confirmed=%s",
                dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "strftime") else str(dt),
                ticker,
                price,
                prob_up,
                prob_down,
                rec,
                raw_pre_signal,
                pre_signal,
                streak,
                signal_confirm_bars,
                confirmed_now,
            )
            if h is not None:
                logger.info(
                    "PRE-MARKET %s | %s hurst=%.3f thresholds long=%.3f short=%.3f",
                    dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "strftime") else str(dt),
                    ticker,
                    h,
                    local_long,
                    local_short,
                )

            broker = st["broker"]
            broker.step_to_target_fraction(dt, price, pre_signal)
            equity = broker.equity(price)

            last_event = broker.trade_log[-1] if broker.trade_log else None

            drawdown = 0.0
            if broker.max_equity > 0:
                drawdown = 1.0 - (equity / broker.max_equity)

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
                "portfolio_equity": None,
                "news_source": st.get("news_source", "unknown"),
                "news_source_breakdown": st.get("news_source_breakdown", ""),
            }
            history_rows.append(row)
            premarket_rows.append(row)

        except Exception as e:
            logger.exception("Error in pre-market for %s: %s", ticker, e)

    if premarket_rows:
        portfolio_equity = 0.0
        for tk, st2 in state.items():
            lp = st2["last_price"]
            if lp is not None:
                portfolio_equity += st2["broker"].equity(lp)

        for r in premarket_rows:
            r["portfolio_equity"] = portfolio_equity

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

    # Check market hours and wait for market open if configured
    _check_market_hours_and_wait(logger, live_cfg)

    # === 5) LIVE LOOP: iterate over all tickers each cycle ===
    while True:
        try:
            # We’ll compute portfolio equity *after* updating all tickers in this cycle
            cycle_rows = []

            for ticker, st in state.items():
                try:
                    _refresh_news_if_due(cfg, st, ticker, logger, force=False)

                    price_df = fetch_market_data(
                        ticker=ticker,
                        lookback_days=mcfg["lookback_days"],
                        interval=mcfg["interval"],
                    )
                    price_source = str(price_df.attrs.get("price_source", "yahoo_bar"))
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
                    block_stale_bar, stale_threshold = _should_block_stale_bar(
                        live_cfg,
                        str(mcfg.get("interval", "5m")),
                        price_source,
                        bar_age_sec,
                    )
                    if bar_age_sec is not None and bar_age_sec > stale_threshold:
                        logger.warning(
                            "Stale market bar for %s: age=%.1fs source=%s interval=%ss (threshold=%.0fs)",
                            ticker,
                            bar_age_sec,
                            price_source,
                            interval_sec,
                            stale_threshold,
                        )

                    if block_stale_bar:
                        broker = st["broker"]
                        equity = broker.equity(price)
                        drawdown = 0.0
                        if broker.max_equity > 0:
                            drawdown = 1.0 - (equity / broker.max_equity)

                        logger.warning(
                            "Blocking live trade for %s because the latest bar is stale.",
                            ticker,
                        )

                        row = {
                            "execution_time": pd.Timestamp.now("UTC"),
                            "execution_time_local": _to_local_time_str(pd.Timestamp.now("UTC"), display_timezone),
                            "datetime": dt,
                            "ticker": ticker,
                            "price": price,
                            "price_source": price_source,
                            "prob_up": None,
                            "prob_down": None,
                            "signal": 0.0,
                            "equity": equity,
                            "cash": broker.cash,
                            "position": broker.position,
                            "event": "stale_bar_block",
                            "shares_delta": 0,
                            "trade_price": None,
                            "drawdown": drawdown,
                            "model_confidence": None,
                            "portfolio_equity": None,
                            "news_source": st.get("news_source", "unknown"),
                            "news_source_breakdown": st.get("news_source_breakdown", ""),
                        }
                        cycle_rows.append(row)
                        continue

                    x_row = latest[st["feature_cols"]]
                    prob_down, prob_up = predict_proba_pca(st["model"], x_row)

                    local_long, local_short, h = _adaptive_thresholds_from_hurst(
                        feat_df["Adj Close"],
                        long_thresh,
                        short_thresh,
                    )

                    if prob_up >= local_long:
                        raw_signal = 1.0
                    elif prob_down >= local_short:
                        raw_signal = -1.0
                    else:
                        raw_signal = 0.0

                    signal, streak, confirmed_now = _apply_signal_confirmation(
                        st,
                        raw_signal,
                        signal_confirm_bars,
                    )

                    if h is not None:
                        logger.info(
                            "LIVE %s | hurst=%.3f thresholds long=%.3f short=%.3f prob_up=%.3f prob_down=%.3f raw=%+.1f exec=%+.1f streak=%d/%d confirmed=%s",
                            ticker,
                            h,
                            local_long,
                            local_short,
                            prob_up,
                            prob_down,
                            raw_signal,
                            signal,
                            streak,
                            signal_confirm_bars,
                            confirmed_now,
                        )

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
                        "news_source": st.get("news_source", "unknown"),
                        "news_source_breakdown": st.get("news_source_breakdown", ""),
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
