# src/live_moomoo_nn_moomoo.py

import sys
import json
import time
from pathlib import Path
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Tuple

import pandas as pd
import pytz
import torch

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.features.feature_builder import build_features
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_free
from src.features.sentiment import add_sentiment
from src.trading.moomoo_broker import MoomooPaperBroker
from src.models.nn_model import (
    train_nn_direction_model,
    predict_proba_nn,
    MLPClassifier,
)
from src.data.free_quotes import get_last_price

# ---------- Notifications (plyer) ----------

try:
    from plyer import notification
except ImportError:
    notification = None


def notify(title: str, msg: str):
    if notification is None:
        print(f"[NOTIFY] {title}: {msg}")
    else:
        try:
            notification.notify(title=title, message=msg, timeout=5)
        except Exception as e:
            print(f"[NOTIFY-ERROR] {e} | {title}: {msg}")


# ---------- Market hours helper ----------


def is_us_market_open(now_utc: datetime) -> bool:
    """
    Basic US regular-hours check: 9:30–16:00 US/Eastern, Mon–Fri.
    (Does not handle holidays; ok for a first pass.)
    """
    eastern = pytz.timezone("US/Eastern")
    now_et = now_utc.astimezone(eastern)
    if now_et.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t


# ---------- Model persistence helpers ----------


def get_model_paths(root: Path, ticker: str) -> Tuple[Path, Path]:
    """
    Returns paths: (weights_path, meta_path) for a ticker.
    """
    model_dir = root / "models" / "nn_live"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / f"{ticker}_nn.pth"
    meta_path = model_dir / f"{ticker}_nn_meta.json"
    return weights_path, meta_path


def save_nn_model(
    root: Path,
    ticker: str,
    model: MLPClassifier,
    context: Dict[str, Any],
    logger,
):
    weights_path, meta_path = get_model_paths(root, ticker)

    torch.save(model.state_dict(), weights_path)

    meta = {
        "feature_cols": context.get("feature_cols", []),
        "last_trained_date": context.get("last_trained_date", None),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    logger.info(
        "[SAVE] Saved NN model for %s to %s (meta: %s)",
        ticker,
        weights_path,
        meta,
    )


def load_nn_model(
    root: Path,
    ticker: str,
    logger,
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Load a saved NN for a ticker. Raises if files are missing/invalid.
    """
    weights_path, meta_path = get_model_paths(root, ticker)

    if not weights_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No saved NN model for {ticker}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols: List[str] = meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError(f"Meta for {ticker} missing feature_cols")

    input_dim = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(input_dim=input_dim).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    context = {
        "feature_cols": feature_cols,
        "device": device.type,
        "last_trained_date": meta.get("last_trained_date"),
    }

    logger.info(
        "[LOAD] Loaded NN model for %s from %s (last_trained_date=%s)",
        ticker,
        weights_path,
        context["last_trained_date"],
    )

    return model, context


# ---------- Universe + training ----------


def load_approved_universe(root: Path) -> List[str]:
    csv_path = root / "approved_universe.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        return []
    return df["ticker"].dropna().astype(str).tolist()


def train_nn_for_ticker(
    cfg: Dict[str, Any],
    logger,
    ticker: str,
    root: Path,
):
    """
    Train a NN model for a single ticker using train_nn_direction_model.
    Also saves weights + meta to disk.
    Returns: model, context dict, and the news_df used.
    """
    mcfg = cfg["market"]
    ncfg = cfg["news"]
    m_model = cfg.get("model", {})

    logger.info("[TRAIN] Loading history for %s ...", ticker)

    price_df = fetch_market_data(
        ticker=ticker,
        lookback_days=mcfg["lookback_days"],
        interval=mcfg["interval"],
    )

    news_df = fetch_news_free(
        query=ncfg.get("query", ticker),
        language=ncfg["language"],
        lookback_days=ncfg["lookback_days"],
        max_articles=ncfg["max_articles"],
    )
    news_df = add_sentiment(news_df)

    feat_df = build_features(price_df, news_df)
    logger.info("[TRAIN] Feature DF for %s: %s", ticker, feat_df.shape)

    if feat_df.empty:
        raise RuntimeError(f"No features for training on {ticker}.")

    test_size = float(m_model.get("test_size", 0.2))
    random_state = int(m_model.get("random_state", 42))
    num_epochs = int(m_model.get("nn_epochs", 50))
    batch_size = int(m_model.get("batch_size", 64))
    lr = float(m_model.get("lr", 1e-3))

    (
        model,
        feature_cols,
        X_train,
        X_test,
        y_train,
        y_test,
        report,
        context,
    ) = train_nn_direction_model(
        feat_df,
        test_size=test_size,
        random_state=random_state,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    logger.info(
        "[TRAIN] %s report: %s | device=%s | n_features=%d",
        ticker,
        report,
        context["device"],
        len(feature_cols),
    )

    # stash metadata
    context["feature_cols"] = feature_cols

    eastern = pytz.timezone("US/Eastern")
    today_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(eastern).date()
    context["last_trained_date"] = today_et.isoformat()

    # save model + meta
    save_nn_model(root, ticker, model, context, logger)

    return model, context, news_df


# ---------- Main live loop ----------


def main():
    logger = setup_logging()
    root = Path(__file__).resolve().parents[1]

    # --- parse --profile=xxx from command line ---
    profile = None
    for arg in sys.argv[1:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]

    cfg = load_config(root, profile)

    # --- load approved universe ---
    tickers = load_approved_universe(root)
    if not tickers:
        logger.error(
            "No approved_universe.csv or empty. Run your multi-backtest to create it."
        )
        return

    logger.info("Approved universe for moomoo live NN: %s", tickers)

    # --- set up broker (moomoo SIMULATE) ---
    mcfg_moo = cfg["moomoo"]
    broker = MoomooPaperBroker(
        host=mcfg_moo.get("host", "127.0.0.1"),
        port=int(mcfg_moo.get("port", 11111)),
        market=mcfg_moo.get("market", "US"),
        security_firm=mcfg_moo.get("security_firm", "FUTUSECURITIES"),
        logger=logger,
    )
    currency = mcfg_moo.get("currency", "USD")

    live_cfg = cfg.get("live", {})
    interval_sec = int(live_cfg.get("interval_seconds", 60))
    max_total_fraction = float(live_cfg.get("max_total_fraction", 0.6))
    per_ticker_fraction = max_total_fraction / max(len(tickers), 1)

    # thresholds for signals
    m_model = cfg.get("model", {})
    long_thresh = float(m_model.get("long_thresh", 0.60))
    short_thresh = float(m_model.get("short_thresh", 0.60))

    # --- try to load existing models; train if missing ---
    state: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        try:
            model, context = load_nn_model(root, ticker, logger)
            # fetch some news_df to use in feature builder
            ncfg = cfg["news"]
            news_df = fetch_news_free(
                query=ncfg.get("query", ticker),
                language=ncfg["language"],
                lookback_days=ncfg["lookback_days"],
                max_articles=ncfg["max_articles"],
            )
            news_df = add_sentiment(news_df)
            state[ticker] = {"model": model, "context": context, "news_df": news_df}
        except Exception as e:
            logger.info(
                "[INIT] No usable saved model for %s (or load failed: %s). Training fresh...",
                ticker,
                e,
            )
            try:
                model, context, news_df = train_nn_for_ticker(cfg, logger, ticker, root)
                state[ticker] = {
                    "model": model,
                    "context": context,
                    "news_df": news_df,
                }
            except Exception as e2:
                logger.exception("Error training NN for %s: %s", ticker, e2)

    if not state:
        logger.error("No tickers successfully initialized. Exiting.")
        return

    # --- logging / history setup ---
    log_path = root / "live_log_moomoo_nn.csv"
    if log_path.exists():
        history_df = pd.read_csv(log_path)
        history_rows = history_df.to_dict(orient="records")
    else:
        history_rows: List[Dict[str, Any]] = []

    notify("AI Trader", "Started moomoo NN paper-trading loop.")
    logger.info(
        "Starting moomoo NN live loop (interval=%ds). Max total exposure=%.2f, per ticker=%.3f",
        interval_sec,
        max_total_fraction,
        per_ticker_fraction,
    )

    last_eod_date = None
    eastern = pytz.timezone("US/Eastern")

    try:
        while True:
            # NOTE: Python warns about utcnow being naive; we immediately
            # attach UTC tzinfo so it's fine here.
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            now_et = now_utc.astimezone(eastern)
            today_et = now_et.date()
            today_str = today_et.isoformat()

            # ---------- 30-min pre-open daily retrain (09:00–09:30 ET) ----------
            if dtime(9, 0) <= now_et.time() < dtime(9, 30):
                for ticker in list(state.keys()):
                    ctx = state[ticker]["context"]
                    last_trained = ctx.get("last_trained_date")
                    if last_trained != today_str:
                        logger.info(
                            "[PRE-MARKET] Retraining NN for %s for today's session...",
                            ticker,
                        )
                        try:
                            model, context, news_df = train_nn_for_ticker(
                                cfg, logger, ticker, root
                            )
                            state[ticker] = {
                                "model": model,
                                "context": context,
                                "news_df": news_df,
                            }
                        except Exception as e_re:
                            logger.exception(
                                "[PRE-MARKET] Error retraining %s: %s", ticker, e_re
                            )

            # ---------- skip trading if market closed ----------
            if not is_us_market_open(now_utc):
                logger.info("Market closed (US). Sleeping %ds.", interval_sec)
                time.sleep(interval_sec)
                continue

            # ---------- live trading per ticker ----------
            for ticker, st in state.items():
                try:
                    # 1) live last price from free quotes (yfinance)
                    price = get_last_price(ticker)

                    # 2) build latest features
                    m_market = cfg["market"]
                    n_news = cfg["news"]

                    price_df = fetch_market_data(
                        ticker=ticker,
                        lookback_days=m_market["lookback_days"],
                        interval=m_market["interval"],
                    )
                    news_df = fetch_news_free(
                        query=n_news.get("query", ticker),
                        language=n_news["language"],
                        lookback_days=n_news["lookback_days"],
                        max_articles=n_news["max_articles"],
                    )
                    news_df = add_sentiment(news_df)
                    feat_df = build_features(price_df, news_df)

                    if feat_df.empty:
                        logger.warning("[LIVE] No features for %s; skipping.", ticker)
                        continue

                    latest = feat_df.iloc[-1]
                    feature_cols = st["context"]["feature_cols"]
                    x_row = latest[feature_cols].values.astype("float32")

                    probs = predict_proba_nn(
                        st["model"], x_row, device=st["context"]["device"]
                    )
                    prob_down = float(probs[0])
                    prob_up = float(probs[1])

                    # ---------- STOP-RUN AWARE LONG THRESHOLD ----------
                    # If we just had a “strong stop run” on high volume,
                    # be slightly more aggressive going long.
                    stop_run_flag = bool(latest.get("stop_run_hi_vol_20", 0) == 1)

                    effective_long_thresh = long_thresh
                    if stop_run_flag:
                        # e.g. 0.65 → 0.60, but never below 0.50
                        effective_long_thresh = max(0.50, long_thresh - 0.05)
                        logger.info(
                            "[STOP-RUN] %s stop_run_hi_vol_20=1, prob_up=%.3f, "
                            "relaxing long_thresh from %.2f to %.2f",
                            ticker,
                            prob_up,
                            long_thresh,
                            effective_long_thresh,
                        )

                    # 3) decide signal
                    # 3) decide signal
                    if prob_up >= long_thresh:
                        signal = 1.0
                    elif prob_down >= short_thresh:
                        signal = -1.0
                    else:
                        signal = 0.0

                    # --- NEW: momentum sanity filter ---
                    ret_5d = float(latest.get("ret_5d", 0.0))
                    if signal == 1.0 and ret_5d < 0:
                        # don't go long against a 5-day downtrend
                        signal = 0.0
                    elif signal == -1.0 and ret_5d > 0:
                        # don't short a 5-day uptrend
                        signal = 0.0


                    # 4) rebalance via moomoo paper broker
                    trade_info = broker.rebalance_to_fraction(
                        ticker=ticker,
                        target_fraction=signal * per_ticker_fraction,
                        max_fraction=per_ticker_fraction,
                        currency=currency,
                        logger=logger,
                        price=price,  # use free quote for sizing
                    )

                    acct = broker.get_account_state(currency)
                    row = {
                        "timestamp_utc": now_utc.isoformat(),
                        "ticker": ticker,
                        "price": price,
                        "prob_up": prob_up,
                        "prob_down": prob_down,
                        "signal": signal,
                        "equity": acct.equity,
                        "cash": acct.cash,
                        "sec_mv": acct.sec_market_value,
                        "stop_run_hi_vol_20": int(stop_run_flag),
                    }

                    if trade_info is not None:
                        row["trade_side"] = trade_info["side"]
                        row["trade_qty"] = trade_info["qty"]
                        notify(
                            "AI Trade Executed",
                            f"{ticker}: {trade_info['side']} {trade_info['qty']} @ ~{price:.2f}",
                        )
                    else:
                        row["trade_side"] = ""
                        row["trade_qty"] = 0

                    history_rows.append(row)
                    pd.DataFrame(history_rows).to_csv(log_path, index=False)

                    logger.info(
                        "[LIVE] %s price=%.2f prob_up=%.3f prob_down=%.3f "
                        "signal=%+.1f stop_run=%d equity=%.2f cash=%.2f",
                        ticker,
                        price,
                        prob_up,
                        prob_down,
                        signal,
                        int(stop_run_flag),
                        acct.equity,
                        acct.cash,
                    )

                except Exception as ex_t:
                    logger.exception("[LIVE] Error in loop for %s: %s", ticker, ex_t)

            # ---------- EOD summary once per ET day, after 16:05 ----------
            if last_eod_date != today_et and now_et.time() >= dtime(16, 5):
                snap = broker.snapshot_portfolio(currency)
                eod_path = root / "eod_summary_moomoo_nn.csv"

                row = {
                    "date": today_et.isoformat(),
                    "equity": snap["equity"],
                    "cash": snap["cash"],
                    "sec_mv": snap["sec_market_value"],
                    "n_positions": len(snap["positions"]),
                }

                if eod_path.exists():
                    eod_df = pd.read_csv(eod_path)
                    eod_df = pd.concat([eod_df, pd.DataFrame([row])], axis=0)
                else:
                    eod_df = pd.DataFrame([row])

                eod_df.to_csv(eod_path, index=False)

                logger.info(
                    "[EOD] %s | equity=%.2f cash=%.2f sec_mv=%.2f n_pos=%d",
                    today_et,
                    snap["equity"],
                    snap["cash"],
                    snap["sec_market_value"],
                    len(snap["positions"]),
                )
                notify(
                    "AI Trader EOD Summary",
                    f"{today_et}: equity={snap['equity']:.2f}, cash={snap['cash']:.2f}",
                )
                last_eod_date = today_et

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Received Ctrl+C. Stopping live moomoo NN loop.")
    finally:
        broker.close()
        notify("AI Trader", "Stopped moomoo NN live loop.")


if __name__ == "__main__":
    main()
