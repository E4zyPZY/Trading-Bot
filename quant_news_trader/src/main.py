from pathlib import Path
import sys
import tkinter as tk
from threading import Thread
import time
import json

import pandas as pd

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.data.market_data import fetch_market_data
from src.data.news_data import fetch_news_for_watchlist
from src.features.sentiment import add_sentiment
from src.features.feature_builder import build_features
from src.models.backtest import compute_performance_stats, compute_risk_metrics
from src.trading.broker import SimBroker
from src.models.option_models import choose_best_action_from_models
from src.models.model import train_pca_direction_model, predict_proba_pca
from src.live_paper import main as run_live_paper

# Import new advanced modules
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.analysis.regime_detection import HurstAnalyzer, RegimeDetector, adaptive_signal_by_regime
from src.models.kalman_ou import KalmanFilter1D, KalmanOUFilter, PairsTrader


LIVE_REFRESH_MS = 10000


def _compute_sentiment_analysis(feat_df: pd.DataFrame, news_df: pd.DataFrame | None = None) -> dict:
    """
    Compute sentiment impact analysis from raw features.
    Works directly with sentiment columns (daily_sentiment, daily_sentiment_ema_5, etc).
    Returns dict with sentiment feature interactions.
    """
    results = {}
    
    # Find sentiment columns
    sentiment_cols = [
        col for col in feat_df.columns
        if "sentiment" in col.lower() and pd.api.types.is_numeric_dtype(feat_df[col])
    ]
    
    if not sentiment_cols:
        return {}
    
    # Find volatility features to interact with sentiment
    vol_cols = [
        col for col in feat_df.columns
        if any(x in col.lower() for x in ['vol', 'atr', 'rolling_vol'])
        and pd.api.types.is_numeric_dtype(feat_df[col])
    ]
    
    if not vol_cols:
        vol_cols = []
    
    # For each sentiment feature, create interactions with vol and price direction
    for sent_col in sentiment_cols:
        sent_series = feat_df[sent_col].dropna()
        if sent_series.empty or float(sent_series.abs().sum()) < 1e-9:
            continue
        
        # Split by sentiment polarity (avoid edge case of all same sign)
        neg_mask = feat_df[sent_col] < 0
        pos_mask = feat_df[sent_col] > 0
        neutral_mask = feat_df[sent_col] == 0
        
        neg_count = int(neg_mask.sum())
        pos_count = int(pos_mask.sum())
        
        # Compute average direction (prob_up) for each sentiment regime
        if "prob_up" in feat_df.columns:
            baseline_prob_up = float(feat_df["prob_up"].mean())
            neg_prob_up = float(feat_df.loc[neg_mask, "prob_up"].mean()) if neg_count > 0 else baseline_prob_up
            pos_prob_up = float(feat_df.loc[pos_mask, "prob_up"].mean()) if pos_count > 0 else baseline_prob_up
            elasticity_up = (pos_prob_up - neg_prob_up) / abs(neg_prob_up) if abs(neg_prob_up) > 1e-9 else 0.0
            
            results[f"{sent_col}_prob_up"] = {
                "negative_impact": neg_prob_up,
                "positive_impact": pos_prob_up,
                "elasticity": elasticity_up,
                "negative_count": neg_count,
                "positive_count": pos_count,
            }
        
        # For each volatility feature, compute interaction
        for vol_col in vol_cols[:3]:  # Limit to 3 vol features to avoid clutter
            baseline_vol = float(feat_df[vol_col].mean())
            vol_neg = float(feat_df.loc[neg_mask, vol_col].mean()) if neg_count > 0 else baseline_vol
            vol_pos = float(feat_df.loc[pos_mask, vol_col].mean()) if pos_count > 0 else baseline_vol
            
            if vol_neg > 1e-9:
                vol_elasticity = (vol_pos - vol_neg) / vol_neg
            else:
                vol_elasticity = 0.0
            
            results[f"{sent_col}_{vol_col}"] = {
                "negative_impact": vol_neg,
                "positive_impact": vol_pos,
                "elasticity": vol_elasticity,
            }
    
    if results:
        return results

    # If feature-level sentiment is unavailable (often due date overlap gaps),
    # provide a news-level fallback so reporting is still informative.
    if news_df is not None and not news_df.empty and "sentiment" in news_df.columns:
        sent = pd.to_numeric(news_df["sentiment"], errors="coerce").dropna()
        if not sent.empty:
            pos = float((sent > 0).mean())
            neg = float((sent < 0).mean())
            neu = float((sent == 0).mean())
            return {
                "news_sentiment_summary": {
                    "negative_impact": float(sent[sent < 0].mean()) if (sent < 0).any() else 0.0,
                    "positive_impact": float(sent[sent > 0].mean()) if (sent > 0).any() else 0.0,
                    "elasticity": float(sent.mean()),
                    "article_count": int(sent.shape[0]),
                    "positive_ratio": pos,
                    "negative_ratio": neg,
                    "neutral_ratio": neu,
                }
            }

    return {}


def _fallback_sentiment_summary(feat_df: pd.DataFrame, news_df: pd.DataFrame | None = None) -> dict:
    """Legacy fallback that calls the new sentiment analysis function."""
    return _compute_sentiment_analysis(feat_df, news_df=news_df)


class BacktestDashboard:
    """Simple backtest results dashboard using tkinter."""

    def __init__(
        self,
        root_tk: tk.Tk,
        backtest_df: pd.DataFrame,
        stats: dict,
        broker,
        ticker: str = "BACKTEST",
        key_variables: dict = None,
        live_log_path: Path | None = None,
    ):
        self.root = root_tk
        self.backtest_df = backtest_df
        self.stats = stats
        self.broker = broker
        self.ticker = ticker
        self.key_variables = key_variables or {}
        self.live_log_path = live_log_path
        self.active_profile = self.key_variables.get("active_profile", "auto")
        self.selected_ticker_count = int(self.key_variables.get("selected_ticker_count", 0) or 0)
        self.max_live_tickers = int(self.key_variables.get("max_live_tickers", 0) or 0)
        self.max_position_fraction = float(self.key_variables.get("max_position_fraction", 0.0) or 0.0)
        self.live_snapshot_path = (
            live_log_path.parent / "portfolio_snapshot_live_nn.json"
            if live_log_path is not None
            else None
        )

        self.root.title(f"Backtest Dashboard - {ticker}")
        self.root.geometry("900x700")  # Increased size for new content

        # Header
        title = tk.Label(
            self.root,
            text=f"Backtest Results - {ticker}",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(pady=10)

        # Stats frame
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(pady=10, padx=10, fill="both", expand=False)

        # Display key metrics
        final_equity = (
            backtest_df["equity"].iloc[-1]
            if len(backtest_df) > 0
            else 0
        )
        buy_hold = (
            backtest_df["buy_hold_cum"].iloc[-1]
            if len(backtest_df) > 0
            else 0
        )

        # Format stats with None handling
        total_return_str = (
            f"{stats.get('total_return', 0) * 100:.2f}%"
            if stats.get('total_return') is not None
            else 'N/A'
        )
        cagr_str = (
            f"{stats.get('cagr', 0) * 100:.2f}%"
            if stats.get('cagr') is not None
            else 'N/A'
        )
        sharpe_str = (
            f"{stats.get('sharpe', 0):.2f}"
            if stats.get('sharpe') is not None
            else 'N/A'
        )
        max_dd_str = f"{stats.get('max_drawdown', 0) * 100:.2f}%"

        metrics_text = (
            f"Active Profile:      {self.active_profile}\n"
            f"Selected Tickers:    {self.selected_ticker_count}\n"
            f"Max Live Tickers:    {self.max_live_tickers}\n"
            f"Max Position Frac:   {self.max_position_fraction:.2f}\n"
            f"Final Equity:        ${final_equity:,.2f}\n"
            f"Buy & Hold Value:    ${buy_hold:,.2f}\n"
            f"Total Return:        {total_return_str}\n"
            f"CAGR:                {cagr_str}\n"
            f"Sharpe Ratio:        {sharpe_str}\n"
            f"Max Drawdown:        {max_dd_str}\n"
            f"VaR (95%):          {self.key_variables.get('VaR_95', 'N/A')}\n"
            f"ES (95%):           {self.key_variables.get('ES_95', 'N/A')}\n"
            f"Total Trades:        {len(broker.trade_log)}"
        )

        self.stats_label = tk.Label(
            stats_frame,
            text=metrics_text,
            justify="left",
            font=("Consolas", 10),
            bg="lightblue",
            padx=10,
            pady=10,
        )
        self.stats_label.pack(fill="both", expand=True)

        # Advanced Analysis Section
        if self.key_variables:
            analysis_frame = tk.Frame(self.root)
            analysis_frame.pack(pady=10, padx=10, fill="both", expand=False)

            analysis_label = tk.Label(
                analysis_frame,
                text="Advanced Analysis:",
                font=("Segoe UI", 12, "bold"),
            )
            analysis_label.pack(anchor="w")

            sentiment_text = "Sentiment Impact:\n"
            sentiment_impact = self.key_variables.get('sentiment_impact', {})
            if sentiment_impact:
                for key, value in list(sentiment_impact.items())[:3]:  # Show first 3
                    if isinstance(value, dict):
                        neg = value.get('negative_impact', 'N/A')
                        pos = value.get('positive_impact', 'N/A')
                        elast = value.get('elasticity', 'N/A')
                        sentiment_text += f"  {key}: Neg={neg:.4f}, Pos={pos:.4f}, Elasticity={elast:.4f}\n"
                    else:
                        sentiment_text += f"  {key}: {value}\n"
            else:
                sentiment_text += "  No sentiment analysis available\n"

            regime = self.key_variables.get('current_regime', 'N/A')
            hurst = self.key_variables.get('avg_hurst', 'N/A')
            theta = self.key_variables.get('kalman_ou_theta', 'N/A')
            theta_floor_hit = self.key_variables.get('theta_floor_hit', 'N/A')
            market_bar_age_days = self.key_variables.get('market_bar_age_days', 'N/A')

            analysis_text = (
                f"{sentiment_text}"
                f"Market Regime: {regime}\n"
                f"Avg Hurst Exponent: {hurst}\n"
                f"Kalman-OU Theta: {theta}\n"
                f"Theta Floor Hit: {theta_floor_hit}\n"
                f"Market Bar Age (days): {market_bar_age_days}"
            )

            analysis_display = tk.Label(
                analysis_frame,
                text=analysis_text,
                justify="left",
                font=("Consolas", 9),
                bg="lightgreen",
                padx=10,
                pady=10,
            )
            analysis_display.pack(fill="both", expand=True)

        # Trade log frame
        log_label = tk.Label(
            self.root,
            text="Trade Log (Last 20):",
            font=("Segoe UI", 10, "bold"),
        )
        log_label.pack(pady=5)

        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=5, padx=10, fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.trade_listbox = tk.Listbox(
            log_frame,
            height=15,
            font=("Consolas", 8),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.trade_listbox.yview)
        self.trade_listbox.pack(side=tk.LEFT, fill="both", expand=True)

        # Populate trade log. If a live multi-ticker log is available, prefer that.
        if self.live_log_path is not None and self.live_log_path.exists():
            try:
                live_df = pd.read_csv(self.live_log_path)
                if not live_df.empty:
                    self.trade_listbox.insert(
                        tk.END,
                        f"{'Exec Time':<20} {'Bar Time':<20} {'Ticker':<8} {'Event':<15} {'Signal':<8} {'Price':<10}",
                    )
                    self.trade_listbox.insert(tk.END, "-" * 100)

                    recent = live_df.tail(20)
                    for _, row in recent.iterrows():
                        exec_ts = str(row.get("execution_time_local", row.get("execution_time", "")))[:19]
                        bar_ts = str(row.get("datetime", ""))[:19]
                        ticker = str(row.get("ticker", ""))[:8]
                        event = str(row.get("event", ""))[:15]
                        signal = float(row.get("signal", 0.0))
                        price = float(row.get("price", 0.0))

                        line = (
                            f"{exec_ts:<20} {bar_ts:<20} {ticker:<8} {event:<15} {signal:<+8.1f} {price:<10.2f}"
                        )
                        self.trade_listbox.insert(tk.END, line)
                else:
                    self.trade_listbox.insert(tk.END, "(live log exists but has no rows yet)")
            except Exception as e:
                self.trade_listbox.insert(tk.END, f"(failed to read live log: {e})")
        elif len(broker.trade_log) > 0:
            self.trade_listbox.insert(
                tk.END,
                f"{'DateTime':<20} {'Event':<15} {'Price':<10} {'Qty':<8} {'Equity':<12}",
            )
            self.trade_listbox.insert(tk.END, "-" * 80)

            # Show last 20 trades
            for trade in broker.trade_log[-20:]:
                dt_str = str(trade.get("datetime"))[:19]
                event = trade.get("event", "")
                price = trade.get("price", 0)
                qty = trade.get("shares_delta", 0)
                equity = trade.get("equity", 0)

                line = (
                    f"{dt_str:<20} {event:<15} {price:<10.2f} {qty:<8} {equity:<12.2f}"
                )
                self.trade_listbox.insert(tk.END, line)
        else:
            self.trade_listbox.insert(tk.END, "(no trades executed)")

        if self.live_log_path is not None:
            self._refresh_live_dashboard()

        # Close button
        close_btn = tk.Button(
            self.root,
            text="Close Dashboard",
            command=self.root.quit,
            bg="red",
            fg="white",
            font=("Segoe UI", 10),
        )
        close_btn.pack(pady=10)

    def _refresh_live_dashboard(self):
        """Refresh live portfolio stats and recent trades from live log/snapshot files."""
        try:
            if self.live_log_path is None or not self.live_log_path.exists():
                self.root.after(LIVE_REFRESH_MS, self._refresh_live_dashboard)
                return

            live_df = pd.read_csv(self.live_log_path)
            if live_df.empty or "portfolio_equity" not in live_df.columns:
                self.root.after(LIVE_REFRESH_MS, self._refresh_live_dashboard)
                return

            live_df["execution_time"] = pd.to_datetime(
                live_df.get("execution_time"),
                utc=True,
                errors="coerce",
            )
            live_df = live_df.sort_values("execution_time")

            pe = pd.to_numeric(live_df["portfolio_equity"], errors="coerce").dropna()
            if pe.empty:
                self.root.after(LIVE_REFRESH_MS, self._refresh_live_dashboard)
                return

            start_eq = float(pe.iloc[0])
            latest_eq = float(pe.iloc[-1])
            total_return = (latest_eq / start_eq - 1.0) if start_eq > 0 else 0.0

            t_valid = live_df["execution_time"].dropna()
            cagr = None
            if len(t_valid) >= 2 and start_eq > 0 and latest_eq > 0:
                elapsed_years = (t_valid.iloc[-1] - t_valid.iloc[0]).total_seconds() / (365.25 * 24 * 3600)
                if elapsed_years > 0:
                    cagr = (latest_eq / start_eq) ** (1.0 / elapsed_years) - 1.0

            total_cash = None
            total_position_value = None
            news_source_line = "N/A"
            sentiment_line = "N/A"
            if self.live_snapshot_path is not None and self.live_snapshot_path.exists():
                try:
                    snap = json.loads(self.live_snapshot_path.read_text(encoding="utf-8"))
                    total_cash = float(snap.get("total_cash", 0.0))
                    total_position_value = float(snap.get("total_position_value", 0.0))
                    news_sources = snap.get("news_sources", {}) or {}
                    if news_sources:
                        counts = {}
                        for _, src in news_sources.items():
                            src_key = str(src or "unknown")
                            counts[src_key] = counts.get(src_key, 0) + 1
                        news_source_line = ", ".join(
                            f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                        )
                    # Extract sentiment metrics
                    sentiment_impact = snap.get("sentiment_impact", {})
                    portfolio_avg_sent = sentiment_impact.get("portfolio_avg_sentiment", 0.0)
                    active_ticker_count = sentiment_impact.get("active_ticker_count", 0)
                    if active_ticker_count > 0:
                        sentiment_line = f"{portfolio_avg_sent:+.4f} ({active_ticker_count} tickers)"
                except Exception:
                    total_cash = None
                    total_position_value = None
                    news_source_line = "N/A"
                    sentiment_line = "N/A"

            cagr_str = f"{cagr * 100:.2f}%" if cagr is not None else "N/A"
            cash_str = f"${total_cash:,.2f}" if total_cash is not None else "N/A"
            sec_str = f"${total_position_value:,.2f}" if total_position_value is not None else "N/A"

            live_metrics_text = (
                f"Active Profile:      {self.active_profile}\n"
                f"Selected Tickers:    {self.selected_ticker_count}\n"
                f"Max Live Tickers:    {self.max_live_tickers}\n"
                f"Max Position Frac:   {self.max_position_fraction:.2f}\n"
                f"Live Portfolio Value: ${latest_eq:,.2f}\n"
                f"Live Cash:            {cash_str}\n"
                f"Live Securities:      {sec_str}\n"
                f"Portfolio Sentiment:  {sentiment_line}\n"
                f"News Source (Live):   {news_source_line}\n"
                f"Total Return (Live):  {total_return * 100:.4f}%\n"
                f"CAGR (Live):          {cagr_str}\n"
                f"Last Update:          {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Rows in Live Log:     {len(live_df)}"
            )
            self.stats_label.config(text=live_metrics_text)

            # Refresh recent trades panel from latest rows.
            self.trade_listbox.delete(0, tk.END)
            self.trade_listbox.insert(
                tk.END,
                f"{'Exec Time':<20} {'Bar Time':<20} {'Ticker':<8} {'News':<10} {'Event':<15} {'Signal':<8} {'Price':<10}",
            )
            self.trade_listbox.insert(tk.END, "-" * 112)
            for _, row in live_df.tail(20).iterrows():
                exec_ts = str(row.get("execution_time_local", row.get("execution_time", "")))[:19]
                bar_ts = str(row.get("datetime", ""))[:19]
                ticker = str(row.get("ticker", ""))[:8]
                news_source = str(row.get("news_source", ""))[:10]
                event = str(row.get("event", ""))[:15]
                signal = float(pd.to_numeric(row.get("signal", 0.0), errors="coerce") or 0.0)
                price = float(pd.to_numeric(row.get("price", 0.0), errors="coerce") or 0.0)
                self.trade_listbox.insert(
                    tk.END,
                    f"{exec_ts:<20} {bar_ts:<20} {ticker:<8} {news_source:<10} {event:<15} {signal:<+8.1f} {price:<10.2f}",
                )

        except Exception:
            # Keep dashboard alive even if one refresh fails.
            pass

        self.root.after(LIVE_REFRESH_MS, self._refresh_live_dashboard)


def main():
    logger = setup_logging()

    # Get root directory
    root = Path(__file__).resolve().parents[1]

    # --- parse --profile=xxx from command line ---
    profile = None
    for arg in sys.argv[1:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]

    cfg = load_config(root, profile)
    # ... rest of your code

    logger.info("Loading market data...")
    mcfg = cfg["market"]
    price_df = fetch_market_data(
        ticker=mcfg["ticker"],
        lookback_days=mcfg["lookback_days"],
        interval=mcfg["interval"],
    )
    logger.info(f"Got {len(price_df)} rows of price data.")

    logger.info("Loading news...")
    ncfg = cfg["news"]
    selected_tickers = cfg.get("market", {}).get("tickers") or [mcfg["ticker"]]
    news_df = fetch_news_for_watchlist(
        base_query=ncfg.get("query"),
        tickers=selected_tickers,
        language=ncfg["language"],
        lookback_days=ncfg["lookback_days"],
        max_articles=ncfg["max_articles"],
        include_macro_news=bool(ncfg.get("include_macro_news", True)),
        macro_queries=ncfg.get("macro_queries"),
        per_query_article_cap=ncfg.get("per_query_article_cap"),
    )
    logger.info(f"Got {len(news_df)} news articles.")

    logger.info("Computing sentiment...")
    news_df = add_sentiment(news_df)

    logger.info("Building features...")
    feat_df = build_features(price_df, news_df)
    logger.info(f"Feature DF shape: {feat_df.shape}")

    # Convert rolling 20-day volatility to annual sigma for option models
    feat_df["sigma_annual"] = feat_df["rolling_vol_20"] * (252 ** 0.5)
    feat_df = feat_df[feat_df["sigma_annual"] > 0].copy()

    if feat_df.empty:
        logger.error("No features after processing. Check data sources.")
        return

    # 4. Generate signals using option-pricing models
    logger.info("Generating signals using Binomial, BSM and Merton models...")
    opt_cfg = cfg.get("option_models", {})

    signals = []
    chosen_models = []
    chosen_scores = []

    for dt, row in feat_df.iterrows():
        S = float(row["Adj Close"])
        sigma_ann = float(row["sigma_annual"])

        signal, details = choose_best_action_from_models(S, sigma_ann, opt_cfg)
        signals.append(signal)

        chosen_models.append(details.get("chosen_model"))
        chosen_scores.append(details.get("chosen_score", 0.0))

    feat_df["signal"] = signals
    feat_df["chosen_model"] = chosen_models
    feat_df["chosen_score"] = chosen_scores

    # 4b. Train PCA+RF on extended factor set and create model signals
    m_model_cfg = cfg.get("model", {})
    pca_pipeline = train_pca_direction_model(
        feat_df,
        n_components=int(m_model_cfg.get("pca_components", 8)),
        test_size=float(m_model_cfg.get("test_size", 0.2)),
        random_state=int(m_model_cfg.get("random_state", 42)),
        n_jobs=int(m_model_cfg.get("n_jobs", 1)),
    )
    feat_df[["prob_down", "prob_up"]] = feat_df.apply(
        lambda row: pd.Series(predict_proba_pca(pca_pipeline, row)),
        axis=1,
    )

    long_thresh = float(m_model_cfg.get("long_threshold", 0.60))
    short_thresh = float(m_model_cfg.get("short_threshold", 0.60))

    def pca_signal(r):
        if r["prob_up"] >= long_thresh:
            return 1.0
        if r["prob_down"] >= short_thresh:
            return -1.0
        return 0.0

    feat_df["pca_signal"] = feat_df.apply(pca_signal, axis=1)

    # Use PCA signal as primary execution signal (mix with option signal if desired)
    feat_df["signal"] = feat_df["pca_signal"]

    # 5. Backtest with SimBroker using these signals
    bcfg = cfg["broker"]
    broker = SimBroker(
        initial_cash=bcfg["initial_cash"],
        commission_per_trade=bcfg["commission_per_trade"],
        commission_rate=bcfg.get("commission_rate", 0.0),
        slippage_bp=bcfg["slippage_bp"],
    )

    logger.info("Running broker backtest...")
    backtest_df = broker.run_backtest(
        price_series=feat_df["Adj Close"],
        signals=feat_df["signal"],
    )

    stats = compute_performance_stats(backtest_df["equity"])
    logger.info(f"Backtest stats: {stats}")

    # Compute risk metrics (VaR, ES)
    returns = backtest_df["equity"].pct_change().dropna()
    risk_metrics = compute_risk_metrics(returns)
    logger.info(f"Risk metrics: {risk_metrics}")

    # Advanced Analysis: SHAP, Hurst, Kalman-OU
    logger.info("Running advanced analysis...")

    # 1. SHAP Analysis for sentiment impact
    # Explain the exact feature space used by the trained PCA+RF model.
    x_shap_raw = feat_df[pca_pipeline["feature_cols"]].fillna(0.0)
    x_shap_scaled = pca_pipeline["scaler"].transform(x_shap_raw)
    x_shap_pca = pca_pipeline["pca"].transform(x_shap_scaled)
    shap_feature_names = [f"PC{i + 1}" for i in range(x_shap_pca.shape[1])]
    x_shap_df = pd.DataFrame(x_shap_pca, index=feat_df.index, columns=shap_feature_names)

    shap_analyzer = SHAPAnalyzer(pca_pipeline["model"], shap_feature_names)
    shap_analyzer.fit(x_shap_df)

    # Get sentiment impact analysis using direct method (more reliable than SHAP)
    # SHAP analyzer sees PCA-transformed features (PC1, PC2, etc) so it can't find "sentiment" columns
    sentiment_impact = _compute_sentiment_analysis(feat_df, news_df=news_df)
    
    # If direct method returns empty, try SHAP as fallback
    if not sentiment_impact:
        sentiment_impact = shap_analyzer.analyze_sentiment_impact(x_shap_df, feat_df["signal"].values)

    # 2. Hurst Exponent and Regime Detection
    # Use shorter lookback for limited data
    short_lookback = min(50, len(feat_df) - 10)
    hurst_values = pd.Series(dtype=float)
    current_regime = None
    adaptive_signals = feat_df["signal"]

    try:
        hurst_analyzer = HurstAnalyzer(lookback=short_lookback)
        # calculate_windowed slices prices[i-window:i], and calculate() uses diff(prices),
        # so the effective returns length is window-1. Use lookback+1 to avoid fallback-to-0.5.
        hurst_window = min(short_lookback + 1, len(feat_df))
        hurst_values = hurst_analyzer.calculate_windowed(feat_df["Adj Close"].values, hurst_window)
    except Exception as e:
        logger.warning(f"Hurst calculation failed: {e}")

    try:
        returns = feat_df["Adj Close"].pct_change().dropna().values
        regime_detector = RegimeDetector(feat_df["Adj Close"].values, returns, lookback=short_lookback)
        current_regime = regime_detector.get_regime()

    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")

    # 3. Kalman-OU Mean-Reversion Analysis
    try:
        kalman_ou = KalmanOUFilter(lookback=short_lookback, z_threshold=2.0)
        fitted_params = kalman_ou.fit(feat_df["Adj Close"].values)

        # Get signal for last price
        last_price = feat_df["Adj Close"].iloc[-1]
        ou_signal, ou_diag = kalman_ou.predict(last_price)
    except Exception as e:
        logger.warning(f"Kalman-OU analysis failed: {e}")
        fitted_params = None
        ou_signal = 0.0

    # Extract key variables for reporting
    latest_bar_ts = pd.to_datetime(feat_df.index[-1], errors="coerce")
    now_utc = pd.Timestamp.now("UTC")
    if pd.isna(latest_bar_ts):
        market_bar_age_days = None
    else:
        if latest_bar_ts.tzinfo is None:
            latest_bar_ts = latest_bar_ts.tz_localize("UTC")
        else:
            latest_bar_ts = latest_bar_ts.tz_convert("UTC")
        market_bar_age_days = float((now_utc - latest_bar_ts).total_seconds() / 86400.0)

    theta_floor_hit = False
    if fitted_params is not None:
        theta_floor_hit = bool(abs(float(fitted_params.theta) - 0.001) < 1e-12)

    key_variables = {
        "sentiment_impact": sentiment_impact,
        "current_regime": current_regime.state.value if current_regime else None,
        "avg_hurst": hurst_values.mean() if len(hurst_values) > 0 else None,
        "kalman_ou_theta": fitted_params.theta if fitted_params else None,
        "theta_floor_hit": theta_floor_hit,
        "market_bar_age_days": market_bar_age_days,
        "VaR_95": risk_metrics["VaR_95"],
        "ES_95": risk_metrics["ES_95"],
        "active_profile": cfg.get("active_profile", profile if profile else "auto"),
        "selected_ticker_count": len(cfg.get("market", {}).get("tickers", []) or []),
        "max_live_tickers": int(cfg.get("live", {}).get("max_live_tickers", 0) or 0),
        "max_position_fraction": float(cfg.get("broker", {}).get("max_position_fraction", 0.0) or 0.0),
    }

    avg_hurst_str = (
        f"{key_variables['avg_hurst']:.4f}"
        if key_variables["avg_hurst"] is not None
        else "N/A"
    )
    kalman_theta_str = (
        f"{key_variables['kalman_ou_theta']:.4f}"
        if key_variables["kalman_ou_theta"] is not None
        else "N/A"
    )
    market_bar_age_days_str = (
        f"{key_variables['market_bar_age_days']:.2f}"
        if key_variables["market_bar_age_days"] is not None
        else "N/A"
    )

    # Write key variables to text file
    output_file = root / "reports" / "key_variables_report.txt"
    with open(output_file, 'w') as f:
        f.write("=== Key Variables Report ===\n\n")
        f.write(f"Sentiment Impact Analysis:\n")
        if sentiment_impact:
            for key, value in sentiment_impact.items():
                if isinstance(value, dict):
                    neg = value.get('negative_impact', 'N/A')
                    pos = value.get('positive_impact', 'N/A')
                    elast = value.get('elasticity', 'N/A')
                    if isinstance(neg, float) and isinstance(pos, float) and isinstance(elast, float):
                        f.write(f"  {key}:\n")
                        f.write(f"    Negative Sentiment Impact: {neg:.6f}\n")
                        f.write(f"    Positive Sentiment Impact: {pos:.6f}\n")
                        f.write(f"    Elasticity: {elast:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        else:
            f.write("  (No sentiment analysis available)\n")

        f.write(f"\nCurrent Market Regime: {key_variables['current_regime']}\n")
        f.write(f"Average Hurst Exponent: {avg_hurst_str}\n")
        f.write(f"Kalman-OU Theta (Mean-Reversion Speed): {kalman_theta_str}\n")
        f.write(f"Kalman-OU Theta Floor Hit: {key_variables['theta_floor_hit']}\n")
        f.write(f"Market Bar Age (days): {market_bar_age_days_str}\n")
        f.write(f"Value at Risk (95%): {key_variables['VaR_95']:.4f}\n")
        f.write(f"Expected Shortfall (95%): {key_variables['ES_95']:.4f}\n")

    logger.info(f"Key variables written to {output_file}")

    print("\n=== Backtest summary (Option-model signals) ===")
    print(f"Final equity:     {backtest_df['equity'].iloc[-1]:.2f}")
    print(f"Buy & hold value: {backtest_df['buy_hold_cum'].iloc[-1]:.2f}")
    if stats["total_return"] is not None:
        print(f"Total return:     {stats['total_return'] * 100:.2f}%")
    else:
        print("Total return:     N/A")
    if stats["cagr"] is not None:
        print(f"CAGR:             {stats['cagr'] * 100:.2f}%")
    else:
        print("CAGR:             N/A")
    if stats["sharpe"] is not None:
        print(f"Sharpe:           {stats['sharpe']:.2f}")
    else:
        print("Sharpe:           N/A")
    print(f"Max drawdown:     {stats['max_drawdown'] * 100:.2f}%")

    print(f"\nNumber of trades: {len(broker.trade_log)}")

    print("\n=== Advanced Analysis Results ===")
    print(f"Current Regime: {key_variables['current_regime']}")
    print(f"Avg Hurst: {avg_hurst_str}")
    print(f"VaR 95%: {key_variables['VaR_95']:.4f}")
    print(f"ES 95%: {key_variables['ES_95']:.4f}")
    print(f"Kalman-OU Theta: {kalman_theta_str}")
    print(f"Kalman Theta Floor Hit: {key_variables['theta_floor_hit']}")
    print(f"Market Bar Age (days): {market_bar_age_days_str}")
    print(f"Sentiment impacts computed: {len(sentiment_impact)} interactions")

    live_cfg = cfg.get("live", {})
    auto_start_paper = bool(live_cfg.get("auto_start_paper_trading", True))
    launch_dashboard_enabled = bool(live_cfg.get("launch_dashboard", True))

    # Optionally auto-start continuous paper trading loop
    if auto_start_paper:
        def launch_live_paper():
            try:
                run_live_paper(profile=profile)
            except Exception as e:
                logger.exception("Live paper loop failed: %s", e)

        logger.info("Auto-start enabled: launching continuous live paper trading...")
        live_thread = Thread(target=launch_live_paper, daemon=False)
        live_thread.start()

    # Launch dashboard in separate thread
    def launch_dashboard():
        dashboard_ticker = "MULTI-TICKER" if auto_start_paper else mcfg["ticker"]
        live_log_path = (root / "logs" / "live_log_multi_nn.csv") if auto_start_paper else None
        tk_root = tk.Tk()
        dashboard = BacktestDashboard(
            tk_root,
            backtest_df,
            stats,
            broker,
            ticker=dashboard_ticker,
            key_variables=key_variables,
            live_log_path=live_log_path,
        )
        tk_root.mainloop()

    if launch_dashboard_enabled:
        logger.info("Launching dashboard...")
        # Run Tkinter in main thread for better Windows compatibility.
        try:
            launch_dashboard()
        except KeyboardInterrupt:
            logger.info("Dashboard closed.")

        # If live paper trading is enabled, keep running continuously.
        if auto_start_paper:
            logger.info("Dashboard closed; live paper trading continues. Press Ctrl+C to stop.")
            try:
                while live_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping live paper trading.")
    elif auto_start_paper:
        # No dashboard, keep alive while continuous live paper trading runs
        logger.info("Dashboard disabled; live paper trading running continuously.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping live paper trading.")


if __name__ == "__main__":
    main()
