# src/portfolio_dashboard.py

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from src.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.trading.moomoo_broker import MoomooPaperBroker
from src.market_outlook import assess_market_outlook
from src.analysis.regime_detection import RegimeDetector
from src.data.market_data import fetch_market_data

REFRESH_MS = 10000  # refresh every 10 seconds


class PortfolioDashboard:
    def __init__(self, root_tk: tk.Tk, broker: MoomooPaperBroker, currency: str = "USD"):
        self.root = root_tk
        self.broker = broker
        self.currency = currency
        self.root_path = Path(__file__).resolve().parents[1]

        self.root.title("🤖 AI Quant Trader Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Initialize data storage for charts
        self.equity_history = []
        self.time_history = []

        self.setup_ui()
        self.refresh()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Header
        header_frame = ttk.Frame(main_frame, style='Header.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)

        title_label = ttk.Label(header_frame, text="🤖 AI Quant Trader Dashboard",
                               font=("Segoe UI", 16, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W)

        self.status_label = ttk.Label(header_frame, text="Initializing...",
                                     font=("Segoe UI", 10), foreground="orange")
        self.status_label.grid(row=0, column=1, sticky=tk.E)

        # Left column - Portfolio & Risk
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Portfolio Summary
        self.create_portfolio_section(left_frame)

        # Risk Metrics
        self.create_risk_section(left_frame)

        # Market Commentary
        self.create_market_section(left_frame)

        # Right column - Charts & Positions
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        # Equity Chart
        self.create_chart_section(right_frame)

        # Positions
        self.create_positions_section(right_frame)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

    def create_portfolio_section(self, parent):
        frame = ttk.LabelFrame(parent, text="💰 Portfolio Summary", padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.portfolio_text = tk.Text(frame, height=6, width=50, font=("Consolas", 9),
                                     bg='#f8f8f8', relief='flat')
        self.portfolio_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        frame.columnconfigure(0, weight=1)

    def create_risk_section(self, parent):
        frame = ttk.LabelFrame(parent, text="⚠️ Risk Metrics", padding="5")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.risk_text = tk.Text(frame, height=8, width=50, font=("Consolas", 9),
                                bg='#fff8f8', relief='flat')
        self.risk_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        frame.columnconfigure(0, weight=1)

    def create_market_section(self, parent):
        frame = ttk.LabelFrame(parent, text="🌍 Market Commentary", padding="5")
        frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.market_text = tk.Text(frame, height=6, width=50, font=("Segoe UI", 9),
                                  bg='#f8fff8', relief='flat', wrap=tk.WORD)
        self.market_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def create_chart_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📈 Portfolio Performance", padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.ax.set_title("Portfolio Equity Over Time", fontsize=10)
        self.ax.set_xlabel("Time", fontsize=8)
        self.ax.set_ylabel(f"Equity ({self.currency})", fontsize=8)
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def create_positions_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📊 Current Positions", padding="5")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Positions listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.positions_listbox = tk.Listbox(
            list_frame,
            width=60,
            height=12,
            font=("Consolas", 8),
            bg='#f8f8f8',
            yscrollcommand=scrollbar.set
        )
        self.positions_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.positions_listbox.yview)

        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Status label (moved to header)
        # self.status = tk.Label(...)  # Removed, now in header

        # kick off auto-refresh
        self.refresh()

    def refresh(self):
        try:
            # Get portfolio data
            snapshot = self.broker.snapshot_portfolio(self.currency)
            self.update_portfolio_display(snapshot)

            # Get risk metrics
            self.update_risk_display()

            # Get market commentary
            self.update_market_display()

            # Update positions
            self.update_positions_display(snapshot)

            # Update chart
            self.update_chart(snapshot)

            self.status_label.config(text="✅ Last update: OK", foreground="green")

        except Exception as e:
            error_msg = f"❌ Error: {str(e)[:50]}..."
            self.status_label.config(text=error_msg, foreground="red")
            print(f"Dashboard refresh error: {e}")

        # schedule next refresh
        self.root.after(REFRESH_MS, self.refresh)

    def update_portfolio_display(self, snapshot):
        equity = float(snapshot.get("equity", 0.0))
        cash = float(snapshot.get("cash", 0.0))
        sec_mv = float(snapshot.get("sec_market_value", 0.0))
        positions_raw = snapshot.get("positions", None)

        # Calculate performance metrics
        total_value = cash + sec_mv
        initial_capital = 1000000  # From config
        total_return = (total_value - initial_capital) / initial_capital * 100

        # Handle positions
        if isinstance(positions_raw, pd.DataFrame):
            positions_df = positions_raw
        elif positions_raw is None:
            positions_df = pd.DataFrame()
        else:
            positions_df = pd.DataFrame(positions_raw)

        portfolio_info = f"""💰 PORTFOLIO SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equity (Net):     ${equity:>12,.2f}
Cash:             ${cash:>12,.2f}
Securities:       ${sec_mv:>12,.2f}
Total Value:      ${total_value:>12,.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Return:     {total_return:>12.2f}%
Positions:        {len(positions_df):>12d}
Initial Capital:  ${initial_capital:>12,.2f}
"""

        self.portfolio_text.delete(1.0, tk.END)
        self.portfolio_text.insert(tk.END, portfolio_info)

    def update_risk_display(self):
        try:
            # Try to read key variables report
            report_path = self.root_path / "reports" / "key_variables_report.txt"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    content = f.read()

                # Extract key metrics
                lines = content.split('\n')
                regime = "Unknown"
                hurst = "N/A"
                var_95 = "N/A"
                es_95 = "N/A"

                for line in lines:
                    if "Current Market Regime:" in line:
                        regime = line.split(":")[1].strip()
                    elif "Average Hurst Exponent:" in line:
                        hurst = line.split(":")[1].strip()
                    elif "Value at Risk (95%):" in line:
                        var_95 = line.split(":")[1].strip()
                    elif "Expected Shortfall (95%):" in line:
                        es_95 = line.split(":")[1].strip()

                risk_info = f"""⚠️ RISK METRICS (95% Confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Market Regime:    {regime:>12}
Hurst Exponent:   {hurst:>12}
VaR (95%):        {var_95:>12}
ES (95%):         {es_95:>12}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:           {'🟢 Low Risk' if regime == 'Trending Low Vol' else '🟡 Moderate' if regime == 'Random Walk' else '🔴 High Risk'}
"""
            else:
                risk_info = """⚠️ RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report not available. Run backtests to generate.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:           ⚪ Unknown
"""

        except Exception as e:
            risk_info = f"""⚠️ RISK METRICS
Error loading risk data: {str(e)[:30]}...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:           ⚪ Unknown
"""

        self.risk_text.delete(1.0, tk.END)
        self.risk_text.insert(tk.END, risk_info)

    def update_market_display(self):
        try:
            # Get current market outlook
            profile = assess_market_outlook(self.root_path)

            # Get SPY performance
            market_data = fetch_market_data("SPY", lookback_days=30, interval="1d")
            if not market_data.empty:
                recent_prices = market_data["Close"]
                current_price = recent_prices.iloc[-1]
                prev_price = recent_prices.iloc[-2] if len(recent_prices) > 1 else current_price
                daily_change = (current_price - prev_price) / prev_price * 100

                market_trend = "📈 Bullish" if daily_change > 0.5 else "📉 Bearish" if daily_change < -0.5 else "➡️ Sideways"
            else:
                daily_change = 0
                market_trend = "❓ Unknown"

            # Generate commentary
            if profile == "aggro":
                strategy = "Aggressive - Capitalizing on strong trends"
                outlook = "Market conditions favorable for active trading"
            elif profile == "balanced":
                strategy = "Balanced - Moderate risk management"
                outlook = "Market showing mixed signals, cautious approach"
            else:  # conservative
                strategy = "Conservative - Risk mitigation priority"
                outlook = "Market volatility high, defensive positioning"

            commentary = f"""🌍 MARKET COMMENTARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Strategy: {strategy}
Market Outlook:   {outlook}

SPY Daily Change: {daily_change:+.2f}%
Market Trend:     {market_trend}

AI Assessment:    {profile.title()} mode activated
Last Update:      {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Portfolio is {'performing well' if daily_change > 0 else 'under pressure'} in current conditions."""

        except Exception as e:
            commentary = f"""🌍 MARKET COMMENTARY
Error generating commentary: {str(e)[:30]}...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please check market data connection."""

        self.market_text.delete(1.0, tk.END)
        self.market_text.insert(tk.END, commentary)

    def update_positions_display(self, snapshot):
        positions_raw = snapshot.get("positions", None)

        if isinstance(positions_raw, pd.DataFrame):
            positions_df = positions_raw
        elif positions_raw is None:
            positions_df = pd.DataFrame()
        else:
            positions_df = pd.DataFrame(positions_raw)

        self.positions_listbox.delete(0, tk.END)

        if len(positions_df) > 0:
            # Header
            self.positions_listbox.insert(tk.END, f"{'Ticker':<8}{'Qty':>8}{'Price':>10}{'Mkt Val':>12}{'Weight':>8}")
            self.positions_listbox.insert(tk.END, "─" * 55)

            total_value = float(snapshot.get("equity", 1.0))

            for _, pos in positions_df.iterrows():
                code = str(pos.get("code", "") or "")
                if "." in code:
                    ticker = code.split(".", 1)[1]
                else:
                    ticker = str(pos.get("ticker", "") or pos.get("symbol", "") or code)

                qty = pos.get("qty") or pos.get("position_qty") or pos.get("can_sell_qty") or 0
                try:
                    qty = int(qty)
                except:
                    qty = 0

                price = pos.get("last_price") or pos.get("price") or pos.get("cost_price") or 0.0
                try:
                    price = float(price)
                except:
                    price = 0.0

                mv = pos.get("market_value") or pos.get("val") or (qty * price)
                try:
                    mv = float(mv)
                except:
                    mv = qty * price

                weight = (mv / total_value * 100) if total_value > 0 else 0

                line = f"{ticker:<8}{qty:>8}{price:>10.2f}${mv:>11,.0f}{weight:>7.1f}%"
                self.positions_listbox.insert(tk.END, line)
        else:
            self.positions_listbox.insert(tk.END, "(no open positions)")

    def update_chart(self, snapshot):
        # Add current equity to history
        current_time = datetime.now()
        equity = float(snapshot.get("equity", 0.0))

        self.time_history.append(current_time)
        self.equity_history.append(equity)

        # Keep only last 50 points for chart
        if len(self.time_history) > 50:
            self.time_history = self.time_history[-50:]
            self.equity_history = self.equity_history[-50:]

        # Update chart
        self.ax.clear()
        if self.equity_history:
            self.ax.plot(self.time_history, self.equity_history, 'b-', linewidth=2, marker='o', markersize=3)
            self.ax.set_title("Portfolio Equity Over Time", fontsize=10, fontweight='bold')
            self.ax.set_xlabel("Time", fontsize=8)
            self.ax.set_ylabel(f"Equity ({self.currency})", fontsize=8)
            self.ax.grid(True, alpha=0.3)
            self.ax.tick_params(axis='both', which='major', labelsize=8)

            # Format x-axis dates
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.fig.autofmt_xdate()

            # Add current value annotation
            if self.equity_history:
                latest_equity = self.equity_history[-1]
                self.ax.annotate(f'${latest_equity:,.0f}',
                               xy=(self.time_history[-1], latest_equity),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                               fontsize=8)

        self.canvas.draw()


def main():
    logger = setup_logging()
    root_path = Path(__file__).resolve().parents[1]

    # parse optional --profile=xxx
    profile = None
    for arg in sys.argv[1:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]

    cfg = load_config(root_path, profile)
    mcfg = cfg["moomoo"]
    currency = mcfg.get("currency", "USD")

    broker = MoomooPaperBroker(
        host=mcfg.get("host", "127.0.0.1"),
        port=int(mcfg.get("port", 11111)),
        market=mcfg.get("market", "US"),
        security_firm=mcfg.get("security_firm", "FUTUSECURITIES"),
        logger=logger,
    )

    root_tk = tk.Tk()
    app = PortfolioDashboard(root_tk, broker, currency=currency)
    root_tk.mainloop()

    broker.close()


if __name__ == "__main__":
    main()
