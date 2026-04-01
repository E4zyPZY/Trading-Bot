"""
Stock Selection Module
======================

Selects stocks to trade based on quant AI criteria.
Currently selects stocks that have trained NN models available.
"""

import yaml
from pathlib import Path
from typing import List
import os
import pandas as pd


def load_universe(root: Path) -> List[str]:
    """
    Load the full NASDAQ universe from YAML file.
    """
    universe_path = root / "data" / "nasdaq_universe.yaml"
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    with universe_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


def select_stocks(root: Path, max_stocks: int = 20) -> List[str]:
    """
    Select stocks to trade based on quant AI criteria.

    Selects top stocks by Sharpe ratio from approved_universe.csv,
    or falls back to stocks with trained NN models.
    """
    universe = load_universe(root)
    approved_path = root / "data" / "approved_universe.csv"

    if approved_path.exists():
        try:
            df = pd.read_csv(approved_path)
            # Sort by sharpe descending
            df_sorted = df.sort_values('sharpe', ascending=False)
            selected = df_sorted['ticker'].head(max_stocks).tolist()
            # Filter to universe
            selected = [t for t in selected if t in universe]
            if selected:
                return selected
        except Exception as e:
            print(f"Error reading approved_universe.csv: {e}")

    # Fallback to models
    models_dir = root / "models" / "nn_live"
    selected = []
    for ticker in universe:
        meta_path = models_dir / f"{ticker}_nn_meta.json"
        model_path = models_dir / f"{ticker}_nn.pth"
        if meta_path.exists() and model_path.exists():
            selected.append(ticker)
            if len(selected) >= max_stocks:
                break

    # If no models, select first max_stocks from universe
    if not selected:
        selected = universe[:max_stocks]

    return selected


def get_selected_tickers(root: Path) -> List[str]:
    """
    Get the list of selected tickers for trading.
    """
    return select_stocks(root)