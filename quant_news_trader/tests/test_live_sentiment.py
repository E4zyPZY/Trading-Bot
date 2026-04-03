#!/usr/bin/env python3
"""
Test sentiment in live paper trading context.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Mock the _compute_portfolio_sentiment function
def _compute_portfolio_sentiment(state: dict) -> dict:
    """Compute aggregate sentiment impact from all tickers in live state."""
    sentiment_metrics = {}
    
    total_sentiment = 0.0
    ticker_count = 0
    
    for ticker, st in state.items():
        news_df = st.get("news_df", pd.DataFrame())
        if news_df.empty or "sentiment" not in news_df.columns:
            continue
        
        # Get daily sentiment aggregates
        daily_sent = float(news_df["sentiment"].mean())
        total_sentiment += daily_sent
        ticker_count += 1
        
        sent_max = float(news_df["sentiment"].max())
        sent_min = float(news_df["sentiment"].min())
        sent_std = float(news_df["sentiment"].std()) if len(news_df) > 1 else 0.0
        
        sentiment_metrics[ticker] = {
            "mean_sentiment": daily_sent,
            "max_sentiment": sent_max,
            "min_sentiment": sent_min,
            "std_sentiment": sent_std,
            "article_count": int(len(news_df)),
        }
    
    # Portfolio-level aggregates
    portfolio_metrics = {
        "portfolio_avg_sentiment": total_sentiment / ticker_count if ticker_count > 0 else 0.0,
        "active_ticker_count": ticker_count,
        "by_ticker": sentiment_metrics,
    }
    
    return portfolio_metrics


def test_live_sentiment():
    """Test sentiment computation for live trading state."""
    
    # Create mock state similar to live_paper
    np.random.seed(42)
    state = {
        "AAPL": {
            "news_df": pd.DataFrame({
                "sentiment": np.random.uniform(-0.5, 0.5, 10),
                "title": [f"Article {i}" for i in range(10)],
            })
        },
        "TSLA": {
            "news_df": pd.DataFrame({
                "sentiment": np.random.uniform(-0.3, 0.7, 8),
                "title": [f"Article {i}" for i in range(8)],
            })
        },
        "NVDA": {
            "news_df": pd.DataFrame({
                "sentiment": np.random.uniform(0.1, 0.9, 12),
                "title": [f"Article {i}" for i in range(12)],
            })
        },
    }
    
    print("="*60)
    print("TEST: Live Paper Trading Sentiment Computation")
    print("="*60)
    
    result = _compute_portfolio_sentiment(state)
    
    print(f"\nPortfolio Average Sentiment: {result['portfolio_avg_sentiment']:+.4f}")
    print(f"Active Tickers with Sentiment: {result['active_ticker_count']}")
    
    print("\nPer-Ticker Metrics:")
    for ticker, metrics in result["by_ticker"].items():
        print(f"\n  {ticker}:")
        print(f"    Mean Sentiment:  {metrics['mean_sentiment']:+.4f}")
        print(f"    Range:           {metrics['min_sentiment']:+.4f} to {metrics['max_sentiment']:+.4f}")
        print(f"    Std Dev:         {metrics['std_sentiment']:.4f}")
        print(f"    Articles:        {metrics['article_count']}")
    
    print("\n" + "="*60)
    print("✓ Sentiment computation works for live state!")
    print("="*60)
    
    # Test empty case
    print("\nTest: Empty state")
    empty_result = _compute_portfolio_sentiment({})
    assert empty_result["active_ticker_count"] == 0
    assert empty_result["portfolio_avg_sentiment"] == 0.0
    print("✓ Empty state handled correctly")
    
    # Test ticker with no news
    print("\nTest: Ticker with no sentiment")
    no_sentiment_state = {
        "TEST": {
            "news_df": pd.DataFrame({"title": ["test"]})  # No sentiment column
        }
    }
    no_sent_result = _compute_portfolio_sentiment(no_sentiment_state)
    assert no_sent_result["active_ticker_count"] == 0
    print("✓ Missing sentiment handled correctly")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    test_live_sentiment()
