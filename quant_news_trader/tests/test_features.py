"""Quick test of extended feature builder."""
import numpy as np
import pandas as pd
from src.features.feature_builder import build_features

# Create minimal synthetic data for testing
dates = pd.date_range('2023-01-01', periods=100, freq='D')
price_data = {
    'Date': dates,
    'Open': np.random.uniform(150, 160, 100),
    'High': np.random.uniform(160, 170, 100),
    'Low': np.random.uniform(140, 150, 100),
    'Adj Close': np.random.uniform(150, 160, 100),
    'Volume': np.random.uniform(1e6, 2e6, 100),
}
price_df = pd.DataFrame(price_data).set_index('Date')

news_data = {
    'published_at': dates[:50],
    'sentiment': np.random.uniform(-1, 1, 50),
}
news_df = pd.DataFrame(news_data)

# Build features
feat_df = build_features(price_df, news_df)
print(f"✓ Feature builder ran successfully")
print(f"✓ Output shape: {feat_df.shape}")
print(f"✓ Total columns: {len(feat_df.columns)}")
print(f"\nNew technical factor columns:")
tech_factors = [
    'ema_10', 'ema_50', 'ema_200',
    'rsi_14', 'atr_14', 'bb_pct',
    'macd_12_26', 'macd_signal_9', 'macd_hist',
    'obv', 'day_of_week', 'month'
]
for col in tech_factors:
    if col in feat_df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} (missing)")

print(f"\nModel-ready sample (last row):")
print(feat_df.iloc[-1][['ema_10', 'rsi_14', 'atr_14', 'macd_hist', 'obv', 'daily_sentiment_ema_5']].to_dict())
