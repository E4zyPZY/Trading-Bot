"""Quick test of PCA model training and prediction."""
import numpy as np
import pandas as pd
from src.features.feature_builder import build_features
from src.models.model import train_pca_direction_model, predict_proba_pca, FEATURE_COLS

# Create synthetic data
dates = pd.date_range('2023-01-01', periods=200, freq='D')
price_data = {
    'Date': dates,
    'Open': 150 + np.cumsum(np.random.uniform(-2, 2, 200)),
    'High': 160 + np.cumsum(np.random.uniform(-2, 2, 200)),
    'Low': 140 + np.cumsum(np.random.uniform(-2, 2, 200)),
    'Adj Close': 150 + np.cumsum(np.random.uniform(-2, 2, 200)),
    'Volume': np.random.uniform(1e6, 2e6, 200),
}
price_df = pd.DataFrame(price_data).set_index('Date')

news_data = {
    'published_at': dates[:100],
    'sentiment': np.random.uniform(-1, 1, 100),
}
news_df = pd.DataFrame(news_data)

# Build features
feat_df = build_features(price_df, news_df)
print(f"✓ Built feature dataframe: {feat_df.shape}")

# Train PCA model
try:
    pipeline = train_pca_direction_model(
        feat_df,
        n_components=8,
        test_size=0.2,
        random_state=42,
    )
    print(f"✓ PCA model trained successfully")
    print(f"  - Scaler: {type(pipeline['scaler']).__name__}")
    print(f"  - PCA components: {pipeline['pca'].n_components_}")
    print(f"  - Explained variance: {np.sum(pipeline['explained_variance_ratio']):.2%}")
    print(f"  - Model type: {type(pipeline['model']).__name__}")
    
    # Test prediction on last row
    last_row = feat_df.iloc[-1]
    prob_down, prob_up = predict_proba_pca(pipeline, last_row)
    print(f"\n✓ Prediction on last row:")
    print(f"  - P(down): {prob_down:.4f}")
    print(f"  - P(up): {prob_up:.4f}")
    print(f"  - Signal: {'LONG' if prob_up > 0.6 else 'SHORT' if prob_down > 0.6 else 'FLAT'}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
