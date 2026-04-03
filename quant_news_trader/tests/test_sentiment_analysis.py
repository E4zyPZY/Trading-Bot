#!/usr/bin/env python3
"""
Test script to verify sentiment analysis is working correctly.
Tests the _compute_sentiment_analysis function with sample data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import _compute_sentiment_analysis


def test_sentiment_analysis():
    """Test sentiment analysis with mock data."""
    # Create mock feature dataframe
    np.random.seed(42)
    n_rows = 100
    
    feat_df = pd.DataFrame({
        'daily_sentiment': np.random.uniform(-0.5, 0.5, n_rows),
        'daily_sentiment_ema_5': np.random.uniform(-0.3, 0.3, n_rows),
        'daily_sentiment_ema_20': np.random.uniform(-0.2, 0.2, n_rows),
        'rolling_vol_5': np.random.uniform(0.01, 0.05, n_rows),
        'rolling_vol_20': np.random.uniform(0.01, 0.05, n_rows),
        'atr_14': np.random.uniform(0.5, 2.0, n_rows),
        'prob_up': np.random.uniform(0.3, 0.7, n_rows),
    })
    
    print("Test Data Created:")
    print(f"  Shape: {feat_df.shape}")
    print(f"  Columns: {list(feat_df.columns)}")
    print(f"  Daily Sentiment stats: min={feat_df['daily_sentiment'].min():.3f}, max={feat_df['daily_sentiment'].max():.3f}")
    
    # Run sentiment analysis
    print("\nRunning sentiment analysis...")
    result = _compute_sentiment_analysis(feat_df)
    
    # Check results
    if not result:
        print("ERROR: Sentiment analysis returned empty dict!")
        return False
    
    print(f"\nSentiment Analysis Results ({len(result)} interactions found):")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            print(f"    Negative Impact: {value.get('negative_impact', 'N/A'):.6f}")
            print(f"    Positive Impact: {value.get('positive_impact', 'N/A'):.6f}")
            print(f"    Elasticity: {value.get('elasticity', 'N/A'):.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Sentiment analysis test PASSED")
    return True


def test_with_all_zero_sentiment():
    """Test that zero sentiment is handled gracefully."""
    n_rows = 100
    feat_df = pd.DataFrame({
        'daily_sentiment': np.zeros(n_rows),  # All zeros - should skip
        'rolling_vol_5': np.random.uniform(0.01, 0.05, n_rows),
        'prob_up': np.random.uniform(0.3, 0.7, n_rows),
    })
    
    print("\n\nTest: All-zero sentiment")
    result = _compute_sentiment_analysis(feat_df)
    print(f"  Result: {len(result)} interactions (expected 0 since sentiment is all zeros)")
    if len(result) == 0:
        print("  ✓ Correctly skipped zero sentiment")
        return True
    else:
        print("  ✗ FAILED: Should skip zero sentiment")
        return False


def test_without_vol_features():
    """Test that missing vol features are handled gracefully."""
    n_rows = 100
    feat_df = pd.DataFrame({
        'daily_sentiment': np.random.uniform(-0.5, 0.5, n_rows),
        # No vol features
        'prob_up': np.random.uniform(0.3, 0.7, n_rows),
    })
    
    print("\n\nTest: Missing vol features")
    result = _compute_sentiment_analysis(feat_df)
    # Should still have sentiment_prob_up interaction
    print(f"  Result: {len(result)} interactions")
    if 'daily_sentiment_prob_up' in result:
        print("  ✓ Correctly computed sentiment_prob_up interaction")
        return True
    else:
        print("  ✗ FAILED: Should have sentiment_prob_up")
        return False


if __name__ == "__main__":
    print("="*60)
    print("SENTIMENT ANALYSIS TEST SUITE")
    print("="*60)
    
    all_passed = True
    all_passed &= test_sentiment_analysis()
    all_passed &= test_with_all_zero_sentiment()
    all_passed &= test_without_vol_features()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("="*60)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("="*60)
        sys.exit(1)
