"""
Test: Advanced Explainability & Regime Detection System
========================================================

Validates:
1. SHAP explainability analysis
2. Kalman-filtered O-U mean-reversion model
3. Hurst exponent regime detection
4. Full integration of all three systems
"""

import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

def test_shap_analyzer():
    """Test SHAP-based model explainability."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: SHAP Explainability Analyzer")
    logger.info("="*60)
    
    try:
        from src.explainability.shap_analyzer import SHAPAnalyzer
        from sklearn.ensemble import RandomForestClassifier
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 8)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Target depends on first 2 features
        
        feature_names = [f"feature_{i}" for i in range(8)]
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        
        # Test SHAP
        analyzer = SHAPAnalyzer(model, feature_names)
        analyzer.fit(pd.DataFrame(X, columns=feature_names), max_samples=50)
        
        # Get feature importance
        importance = analyzer.get_feature_importance(pd.DataFrame(X, columns=feature_names))
        
        logger.info("✓ SHAP analyzer initialized")
        logger.info(f"✓ Feature importance calculated:")
        logger.info(f"\n{importance.head()}")
        
        # Test explanation of single prediction
        x_test = pd.Series(X[0], index=feature_names, name="test_sample")
        explanation = analyzer.explain_prediction(x_test)
        
        logger.info(f"✓ Single prediction explained (sample)")
        logger.info(f"  Feature contributions: {len(explanation)} features")
        
        logger.info("✓ SHAP Explainability: PASSED")
        return True
    
    except ImportError as e:
        logger.warning(f"⚠ SHAP not installed: {e}")
        logger.warning("  Install with: pip install shap")
        return False
    except Exception as e:
        logger.error(f"✗ SHAP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kalman_ou_filter():
    """Test Kalman filter + O-U mean-reversion model."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Kalman-Filtered Ornstein-Uhlenbeck Model")
    logger.info("="*60)
    
    try:
        from src.models.kalman_ou import KalmanOUFilter, PairsTrader, OUParams
        
        # Generate mean-reverting price series
        np.random.seed(42)
        ou_params = OUParams(theta=0.05, mu=100, sigma=2)
        
        prices = [100]
        for _ in range(500):
            dt = 1.0
            dx = ou_params.theta * (ou_params.mu - prices[-1]) * dt + \
                 ou_params.sigma * np.random.normal(0, np.sqrt(dt))
            prices.append(prices[-1] + dx)
        
        prices = np.array(prices)
        
        # Test KalmanOUFilter
        kou = KalmanOUFilter(lookback=252, z_threshold=1.5)
        fitted_params = kou.fit(prices[:250])
        
        logger.info(f"✓ Kalman filter fitted")
        logger.info(f"  O-U parameters: {fitted_params}")
        
        # Get signal
        signal, diag = kou.predict(prices[250])
        
        logger.info(f"✓ Mean-reversion signal generated")
        logger.info(f"  Signal: {signal:.2f} (range -1 to 1)")
        logger.info(f"  Equilibrium: {diag['equilibrium']:.2f}")
        logger.info(f"  Current spread: {diag['spread']:.2f}")
        logger.info(f"  Z-score: {diag['z_score']:.2f}")
        
        half_life = kou.half_life_reversion()
        logger.info(f"✓ Half-life of reversion: {half_life:.1f} periods")
        
        # Test PairsTrader
        prices_a = prices[:300]
        prices_b = prices[:300] * 0.98 + np.random.normal(0, 2, 300)  # Correlated series
        
        pairs = PairsTrader('ASSET_A', 'ASSET_B')
        hedge_ratio = pairs.fit(prices_a, prices_b)
        
        logger.info(f"✓ Pairs trader fitted")
        logger.info(f"  Hedge ratio: {hedge_ratio:.4f}")
        
        position = pairs.get_position_size(prices_a[-1], prices_b[-1], 100000)
        logger.info(f"✓ Pairs position size calculated")
        logger.info(f"  Quantity A: {position['quantity_a']:.0f}")
        logger.info(f"  Quantity B: {position['quantity_b']:.0f}")
        
        logger.info("✓ Kalman O-U Model: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Kalman O-U test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regime_detection():
    """Test Hurst exponent and regime detection."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Hurst Exponent & Regime Detection")
    logger.info("="*60)
    
    try:
        from src.analysis.regime_detection import (
            HurstAnalyzer, RegimeDetector, adaptive_signal_by_regime
        )
        
        # Generate price series with mixed regimes
        np.random.seed(42)
        prices = [100]
        
        for i in range(1000):
            # Trending regime (first 250), mean-reverting (next 250), etc.
            segment = i // 250
            
            if segment % 2 == 0:
                # Trending: momentum
                ret = np.random.normal(0.0005, 0.02)
            else:
                # Mean-reverting: oscillating
                ret = np.random.normal(0, 0.012)
            
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        
        # Test HurstAnalyzer
        hurst = HurstAnalyzer(lookback=252)
        h_exp = hurst.calculate(prices[-252:])
        
        logger.info(f"✓ Hurst exponent calculated: {h_exp:.4f}")
        if h_exp > 0.55:
            interpretation = "Trending"
        elif h_exp < 0.45:
            interpretation = "Mean-reverting"
        else:
            interpretation = "Random walk"
        logger.info(f"  Interpretation: {interpretation}")
        
        # Test RegimeDetector
        detector = RegimeDetector(prices, returns, lookback=252)
        regime = detector.get_regime()
        
        logger.info(f"✓ Regime detected: {regime.state.value}")
        logger.info(f"  Hurst: {regime.hurst_exponent:.4f}")
        logger.info(f"  Volatility: {regime.volatility:.4f}")
        logger.info(f"  Regime score: {regime.regime_score:.2f}")
        logger.info(f"  Strategy: {regime.suggested_strategy}")
        
        # Test position sizing multiplier
        multiplier = detector.get_position_sizing_multiplier()
        logger.info(f"✓ Position sizing multiplier: {multiplier:.2f}")
        
        # Test adaptive signal adjustment
        base_signal = 0.8
        adjusted_signal = adaptive_signal_by_regime(base_signal, prices, returns)
        logger.info(f"✓ Signal adjustment")
        logger.info(f"  Base signal: {base_signal:.2f}")
        logger.info(f"  Adjusted signal: {adjusted_signal:.2f}")
        
        logger.info("✓ Regime Detection: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Regime detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration of all three systems."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Full Integration (SHAP + Kalman-OU + Regime)")
    logger.info("="*60)
    
    try:
        from src.explainability.shap_analyzer import SHAPAnalyzer
        from src.models.kalman_ou import KalmanOUFilter
        from src.analysis.regime_detection import RegimeDetector
        from src.models.model import train_pca_direction_model
        from sklearn.preprocessing import StandardScaler
        
        # Generate synthetic trading data
        np.random.seed(42)
        n_samples = 200
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + 0.5*X[:, 1] > 0).astype(int)
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X[:150], y[:150])
        
        logger.info("✓ Model trained")
        
        # 1. SHAP explainability
        try:
            analyzer = SHAPAnalyzer(model, feature_names)
            analyzer.fit(X_df.iloc[:50], max_samples=30)
            importance = analyzer.get_feature_importance(X_df.iloc[150:])
            logger.info(f"✓ SHAP: Top feature = {importance.iloc[0]['feature']}")
        except ImportError:
            logger.warning("⚠ SHAP skipped (not installed)")
        
        # 2. Kalman O-U mean-reversion
        price_series = np.exp(np.cumsum(np.random.normal(0, 0.02, 300)))
        kou = KalmanOUFilter(lookback=150)
        kou.fit(price_series[:150])
        signal, _ = kou.predict(price_series[150])
        logger.info(f"✓ Kalman O-U: Mean-reversion signal = {signal:.2f}")
        
        # 3. Regime detection
        returns = np.diff(np.log(price_series))
        detector = RegimeDetector(price_series, returns)
        regime = detector.get_regime()
        logger.info(f"✓ Regime: {regime.state.value} (H={regime.hurst_exponent:.3f})")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Full Integration: PASSED")
        logger.info("="*60)
        return True
    
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "█"*60)
    logger.info("ADVANCED EXPLAINABILITY & REGIME DETECTION TEST SUITE")
    logger.info("█"*60)
    
    results = {
        'SHAP': test_shap_analyzer(),
        'Kalman-OU': test_kalman_ou_filter(),
        'Regime Detection': test_regime_detection(),
        'Integration': test_integration(),
    }
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL SYSTEMS OPERATIONAL ✓")
        return 0
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
