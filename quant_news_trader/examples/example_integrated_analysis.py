"""
Complete Integration Example: SHAP + Kalman O-U + Regime Detection
==================================================================

Shows how to use all three advanced modules together in a trading system.

This example demonstrates:
1. SHAP explainability on model predictions
2. Kalman-filtered mean-reversion signals
3. Hurst exponent regime detection
4. Signal combination and risk management
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def demonstrate_shap_analysis():
    """
    Example 1: Explain model predictions with SHAP.
    
    Shows which factors drive buy/sell signals and how sentiment impacts volatility.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: SHAP Model Explainability")
    logger.info("="*70)
    
    try:
        from src.explainability.shap_analyzer import SHAPAnalyzer
        from sklearn.ensemble import RandomForestClassifier
        
        # Simulate: Load trained PCA+RF model and test data
        np.random.seed(42)
        X_test = np.random.randn(50, 8)
        feature_names = ['ema_50', 'rsi_14', 'daily_sentiment_ema_5', 
                        'atr_14', 'bb_pct', 'macd_12_26', 'obv', 'momentum']
        
        # In production, load your saved model and data
        model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
        model.fit(X_test, np.random.randint(0, 2, 50))
        
        # Initialize and explain
        analyzer = SHAPAnalyzer(model, feature_names)
        analyzer.fit(pd.DataFrame(X_test, columns=feature_names), max_samples=30)
        
        # Get rankings
        importance = analyzer.get_feature_importance(pd.DataFrame(X_test, columns=feature_names))
        logger.info(f"\n📊 Feature Importance Ranking:")
        logger.info(importance.to_string(index=False))
        
        # Explain one prediction
        x_test = pd.Series(X_test[0], index=feature_names, name="sample_0")
        explanation = analyzer.explain_prediction(x_test)
        
        logger.info(f"\n🔍 Explanation for sample 0:")
        logger.info(f"   Top contributing factors to this prediction:")
        for i, (feat, val) in enumerate(list(explanation.items())[:3], 1):
            logger.info(f"   {i}. {feat}: SHAP value {val['shap_value']:.4f}")
        
        logger.info("✓ SHAP analysis complete\n")
        
    except ImportError as e:
        logger.warning(f"⚠ SHAP not available: {e}")
    except Exception as e:
        logger.error(f"✗ SHAP example failed: {e}")


def demonstrate_kalman_ou_meanreversion():
    """
    Example 2: Mean-reversion detection with Kalman-filtered O-U process.
    
    Shows how to:
    - Detect mean-reversion opportunities
    - Calculate half-life of reversion
    - Set optimal exit targets
    - Implement pairs trading
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Kalman-Filtered Ornstein-Uhlenbeck Mean-Reversion")
    logger.info("="*70)
    
    try:
        from src.models.kalman_ou import KalmanOUFilter, PairsTrader
        
        # Generate mean-reverting price series
        np.random.seed(42)
        prices = [100]
        for _ in range(300):
            # Ornstein-Uhlenbeck: reverts to 100 with volatility
            dt = 1.0
            theta = 0.04  # Fast mean-reversion
            mu = 100
            sigma = 2
            dx = theta * (mu - prices[-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            prices.append(prices[-1] + dx)
        
        prices = np.array(prices)
        
        # Fit Kalman O-U filter
        kou = KalmanOUFilter(lookback=200, z_threshold=1.5)
        kou.fit(prices[:-50])
        
        logger.info(f"\n📈 Kalman-O-U Filter Results:")
        logger.info(f"   O-U Parameters: {kou.ou_params}")
        logger.info(f"   Half-life of reversion: {kou.half_life_reversion():.1f} periods")
        
        # Generate signals for recent prices
        recent_signals = []
        for i in range(len(prices)-50, len(prices)):
            signal, diag = kou.predict(prices[i])
            recent_signals.append({
                'price': prices[i],
                'signal': signal,
                'z_score': diag['z_score'],
                'spread': diag['spread']
            })
        
        signals_df = pd.DataFrame(recent_signals)
        logger.info(f"\n💡 Recent signals (last 10 periods):")
        logger.info(signals_df.tail(10).to_string(index=False))
        
        # Example: Pairs trading
        logger.info(f"\n🤝 Pairs Trading Example:")
        prices_a = prices[:250]
        prices_b = prices[:250] * 0.99 + np.random.normal(0, 1.5, 250)  # Correlated
        
        pairs = PairsTrader('ASSET_A', 'ASSET_B')
        hedge_ratio = pairs.fit(prices_a, prices_b)
        
        logger.info(f"   Hedge ratio (β): {hedge_ratio:.4f}")
        logger.info(f"   Interpretation: Buy 1 A, Short {hedge_ratio:.2f} B for delta-neutral")
        
        # Position sizing
        position = pairs.get_position_size(prices_a[-1], prices_b[-1], 
                                           notional_capital=100000)
        logger.info(f"   Position for $100k capital:")
        logger.info(f"     - Asset A: {position['quantity_a']:.0f} shares")
        logger.info(f"     - Asset B: {position['quantity_b']:.0f} shares (short)")
        
        logger.info("✓ Mean-reversion analysis complete\n")
        
    except Exception as e:
        logger.error(f"✗ Kalman O-U example failed: {e}")


def demonstrate_regime_detection():
    """
    Example 3: Hurst exponent and market regime detection.
    
    Shows how to:
    - Calculate Hurst exponent (trending vs mean-reverting)
    - Detect 5 market regimes (volatility × trend combinations)
    - Adapt trading signals by regime
    - Size positions based on regime confidence
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Hurst Exponent & Market Regime Detection")
    logger.info("="*70)
    
    try:
        from src.analysis.regime_detection import (
            HurstAnalyzer, RegimeDetector, MarketRegime, adaptive_signal_by_regime
        )
        
        # Generate price series with mixed regimes
        np.random.seed(42)
        prices = [100]
        
        # Segment 1 (0-250): Trending up
        for _ in range(250):
            ret = np.random.normal(0.0008, 0.015)
            prices.append(prices[-1] * (1 + ret))
        
        # Segment 2 (250-500): Mean-reverting
        for _ in range(250):
            ret = np.random.normal(0, 0.008)
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        
        # Calculate Hurst for different periods
        logger.info(f"\n📊 Hurst Exponent Calculation:")
        hurst = HurstAnalyzer(lookback=150)
        
        h_first_half = hurst.calculate(prices[:250])
        h_second_half = hurst.calculate(prices[250:])
        
        logger.info(f"   Period 1 (0-250) Hurst: {h_first_half:.3f} → Trending")
        logger.info(f"   Period 2 (250-500) Hurst: {h_second_half:.3f} → Mean-reverting")
        logger.info(f"   Threshold: <0.45=Revert, 0.45-0.55=Random, >0.55=Trending")
        
        interaction = "✓ Correct" if (h_first_half > 0.55) and (h_second_half < 0.45) else "✗ Wrong"
        logger.info(f"   Regime detection: {interaction}")
        
        # Full regime detection
        logger.info(f"\n🎯 Market Regime Classification:")
        detector = RegimeDetector(prices, returns, lookback=200)
        regime = detector.get_regime()
        
        logger.info(f"   Current regime: {regime.state.value}")
        logger.info(f"   Hurst exponent: {regime.hurst_exponent:.4f}")
        logger.info(f"   Volatility: {regime.volatility:.4f}")
        logger.info(f"   Regime confidence: {regime.regime_score:.0%}")
        logger.info(f"   Suggested strategy: {regime.suggested_strategy}")
        
        # Position sizing by regime
        multiplier = detector.get_position_sizing_multiplier()
        logger.info(f"\n📍 Position Sizing:")
        logger.info(f"   Base position size: 1000 shares")
        logger.info(f"   Regime multiplier: {multiplier:.2f}")
        logger.info(f"   Actual position: {1000 * multiplier:.0f} shares")
        
        # Signal adaptation
        logger.info(f"\n🔄 Signal Adaptation by Regime:")
        base_signals = [0.7, -0.5, 0.3]
        
        for base_signal in base_signals:
            adjusted = adaptive_signal_by_regime(base_signal, prices, returns)
            change = ((adjusted - base_signal) / base_signal * 100) if base_signal != 0 else 0
            direction = "↑ Strengthened" if adjusted > base_signal else "↓ Weakened"
            logger.info(f"   Base {base_signal:+.1f} → Adjusted {adjusted:+.1f} ({change:+.0f}%) {direction}")
        
        logger.info("✓ Regime detection complete\n")
        
    except Exception as e:
        logger.error(f"✗ Regime detection example failed: {e}")


def demonstrate_integrated_trading_signal():
    """
    Example 4: Combine all three modules for final trading signal.
    
    Shows the complete flow:
    1. Generate base ML signal (PCA+RF)
    2. Add O-U mean-reversion signal
    3. Adapt by market regime
    4. Apply risk limits
    5. Execute with regime-aware sizing
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Complete Integrated Trading Signal")
    logger.info("="*70)
    
    try:
        from src.analysis.regime_detection import RegimeDetector, MarketRegime
        from src.models.kalman_ou import KalmanOUFilter
        
        np.random.seed(42)
        
        # Simulate price and signal data
        prices = np.exp(np.cumsum(np.random.normal(0, 0.02, 300)))
        returns = np.diff(np.log(prices))
        
        # Base ML signal (from PCA+RF model)
        base_ml_signal = np.random.uniform(-1, 1)
        
        # O-U mean-reversion signal
        kou = KalmanOUFilter(lookback=200)
        kou.fit(prices[:200])
        ou_signal, ou_diag = kou.predict(prices[200])
        
        # Detect regime
        detector = RegimeDetector(prices, returns)
        regime = detector.get_regime()
        
        logger.info(f"\n📊 Signal Combination Pipeline:")
        logger.info(f"\n   Step 1: Base ML Signal (PCA+RF)")
        logger.info(f"      Signal: {base_ml_signal:+.3f}")
        logger.info(f"      Interpretation: {'BUY' if base_ml_signal > 0 else 'SELL'}")
        
        logger.info(f"\n   Step 2: Kalman O-U Mean-Reversion Signal")
        logger.info(f"      Signal: {ou_signal:+.3f}")
        logger.info(f"      Z-score: {ou_diag['z_score']:.2f}")
        logger.info(f"      Distance from equilibrium: ${ou_diag['spread']:.2f}")
        
        logger.info(f"\n   Step 3: Market Regime Detection")
        logger.info(f"      Regime: {regime.state.value}")
        logger.info(f"      Hurst: {regime.hurst_exponent:.3f}")
        logger.info(f"      Confidence: {regime.regime_score:.0%}")
        
        # Weighted combination based on regime
        logger.info(f"\n   Step 4: Combine Signals by Regime")
        
        if regime.state == MarketRegime.TRENDING_LOW_VOL:
            # Trust ML signal more
            combined = 0.75 * base_ml_signal + 0.25 * ou_signal
            logger.info(f"      Weights: 75% ML, 25% O-U (trending regime)")
        
        elif regime.state == MarketRegime.MEAN_REVERT_LOW_VOL:
            # Use O-U, reverse ML signal
            combined = 0.25 * base_ml_signal - 0.75 * ou_signal
            logger.info(f"      Weights: 25% ML, 75% reversed O-U (mean-revert)")
        
        else:
            # Balanced in uncertain conditions
            combined = 0.5 * base_ml_signal + 0.5 * ou_signal
            logger.info(f"      Weights: 50% ML, 50% O-U (neutral/uncertain)")
        
        logger.info(f"      Combined signal: {combined:+.3f}")
        
        # Apply regime confidence multiplier
        logger.info(f"\n   Step 5: Apply Regime Confidence")
        multiplier = detector.get_position_sizing_multiplier()
        final_signal = combined * multiplier
        
        logger.info(f"      Confidence multiplier: {multiplier:.2f}")
        logger.info(f"      Final signal: {final_signal:+.3f}")
        
        # Position sizing
        logger.info(f"\n   Step 6: Position Sizing")
        base_capital = 100000
        base_position = 1000
        
        risk_per_trade = 0.02  # 2% of capital
        actual_position = base_position * multiplier
        
        logger.info(f"      Base capital: ${base_capital:,}")
        logger.info(f"      Base position: {base_position} shares")
        logger.info(f"      Regime multiplier: {multiplier:.2f}x")
        logger.info(f"      Actual position: {actual_position:.0f} shares")
        logger.info(f"      Dollar risk: ${actual_position * 1.5:.0f} (assuming $1.50 stop)")
        logger.info(f"      % of capital risk: {(actual_position * 1.5) / base_capital:.2%}")
        
        # Final decision
        logger.info(f"\n   Step 7: Final Decision")
        
        action = "PASS" if abs(final_signal) > 0.3 else "SKIP"
        logger.info(f"      Signal threshold: 0.3")
        logger.info(f"      Final signal: {final_signal:+.3f}")
        logger.info(f"      Decision: {action} trade")
        
        if action == "PASS":
            side = "BUY" if final_signal > 0 else "SELL"
            logger.info(f"      Trade: {side} {actual_position:.0f} shares")
            logger.info(f"      Strategy: {regime.suggested_strategy}")
            logger.info(f"      Risk: VaR check needed before execution")
        
        logger.info("✓ Integrated signal generation complete\n")
        
    except Exception as e:
        logger.error(f"✗ Integrated example failed: {e}")


def main():
    """Run all demonstration examples."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    logger.info("\n" + "█"*70)
    logger.info("ADVANCED QUANTITATIVE TRADING: COMPLETE INTEGRATION GUIDE")
    logger.info("█"*70)
    logger.info("\nPhase 3: SHAP + Kalman O-U + Regime Detection")
    logger.info("All three systems working together for superior signals\n")
    
    # Run demonstrations
    demonstrate_shap_analysis()
    demonstrate_kalman_ou_meanreversion()
    demonstrate_regime_detection()
    demonstrate_integrated_trading_signal()
    
    logger.info("\n" + "█"*70)
    logger.info("✓ ALL DEMONSTRATIONS COMPLETE")
    logger.info("█"*70)
    logger.info("\nNext Steps:")
    logger.info("1. Run: python test_advanced_systems.py (validate modules)")
    logger.info("2. Read: PHASE3_ADVANCED_ANALYSIS.md (complete documentation)")
    logger.info("3. Integrate into: src/main.py and src/live_paper.py")
    logger.info("4. Backtest with regime-aware signal generation")
    logger.info("5. Monitor SHAP importance and regime switches in production\n")


if __name__ == "__main__":
    main()
