# Advanced Explainability & Regime Detection Integration
## Phase 3: Interpretability + Mean-Reversion + Regime Adaptation

### Summary

This phase adds three advanced quantitative modules (without L2 data requirement):

1. **SHAP Explainability** - Understand what drives model predictions
2. **Kalman-Filtered O-U Mean-Reversion** - Identify arbitrage opportunities
3. **Hurst Exponent Regime Detection** - Adapt strategy to market conditions

---

## Module 1: SHAP Explainability Analysis

### Purpose
Explains PCA+RF model decisions through SHAP decomposition. Answers:
- Which factors drive buy/sell signals?
- How do negative vs positive news impact volatility?
- Which factor interactions matter most?

### Key Classes

**`SHAPAnalyzer`** - Main class
```python
from src.explainability.shap_analyzer import SHAPAnalyzer

# Initialize
analyzer = SHAPAnalyzer(model, feature_names=['ema_10', 'rsi_14', ...])
analyzer.fit(X_train, max_samples=100)  # Background data for SHAP

# Get importance ranking
importance = analyzer.get_feature_importance(X_test)
#   Returns: DataFrame (feature, importance, rank)

# Analyze sentiment impact
sentiment_impact = analyzer.analyze_sentiment_impact(X_test, y_test)
#   Returns: Dict with negative_impact, positive_impact, elasticity
```

### Use Cases

**1. Feature importance ranking** (which factors matter?)
```python
importance = analyzer.get_feature_importance(X_test)
print(importance.head())
# Output:
#          feature  importance  rank
# 0  daily_sentiment  0.0342     1
# 1      ema_50      0.0289     2
# 2      rsi_14      0.0156     3
```

**2. Sentiment analysis** (negative vs positive news volatility)
```python
impact = analyzer.analyze_sentiment_impact(X_test, y_test, 
                                          sentiment_features=['daily_sentiment_ema_5'])

# Output: {
#   'daily_sentiment_ema_5_atr_14': {
#     'negative_impact': 0.012,      # Volatility when sentiment < 0
#     'positive_impact': 0.008,      # Volatility when sentiment > 0
#     'elasticity': -0.33            # Positive sentiment reduces vol impact
#   }
# }
```

**3. Explain individual predictions**
```python
explanation = analyzer.explain_prediction(x_test)
# Shows which features caused specific buy/sell decision
```

**4. Compare model behaviors**
```python
comparison = compare_model_behaviors(X_train, X_test, 
                                     model_old=nn_model,
                                     model_new=pca_rf_model,
                                     feature_names=...)
# Shows which factors stabilized/destabilized after model upgrade
```

### Integration into Main Pipeline

```python
# In src/main.py, after training PCA+RF model:

from src.explainability.shap_analyzer import SHAPAnalyzer

# Create and fit analyzer
analyzer = SHAPAnalyzer(model, feature_names=feature_builder.FEATURE_NAMES)
analyzer.fit(X_train[:100], max_samples=50)

# Save plots
analyzer.plot_summary(X_test, output_path='shap_summary.png')

# Analyze sentiment
sentiment_report = analyzer.analyze_sentiment_impact(X_test, y_test)
logger.info(f"Sentiment impact analysis:\n{sentiment_report}")
```

---

## Module 2: Kalman-Filtered Ornstein-Uhlenbeck Mean-Reversion

### Purpose
Identifies mean-reversion opportunities using Kalman filtering + O-U process.

**O-U Process Model**: `dx = θ(μ - x)dt + σ dW`
- **θ** (theta): Mean-reversion speed (days to recover 50%)
- **μ** (mu): Long-term equilibrium
- **σ** (sigma): Volatility

### Key Classes

**`KalmanOUFilter`** - Single asset mean-reversion
```python
from src.models.kalman_ou import KalmanOUFilter

# Initialize
kou = KalmanOUFilter(lookback=252, z_threshold=2.0)

# Fit on historical prices
kou.fit(prices_history)

# Generate signal on current price
signal, diagnostics = kou.predict(current_price)

# signal: -1.0 (strong sell) to +1.0 (strong buy)
# diagnostics: {'equilibrium', 'spread', 'z_score', 'half_life_reversion', ...}
```

**`PairsTrader`** - Statistical arbitrage (two correlated stocks)
```python
from src.models.kalman_ou import PairsTrader

# Initialize
pairs = PairsTrader('AAPL', 'MSFT', lookback=252)

# Fit hedge ratio
hedge_ratio = pairs.fit(prices_aapl, prices_msft)
# Returns β where: MSFT = β * AAPL + α + ε

# Get spread signal
hedge_info = pairs.get_spread(price_a=150, price_b=300)

# Get position sizing
position = pairs.get_position_size(150, 300, capital=100000)
# Returns: {'quantity_a': 500, 'quantity_b': -450, 'signal': 0.7}
```

### Use Cases

**1. Detect mean-reversion opportunities on single stock**
```python
kou = KalmanOUFilter(lookback=252)
kou.fit(prices[-252:])

signal, diag = kou.predict(current_price)

if signal > 0.5:  # Strong mean-reversion buy signal
    position_size = capital * 0.02 / (diag['ou_std'] * current_price)
    broker.buy(ticker, position_size)
elif signal < -0.5:  # Strong mean-reversion sell signal
    broker.sell(ticker, position_size)
```

**2. Pairs trading (delta-neutral arbitrage)**
```python
pairs = PairsTrader('STOCK_A', 'STOCK_B')
pairs.fit(prices_a[:-100], prices_b[:-100])

position = pairs.get_position_size(prices_a[-1], prices_b[-1], 
                                   notional_capital=100000)

if abs(position['signal']) > 0.3:
    # Buy A, short B (delta hedged)
    broker.buy('STOCK_A', position['quantity_a'])
    broker.short('STOCK_B', abs(position['quantity_b']))
```

**3. Mean-reversion half-life**
```python
half_life = kou.half_life_reversion()
# Days for price to revert halfway to equilibrium
# Use to set exit targets: exit at ±2σ or after half_life periods
```

### Integration into Main Pipeline

```python
# In src/main.py, add O-U signals:

from src.models.kalman_ou import portfolio_ou_filter

# After generating PCA+RF signals:
ou_signals = portfolio_ou_filter(price_df, lookback=252)

# Combine signals
combined_signal = 0.6 * pca_signal + 0.4 * ou_signal

# Execute trade if signals align
if combined_signal > 0.7:
    broker.buy(...)
```

---

## Module 3: Hurst Exponent & Regime Detection

### Purpose
Identifies market regime (trending vs mean-reverting) and adapts strategy.

**Hurst Exponent H:**
- **H > 0.55**: Trending (momentum works)
- **0.45 < H < 0.55**: Random walk (no alpha)
- **H < 0.45**: Mean-reverting (pairs/reversion works)

### Key Classes

**`HurstAnalyzer`** - Calculate Hurst exponent
```python
from src.analysis.regime_detection import HurstAnalyzer

# Initialize
hurst = HurstAnalyzer(lookback=252)

# Calculate for price series
h_exp = hurst.calculate(prices)  # Returns 0-1

# Rolling calculation
h_rolling = hurst.calculate_windowed(prices, window=252)
```

**`RegimeDetector`** - Full regime classification
```python
from src.analysis.regime_detection import RegimeDetector, MarketRegime

# Initialize
detector = RegimeDetector(prices, returns, lookback=252)

# Get current regime
regime = detector.get_regime()

# regime contains:
# - state: MarketRegime.TRENDING_LOW_VOL (enum)
# - hurst_exponent: 0.623 (float)
# - volatility: 0.0156 (float)
# - regime_score: 0.80 (confidence 0-1)
# - suggested_strategy: "MOMENTUM: Follow trend, wide stops"
```

### 5 Market Regimes

| Regime | Hurst | Vol | Strategy | Position Multiplier |
|--------|-------|-----|----------|-------------------|
| Trending Low Vol | >0.55 | low | Momentum: follow trend | 1.0 |
| Trending High Vol | >0.55 | high | Momentum with caution | 0.6 |
| Mean-Revert Low | <0.45 | low | Pairs/reversion: comfortable | 0.85 |
| Mean-Revert High | <0.45 | high | Reversion with risk check | 0.5 |
| Random Walk | 0.45-0.55 | any | AVOID: no alpha | 0.3 |

### Use Cases

**1. Adapt signals by regime**
```python
from src.analysis.regime_detection import adaptive_signal_by_regime

detector = RegimeDetector(prices, returns)
base_signal = 0.8  # Strong buy from ML model

# Adjust by regime
adjusted_signal = adaptive_signal_by_regime(base_signal, prices, returns)

# In trending low-vol: signal might become 0.96 (stronger)
# In random walk: signal might become 0.24 (weaker)
# In mean-revert high-vol: signal might become 0.40 (cautious)
```

**2. Position sizing by regime confidence**
```python
regime = detector.get_regime()
multiplier = detector.get_position_sizing_multiplier()  # 0-1

base_position = 1000
actual_position = base_position * multiplier

# In high-confidence regime: use full size
# In random walk: use 30% size only
```

**3. Different signals for different regimes**
```python
regime = detector.get_regime()

if regime.state == MarketRegime.TRENDING_LOW_VOL:
    # Use momentum strategy
    signal = pca_signal  # Follow ML model
elif regime.state == MarketRegime.MEAN_REVERT_LOW_VOL:
    # Reverse signal (mean-reversion)
    signal = -pca_signal
else:
    # Reduce signal in uncertain regimes
    signal = pca_signal * 0.5
```

### Integration into Main Pipeline

```python
# In src/main.py, add regime adaptation:

from src.analysis.regime_detection import RegimeDetector

detector = RegimeDetector(prices, returns)
regime = detector.get_regime()

logger.info(f"Market regime: {regime.state.value}")
logger.info(f"Strategy: {regime.suggested_strategy}")

# Adjust PCA signal by regime
signal = adaptive_signal_by_regime(pca_signal, prices, returns)

# Execute with regime-aware position sizing
if abs(signal) > 0.5:
    position_size = base_size * detector.get_position_sizing_multiplier()
    broker.execute_trade(signal, position_size)
```

---

## Full Integration Example

```python
# src/main.py: Integrate all three systems

from src.explainability.shap_analyzer import SHAPAnalyzer
from src.models.kalman_ou import KalmanOUFilter, portfolio_ou_filter
from src.analysis.regime_detection import RegimeDetector, adaptive_signal_by_regime

# === STEP 1: Train PCA+RF and explain ===
pca_model = train_pca_direction_model(X_train, y_train)
analyzer = SHAPAnalyzer(pca_model, feature_names)
analyzer.fit(X_train)
importance = analyzer.get_feature_importance(X_test)
logger.info(f"Top features: {importance.head()}")

# === STEP 2: Generate signals (PCA + O-U) ===
pca_signals = model.predict_proba_pca(X_test)[:, 1] * 2 - 1  # -1 to 1
ou_signals = portfolio_ou_filter(price_df)

# === STEP 3: Detect regime and adapt ===
detector = RegimeDetector(prices[-252:], returns[-252:])
regime = detector.get_regime()

# Combine signals
if regime.state == MarketRegime.TRENDING_LOW_VOL:
    combined = 0.8 * pca_signals + 0.2 * ou_signals
elif regime.state == MarketRegime.MEAN_REVERT_LOW_VOL:
    combined = 0.2 * pca_signals + 0.8 * (-ou_signals)  # Reverse O-U
else:
    combined = 0.5 * pca_signals + 0.5 * ou_signals

# Adjust by regime confidence
final_signal = adaptive_signal_by_regime(combined, prices, returns)

# === STEP 4: Execute with risk management ===
if abs(final_signal) > 0.5:
    position_multiplier = detector.get_position_sizing_multiplier()
    position_size = base_size * position_multiplier
    
    # Add VaR check (from Phase 2)
    rm = RiskManager(capital)
    is_allowed, var_loss, _ = rm.compute_var_limit(current_price, position_size, vol)
    
    if is_allowed and position_size > 0:
        broker.execute_trade(final_signal, position_size)
        logger.info(f"Signal: {final_signal:.2f}, Regime: {regime.state.value}, " +
                   f"Size: {position_size:.0f}, VaR: ${var_loss:.0f}")
```

---

## Key Metrics & Diagnostics

### SHAP Analysis Output
```
Feature Importance:
  1. daily_sentiment_ema_5: 0.0342 (11% of decisions)
  2. ema_50:                0.0289 (9%)
  3. rsi_14:               0.0156 (5%)

Sentiment Impact (Volatility):
  - Negative news:  +1.2x impact
  - Positive news:  -0.8x impact (dampens vol)
  - Elasticity:     -33% (aggressive dampening)
```

### O-U Mean-Reversion Output
```
O-U Parameters:
  θ (reversion speed): 0.0051 (half-life: 136 days)
  μ (equilibrium):     $102.34
  σ (volatility):      $2.15

Current Signal:
  Price:           $105.62
  Spread:          +$3.28 (expensive)
  Z-score:         +1.53 (mild sell)
  Signal:          -0.50 (reduce long position)
```

### Regime Detection Output
```
Market Regime: Trending Low Vol
  Hurst Exponent: 0.628 (strong trending)
  Volatility:     0.0108 (quiet market)
  Regime Score:   0.80 (high confidence)
  
Position Sizing Multiplier: 0.80
Suggested Strategy: "MOMENTUM: Follow trend, wide stops"

Signal Adjustment:
  Base signal:     +0.70
  After regime adjust: +0.74 (strengthened)
```

---

## File Structure

```
src/
├── explainability/
│   ├── __init__.py
│   └── shap_analyzer.py (340 lines) ← SHAP explanations
├── analysis/
│   ├── __init__.py
│   └── regime_detection.py (500 lines) ← Hurst + regime
├── models/
│   └── kalman_ou.py (480 lines) ← O-U mean-reversion
├── config.yaml (updated)
```

---

## Testing & Validation

All modules tested and passing:
```
✓ SHAP Explainability: Feature importance, sentiment analysis working
✓ Kalman O-U: Single-asset and pairs trading signals generated
✓ Regime Detection: 5 regimes classified, signals adjusted
✓ Full Integration: All three systems working together
```

Run tests:
```bash
python test_advanced_systems.py
```

---

## Next Steps

1. **Integrate into live trading** (src/live_paper.py)
   - Add SHAP explanation reports
   - Use O-U signals for mean-reversion pairs
   - Adapt signals by regime

2. **Backtest with regime adaptation** (src/main.py)
   - Compare performance: no regime vs regime-aware
   - Track Sharpe ratio by regime
   - Analyze factor interactions

3. **Monitor production signals**
   - Log SHAP importance daily
   - Track regime switches
   - Alert on unusual factor interactions

4. **Extend to multi-asset portfolios**
   - Portfolio-level O-U (spread baskets)
   - Cross-asset sentiment analysis
   - Regime consensus (majority voting)

---

## Configuration (config.yaml)

```yaml
model:
  # ... existing PCA config ...
  
explainability:
  enable_shap: true
  shap_background_samples: 100
  shap_plot_max_features: 15
  
mean_reversion:
  enable_ou_filter: true
  ou_lookback: 252
  ou_z_threshold: 2.0  # Z-score for signal
  
regime:
  enable_detection: true
  hurst_lookback: 252
  hurst_lags: 50
  regime_vol_window: 20
```

---

**PHASE 3 COMPLETE**: Explainability + Advanced Quant Models 🚀
