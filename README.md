# Quant News Trader

Advanced quantitative trading system with news sentiment analysis, machine learning models, and risk management.

## Project Structure

```
quant_news_trader/
├── configs/           # Configuration files (YAML)
├── data/             # Static data files (CSV)
├── examples/         # Example scripts and demonstrations
├── logs/             # Log files and trading records
├── models/           # Pre-trained ML models and metadata
├── notebooks/        # Jupyter notebooks for analysis
├── reports/          # Generated reports and outputs
├── src/              # Source code
│   ├── data/         # Data loading and processing
│   ├── explainability/  # SHAP analysis and model interpretation
│   ├── features/     # Feature engineering
│   ├── models/       # ML models and backtesting
│   ├── trading/      # Broker interfaces and execution
│   └── utils/        # Utilities and logging
├── tests/            # Test suites
│   ├── unit/         # Fast unit/smoke checks
│   ├── integration/  # Cross-module integration checks
│   └── system/       # End-to-end and subsystem validation
├── requirements.txt  # Python dependencies
└── .env.example      # Environment variables template

docs/                 # Documentation and guides
scripts/              # Setup and utility scripts
venv/                 # Python virtual environment
```

## Quick Start

1. See `docs/QUICK_START.md` for setup instructions
2. See `docs/SETUP_AND_RUN.md` for detailed configuration
3. Run `python -m src.main` for backtesting with dashboard
4. Run `python src/live_paper.py` for paper trading simulation

## Key Features

- **News Sentiment Analysis**: Real-time news processing with sentiment scoring
- **ML Models**: PCA + Random Forest for price prediction
- **Risk Management**: VaR, Expected Shortfall, position limits
- **Advanced Analysis**: SHAP explainability, Hurst regime detection, Kalman-OU mean-reversion
- **Live Trading**: Integration with Moomoo broker API
- **Backtesting**: Comprehensive performance analysis with visualizations
- **Automated Updates**: Forward-rolling model improvement system

## Automated Model Updates

The system includes intelligent model updates that run automatically:

- **Micro Updates** (every 3 days): Incremental learning on new data
- **Macro Updates** (every 7 days): Full model retraining
- **Major Updates** (every 30 days): Hyperparameter optimization
- **Epoch Updates** (every 365 days): Complete model overhaul

See `docs/MODEL_UPDATE_SYSTEM.md` for detailed information on configuring and monitoring automated updates.

## Configuration

- `configs/config.yaml`: Main configuration
- `configs/config_aggro.yaml`: Aggressive risk settings
- `configs/config_conservative.yaml`: Conservative risk settings
- `configs/config_balanced.yaml`: Balanced risk settings

## Testing

Run tests with:
```bash
python -m pytest tests/
```

Or run individual test files:
```bash
python tests/system/test_advanced_systems.py
```