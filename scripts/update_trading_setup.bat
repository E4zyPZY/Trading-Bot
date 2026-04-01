@echo off
REM Automated Trading Setup Update Script
REM This script updates the trading stocks and aggressiveness mode based on current market conditions

cd /d %~dp0
cd ..

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Change to quant_news_trader directory
cd quant_news_trader

REM Run the dynamic config test
python -c "
from src.config_loader import load_config
from pathlib import Path
cfg = load_config(Path('.'))
print('=== Automated Trading Setup Update ===')
print('Selected Stocks:', cfg['market']['tickers'])
print('Market Profile: Determined dynamically based on current regime')
print('Update completed successfully!')
"

pause