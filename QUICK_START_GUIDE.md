# 🚀 Quant News Trader - Complete Setup & Run Guide

## 📋 Overview

This guide provides **step-by-step instructions** to set up and run the Quant News Trader system on any Windows PC. The system includes automated model improvement that runs periodically without requiring 24/7 operation.

**⚠️ IMPORTANT**: Follow sections in order. One-off setup steps only need to be done once per machine. Daily operation steps are done each time you want to run the system.

---

## 🔧 PART 1: ONE-OFF ENVIRONMENT SETUP (Do This Only Once Per Machine)

### Step 1.1: Verify System Requirements
**Required Software:**
- Windows 10/11 (64-bit)
- Python 3.11+ (if not installed)
- PowerShell (comes with Windows)
- Windows Task Scheduler (built-in)

**Check Python Version:**
```cmd
python --version
```
If Python is not installed, download from: https://python.org/downloads/

### Step 1.2: Extract Project Files
1. Copy the entire `/Qant` folder to your desired location (e.g., `C:\Projects\Qant`)
2. Ensure all subfolders are present:
   ```
   Qant/
   ├── docs/           # Documentation
   ├── quant_news_trader/  # Main application
   ├── scripts/        # Automation scripts
   ├── venv/          # Python environment (will be created)
   └── README.md      # This file
   ```

### Step 1.3: Set Up Python Virtual Environment
**Navigate to project root:**
```cmd
cd C:\Path\To\Your\Qant
```

**Create virtual environment:**
```cmd
python -m venv venv
```

**Activate virtual environment:**
```cmd
venv\Scripts\activate
```

**Install dependencies:**
```cmd
cd quant_news_trader
pip install -r requirements.txt
```

**Verify installation:**
```cmd
python -c "import pandas, numpy, sklearn, shap; print('✅ All dependencies installed')"
```

### Step 1.4: Configure API Keys and Settings
**Edit configuration file:**
1. Open `quant_news_trader\configs\config.yaml`
2. Update the following settings:

```yaml
# Required: Update these with your actual values
market:
  ticker: "AAPL"  # Your preferred stock symbol

news:
  query: "Apple Inc"  # Company name for news search

broker:
   initial_cash: 1000000  # Default paper-trading starting capital

live:
   resume_state: true  # Keep cash/positions continuity after restart
   display_timezone: "Asia/Shanghai"  # UTC+8 display time (SG/Shanghai)
   use_moomoo: true  # Prefer moomoo snapshot quotes for fresher live prices

moomoo:
   host: "127.0.0.1"
   port: 11111
   market: "US"
   security_firm: "FUTUSECURITIES"

# Optional: Adjust risk settings
risk:
  max_var_loss_pct: 2.0  # Maximum daily loss limit
```

**Environment variables (recommended for better news coverage):**
1. Create `quant_news_trader\.env` (same folder as `requirements.txt`)
2. Add your NewsAPI key:

```env
NEWSAPI_KEY=YOUR_NEWSAPI_KEY_HERE
```

3. Optional: you can also add GNews key if you use it:

```env
GNEWS_API_KEY=YOUR_GNEWS_API_KEY_HERE
```

**News source priority used by the app:**
1. NewsAPI (if `NEWSAPI_KEY` is present)
2. GNews (if `GNEWS_API_KEY` is present)
3. Google News RSS fallback (no API key required)

This means the system still runs without any key, but adding `NEWSAPI_KEY` usually improves article count and relevance.

### Step 1.5: Test Initial Setup
**Run a quick test:**
```cmd
cd quant_news_trader
python -c "from src.config_loader import load_config; from pathlib import Path; cfg = load_config(Path('.')); print('✅ Configuration loaded successfully')"
```

**Run basic backtest:**
```cmd
python -m src.main
```
This should complete without errors and generate reports in `reports/` folder.

### Step 1.6: Set Up Automated Model Updates (Optional but Recommended)
**Configure Windows Task Scheduler:**

1. **Open Task Scheduler:**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create New Task:**
   - Click "Create Task..." in the right panel
   - Name: "Q"
   - Check "Run with highest privileges"

3. **Triggers Tab:**
   - Click "New..."
   - Begin the task: "On a schedule"
   - Settings: Daily, Start time: 6:00 AM (or your preferred time)
   - Recur every: 1 days
   - Click OK

4. **Actions Tab:**
   - Click "New..."
   - Action: "Start a program"
   - Program/script: `C:\Path\To\Your\Qant\scripts\run_model_updates.bat`
   - Start in: `C:\Path\To\Your\Qant`
   - Click OK

5. **Conditions Tab:**
   - Uncheck "Start the task only if the computer is on AC power" (if you want it to run on battery)
   - Check other options as preferred

6. **Settings Tab:**
   - Check "Run task as soon as possible after a scheduled start is missed"
   - Check "If the task fails, restart every: 1 hour, up to 3 times"
   - Click OK

**Test the scheduled task:**
- Right-click your new task → "Run"
- Check `quant_news_trader\logs\` for execution logs

---

## 🎯 PART 2: DAILY OPERATION (Do These Steps Every Time You Run)

### Step 2.1: Daily Backtesting & Analysis
**Navigate to project:**
```cmd
cd C:\Path\To\Your\Qant
venv\Scripts\activate
cd quant_news_trader
```

**Run full analysis:**
```cmd
python -m src.main
```

**What this does:**
- Fetches latest market data and news
- Runs backtest with current model
- Performs advanced analysis (SHAP, Hurst, Kalman-OU)
- Generates reports in `reports/` folder
- Updates dashboard with key metrics
- Saves key variables to `reports/key_variables_report.txt`

### Step 2.2: Check Results
**View results:**
- Open `reports/key_variables_report.txt` for key metrics
- Check `reports/model_update_*.json` for latest analysis
- Review console output for performance summary

**Expected output includes:**
```
=== Backtest summary ===
Final equity: $110581.14
VaR 95%: -0.0002
ES 95%: -0.0009
Current Market Regime: Trending High Vol
```

### Step 2.3: Live Paper Trading (Optional)
**Run paper trading simulation:**
```cmd
python src/live_paper.py
```

**What this does:**
- Simulates live trading with paper money
- Uses approved universe from `data/approved_universe.csv`
- Logs trades to `logs/live_log_multi_nn.csv`
- Writes trade-only transaction ledger to `logs/transactions_live_multi_nn.csv`
- Writes current portfolio snapshot (continuously updated) to `logs/portfolio_snapshot_live_nn.json`
- Restores cash/positions from prior log on restart when `live.resume_state: true`
- Uses UTC+8 display timestamps by default via `live.display_timezone: "Asia/Shanghai"`
- Uses moomoo snapshot quotes when `live.use_moomoo: true` (with Yahoo fallback)
- Logs stale bar warnings if latest market bar timestamp is too old for the configured interval
- Can run continuously or for specific periods

**Reset continuity (start fresh at initial cash):**
- Stop the live process
- Delete `quant_news_trader\logs\live_log_multi_nn.csv`
- Start `python src/live_paper.py` again

### Step 2.4: Schedule Main 30 Minutes Before Market Open
If you want Windows to start the full system automatically each trading day, use the included launcher:

```cmd
scripts\run_main.bat
```

**Recommended Task Scheduler settings for US equities:**
1. Open Task Scheduler and click **Create Task...**
2. Name it something like `Qant Main Pre-Market`
3. In **General**:
   - Check **Run whether user is logged on or not**
   - Check **Run with highest privileges**
4. In **Triggers**:
   - Click **New...**
   - Begin the task: **On a schedule**
   - Set **Daily**
   - Set start time to **9:00:00 AM** if you want it 30 minutes before the regular US market open at 9:30 AM Eastern
   - Enable the trigger
5. In **Actions**:
   - Click **New...**
   - Action: **Start a program**
   - Program/script: `C:\Path\To\Your\Qant\scripts\run_main.bat`
   - Start in: `C:\Path\To\Your\Qant`
6. In **Conditions**:
   - Uncheck power-related restrictions if you want it to run on battery
7. In **Settings**:
   - Check **Run task as soon as possible after a scheduled start is missed**
   - Check **If the task fails, restart every** and choose something like `5 minutes`
   - Set retry attempts to `3`

**Logs for scheduled main runs:**
- The launcher writes timestamped logs into `quant_news_trader\logs\main_scheduler_YYYYMMDD_HHMMSS.log`

**If you meant repeat every 30 minutes instead of start 30 minutes before open:**
- In the trigger, enable **Repeat task every:** `30 minutes`
- Set **for a duration of:** `8 hours`
- Keep the same 9:00 AM start time

---

## 🤖 PART 3: AUTOMATED SELF-IMPROVEMENT SYSTEM

### How It Works (No 24/7 Operation Required!)

**❌ NOT 24/7**: The system does NOT need to run continuously. Model updates happen automatically via Windows Task Scheduler.

**✅ Scheduled Updates**: The system improves itself through periodic updates:
- **Every 3 days**: Micro updates (small improvements)
- **Every 7 days**: Macro updates (full retraining)
- **Every 30 days**: Major updates (hyperparameter tuning)
- **Every 365 days**: Epoch updates (complete overhaul)

**🔄 Process Flow:**
1. **Daily**: You run backtests manually to see current performance
2. **Scheduled**: Windows Task Scheduler runs update checks automatically
3. **Conditional**: Updates only run when due (based on last update time)
4. **Automatic**: New improved models are saved and become active

### Monitoring Self-Improvement

**Check update status:**
```cmd
# View update history
type quant_news_trader\logs\model_update_history.jsonl

# View latest reports
dir quant_news_trader\reports\model_update_*.json

# Check scheduler logs
type quant_news_trader\logs\model_update_scheduler_*.log
```

**Update Types Explained:**
- **Micro**: Quick improvements with new data
- **Macro**: Full model retraining with extended data
- **Major**: Advanced optimization (takes longer)
- **Epoch**: Complete model architecture review (rare)

---

## 🛠️ PART 4: TROUBLESHOOTING & MAINTENANCE

### Common Issues

**Issue: "Module not found" errors**
```
Solution: Ensure virtual environment is activated
venv\Scripts\activate
pip install -r requirements.txt
```

**Issue: "Config file not found"**
```
Solution: Check file paths in quant_news_trader\configs\config.yaml
Ensure you're running from correct directory
```

**Issue: Model updates not running**
```
Solution: Check Windows Task Scheduler
- Task should be enabled
- Check "Last Run Result" in Task Scheduler
- Review logs in quant_news_trader\logs\
```

**Issue: Poor backtest performance**
```
Solution: Check data quality and market conditions
Review config.yaml settings
Run: python -m src.model_update --update-type macro
```

### Maintenance Tasks

**Monthly:**
- Review `reports/` folder for performance trends
- Check `logs/model_update_history.jsonl` for update success
- Clean old log files if disk space is concern

**Quarterly:**
- Update Python dependencies: `pip install -r requirements.txt --upgrade`
- Review and adjust risk parameters in `configs/config.yaml`

**Annually:**
- Consider major configuration changes
- Review overall system performance
- Update approved universe in `data/approved_universe.csv`

---

## 📁 FILE ORGANIZATION REFERENCE

After setup, your folder structure should be:
```
Qant/
├── docs/                          # Documentation
│   ├── *.md files
├── quant_news_trader/             # Main application
│   ├── configs/                   # Configuration files
│   ├── data/                      # Static data files
│   ├── examples/                  # Example scripts
│   ├── logs/                      # Log files (created during runs)
│   ├── models/                    # ML models (created during runs)
│   ├── notebooks/                 # Jupyter notebooks
│   ├── reports/                   # Analysis reports (created during runs)
│   ├── src/                       # Source code
│   ├── tests/                     # Test files
│   │   ├── unit/                  # Fast unit/smoke checks
│   │   ├── integration/           # Cross-module integration checks
│   │   └── system/                # End-to-end and subsystem validation
│   ├── requirements.txt
│   └── .env.example
├── scripts/                       # Automation scripts
│   ├── schedule_model_updates.ps1
│   ├── run_model_updates.bat
│   └── run_main.bat
├── venv/                          # Python environment
└── README.md                      # This file
```

---

## 🚦 QUICK REFERENCE

### One-Off Setup (Do Once)
```cmd
cd C:\Path\To\Qant
python -m venv venv
venv\Scripts\activate
cd quant_news_trader
pip install -r requirements.txt
# Configure config.yaml
# Set up Windows Task Scheduler
```

### Daily Operation (Do Every Time)
```cmd
cd C:\Path\To\Qant
venv\Scripts\activate
cd quant_news_trader
python -m src.main
# Check reports/ folder for results
```

### Self-Improvement
- **Automatic**: Windows Task Scheduler handles updates
- **No 24/7 required**: Updates run on schedule
- **Monitoring**: Check `logs/` and `reports/` folders

---

## 📞 SUPPORT

If you encounter issues:
1. Check the troubleshooting section above
2. Review log files in `quant_news_trader\logs\`
3. Ensure all dependencies are installed
4. Verify configuration files are correct

**Success Indicators:**
- Backtest completes without errors
- Files created in `reports/`, `models/`, `logs/` folders
- Key variables report shows reasonable values
- Model updates run successfully via Task Scheduler

---

**🎉 You're all set! The system will now run daily and continuously improve itself automatically.**