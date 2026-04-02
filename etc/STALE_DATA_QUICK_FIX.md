# Quick Reference: Stale Market Data Fix

## The Problem
```
Stale market bar for CRWD: age=1695.5s source=yahoo_bar interval=60s
```

Bot was running at **2 AM UTC** (after market close) trying to fetch fresh bars that don't exist.

## The Solution

Three configuration options in `configs/config.yaml`:

### Default (Recommended): Wait for Market to Open
```yaml
live:
  wait_for_market_open: true  # ← Current default
```
**Result**: Bot waits until 9:30 AM ET automatically

### Option 2: Start Now, Tolerate Stale Data  
```yaml
live:
  wait_for_market_open: false
  stale_bar_threshold_afterhours: 3600  # ← Use 1-hour threshold instead of 3-min
```
**Result**: Bot starts immediately, fewer warnings

### Option 3: Strict Market-Hours Only
```yaml
live:
  wait_for_market_open: true
  skip_afterhours: true  # TODO: needs implementation
```
**Result**: Bot stops trading at 4 PM ET

## Testing

Check market status anytime:
```python
from src.utils.market_hours import get_us_market_status, is_us_market_open
print(get_us_market_status())  # Shows "OPEN (closes in X min)" or "CLOSED"
print(is_us_market_open())     # Boolean
```

## Run Commands

```bash
# Start with default (wait for market)
python -m src.live_paper

# Start aggressive config (waits for market)  
python -m src.live_paper --profile=aggro

# Override to start now (testing only)
# Edit config.yaml: wait_for_market_open: false
python -m src.live_paper
```

## Files Changed

✅ Created: `src/utils/market_hours.py` - Market hours detection

✅ Updated: `src/live_paper.py` - Market hours integration

✅ Updated: All configs - Added market hours settings

## Key Timestamps

- **Pre-Market**: 4:00 AM - 9:30 AM ET
- **Regular Hours**: 9:30 AM - 4:00 PM ET ← Target for trading
- **After-Hours**: 4:00 PM - 8:00 PM ET  
- **Closed**: 8:00 PM - 4:00 AM ET (next day)
- **Weekends**: Saturday-Sunday (always closed)

## Logging Examples

### Market Open
```
INFO - US Stock Market Status: OPEN (closes in 120 min)
INFO - Starting live NN multi-ticker paper-trading loop...
```

### Market Closed  
```
WARNING - US Stock Market Status: CLOSED (opens in 420 min)
INFO - Market is closed. Waiting 420 minutes for market to open...
INFO - Market is now open! Starting live trading.
```

---

**TLDR**: Your bot now automatically waits for market hours before trading. No more stale data warnings during market close! 🎉
