#!/usr/bin/env python
"""
Diagnostic script for market hours detection and live paper trading setup.

Usage:
    python etc/diagnose_market_hours.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root/src to path.
project_root = Path(__file__).resolve().parents[1] / "quant_news_trader"
sys.path.insert(0, str(project_root))

try:
    from src.utils.market_hours import (
        is_us_market_open,
        is_us_market_premarket, 
        is_us_market_afterhours,
        get_us_market_status,
        minutes_until_market_open,
        minutes_until_market_close,
        get_current_et_time,
        get_current_et_date,
    )
    print("✅ market_hours module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import market_hours: {e}")
    sys.exit(1)

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    print_section("MARKET HOURS DIAGNOSTICS")
    
    # System time
    print_section("System Time Information")
    now_utc = datetime.now(timezone.utc)
    now_et = get_current_et_time()
    
    print(f"Current UTC Time:    {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current ET Time:     {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current ET Date:     {get_current_et_date()}")
    
    # Market status
    print_section("Market Status")
    print(f"Market Status String: {get_us_market_status()}")
    print(f"Is Market Open?:      {is_us_market_open()}")
    print(f"Is Pre-Market?:       {is_us_market_premarket()}")
    print(f"Is After-Hours?:      {is_us_market_afterhours()}")
    
    # Time calculations
    print_section("Market Time Calculations")
    mins_to_open = minutes_until_market_open()
    mins_to_close = minutes_until_market_close()
    
    print(f"Minutes Until Open:  {mins_to_open:.1f}")
    print(f"Minutes Until Close: {mins_to_close:.1f}")
    
    if mins_to_open > 0:
        hours = int(mins_to_open) // 60
        mins = int(mins_to_open) % 60
        print(f"  → Market opens in {hours}h {mins}m")
    
    if mins_to_close > 0:
        hours = int(mins_to_close) // 60
        mins = int(mins_to_close) % 60
        print(f"  → Market closes in {hours}h {mins}m")
    
    # Market hours reference
    print_section("US Stock Market Hours (ET - Eastern Time)")
    print("Pre-Market:     04:00 - 09:30 ET")
    print("Regular Hours:  09:30 - 16:00 ET")
    print("After-Hours:    16:00 - 20:00 ET")
    print("Closed:         20:00 - 04:00 ET")
    print("Closed:         Weekends (Sat-Sun)")
    print("Closed:         US Federal Holidays")
    
    # Recommendation
    print_section("Recommendation")
    
    if is_us_market_open():
        print("✅ Market is OPEN - Good time for live trading!")
        print(f"   You have {mins_to_close:.0f} minutes of trading time remaining.")
    else:
        if is_us_market_premarket():
            print("⏳ Pre-Market Hours")
            print(f"   Regular market opens in {mins_to_open:.0f} minutes")
            print("   Recommendation: Consider using strategy for pre-market trading")
        elif is_us_market_afterhours():
            print("⏳ After-Hours Trading")
            print(f"   Next regular session opens in {mins_to_open:.0f} minutes")
            print("   Recommendation: Use limited after-hours strategies with broker support")
        else:
            print("❌ Market is CLOSED (overnight/weekend)")
            print(f"   Market opens in {mins_to_open:.0f} minutes")
            print("   Recommendation: Wait for market open or test with historical data")
    
    # Live trading config recommendation
    print_section("Recommended Config Settings")
    if not is_us_market_open():
        print("Since market is not open, your config should have:")
        print("  wait_for_market_open: true")
        print("  (This will make bot wait until 9:30 AM ET automatically)")
    else:
        print("Market is currently open, so 'wait_for_market_open' won't trigger.")
        print("Current config value doesn't matter right now.")
    
    print_section("Next Steps")
    print("1. If market is closed: Your bot will wait for 9:30 AM ET to start")
    print("2. If market is open:  Your bot will start immediately")
    print("3. Stale data warnings: Adaptive thresholds based on market hours")
    print("   - Market open:  180 seconds (3 min)")
    print("   - After-hours:  3600 seconds (1 hour)")

if __name__ == "__main__":
    main()
