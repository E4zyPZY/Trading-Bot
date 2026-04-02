"""
Market hours and timezone utilities for US stock market.

US Stock Market Hours (all times Eastern):
- Regular: Monday-Friday, 9:30 AM - 4:00 PM ET
- Pre-market: 4:00 AM - 9:30 AM ET  
- After-hours: 4:00 PM - 8:00 PM ET

No external dependencies - uses only stdlib datetime/timezone.
"""

from datetime import datetime, time as dt_time, timedelta, timezone


def _is_dst(date_obj):
    """
    Check if date is in US Eastern Daylight Time.
    DST: 2nd Sunday in March to 1st Sunday in November
    """
    year = date_obj.year
    
    # 2nd Sunday in March (start of DST)
    march_1 = datetime(year, 3, 1)
    # Find first Sunday
    days_to_sunday = (6 - march_1.weekday()) % 7
    if days_to_sunday == 0:
        first_sunday = march_1
    else:
        first_sunday = march_1 + timedelta(days=days_to_sunday)
    # 2nd Sunday
    dst_start = first_sunday + timedelta(days=7)
    
    # 1st Sunday in November (end of DST)
    nov_1 = datetime(year, 11, 1)
    days_to_sunday = (6 - nov_1.weekday()) % 7
    dst_end = nov_1 + timedelta(days=days_to_sunday)
    
    return dst_start.date() <= date_obj < dst_end.date()


def _get_et_tz_offset():
    """
    Get ET timezone offset from UTC.
    Returns timezone object for current DST status.
    """
    now = datetime.now(timezone.utc)
    if _is_dst(now.date()):
        return timezone(timedelta(hours=-4))  # EDT (UTC-4)
    else:
        return timezone(timedelta(hours=-5))  # EST (UTC-5)


def get_current_et_time() -> datetime:
    """Get current time in US Eastern Time."""
    now_utc = datetime.now(timezone.utc)
    et_tz = _get_et_tz_offset()
    return now_utc.astimezone(et_tz)


def get_current_et_date():
    """Get current date in US Eastern Time."""
    return get_current_et_time().date()


def _normalize_to_et(dt: datetime) -> datetime:
    """Normalize any datetime to US Eastern Time."""
    if dt is None:
        return get_current_et_time()
    
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to ET
    et_tz = _get_et_tz_offset()
    return dt.astimezone(et_tz)


def is_us_market_open(dt: datetime = None) -> bool:
    """
    Check if US stock market is open at given time.
    
    Args:
        dt: datetime object (UTC, ET, or naive). If None, uses current time.
        
    Returns:
        True if market is open (9:30 AM - 4 PM ET on weekdays)
    """
    et_time = _normalize_to_et(dt)
    
    # Market closed on weekends (5=Saturday, 6=Sunday)
    if et_time.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    
    return market_open <= et_time.time() < market_close


def is_us_market_premarket(dt: datetime = None) -> bool:
    """Check if in pre-market hours (4 AM - 9:30 AM ET)."""
    et_time = _normalize_to_et(dt)
    
    if et_time.weekday() >= 5:
        return False
    
    premarket_start = dt_time(4, 0)
    market_open = dt_time(9, 30)
    
    return premarket_start <= et_time.time() < market_open


def is_us_market_afterhours(dt: datetime = None) -> bool:
    """Check if in after-hours trading (4 PM - 8 PM ET)."""
    et_time = _normalize_to_et(dt)
    
    if et_time.weekday() >= 5:
        return False
    
    market_close = dt_time(16, 0)
    afterhours_end = dt_time(20, 0)
    
    return market_close <= et_time.time() < afterhours_end


def minutes_until_market_open() -> float:
    """
    Calculate minutes until US market opens.
    
    Returns:
        Minutes until market open. Returns 0 if market is currently open.
    """
    now = get_current_et_time()
    
    if is_us_market_open(now):
        return 0.0
    
    today = now.date()
    et_tz = _get_et_tz_offset()
    market_open_time = datetime.combine(today, dt_time(9, 30))
    market_open = market_open_time.replace(tzinfo=et_tz)
    
    # If past market open today, calculate for tomorrow
    if now > market_open:
        # Find next weekday
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        
        market_open = datetime.combine(next_day, dt_time(9, 30)).replace(tzinfo=et_tz)
    
    delta = market_open - now
    return delta.total_seconds() / 60.0


def minutes_until_market_close() -> float:
    """
    Calculate minutes until US market closes.
    
    Returns:
        Minutes until market close. Returns 0 if market not open or already closed.
    """
    now = get_current_et_time()
    
    if not is_us_market_open(now):
        return 0.0
    
    today = now.date()
    et_tz = _get_et_tz_offset()
    market_close = datetime.combine(today, dt_time(16, 0)).replace(tzinfo=et_tz)
    
    delta = market_close - now
    return delta.total_seconds() / 60.0


def should_wait_for_market_open(warn_logger=None) -> bool:
    """
    Check if should wait for market to open before running live trading.
    
    Logs a message if provided logger.
    
    Returns:
        True if should wait (market not open)
    """
    if not is_us_market_open():
        minutes = minutes_until_market_open()
        if warn_logger:
            if minutes > 0:
                warn_logger(
                    f"Market is closed. Will open in {minutes:.1f} minutes. "
                    f"Consider scheduling trades during 9:30 AM - 4:00 PM ET."
                )
            else:
                warn_logger("Market is closed. No trades scheduled for after-hours.")
        return True
    return False


def get_us_market_status() -> str:
    """Get human-readable US market status."""
    if is_us_market_open():
        mins = minutes_until_market_close()
        return f"OPEN (closes in {mins:.0f} min)"
    elif is_us_market_premarket():
        mins = minutes_until_market_open()
        return f"PRE-MARKET (opens in {mins:.0f} min)"
    elif is_us_market_afterhours():
        return "AFTER-HOURS"
    else:
        et_now = get_current_et_time()
        if et_now.weekday() >= 5:
            return "CLOSED (Weekend)"
        else:
            mins = minutes_until_market_open()
            return f"CLOSED (opens in {mins:.0f} min)"
