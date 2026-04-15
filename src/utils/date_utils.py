"""
date_utils.py — Date and time helpers for the trading bot.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List

import pytz

_EASTERN = pytz.timezone("America/New_York")
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 30
_MARKET_CLOSE_HOUR = 16


def now_eastern() -> datetime:
    """Return the current time in US/Eastern (NYSE timezone).

    Returns:
        Timezone-aware datetime in the America/New_York zone.
    """
    return datetime.now(_EASTERN)


def is_market_hours() -> bool:
    """Determine whether the current time is within NYSE regular hours.

    Does not account for market holidays — use SchwabFetcher.is_market_open()
    for definitive checks.

    Returns:
        True if the current Eastern time is Mon–Fri between 09:30 and 16:00.
    """
    now = now_eastern()
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    open_time = now.replace(
        hour=_MARKET_OPEN_HOUR,
        minute=_MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    close_time = now.replace(
        hour=_MARKET_CLOSE_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    return open_time <= now < close_time


def business_days_ago(n: int) -> date:
    """Return the date that is *n* business days before today.

    Args:
        n: Number of business days to look back.

    Returns:
        Date object representing the target business day.
    """
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def date_range_business_days(start: date, end: date) -> List[date]:
    """Return all business days between *start* and *end* (inclusive).

    Args:
        start: Start date.
        end: End date.

    Returns:
        List of dates that are Monday–Friday within the range.
    """
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def to_epoch_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds since Unix epoch.

    Args:
        dt: Datetime to convert.  If naive, UTC is assumed.

    Returns:
        Integer milliseconds since epoch.
    """
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    epoch = datetime(1970, 1, 1, tzinfo=pytz.utc)
    return int((dt - epoch).total_seconds() * 1000)
