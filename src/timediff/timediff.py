"""Hour differences between timestamps of the form ``2017/05/13 12:00``."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
WORKDAY_START = time(9, 0)
WORKDAY_END = time(17, 0)


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse ``YYYY/MM/DD HH:MM`` (surrounding whitespace allowed)."""
    if isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Invalid timestamp {value!r}; expected format 'YYYY/MM/DD HH:MM'"
        ) from exc


def _round_hours(seconds: float) -> int:
    """Round a duration to whole hours, halves rounded up (Python's round() is banker's)."""
    return int((Decimal(seconds) / Decimal(3600)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def hours_between(start: str | datetime, end: str | datetime) -> int:
    """Rounded number of full hours between two timestamps.

    The order of the arguments does not matter; the absolute difference is returned.

    >>> hours_between("2022/02/15 00:05", "2022/02/15 01:00")
    1
    """
    a, b = sorted((parse_timestamp(start), parse_timestamp(end)))
    return _round_hours((b - a).total_seconds())


def business_seconds_between(
    start: str | datetime,
    end: str | datetime,
    workday_start: time = WORKDAY_START,
    workday_end: time = WORKDAY_END,
    holidays: Iterable[date] = (),
) -> float:
    """Seconds between two timestamps that fall within working hours on weekdays."""
    a, b = sorted((parse_timestamp(start), parse_timestamp(end)))
    if workday_start >= workday_end:
        raise ValueError("workday_start must be before workday_end")
    skip = set(holidays)

    total = 0.0
    day = a.date()
    while day <= b.date():
        if day.weekday() < 5 and day not in skip:  # Mon=0 … Fri=4
            window_start = max(a, datetime.combine(day, workday_start))
            window_end = min(b, datetime.combine(day, workday_end))
            if window_end > window_start:
                total += (window_end - window_start).total_seconds()
        day += timedelta(days=1)
    return total


def business_hours_between(
    start: str | datetime,
    end: str | datetime,
    workday_start: time = WORKDAY_START,
    workday_end: time = WORKDAY_END,
    holidays: Iterable[date] = (),
) -> int:
    """Like :func:`hours_between`, but only counting 09:00–17:00 on weekdays.

    Optionally pass ``holidays`` (dates) to exclude public holidays as well.

    >>> business_hours_between("2022/02/18 16:00", "2022/02/21 10:00")  # Fri -> Mon
    2
    """
    return _round_hours(
        business_seconds_between(start, end, workday_start, workday_end, holidays)
    )
