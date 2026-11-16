"""The business clock. A single seam, so tests can freeze time in one place."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from voice_agent import config


def now_local() -> datetime:
    """Current time in the booking timezone (``BOOKING_TIMEZONE``)."""
    return datetime.now(ZoneInfo(config.BOOKING_TIMEZONE))
