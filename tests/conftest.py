"""
Shared test fixtures.

Tests never touch the network or the developer's real data: provider keys are
blanked before the application config is imported (so a local ``.env`` cannot
leak in), bookings go to a throw-away SQLite file, and the clock is frozen.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Must happen before anything imports voice_agent.config.
for _key in (
    "SARVAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "CAL_API_KEY", "ADMIN_API_TOKEN",
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OUTBOUND_TRUNK_ID",
):
    os.environ[_key] = ""
os.environ["APP_ENV"] = "development"
os.environ["CAL_EVENT_TYPE_ID"] = "0"
os.environ["BOOKING_TIMEZONE"] = "Asia/Kolkata"
os.environ["OFFICE_START_HOUR"] = "9"
os.environ["OFFICE_END_HOUR"] = "19"

import pytest  # noqa: E402

from voice_agent import clock, config  # noqa: E402
from voice_agent.booking import store  # noqa: E402

# A Wednesday morning, so "tomorrow" is a weekday and "Saturday" is 3 days out.
FROZEN_NOW = datetime(2026, 4, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(clock, "now_local", lambda: FROZEN_NOW)
    return FROZEN_NOW


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BOOKINGS_DB_PATH", str(tmp_path / "bookings.db"))
    yield
    store.close()


@pytest.fixture(autouse=True)
def mock_calendar(monkeypatch):
    """Cal.com mock mode unless a test opts in to the real client."""
    monkeypatch.setattr(config, "CAL_API_KEY", "")
