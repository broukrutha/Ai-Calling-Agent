"""
Check your Cal.com credentials and event type.

    python scripts/check_calcom.py

Lists your event types (so you can find CAL_EVENT_TYPE_ID) and shows the open
slots for the next few days through the same client the agent uses.
"""

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent import clock, config  # noqa: E402
from voice_agent.booking import calcom  # noqa: E402


async def main() -> int:
    if not config.CAL_API_KEY:
        print("CAL_API_KEY is not set in .env — the agent is running in mock mode.")
        return 1

    print("Event types on this account:")
    async with httpx.AsyncClient(timeout=10.0, trust_env=config.HTTP_TRUST_ENV) as client:
        resp = await client.get(
            f"{config.CAL_BASE_URL}/event-types",
            headers={"Authorization": f"Bearer {config.CAL_API_KEY}", "cal-api-version": "2024-06-14"},
        )
    if resp.status_code != 200:
        print(f"  Cal.com returned HTTP {resp.status_code}. Check that the API key is valid.")
        return 1
    data = resp.json().get("data") or []
    groups = data.get("eventTypeGroups", []) if isinstance(data, dict) else [{"eventTypes": data}]
    for group in groups:
        for event_type in group.get("eventTypes", []):
            marker = "  <-- configured" if event_type.get("id") == config.CAL_EVENT_TYPE_ID else ""
            minutes = event_type.get("lengthInMinutes") or event_type.get("length") or "?"
            print(f"  id={event_type.get('id')}  slug={event_type.get('slug')}  {minutes} min{marker}")

    if config.CAL_EVENT_TYPE_ID <= 0:
        print("\nSet CAL_EVENT_TYPE_ID in .env to one of the ids above.")
        return 1

    print(f"\nOpen slots for event type {config.CAL_EVENT_TYPE_ID} ({config.BOOKING_TIMEZONE}):")
    today = clock.now_local().date()
    slots = await calcom.get_slots_in_range(today, days=5)
    for day in sorted(slots):
        print(f"  {day:%a %d %b}: {', '.join(slots[day]) or 'none'}")
    await calcom.close()
    if not any(slots.values()):
        print(f"\nNo slots between {today} and {today + timedelta(days=4)}. Check the event type's availability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
