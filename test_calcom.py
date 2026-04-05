"""
test_calcom.py – Run this to debug your Cal.com API connection.
Usage:  python test_calcom.py

Paste your keys directly below or it will read from .env
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Paste your values here if .env is not loading ──────────────────
CAL_API_KEY      = os.getenv("CAL_API_KEY", "PASTE_YOUR_KEY_HERE")
CAL_USERNAME     = os.getenv("CAL_USERNAME", "broukrutha")
CAL_EVENT_SLUG   = os.getenv("CAL_EVENT_SLUG", "appointments")
CAL_EVENT_TYPE_ID = int(os.getenv("CAL_EVENT_TYPE_ID", "5260220"))
# ───────────────────────────────────────────────────────────────────

IST = ZoneInfo("Asia/Kolkata")
BASE = "https://api.cal.com/v2"

HEADERS_SLOTS = {
    "Authorization": f"Bearer {CAL_API_KEY}",
    "cal-api-version": "2024-09-04",
}
HEADERS_BOOKINGS = {
    "Authorization": f"Bearer {CAL_API_KEY}",
    "Content-Type": "application/json",
    "cal-api-version": "2024-08-13",
}


async def test_slots(date_str: str):
    """Try multiple endpoint + param combinations to find what works."""
    IST = ZoneInfo("Asia/Kolkata")
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=IST)
    end   = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=IST)
    start_utc = start.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc   = end.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    attempts = [
        {
            "label": "✅ CORRECT: /v2/slots  eventTypeId + start + end",
            "url":   f"{BASE}/slots",
            "params": {
                "eventTypeId": CAL_EVENT_TYPE_ID,
                "start":       start_utc,
                "end":         end_utc,
            },
        },
        {
            "label": "v2/slots  username + eventTypeSlug + start + end",
            "url":   f"{BASE}/slots",
            "params": {
                "username":      CAL_USERNAME,
                "eventTypeSlug": CAL_EVENT_SLUG,
                "start":         start_utc,
                "end":           end_utc,
            },
        },
    ]

    print(f"\n{'='*60}")
    print(f"  Testing Cal.com slots for date: {date_str}")
    print(f"  UTC window: {start_utc}  →  {end_utc}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in attempts:
            print(f"  ── {attempt['label']}")
            print(f"     URL:    {attempt['url']}")
            print(f"     Params: {attempt['params']}")
            try:
                resp = await client.get(
                    attempt["url"],
                    headers=HEADERS_SLOTS,
                    params=attempt["params"],
                )
                print(f"     Status: {resp.status_code}")
                try:
                    data = resp.json()
                    # Pretty-print, truncate if huge
                    text = json.dumps(data, indent=2)
                    print(f"     Response:\n{text[:800]}")
                    # Check if we got slots
                    slots = data.get("data", {}).get("slots", {})
                    if slots:
                        print(f"\n  ✅ THIS ENDPOINT WORKS! Slots found: {list(slots.keys())}")
                    else:
                        print(f"  ⚠️  No slots in response (data.slots is empty or missing)")
                except Exception:
                    print(f"     Raw: {resp.text[:300]}")
            except Exception as e:
                print(f"     ERROR: {e}")
            print()


async def test_event_types():
    """List all event types to verify eventTypeId."""
    print(f"\n{'='*60}")
    print("  Fetching your Cal.com event types")
    print(f"{'='*60}\n")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{BASE}/event-types",
            headers=HEADERS_BOOKINGS,
        )
        print(f"  Status: {resp.status_code}")
        try:
            data = resp.json()
            event_types = data.get("data", {}).get("eventTypeGroups", [])
            for group in event_types:
                for et in group.get("eventTypes", []):
                    print(f"  📅 ID={et.get('id')}  slug={et.get('slug')}  title={et.get('title')}  length={et.get('length')}min")
            if not event_types:
                # Try flat list
                flat = data.get("data", [])
                if isinstance(flat, list):
                    for et in flat:
                        print(f"  📅 ID={et.get('id')}  slug={et.get('slug')}  title={et.get('title')}")
                else:
                    print(f"  Raw: {json.dumps(data, indent=2)[:600]}")
        except Exception:
            print(f"  Raw: {resp.text[:400]}")


async def main():
    print("\n🔍 Cal.com API Diagnostic Tool")
    print(f"   API Key : {CAL_API_KEY[:12]}...{CAL_API_KEY[-4:]}")
    print(f"   Username: {CAL_USERNAME}")
    print(f"   Slug    : {CAL_EVENT_SLUG}")
    print(f"   TypeID  : {CAL_EVENT_TYPE_ID}")

    # Test event types first
    await test_event_types()

    # Test next Monday (most likely to have slots)
    today = datetime.now(IST).date()
    days_ahead = (7 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_monday = today + timedelta(days=days_ahead)
    await test_slots(next_monday.strftime("%Y-%m-%d"))

    # Also test tomorrow
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() < 5:
        await test_slots(tomorrow.strftime("%Y-%m-%d"))

    print("\n✅ Diagnostic complete. Share the output above to identify the issue.\n")


if __name__ == "__main__":
    asyncio.run(main())