"""
cal_booking.py – Cal.com booking integration for the Sarvam AI Voice Agent.

Handles:
  - Checking available slots on a given date
  - Finding next available slot if requested slot is taken
  - Creating bookings via Cal.com API v2
  - Cancelling bookings via Cal.com API v2
  - Listing bookings from Cal.com
  - Detecting booking intent from user queries
  - Multi-turn dialogue to collect attendee name + email before booking
  - Local SQLite persistence of all booking records

Office hours: 9 AM – 5 PM (Asia/Kolkata / IST)
Slot duration: 60 minutes

FIXES IN THIS VERSION:
  [FIX-1]  _do_booking no longer loses attendee_email after session.reset().
  [FIX-2]  Config loaded from voice_agent.config (single source of truth).
  [FIX-3]  Weekend follow-up: user can say "yes" to auto-suggested Monday.
  [FIX-4]  cancel_booking() added — calls Cal.com API + updates local DB.
  [FIX-5]  list_bookings_from_cal() added — fetches bookings from Cal.com.
  [FIX-6]  All successful bookings saved to local SQLite via booking_db.
  [FIX-7]  handle_booking_query() accepts optional session parameter — works
           both from voice (with persistent session) and REST (session-less).
"""

import json
import logging
import re
from datetime import datetime, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from voice_agent.config import (
    CAL_API_KEY,
    CAL_USERNAME,
    CAL_EVENT_SLUG,
    CAL_EVENT_TYPE_ID,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

CAL_BASE_URL   = "https://api.cal.com/v2"
IST            = ZoneInfo("Asia/Kolkata")
OFFICE_START_H = 9    # 9 AM IST
OFFICE_END_H   = 19   # 7 PM IST — matches Cal.com schedule


# ──────────────────────────────────────────────
# Multi-turn booking session state
# ──────────────────────────────────────────────

class BookingSession:
    """
    Holds state across multiple voice turns while collecting
    the details needed to complete a booking or cancellation.

    Booking states:
      idle             – no operation in progress
      need_name        – slot confirmed available, waiting for user's name
      need_email       – have name, waiting for email
      confirming       – have all details, confirm before booking
      weekend_redirect – offered next weekday, waiting for yes/no

    Cancel states:
      cancel_need_uid  – cancel intent detected, waiting for booking UID
      cancel_confirm   – have UID, asking user to confirm cancellation

    Note: There is intentionally NO "done" state. After any terminal
    operation (booking created, cancelled, aborted) the session is
    reset() back to "idle" so the next turn starts fresh.
    """

    def __init__(self):
        self.state:            str            = "idle"
        self.req_date:         Optional[date] = None
        self.req_slot:         str            = ""    # "HH:MM"
        self.attendee_name:    str            = ""
        self.attendee_email:   str            = ""
        self.suggested_date:   Optional[date] = None  # for weekend redirect
        self.cancel_uid:       str            = ""    # UID to cancel (cancel flow)

    def reset(self):
        self.__init__()

    def is_active(self) -> bool:
        return self.state != "idle"


# ──────────────────────────────────────────────
# Booking intent detection
# ──────────────────────────────────────────────

BOOKING_KEYWORDS = [
    "book", "booking", "appointment", "schedule", "reserve",
    "slot", "meeting", "fix a time", "set up a meeting",
    "can i book", "i want to book", "book me", "make an appointment",
    "check availability", "available slot", "free slot", "open slot",
    "can you book", "please book", "i need an appointment",
    "set an appointment", "get an appointment", "arrange a meeting",
    "tomorrow", "next week", "next monday", "next tuesday",
    "monday at", "tuesday at", "wednesday at", "thursday at", "friday at",
    "at 9", "at 10", "at 11", "at 12", "at 1", "at 2", "at 3", "at 4",
    "9 am", "10 am", "11 am", "12 pm", "1 pm", "2 pm", "3 pm", "4 pm",
]


def is_booking_intent(text: str) -> bool:
    """Return True if the user's message is about booking an appointment."""
    t = text.lower()
    strong = ["book", "appointment", "schedule", "reserve", "slot", "meeting"]
    if any(kw in t for kw in strong):
        return True
    time_words = [
        "tomorrow", "today", "monday", "tuesday", "wednesday",
        "thursday", "friday", "am", "pm", "next week",
        "morning", "afternoon", "evening", "night",
    ]
    availability_words = ["available", "free", "open", "check", "availability"]
    if any(a in t for a in availability_words) and any(tw in t for tw in time_words):
        return True
    # Catch "check April 6th", "check Monday" etc.
    if t.strip().startswith("check") and any(tw in t for tw in time_words + [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]):
        return True
    return False


def is_cancel_intent(text: str) -> bool:
    """Return True if the user wants to cancel an existing booking."""
    t = text.lower()
    cancel_phrases = [
        "cancel my appointment", "cancel my booking", "cancel the appointment",
        "cancel the booking", "remove my booking", "delete my appointment",
        "i want to cancel", "please cancel", "need to cancel", "want to cancel",
        "cancel it", "cancel booking",
    ]
    return any(p in t for p in cancel_phrases)


def _extract_booking_uid(text: str) -> Optional[str]:
    """
    Try to pull a Cal.com booking UID or numeric ID from user speech.

    Cal.com UIDs look like: "abc123xyz" or "Wui3kB9Qn8rFp2K" (alphanumeric, 6-40 chars).
    Numeric IDs look like: "12345".
    Common English words are filtered out.
    """
    STOP_WORDS = {
        "cancel", "booking", "appointment", "please", "want", "would", "like",
        "the", "my", "okay", "sure", "yes", "yeah", "no", "that", "this",
        "it", "is", "was", "are", "have", "has", "had", "with", "for",
    }
    # Try numeric ID first (e.g., "booking number 12345")
    m = re.search(r'\b(\d{4,12})\b', text)
    if m:
        return m.group(1)

    # Try alphanumeric UID (min 6 chars, letters+digits, maybe with dashes)
    for token in re.findall(r'\b([A-Za-z0-9][A-Za-z0-9_-]{5,63})\b', text):
        if token.lower() not in STOP_WORDS and not token.isalpha():
            return token  # Mixed alphanumeric → likely a UID

    # Last resort: any token 6-40 chars that isn't a stop word
    for token in re.findall(r'\b([A-Za-z]{6,40})\b', text):
        if token.lower() not in STOP_WORDS:
            return token

    return None


# ──────────────────────────────────────────────
# Date/time parsing
# ──────────────────────────────────────────────

def parse_booking_request(text: str) -> dict:
    """
    Extract date and time from natural language.
    Returns dict with keys: date (date), hour (int 0-23)
    """
    now  = datetime.now(IST)
    t    = text.lower()
    result_date: Optional[date] = None
    result_hour: Optional[int]  = None

    # ── Date extraction ────────────────────────────────────────────
    if "today" in t:
        result_date = now.date()
    elif "day after tomorrow" in t:
        result_date = (now + timedelta(days=2)).date()
    elif "tomorrow" in t:
        result_date = (now + timedelta(days=1)).date()
    else:
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }
        for day_name, day_num in day_map.items():
            if day_name in t:
                days_ahead = (day_num - now.weekday()) % 7
                # "next <day>" always means the FOLLOWING week, never today/tomorrow
                if "next" in t or days_ahead == 0:
                    days_ahead += 7
                result_date = (now + timedelta(days=days_ahead)).date()
                break

    # Check for explicit date patterns like "April 5" or "5th April"
    if result_date is None:
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }
        for month_name, month_num in month_map.items():
            # "April 5" or "April 5th"
            m = re.search(rf'{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?', t)
            if m:
                day = int(m.group(1))
                year = now.year
                try:
                    result_date = date(year, month_num, day)
                    if result_date < now.date():
                        result_date = date(year + 1, month_num, day)
                except ValueError:
                    pass
                break
            # "5th April" or "5 April"
            m = re.search(rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}', t)
            if m:
                day = int(m.group(1))
                year = now.year
                try:
                    result_date = date(year, month_num, day)
                    if result_date < now.date():
                        result_date = date(year + 1, month_num, day)
                except ValueError:
                    pass
                break

    # "next week" without specific day → next Monday
    if result_date is None and "next week" in t:
        days_to_monday = (7 - now.weekday()) % 7
        if days_to_monday == 0:
            days_to_monday = 7
        result_date = (now + timedelta(days=days_to_monday)).date()

    if result_date is None:
        result_date = (now + timedelta(days=1)).date()

    # ── Time extraction ────────────────────────────────────────────
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*(am|pm)',
        r'(\d{1,2})\s*(am|pm)',
        r'(\d{1,2}):(\d{2})',
    ]
    for pattern in time_patterns:
        m = re.search(pattern, t)
        if m:
            groups = m.groups()
            hour = int(groups[0])
            ampm = None
            if len(groups) == 3:
                ampm = groups[2]
            elif len(groups) == 2:
                if groups[1] in ("am", "pm"):
                    ampm = groups[1]
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            result_hour = hour
            break

    # ── Vague time expressions ─────────────────────────────────────────
    # Check these BEFORE numeric fallbacks so "morning" etc. always win
    if result_hour is None:
        if any(w in t for w in ["morning", "early morning"]):
            result_hour = 9    # 9 AM
        elif any(w in t for w in ["noon", "midday", "lunch"]):
            result_hour = 12   # 12 PM
        elif any(w in t for w in ["afternoon"]):
            result_hour = 14   # 2 PM
        elif any(w in t for w in ["evening", "evening time"]):
            result_hour = 18   # 6 PM (now within office hours)
        elif any(w in t for w in ["night", "tonight", "late"]):
            result_hour = 18   # 6 PM (last comfortable slot)

    # Check for phrases like "at 3" or "at three"
    if result_hour is None:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12,
        }
        m = re.search(r'at\s+(\w+)', t)
        if m:
            word = m.group(1)
            if word in word_to_num:
                h = word_to_num[word]
                # Assume PM for hours 1-5 (business hours), AM for 6-12
                result_hour = h + 12 if 1 <= h <= 5 else h

    if result_hour is None:
        result_hour = 9

    result_hour = max(OFFICE_START_H, min(result_hour, OFFICE_END_H - 1))

    return {"date": result_date, "hour": result_hour}


# ──────────────────────────────────────────────
# Name / email extraction helpers
# ──────────────────────────────────────────────

def extract_name(text: str) -> str:
    """
    Pull a person's name out of a reply like:
      "My name is Rahul Sharma"  → "Rahul Sharma"
      "I am Priya"               → "Priya"
      "Rahul"                    → "Rahul"
    """
    t = text.strip()
    patterns = [
        r"(?:my name is|i am|i'm|this is|call me)\s+([A-Za-z][\w\s]{1,40})",
        r"^([A-Za-z][\w\s]{1,40})$",   # bare name
    ]
    for pat in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Remove trailing filler words
            name = re.sub(
                r"\s+(please|thanks|thank you|sir|ma'am)$", "",
                name, flags=re.IGNORECASE,
            )
            if 2 <= len(name) <= 50:
                return name.title()
    # Last resort: take first 1-3 capitalised words
    words = [w for w in t.split() if w[0].isupper()] if t else []
    if words:
        return " ".join(words[:3])
    return t.title()[:50]


def extract_email(text: str) -> Optional[str]:
    """Pull an email address out of the user's reply."""
    # Handle spoken email: "rahul at gmail dot com" → "rahul@gmail.com"
    spoken = text.lower().strip()
    spoken = re.sub(r'\b(at the rate|at the rate of)\b', '@', spoken)
    spoken = re.sub(r'\bat\b', '@', spoken)
    spoken = re.sub(r'\bdot\b', '.', spoken)
    spoken = re.sub(r'\s+', '', spoken)   # remove all spaces

    m = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', spoken)
    if m:
        return m.group(0).lower()

    # Try original text as fallback
    m = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    return m.group(0).lower() if m else None


# ──────────────────────────────────────────────
# Cal.com API calls
# ──────────────────────────────────────────────

def _headers_slots():
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-09-04",
    }


def _headers_bookings():
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-08-13",
    }


async def get_available_slots(target_date: date) -> list[str]:
    """
    Fetch available slots from Cal.com for a given date.
    Returns list of HH:MM strings in IST within office hours.
    """
    if not CAL_API_KEY:
        logger.warning("CAL_API_KEY not set — returning mock slots")
        return _mock_slots(target_date)

    day_start_ist = datetime(
        target_date.year, target_date.month, target_date.day,
        0, 0, 0, tzinfo=IST,
    )
    day_end_ist = datetime(
        target_date.year, target_date.month, target_date.day,
        23, 59, 59, tzinfo=IST,
    )

    start_utc = day_start_ist.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc   = day_end_ist.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "eventTypeId": CAL_EVENT_TYPE_ID,   # numeric ID — most reliable
        "start":       start_utc,            # ← NOT startTime
        "end":         end_utc,              # ← NOT endTime
    }

    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.get(
                f"{CAL_BASE_URL}/slots",     # ← NOT /slots/available
                headers=_headers_slots(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        logger.info(f"Cal.com slots raw response for {target_date}: {json.dumps(data)[:500]}")

        # Response structure: {"data": {"2026-04-06": [{"start": "...UTC"}, ...]}}
        # NOT data.slots — the date keys are directly under data
        slots_raw = data.get("data", {})
        if isinstance(slots_raw, dict) and "slots" in slots_raw:
            slots_raw = slots_raw["slots"]   # handle both response shapes

        all_slots = []
        for _date_key, slot_list in slots_raw.items():
            for slot in slot_list:
                # Cal.com v2 uses "start", older versions used "time"
                slot_time = slot.get("start") or slot.get("time", "")
                if slot_time:
                    all_slots.append(slot_time)

        office_slots = []
        for slot_utc_str in all_slots:
            try:
                slot_dt_utc = datetime.fromisoformat(
                    slot_utc_str.replace("Z", "+00:00")
                )
                slot_dt_ist = slot_dt_utc.astimezone(IST)
                if OFFICE_START_H <= slot_dt_ist.hour < OFFICE_END_H:
                    office_slots.append(slot_dt_ist.strftime("%H:%M"))
            except Exception:
                continue

        office_slots.sort()
        logger.info(f"Available slots on {target_date}: {office_slots}")
        return office_slots

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Cal.com slots API error: {e.response.status_code} {e.response.text}"
        )
        return []
    except Exception as e:
        logger.error(f"Cal.com slots fetch failed: {e}")
        return []


def _mock_slots(target_date: date) -> list[str]:
    """Return mock slots when CAL_API_KEY is not set (development mode)."""
    if target_date.weekday() >= 5:
        return []  # no weekend slots
    all_slots = [f"{h:02d}:00" for h in range(OFFICE_START_H, OFFICE_END_H)]
    busy = ["10:00", "13:00", "15:00"]
    return [s for s in all_slots if s not in busy]


async def create_booking(
    slot_time_ist: str,
    target_date: date,
    attendee_name: str,
    attendee_email: str,
    timezone: str = "Asia/Kolkata",
    source: str = "voice",
) -> dict:
    """
    Create a booking on Cal.com and save to local SQLite.
    Returns dict with success, booking_id, booking_uid, message.
    """
    h, m = map(int, slot_time_ist.split(":"))
    start_ist = datetime(
        target_date.year, target_date.month, target_date.day,
        h, m, 0, tzinfo=IST,
    )
    start_utc = start_ist.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    readable  = start_ist.strftime("%B %d, %Y at %I:%M %p IST")

    if not CAL_API_KEY:
        logger.warning("CAL_API_KEY not set — simulating booking")
        mock_uid = f"mock-{target_date.isoformat()}-{slot_time_ist.replace(':', '')}"
        # Save mock booking to local DB
        try:
            from voice_agent.booking_db import save_booking
            save_booking(
                cal_booking_uid=mock_uid,
                cal_booking_id=0,
                event_type_id=CAL_EVENT_TYPE_ID,
                booking_date=target_date.isoformat(),
                time_ist=slot_time_ist,
                start_utc=start_utc,
                attendee_name=attendee_name,
                attendee_email=attendee_email,
                timezone=timezone,
                source=source,
                cal_response="mock",
            )
        except Exception as e:
            logger.warning(f"Failed to save mock booking to DB: {e}")

        return {
            "success":     True,
            "booking_id":  "mock-0",
            "booking_uid": mock_uid,
            "message": (
                f"Appointment booked for {readable} "
                f"(mock mode — set CAL_API_KEY for real bookings)"
            ),
        }

    payload = {
        "eventTypeId": CAL_EVENT_TYPE_ID,
        "start":       start_utc,
        "attendee": {
            "name":     attendee_name,
            "email":    attendee_email,
            "timeZone": timezone,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.post(
                f"{CAL_BASE_URL}/bookings",
                headers=_headers_bookings(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        booking_data = data.get("data", {})
        booking_id   = booking_data.get("id") or data.get("id", 0)
        booking_uid  = booking_data.get("uid") or str(booking_id)

        logger.info(f"Booking created: id={booking_id} uid={booking_uid} for {readable}")

        # [FIX-6] Save to local SQLite database
        try:
            from voice_agent.booking_db import save_booking
            save_booking(
                cal_booking_uid=booking_uid,
                cal_booking_id=int(booking_id) if str(booking_id).isdigit() else 0,
                event_type_id=CAL_EVENT_TYPE_ID,
                booking_date=target_date.isoformat(),
                time_ist=slot_time_ist,
                start_utc=start_utc,
                attendee_name=attendee_name,
                attendee_email=attendee_email,
                timezone=timezone,
                source=source,
                cal_response=json.dumps(booking_data),
            )
        except Exception as db_err:
            logger.error(f"Failed to save booking to local DB: {db_err}")
            # Don't fail the whole booking just because DB save failed

        return {
            "success":     True,
            "booking_id":  str(booking_id),
            "booking_uid": booking_uid,
            "message":     f"Appointment confirmed for {readable}. Booking ID: {booking_uid}",
        }

    except httpx.HTTPStatusError as e:
        err = e.response.text
        logger.error(f"Cal.com booking error: {e.response.status_code} {err}")
        return {
            "success": False, "booking_id": None, "booking_uid": None,
            "message": f"Booking failed: {err}",
        }
    except Exception as e:
        logger.error(f"Cal.com booking failed: {e}")
        return {
            "success": False, "booking_id": None, "booking_uid": None,
            "message": f"Booking failed: {str(e)}",
        }


# ──────────────────────────────────────────────
# Cancel booking
# ──────────────────────────────────────────────

async def cancel_booking(booking_uid: str) -> dict:
    """
    Cancel a booking on Cal.com and update local DB.
    Returns dict with success and message.
    """
    if not CAL_API_KEY:
        # Mock mode
        try:
            from voice_agent.booking_db import mark_cancelled
            mark_cancelled(booking_uid)
        except Exception:
            pass
        return {
            "success": True,
            "message": f"Booking {booking_uid} cancelled (mock mode).",
        }

    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.post(
                f"{CAL_BASE_URL}/bookings/{booking_uid}/cancel",
                headers=_headers_bookings(),
                json={"cancellationReason": "Cancelled by user via AI assistant"},
            )
            resp.raise_for_status()

        # Update local DB
        try:
            from voice_agent.booking_db import mark_cancelled
            mark_cancelled(booking_uid)
        except Exception as db_err:
            logger.warning(f"Failed to update local DB for cancel: {db_err}")

        logger.info(f"Booking cancelled: uid={booking_uid}")
        return {
            "success": True,
            "message": f"Booking {booking_uid} has been successfully cancelled.",
        }

    except httpx.HTTPStatusError as e:
        err = e.response.text
        logger.error(f"Cal.com cancel error: {e.response.status_code} {err}")
        return {"success": False, "message": f"Cancel failed: {err}"}
    except Exception as e:
        logger.error(f"Cancel booking failed: {e}")
        return {"success": False, "message": f"Cancel failed: {str(e)}"}


# ──────────────────────────────────────────────
# List bookings from Cal.com
# ──────────────────────────────────────────────

async def list_bookings_from_cal(
    status: str = "upcoming",
    attendee_email: Optional[str] = None,
) -> list[dict]:
    """
    Fetch bookings from Cal.com API.
    status: 'upcoming', 'past', 'cancelled', 'recurring'
    """
    if not CAL_API_KEY:
        # Return from local DB in mock mode
        try:
            from voice_agent.booking_db import get_bookings
            return get_bookings(email=attendee_email, limit=20)
        except Exception:
            return []

    params = {"status": status}
    if attendee_email:
        params["attendeeEmail"] = attendee_email

    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.get(
                f"{CAL_BASE_URL}/bookings",
                headers=_headers_bookings(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        bookings = data.get("data", [])
        result = []
        for b in bookings:
            start_str = b.get("start", "")
            end_str   = b.get("end", "")
            try:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                start_ist = start_dt.astimezone(IST)
                date_str = start_ist.strftime("%Y-%m-%d")
                time_str = start_ist.strftime("%H:%M")
                readable = start_ist.strftime("%B %d, %Y at %I:%M %p IST")
            except Exception:
                date_str = start_str
                time_str = ""
                readable = start_str

            attendees = b.get("attendees", [])
            attendee_info = attendees[0] if attendees else {}

            result.append({
                "uid":    b.get("uid", ""),
                "id":     b.get("id", ""),
                "status": b.get("status", ""),
                "date":   date_str,
                "time":   time_str,
                "readable": readable,
                "title":  b.get("title", ""),
                "attendee_name":  attendee_info.get("name", ""),
                "attendee_email": attendee_info.get("email", ""),
            })

        return result

    except Exception as e:
        logger.error(f"Failed to list bookings from Cal.com: {e}")
        # Fallback to local DB
        try:
            from voice_agent.booking_db import get_bookings
            return get_bookings(email=attendee_email, limit=20)
        except Exception:
            return []


# ──────────────────────────────────────────────
# Multi-turn booking handler (called by VoiceAgent)
# ──────────────────────────────────────────────

async def handle_booking_query(
    user_text: str,
    session: Optional[BookingSession] = None,
    attendee_name: str = "",
    attendee_email: str = "",
) -> tuple[str, bool]:
    """
    Multi-turn booking + cancellation handler.

    session parameter is Optional:
      - VoiceAgent passes its persistent BookingSession (for multi-turn voice)
      - REST endpoints can pass None (a temporary one-shot session is created)

    Returns:
        (response_text, operation_completed)
        operation_completed=True means a booking was created OR cancelled successfully.
    """
    # Create temp session if none provided (for REST endpoint)
    if session is None:
        session = BookingSession()
        if attendee_name and attendee_name != "User":
            session.attendee_name = attendee_name
        if attendee_email and attendee_email != "user@example.com":
            session.attendee_email = attendee_email

    # ── CANCEL FLOW: waiting for booking UID ──────────────────────────────
    if session.state == "cancel_need_uid":
        if _is_abort(user_text):
            session.reset()
            return "No problem. Let me know if there's anything else I can help you with.", False

        uid = _extract_booking_uid(user_text)
        if not uid:
            return (
                "I couldn't find a booking ID in that. "
                "Could you please say or spell out your booking reference number?",
                False,
            )
        session.cancel_uid = uid
        session.state = "cancel_confirm"
        return (
            f"Just to confirm — you'd like to cancel booking {uid}. "
            f"Shall I go ahead?",
            False,
        )

    # ── CANCEL FLOW: waiting for confirmation ─────────────────────────────
    if session.state == "cancel_confirm":
        t = user_text.lower()
        if any(w in t for w in ["yes", "yeah", "yep", "ya", "yah", "sure", "go ahead",
                                  "confirm", "ok", "okay", "please", "do it", "alright",
                                  "all right", "sounds good", "of course", "absolutely"]):
            uid = session.cancel_uid
            session.reset()
            result = await cancel_booking(uid)
            return result["message"], result["success"]
        elif any(w in t for w in ["no", "stop", "nevermind", "never mind", "don't", "nah"]):
            session.reset()
            return "Cancellation aborted. Your booking is still active. Let me know if you need anything else.", False
        else:
            return (
                f"Sorry, I didn't catch that. "
                f"Please say yes to cancel booking {session.cancel_uid}, or no to keep it.",
                False,
            )

    # ── WEEKEND REDIRECT: user responding to "shall I check Monday?" ───────
    if session.state == "weekend_redirect":
        t = user_text.lower()
        AFFIRMATIVES = [
            "yes", "yeah", "yep", "ya", "yah", "sure", "ok", "okay",
            "please", "go ahead", "check", "alright", "all right",
            "sounds good", "go for it", "do it", "why not", "of course",
        ]
        # If user explicitly mentions a date (e.g. "yeah check 7th April"),
        # extract that date and use it instead of the suggested one
        parsed_new = parse_booking_request(user_text)
        user_mentioned_date = any(
            m in t for m in [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "tomorrow", "today", "next week",
            ]
        )

        if user_mentioned_date:
            # User said "yeah check 7th April" — use their date
            session.state = "idle"
            session.suggested_date = None
            fake_text = f"book appointment on {parsed_new['date'].strftime('%A %B %d')} at {parsed_new['hour']:02d}:00"
            return await handle_booking_query(fake_text, session, attendee_name, attendee_email)

        elif any(w in t for w in AFFIRMATIVES):
            session.state = "idle"
            suggested = session.suggested_date
            session.suggested_date = None
            if suggested:
                fake_text = f"book appointment on {suggested.strftime('%A %B %d')} at 9 am"
                return await handle_booking_query(fake_text, session, attendee_name, attendee_email)
            else:
                session.reset()
                return "Sorry, something went wrong. Please try your booking again.", False
        else:
            session.reset()
            return "No problem. Let me know whenever you'd like to book.", False

    # ── IDLE: detect what the user wants ──────────────────────────────────
    if session.state == "idle":

        # ── Cancel intent branch ───────────────────────────────────────
        if is_cancel_intent(user_text):
            uid = _extract_booking_uid(user_text)
            if uid:
                # UID provided in the same utterance — ask to confirm
                session.cancel_uid = uid
                session.state = "cancel_confirm"
                return (
                    f"Just to confirm — you'd like to cancel booking {uid}. "
                    f"Shall I go ahead?",
                    False,
                )
            else:
                # Need to collect UID
                session.state = "cancel_need_uid"
                return (
                    "I can help you cancel your booking. "
                    "Could you please provide your booking ID or reference number? "
                    "You would have received it in your confirmation email.",
                    False,
                )

        # ── New booking intent branch ──────────────────────────────────
        parsed   = parse_booking_request(user_text)
        req_date = parsed["date"]
        req_hour = parsed["hour"]
        req_slot = f"{req_hour:02d}:00"

        # Weekend check
        if req_date.weekday() >= 5:
            days_to_monday = 7 - req_date.weekday()
            next_monday = req_date + timedelta(days=days_to_monday)
            session.state = "weekend_redirect"
            session.suggested_date = next_monday
            return (
                f"Appointments are only available on weekdays. "
                f"The next available weekday is {next_monday.strftime('%A, %B %d')}. "
                f"Would you like me to check availability for that day?",
                False,
            )

        # Past date check
        today = datetime.now(IST).date()
        if req_date < today:
            return (
                f"That date ({req_date.strftime('%B %d')}) has already passed. "
                f"Would you like to book for a future date instead?",
                False,
            )

        available = await get_available_slots(req_date)

        if not available:
            next_date = req_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            # Store suggested date so user can say "yes" to check it
            session.state = "weekend_redirect"
            session.suggested_date = next_date
            return (
                f"All slots for {req_date.strftime('%A, %B %d')} are fully booked. "
                f"Would you like me to check availability for "
                f"{next_date.strftime('%A, %B %d')} instead?",
                False,
            )

        def fmt(t):
            return datetime.strptime(t, "%H:%M").strftime("%I:%M %p")

        # Requested slot is taken — suggest nearest alternatives
        if req_slot not in available:
            before = [s for s in available if s < req_slot]
            after  = [s for s in available if s > req_slot]
            suggestions = []
            if after:  suggestions.append(after[0])
            if before: suggestions.append(before[-1])
            suggestions = suggestions[:2]
            slot_strs = " or ".join(fmt(s) for s in suggestions)
            return (
                f"That slot — {fmt(req_slot)} on "
                f"{req_date.strftime('%A, %B %d')} — is already booked. "
                f"The nearest available times are {slot_strs}. "
                f"Would you like to book one of these instead?",
                False,
            )

        # Slot is available — save to session and start collecting details
        session.req_date = req_date
        session.req_slot = req_slot

        if attendee_name and attendee_name != "User":
            session.attendee_name = attendee_name
        if attendee_email and attendee_email != "user@example.com":
            session.attendee_email = attendee_email

        date_str = req_date.strftime("%A, %B %d")
        time_str = fmt(req_slot)

        if session.attendee_name and session.attendee_email:
            return await _do_booking(session)

        if session.attendee_name:
            session.state = "need_email"
            return (
                f"Great! {time_str} on {date_str} is available. "
                f"What is your email address so we can send you a confirmation?",
                False,
            )

        session.state = "need_name"
        return (
            f"Great! {time_str} on {date_str} is available. "
            f"May I have your name please?",
            False,
        )

    # ── TURN 2: Collecting name ────────────────────────────────────────────
    if session.state == "need_name":
        if _is_abort(user_text):
            session.reset()
            return "No problem, booking cancelled. Let me know if you need anything else.", False

        name = extract_name(user_text)
        if not name or len(name) < 2:
            return "Sorry, I didn't catch your name. Could you please repeat it?", False

        session.attendee_name = name
        session.state = "need_email"
        return (
            f"Thank you, {name}. "
            f"What is your email address so we can send you a confirmation?",
            False,
        )

    # ── TURN 3: Collecting email ───────────────────────────────────────────
    if session.state == "need_email":
        if _is_abort(user_text):
            session.reset()
            return "No problem, booking cancelled. Let me know if you need anything else.", False

        email = extract_email(user_text)
        if not email:
            return (
                "Sorry, I didn't catch a valid email address. "
                "Could you please say your email again? "
                "For example: rahul at gmail dot com.",
                False,
            )

        session.attendee_email = email
        session.state = "confirming"

        fmt_time = datetime.strptime(session.req_slot, "%H:%M").strftime("%I:%M %p")
        fmt_date = session.req_date.strftime("%A, %B %d")
        return (
            f"Just to confirm — booking {fmt_time} on {fmt_date} "
            f"for {session.attendee_name} at {session.attendee_email}. "
            f"Shall I go ahead?",
            False,
        )

    # ── TURN 4: Final confirmation ────────────────────────────────────────
    if session.state == "confirming":
        t = user_text.lower()
        affirmatives = [
            "yes", "yeah", "yep", "ya", "yah", "sure", "go ahead",
            "confirm", "ok", "okay", "correct", "please", "do it",
            "that's right", "absolutely", "book it", "alright",
            "all right", "sounds good", "of course", "go for it",
        ]
        negatives = [
            "no", "cancel", "stop", "nevermind", "never mind",
            "change", "don't", "nah",
        ]
        if any(w in t for w in affirmatives):
            return await _do_booking(session)
        elif any(w in t for w in negatives):
            session.reset()
            return (
                "Booking cancelled. Let me know if you'd like to pick a different time.",
                False,
            )
        else:
            fmt_time = datetime.strptime(session.req_slot, "%H:%M").strftime("%I:%M %p")
            fmt_date = session.req_date.strftime("%A, %B %d")
            return (
                f"Sorry, I didn't catch that. "
                f"Please say yes to confirm the {fmt_time} on {fmt_date} "
                f"appointment for {session.attendee_name}, or no to cancel.",
                False,
            )

    # Fallback — unknown state, reset cleanly
    session.reset()
    return "Something went wrong. Please say your request again.", False


def _is_abort(text: str) -> bool:
    """Check if user wants to abort the current flow (booking or cancel)."""
    return any(
        w in text.lower()
        for w in ["cancel", "nevermind", "never mind", "forget it", "no thanks", "stop", "abort", "quit"]
    )


async def _do_booking(session: BookingSession) -> tuple[str, bool]:
    """
    Execute the actual Cal.com booking and reset the session to idle.

    IMPORTANT: All required fields are copied BEFORE session.reset() is called
    so the confirmation message can still reference them. The session is always
    left in the "idle" state (never "done") so the next booking intent starts
    a clean new flow without hitting the fallback branch.
    """
    # Copy all fields before reset — session.reset() wipes everything
    booking_name  = session.attendee_name
    booking_email = session.attendee_email
    booking_date  = session.req_date
    booking_slot  = session.req_slot

    result = await create_booking(
        slot_time_ist=booking_slot,
        target_date=booking_date,
        attendee_name=booking_name,
        attendee_email=booking_email,
        source="voice",
    )

    # Reset to idle — NOT to "done". This ensures a subsequent booking intent
    # correctly enters the "idle" branch instead of hitting the fallback.
    session.reset()

    if result["success"]:
        readable_date = booking_date.strftime("%A, %B %d, %Y")
        readable_time = datetime.strptime(booking_slot, "%H:%M").strftime("%I:%M %p")
        return (
            f"Your appointment has been booked for {readable_time} on "
            f"{readable_date}. A confirmation will be sent to {booking_email}. "
            f"Your booking ID is {result.get('booking_uid', result.get('booking_id', 'N/A'))}. "
            f"Is there anything else I can help you with?",
            True,
        )
    else:
        return (
            f"I'm sorry, the booking could not be completed. "
            f"{result['message']}",
            False,
        )