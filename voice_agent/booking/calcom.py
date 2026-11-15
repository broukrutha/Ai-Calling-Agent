"""
Cal.com API v2 client.

Checks availability, creates, cancels and lists bookings, and mirrors every
successful change into the local database.

Without ``CAL_API_KEY`` the client runs in **mock mode**: weekday office-hour
slots, minus whatever is already booked locally. That makes the whole product
usable (and testable) without a Cal.com account.

User-facing messages returned from here are always safe to speak aloud: raw
API error bodies go to the log, never to the caller.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from voice_agent import clock, config
from voice_agent.booking import nlu, store
from voice_agent.logging_config import mask_email

logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")
_SLOTS_API_VERSION = "2024-09-04"
_BOOKINGS_API_VERSION = "2024-08-13"

_client: httpx.AsyncClient | None = None


def _tz() -> ZoneInfo:
    return ZoneInfo(config.BOOKING_TIMEZONE)


def _http() -> httpx.AsyncClient:
    """One pooled client for all Cal.com calls: no TLS handshake per request."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            trust_env=config.HTTP_TRUST_ENV,
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def is_mock_mode() -> bool:
    return not config.CAL_API_KEY


def _headers(api_version: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.CAL_API_KEY}",
        "cal-api-version": api_version,
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _to_utc_string(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Availability ─────────────────────────────────────────────────────────────

def _mock_slots(target_date: date) -> list[str]:
    if target_date.weekday() >= 5:
        return []
    taken = set(store.get_booked_slots(target_date.isoformat()))
    return [
        f"{hour:02d}:00"
        for hour in range(config.OFFICE_START_HOUR, config.OFFICE_END_HOUR)
        if f"{hour:02d}:00" not in taken
    ]


def _drop_past(slots_by_date: dict[date, list[str]]) -> dict[date, list[str]]:
    """Remove slots that have already started today."""
    now = clock.now_local()
    today = now.date()
    if today in slots_by_date:
        cutoff = now.strftime("%H:%M")
        slots_by_date[today] = [slot for slot in slots_by_date[today] if slot > cutoff]
    return slots_by_date


async def get_slots_in_range(start_date: date, days: int = 1) -> dict[date, list[str]]:
    """
    Open slots for ``days`` consecutive dates starting at ``start_date``.

    Returns ``{date: ["09:00", "10:00", ...]}`` in the booking timezone, limited
    to office hours, sorted, with past times removed. Dates with no openings
    map to an empty list. A Cal.com failure is logged and reported as "no
    openings" so a call can continue gracefully.
    """
    dates = [start_date + timedelta(days=offset) for offset in range(days)]
    result: dict[date, list[str]] = {day: [] for day in dates}

    if is_mock_mode():
        for day in dates:
            result[day] = _mock_slots(day)
        return _drop_past(result)

    tz = _tz()
    window_start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=tz)
    window_end = window_start + timedelta(days=days) - timedelta(seconds=1)
    params = {
        "eventTypeId": config.CAL_EVENT_TYPE_ID,
        "start": _to_utc_string(window_start),
        "end": _to_utc_string(window_end),
    }

    try:
        resp = await _http().get(
            f"{config.CAL_BASE_URL}/slots", headers=_headers(_SLOTS_API_VERSION), params=params
        )
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Cal.com slots error %s: %s", exc.response.status_code, exc.response.text[:500])
        return result
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Cal.com slots request failed: %s", exc)
        return result

    # Shape: {"data": {"2026-04-06": [{"start": "...Z"}, ...]}}; older responses
    # nest the same mapping under data.slots and use "time" instead of "start".
    raw = payload.get("data") or {}
    if isinstance(raw, dict) and isinstance(raw.get("slots"), dict):
        raw = raw["slots"]
    if not isinstance(raw, dict):
        return result

    for slot_list in raw.values():
        for slot in slot_list or []:
            stamp = (slot.get("start") or slot.get("time") or "") if isinstance(slot, dict) else ""
            try:
                local = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(tz)
            except ValueError:
                continue
            if local.date() in result and config.OFFICE_START_HOUR <= local.hour < config.OFFICE_END_HOUR:
                result[local.date()].append(local.strftime("%H:%M"))

    for day in result:
        result[day] = sorted(set(result[day]))
    return _drop_past(result)


async def get_available_slots(target_date: date) -> list[str]:
    """Open slots on one date as sorted ``HH:MM`` strings."""
    return (await get_slots_in_range(target_date, days=1))[target_date]


async def find_next_available_date(after: date, search_days: int = 14) -> date | None:
    """The first date after ``after`` that has at least one open slot."""
    slots = await get_slots_in_range(after + timedelta(days=1), days=search_days)
    for day in sorted(slots):
        if slots[day]:
            return day
    return None


# ── Create ───────────────────────────────────────────────────────────────────

def _failure(message: str, code: str) -> dict[str, Any]:
    return {
        "success": False, "booking_id": None, "booking_uid": None, "ref_code": None,
        "error_code": code, "message": message,
    }


async def create_booking(
    slot_time_ist: str,
    target_date: date,
    attendee_name: str,
    attendee_email: str,
    timezone: str | None = None,
    source: str = "voice",
) -> dict[str, Any]:
    """
    Book ``slot_time_ist`` (HH:MM, booking timezone) on ``target_date``.

    Returns a dict with ``success``, ``booking_uid``, ``ref_code``, a speakable
    ``message`` and, on failure, an ``error_code`` (``invalid_input``,
    ``slot_unavailable`` or ``provider_error``).
    """
    attendee_name = " ".join(attendee_name.split())[:100]
    attendee_email = attendee_email.strip().lower()
    timezone = timezone or config.BOOKING_TIMEZONE

    if not attendee_name:
        return _failure("A name is required to book an appointment.", "invalid_input")
    if not nlu.is_valid_email(attendee_email):
        return _failure("That email address doesn't look valid.", "invalid_input")
    try:
        hour, minute = (int(part) for part in slot_time_ist.split(":"))
        start_local = datetime(
            target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=_tz()
        )
    except ValueError:
        return _failure("That time isn't valid. Use HH:MM, for example 14:00.", "invalid_input")
    if start_local <= clock.now_local():
        return _failure("That time has already passed.", "invalid_input")

    start_utc = _to_utc_string(start_local)
    spoken_when = f"{nlu.speak_time(slot_time_ist)} on {nlu.speak_date(target_date)}"

    if is_mock_mode():
        if slot_time_ist in store.get_booked_slots(target_date.isoformat()):
            return _failure(f"{spoken_when} is no longer available.", "slot_unavailable")
        booking_uid = f"mock-{uuid.uuid4().hex[:12]}"
        booking_id = 0
        stored_response = json.dumps({"mock": True})
    else:
        payload = {
            "eventTypeId": config.CAL_EVENT_TYPE_ID,
            "start": start_utc,
            "attendee": {"name": attendee_name, "email": attendee_email, "timeZone": timezone},
        }
        try:
            resp = await _http().post(
                f"{config.CAL_BASE_URL}/bookings",
                headers=_headers(_BOOKINGS_API_VERSION, json_body=True),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.error("Cal.com booking error %s: %s", exc.response.status_code, body)
            if exc.response.status_code in (400, 409) and any(
                marker in body.lower()
                for marker in ("not available", "already", "no_available_users", "booked", "conflict")
            ):
                return _failure(f"{spoken_when} is no longer available.", "slot_unavailable")
            return _failure(
                "The calendar service rejected the booking. Please try again shortly.",
                "provider_error",
            )
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("Cal.com booking request failed: %s", exc)
            return _failure(
                "I couldn't reach the calendar service. Please try again shortly.",
                "provider_error",
            )

        booking = data.get("data") or {}
        raw_id = booking.get("id") or data.get("id") or 0
        booking_id = int(raw_id) if str(raw_id).isdigit() else 0
        booking_uid = str(booking.get("uid") or raw_id)
        # Keep just enough of the response to reconcile with Cal.com later.
        stored_response = json.dumps(
            {key: booking.get(key) for key in ("id", "uid", "status", "start", "end")}
        )

    ref_code: str | None = None
    try:
        saved = store.save_booking(
            cal_booking_uid=booking_uid,
            cal_booking_id=booking_id,
            event_type_id=config.CAL_EVENT_TYPE_ID,
            booking_date=target_date.isoformat(),
            time_ist=slot_time_ist,
            start_utc=start_utc,
            attendee_name=attendee_name,
            attendee_email=attendee_email,
            timezone=timezone,
            source=source,
            cal_response=stored_response,
        )
        ref_code = saved["ref_code"]
    except Exception:
        # The appointment exists on the calendar; a local bookkeeping failure
        # must not turn a successful booking into an error for the caller.
        logger.exception("Booking %s created but could not be saved locally", booking_uid)

    logger.info(
        "Booking created: uid=%s %s %s for %s (%s)",
        booking_uid, target_date, slot_time_ist, mask_email(attendee_email), source,
    )
    return {
        "success": True,
        "booking_id": str(booking_id),
        "booking_uid": booking_uid,
        "ref_code": ref_code,
        "error_code": None,
        "mock": is_mock_mode(),
        "message": f"Appointment confirmed for {spoken_when}.",
    }


# ── Cancel ───────────────────────────────────────────────────────────────────

async def cancel_booking(booking_uid: str, reason: str = "") -> dict[str, Any]:
    """Cancel by Cal.com booking UID and update the local record."""
    if not nlu.is_valid_booking_uid(booking_uid):
        return {"success": False, "error_code": "invalid_input", "message": "That booking reference isn't valid."}

    if is_mock_mode():
        if not store.mark_cancelled(booking_uid):
            return {
                "success": False, "error_code": "not_found",
                "message": "I couldn't find an active booking with that reference.",
            }
        return {"success": True, "error_code": None, "message": "The appointment has been cancelled."}

    try:
        resp = await _http().post(
            f"{config.CAL_BASE_URL}/bookings/{booking_uid}/cancel",
            headers=_headers(_BOOKINGS_API_VERSION, json_body=True),
            json={"cancellationReason": (reason or "Cancelled by user via AI assistant")[:200]},
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Cal.com cancel error %s: %s", exc.response.status_code, exc.response.text[:500])
        if exc.response.status_code == 404:
            return {
                "success": False, "error_code": "not_found",
                "message": "I couldn't find a booking with that reference.",
            }
        return {
            "success": False, "error_code": "provider_error",
            "message": "The calendar service couldn't cancel that booking. Please try again shortly.",
        }
    except httpx.HTTPError as exc:
        logger.error("Cal.com cancel request failed: %s", exc)
        return {
            "success": False, "error_code": "provider_error",
            "message": "I couldn't reach the calendar service. Please try again shortly.",
        }

    try:
        store.mark_cancelled(booking_uid)
    except Exception:
        logger.exception("Booking %s cancelled but the local record was not updated", booking_uid)

    logger.info("Booking cancelled: uid=%s", booking_uid)
    return {"success": True, "error_code": None, "message": "The appointment has been cancelled."}


# ── List ─────────────────────────────────────────────────────────────────────

def _local_bookings(attendee_email: str | None) -> list[dict[str, Any]]:
    return [
        {
            "uid": row["cal_booking_uid"],
            "id": row["cal_booking_id"],
            "status": row["status"],
            "date": row["date"],
            "time": row["time_ist"],
            "title": "",
            "attendee_name": row["attendee_name"],
            "attendee_email": row["attendee_email"],
        }
        for row in store.get_bookings(email=attendee_email, limit=50)
    ]


async def list_bookings(status: str = "upcoming", attendee_email: str | None = None) -> list[dict[str, Any]]:
    """Bookings from Cal.com, falling back to the local database."""
    if is_mock_mode():
        return _local_bookings(attendee_email)

    params = {"status": status}
    if attendee_email:
        params["attendeeEmail"] = attendee_email
    try:
        resp = await _http().get(
            f"{config.CAL_BASE_URL}/bookings",
            headers=_headers(_BOOKINGS_API_VERSION, json_body=True),
            params=params,
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Cal.com list bookings failed, using local records: %s", exc)
        return _local_bookings(attendee_email)

    tz = _tz()
    bookings = []
    for row in rows:
        start = row.get("start") or ""
        try:
            local = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(tz)
            day, clock = local.strftime("%Y-%m-%d"), local.strftime("%H:%M")
        except ValueError:
            day, clock = start, ""
        attendee = (row.get("attendees") or [{}])[0]
        bookings.append({
            "uid": row.get("uid", ""),
            "id": row.get("id", ""),
            "status": row.get("status", ""),
            "date": day,
            "time": clock,
            "title": row.get("title", ""),
            "attendee_name": attendee.get("name", ""),
            "attendee_email": attendee.get("email", ""),
        })
    return bookings
