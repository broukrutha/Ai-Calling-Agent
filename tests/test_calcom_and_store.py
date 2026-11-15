"""Cal.com client (against a fake HTTP transport) and the SQLite store."""

from datetime import date

import httpx
import pytest

from voice_agent import config
from voice_agent.booking import calcom, store


@pytest.fixture
def live_calendar(monkeypatch):
    """Switch the client out of mock mode and capture what it sends."""
    monkeypatch.setattr(config, "CAL_API_KEY", "cal_test_key")
    monkeypatch.setattr(config, "CAL_EVENT_TYPE_ID", 4242)
    calls: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0)

    monkeypatch.setattr(calcom, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return calls, responses


async def test_slots_are_converted_to_local_office_hours(live_calendar):
    calls, responses = live_calendar
    responses.append(httpx.Response(200, json={"data": {"2026-04-06": [
        {"start": "2026-04-06T03:30:00.000Z"},     # 09:00 IST
        {"start": "2026-04-06T08:30:00.000Z"},     # 14:00 IST
        {"start": "2026-04-06T15:30:00.000Z"},     # 21:00 IST → outside office hours
        {"start": "not-a-date"},
    ]}}))

    assert await calcom.get_available_slots(date(2026, 4, 6)) == ["09:00", "14:00"]

    request = calls[0]
    assert request.url.path == "/v2/slots"
    assert request.url.params["eventTypeId"] == "4242"
    assert request.url.params["start"] == "2026-04-05T18:30:00Z"      # midnight IST in UTC
    assert request.url.params["end"] == "2026-04-06T18:29:59Z"
    assert request.headers["authorization"] == "Bearer cal_test_key"
    assert request.headers["cal-api-version"] == "2024-09-04"


async def test_legacy_slots_shape_is_still_understood(live_calendar):
    _, responses = live_calendar
    responses.append(httpx.Response(200, json={"data": {"slots": {"2026-04-06": [
        {"time": "2026-04-06T04:30:00Z"},
    ]}}}))
    assert await calcom.get_available_slots(date(2026, 4, 6)) == ["10:00"]


async def test_calendar_outage_reads_as_no_openings(live_calendar):
    _, responses = live_calendar
    responses.append(httpx.Response(500, text="upstream exploded"))
    assert await calcom.get_available_slots(date(2026, 4, 6)) == []


async def test_create_booking_sends_expected_payload_and_saves_locally(live_calendar):
    calls, responses = live_calendar
    responses.append(httpx.Response(201, json={"status": "success", "data": {
        "id": 991, "uid": "bXyZ12abCDef34GhIj5kLm", "status": "accepted",
        "start": "2026-04-06T08:30:00Z", "end": "2026-04-06T09:30:00Z",
        "attendees": [{"name": "Rahul", "email": "rahul@example.com", "phoneNumber": "+910000000000"}],
    }}))

    result = await calcom.create_booking("14:00", date(2026, 4, 6), "Rahul", "Rahul@Example.com")

    assert result["success"] and result["booking_uid"] == "bXyZ12abCDef34GhIj5kLm"
    assert result["message"] == "Appointment confirmed for 2 PM on Monday, April 6."
    sent = __import__("json").loads(calls[0].content)
    assert sent == {
        "eventTypeId": 4242,
        "start": "2026-04-06T08:30:00Z",
        "attendee": {"name": "Rahul", "email": "rahul@example.com", "timeZone": "Asia/Kolkata"},
    }

    saved = store.get_booking_by_uid("bXyZ12abCDef34GhIj5kLm")
    assert saved["cal_booking_id"] == 991 and saved["ref_code"] == result["ref_code"]
    assert "phoneNumber" not in saved["cal_response"]      # only reconciliation fields are kept


async def test_provider_errors_are_never_spoken_verbatim(live_calendar):
    _, responses = live_calendar
    responses.append(httpx.Response(500, text='{"error":"TRPCError: secret-internal-detail"}'))
    result = await calcom.create_booking("14:00", date(2026, 4, 6), "Rahul", "rahul@example.com")
    assert not result["success"] and result["error_code"] == "provider_error"
    assert "secret-internal-detail" not in result["message"]


async def test_conflict_is_reported_as_slot_unavailable(live_calendar):
    _, responses = live_calendar
    responses.append(httpx.Response(400, json={"error": {"message": "User either already has booking at this time or is not available"}}))
    result = await calcom.create_booking("14:00", date(2026, 4, 6), "Rahul", "rahul@example.com")
    assert result["error_code"] == "slot_unavailable"


@pytest.mark.parametrize("slot,day,name,email", [
    ("14:00", date(2026, 4, 6), "", "rahul@example.com"),
    ("14:00", date(2026, 4, 6), "Rahul", "not-an-email"),
    ("25:00", date(2026, 4, 6), "Rahul", "rahul@example.com"),
    ("09:00", date(2026, 4, 1), "Rahul", "rahul@example.com"),     # earlier today
])
async def test_create_booking_validates_input(slot, day, name, email):
    result = await calcom.create_booking(slot, day, name, email)
    assert not result["success"] and result["error_code"] == "invalid_input"
    assert store.get_bookings() == []


async def test_cancel_rejects_references_that_could_alter_the_request_path(live_calendar):
    calls, _ = live_calendar
    result = await calcom.cancel_booking("../event-types/1")
    assert not result["success"] and result["error_code"] == "invalid_input"
    assert calls == []                                      # nothing was sent to Cal.com


async def test_cancel_updates_calendar_then_local_record(live_calendar):
    calls, responses = live_calendar
    store.save_booking(
        cal_booking_uid="uid12345", cal_booking_id=1, event_type_id=4242, booking_date="2026-04-06",
        time_ist="14:00", start_utc="2026-04-06T08:30:00Z", attendee_name="R", attendee_email="r@example.com",
    )
    responses.append(httpx.Response(200, json={"status": "success"}))

    result = await calcom.cancel_booking("uid12345")
    assert result["success"]
    assert calls[0].url.path == "/v2/bookings/uid12345/cancel"
    assert store.get_booking_by_uid("uid12345")["status"] == "cancelled"


async def test_mock_mode_double_booking_is_refused():
    first = await calcom.create_booking("14:00", date(2026, 4, 6), "A", "a@example.com")
    second = await calcom.create_booking("14:00", date(2026, 4, 6), "B", "b@example.com")
    assert first["success"] and first["mock"]
    assert not second["success"] and second["error_code"] == "slot_unavailable"
    assert "14:00" not in await calcom.get_available_slots(date(2026, 4, 6))


async def test_next_available_date_skips_the_weekend():
    assert await calcom.find_next_available_date(date(2026, 4, 3)) == date(2026, 4, 6)


# ── Store ────────────────────────────────────────────────────────────────────

def _save(uid, day="2026-04-06", slot="10:00", email="a@example.com"):
    return store.save_booking(
        cal_booking_uid=uid, cal_booking_id=0, event_type_id=1, booking_date=day, time_ist=slot,
        start_utc="", attendee_name="A", attendee_email=email,
    )


def test_stats_and_filters():
    _save("u1")
    _save("u2", day="2026-03-01")           # in the past
    _save("u3", email="b@example.com")
    store.mark_cancelled("u3")

    assert store.booking_stats() == {"total": 3, "confirmed": 2, "cancelled": 1, "upcoming": 1}
    assert [b["cal_booking_uid"] for b in store.get_upcoming_bookings()] == ["u1"]
    assert len(store.get_bookings(email="A@Example.com")) == 2
    assert len(store.get_bookings(status="cancelled")) == 1
    assert store.mark_cancelled("u3") is False              # already cancelled
    assert store.booking_stats(), "works on an empty result too"


def test_reference_codes_are_unique_six_digit_and_only_match_live_bookings():
    codes = {_save(f"u{i}", slot=f"{9 + i}:00")["ref_code"] for i in range(8)}
    assert len(codes) == 8 and all(len(c) == 6 and c.isdigit() for c in codes)

    past = _save("old", day="2026-03-01")
    assert store.get_booking_by_ref(past["ref_code"]) is None
    live = _save("live")
    assert store.get_booking_by_ref(live["ref_code"])["cal_booking_uid"] == "live"
    store.mark_cancelled("live")
    assert store.get_booking_by_ref(live["ref_code"]) is None


def test_existing_database_without_ref_code_column_is_migrated(tmp_path, monkeypatch):
    import sqlite3

    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE bookings (id INTEGER PRIMARY KEY AUTOINCREMENT, cal_booking_uid TEXT UNIQUE, "
        "cal_booking_id INTEGER, event_type_id INTEGER, date TEXT NOT NULL, time_ist TEXT NOT NULL, "
        "start_utc TEXT, attendee_name TEXT NOT NULL, attendee_email TEXT NOT NULL, timezone TEXT, "
        "status TEXT DEFAULT 'confirmed', source TEXT, cal_response TEXT, created_at TEXT, "
        "updated_at TEXT, cancelled_at TEXT)"
    )
    conn.execute(
        "INSERT INTO bookings (cal_booking_uid, date, time_ist, attendee_name, attendee_email) "
        "VALUES ('legacy-1', '2026-04-06', '09:00', 'Old', 'old@example.com')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(config, "BOOKINGS_DB_PATH", str(legacy))
    assert store.get_booking_by_uid("legacy-1")["ref_code"] is None    # old row kept, column added
    assert _save("new-1")["ref_code"]


def test_call_log_lifecycle():
    store.record_call("room-1", "outbound", "+919988776655", status="ringing")
    store.record_call("room-1", "outbound", status="ringing")           # re-dispatch keeps the number
    store.mark_call_answered("room-1")
    store.finish_call("room-1", "completed", turns=4)
    store.finish_call("room-1", "failed", detail="late duplicate")       # ignored: already closed

    [call] = store.get_calls()
    assert (call["phone"], call["status"], call["turns"]) == ("+919988776655", "completed", 4)
    assert call["duration_sec"] == 0 and call["ended_at"]

    store.record_call("room-2", "inbound", "+911111111111")
    store.finish_call("room-2", "failed", detail="no answer")
    stats = store.call_stats()
    assert (stats["total"], stats["inbound"], stats["outbound"], stats["failed"]) == (2, 1, 1, 1)
