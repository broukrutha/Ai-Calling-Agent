from __future__ import annotations

import asyncio
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from voice_agent.booking import calcom, handle_booking_query, store
from voice_agent.server.schemas import BookingCancelRequest, BookingCreateRequest, BookingIntentRequest
from voice_agent.server.security import rate_limit, require_admin

router = APIRouter(prefix="/booking", tags=["Booking"])

_PUBLIC_FIELDS = (
    "id", "cal_booking_uid", "ref_code", "date", "time_ist", "attendee_name", "attendee_email",
    "timezone", "status", "source", "created_at", "cancelled_at",
)
_STATUS_FOR_ERROR = {"invalid_input": 400, "slot_unavailable": 409, "not_found": 404, "provider_error": 502}


def _public(row: dict) -> dict:
    return {key: row.get(key) for key in _PUBLIC_FIELDS}


def _raise_for(result: dict) -> None:
    if not result["success"]:
        raise HTTPException(
            status_code=_STATUS_FOR_ERROR.get(result.get("error_code"), 400), detail=result["message"]
        )


@router.get("/slots", dependencies=[Depends(rate_limit())])
async def slots(day: date = Query(alias="date", description="YYYY-MM-DD")) -> dict:
    """Open appointment times on a date, as HH:MM in the booking timezone."""
    return {"status": "ok", "date": day.isoformat(), "available_slots": await calcom.get_available_slots(day)}


@router.post("/intent", dependencies=[Depends(rate_limit("expensive"))])
async def intent(req: BookingIntentRequest) -> dict:
    """
    One-shot natural-language booking: "Book me in tomorrow at 2 PM".
    Books immediately when the request includes a name and an email.
    """
    name = "" if req.name.strip() == "User" else req.name.strip()
    email = "" if req.email.strip() == "user@example.com" else req.email.strip()
    result = await handle_booking_query(req.query, session=None, attendee_name=name, attendee_email=email)
    return {
        "status": "ok",
        "response": result.text,
        "booking_completed": result.completed,
        "ref_code": result.ref_code,
    }


@router.post("/create", dependencies=[Depends(require_admin)])
async def create(req: BookingCreateRequest) -> dict:
    result = await calcom.create_booking(
        slot_time_ist=req.time, target_date=req.date,
        attendee_name=req.name, attendee_email=req.email, source="ui",
    )
    _raise_for(result)
    return {"status": "ok", **result}


@router.post("/cancel", dependencies=[Depends(require_admin)])
async def cancel(req: BookingCancelRequest) -> dict:
    result = await calcom.cancel_booking(req.booking_uid, reason=req.reason)
    _raise_for(result)
    return {"status": "ok", **result}


@router.get("/list", dependencies=[Depends(require_admin)])
async def list_bookings(
    status: str = Query("upcoming", pattern="^(upcoming|past|cancelled|recurring|unconfirmed)$"),
    email: str | None = Query(None, max_length=254),
) -> dict:
    bookings = await calcom.list_bookings(status=status, attendee_email=email)
    return {"status": "ok", "bookings": bookings, "count": len(bookings)}


@router.get("/history", dependencies=[Depends(require_admin)])
async def history(
    email: str | None = Query(None, max_length=254),
    status: str | None = Query(None, pattern="^(confirmed|cancelled)$"),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    rows = await asyncio.to_thread(
        store.get_bookings,
        email=email, status=status,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
        limit=limit,
    )
    return {"status": "ok", "bookings": [_public(row) for row in rows], "count": len(rows)}


@router.get("/stats", dependencies=[Depends(require_admin)])
async def stats() -> dict:
    summary = await asyncio.to_thread(store.booking_stats)
    upcoming = await asyncio.to_thread(store.get_upcoming_bookings, 5)
    return {
        "status": "ok",
        "stats": {**summary, "mode": "mock" if calcom.is_mock_mode() else "cal.com"},
        "next_appointments": [_public(row) for row in upcoming],
    }
