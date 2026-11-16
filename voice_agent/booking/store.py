"""
SQLite persistence for bookings and phone-call records.

One small database file holds everything the dashboard needs to show. The
schema is created (and migrated) on first use, so there is nothing to set up.

Connections are per-thread and use WAL mode, which lets the web server and
the telephony worker (a separate process) share the same file safely.
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from voice_agent import clock, config
from voice_agent.logging_config import mask_email, mask_phone

logger = logging.getLogger(__name__)

_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    cal_booking_uid   TEXT    UNIQUE,
    cal_booking_id    INTEGER,
    event_type_id     INTEGER,
    date              TEXT    NOT NULL,                 -- YYYY-MM-DD (booking timezone)
    time_ist          TEXT    NOT NULL,                 -- HH:MM      (booking timezone)
    start_utc         TEXT,                             -- ISO-8601 UTC
    attendee_name     TEXT    NOT NULL,
    attendee_email    TEXT    NOT NULL,
    timezone          TEXT    DEFAULT 'Asia/Kolkata',
    status            TEXT    DEFAULT 'confirmed',      -- confirmed | cancelled
    source            TEXT    DEFAULT 'voice',          -- voice | chat | phone | ui | api
    cal_response      TEXT,
    created_at        TEXT    DEFAULT (datetime('now')),
    updated_at        TEXT    DEFAULT (datetime('now')),
    cancelled_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_bookings_email  ON bookings(attendee_email);
CREATE INDEX IF NOT EXISTS idx_bookings_date   ON bookings(date);
CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);

CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    room          TEXT    UNIQUE NOT NULL,
    direction     TEXT    NOT NULL,                     -- inbound | outbound
    phone         TEXT    NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'queued',    -- queued | ringing | in_progress | completed | failed
    detail        TEXT    NOT NULL DEFAULT '',
    turns         INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    DEFAULT (datetime('now')),
    answered_at   TEXT,
    ended_at      TEXT,
    duration_sec  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_calls_created ON calls(created_at);
"""

# Columns added after the first release: (table, column, DDL).
_MIGRATIONS = [
    ("bookings", "ref_code", "ALTER TABLE bookings ADD COLUMN ref_code TEXT"),
]


def _db_path() -> str:
    return os.path.abspath(config.BOOKINGS_DB_PATH)


def _today() -> str:
    return clock.now_local().strftime("%Y-%m-%d")


def _open(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    for table, column, ddl in _MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(ddl)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_ref ON bookings(ref_code) "
        "WHERE ref_code IS NOT NULL"
    )
    conn.commit()
    logger.info("Booking database opened: %s", path)
    return conn


@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    """Thread-local connection, committed on success and rolled back on error."""
    path = _db_path()
    if getattr(_local, "path", None) != path:
        close()
        _local.conn = _open(path)
        _local.path = path
    conn: sqlite3.Connection = _local.conn
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close() -> None:
    """Close this thread's connection (used at shutdown and between tests)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn = None
    _local.path = None


# ── Bookings: write ──────────────────────────────────────────────────────────

def _new_ref_code(conn: sqlite3.Connection) -> str:
    """Six digits: short enough to say on a phone call, unique among bookings."""
    for _ in range(20):
        code = f"{secrets.randbelow(900_000) + 100_000}"
        if not conn.execute("SELECT 1 FROM bookings WHERE ref_code = ?", (code,)).fetchone():
            return code
    raise RuntimeError("Could not allocate a unique booking reference code.")


def save_booking(
    *,
    cal_booking_uid: str,
    cal_booking_id: int,
    event_type_id: int,
    booking_date: str,
    time_ist: str,
    start_utc: str,
    attendee_name: str,
    attendee_email: str,
    timezone: str = "Asia/Kolkata",
    source: str = "voice",
    cal_response: str = "",
) -> dict[str, Any]:
    """Persist a new booking. Returns ``{"id": ..., "ref_code": ...}``."""
    with _get_conn() as conn:
        ref_code = _new_ref_code(conn)
        cursor = conn.execute(
            """
            INSERT INTO bookings
                (cal_booking_uid, cal_booking_id, event_type_id, date, time_ist, start_utc,
                 attendee_name, attendee_email, timezone, source, cal_response, ref_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cal_booking_uid, cal_booking_id, event_type_id, booking_date, time_ist, start_utc,
                attendee_name, attendee_email, timezone, source, cal_response, ref_code,
            ),
        )
        logger.info(
            "Booking saved: id=%s ref=%s %s %s %s",
            cursor.lastrowid, ref_code, booking_date, time_ist, mask_email(attendee_email),
        )
        return {"id": cursor.lastrowid, "ref_code": ref_code}


def mark_cancelled(cal_booking_uid: str) -> bool:
    """Flag a booking as cancelled. Returns True when a confirmed row was updated."""
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            UPDATE bookings
               SET status = 'cancelled', cancelled_at = datetime('now'), updated_at = datetime('now')
             WHERE cal_booking_uid = ? AND status = 'confirmed'
            """,
            (cal_booking_uid,),
        )
        return cursor.rowcount > 0


# ── Bookings: read ───────────────────────────────────────────────────────────

def get_bookings(
    email: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Bookings matching the filters, newest appointment first."""
    query = "SELECT * FROM bookings WHERE 1=1"
    params: list[Any] = []
    if email:
        query += " AND attendee_email = ?"
        params.append(email.lower())
    if status:
        query += " AND status = ?"
        params.append(status)
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    query += " ORDER BY date DESC, time_ist DESC LIMIT ?"
    params.append(limit)

    with _get_conn() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_booking_by_uid(cal_booking_uid: str) -> dict[str, Any] | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM bookings WHERE cal_booking_uid = ?", (cal_booking_uid,)
        ).fetchone()
        return dict(row) if row else None


def get_booking_by_ref(ref_code: str) -> dict[str, Any] | None:
    """Look up an upcoming, still-confirmed booking by its 6-digit reference."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM bookings WHERE ref_code = ? AND status = 'confirmed' AND date >= ?",
            (ref_code, _today()),
        ).fetchone()
        return dict(row) if row else None


def get_booked_slots(booking_date: str) -> list[str]:
    """HH:MM start times of confirmed bookings on a date (used by mock mode)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT time_ist FROM bookings WHERE date = ? AND status = 'confirmed'",
            (booking_date,),
        ).fetchall()
        return [row["time_ist"] for row in rows]


def get_upcoming_bookings(limit: int = 10) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM bookings
             WHERE status = 'confirmed' AND date >= ?
             ORDER BY date ASC, time_ist ASC
             LIMIT ?
            """,
            (_today(), limit),
        ).fetchall()
        return [dict(row) for row in rows]


def booking_stats() -> dict[str, int]:
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)                                                     AS total,
                   COALESCE(SUM(status = 'confirmed'), 0)                       AS confirmed,
                   COALESCE(SUM(status = 'cancelled'), 0)                       AS cancelled,
                   COALESCE(SUM(status = 'confirmed' AND date >= ?), 0)         AS upcoming
              FROM bookings
            """,
            (_today(),),
        ).fetchone()
        return dict(row)


# ── Calls ────────────────────────────────────────────────────────────────────

def record_call(room: str, direction: str, phone: str = "", status: str = "queued") -> None:
    """Create the call row, or update phone/status if the room is already known."""
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO calls (room, direction, phone, status) VALUES (?, ?, ?, ?)
            ON CONFLICT(room) DO UPDATE SET
                status = excluded.status,
                phone  = CASE WHEN excluded.phone != '' THEN excluded.phone ELSE calls.phone END
            """,
            (room, direction, phone, status),
        )
    logger.info("Call %s: %s %s (%s)", status, direction, mask_phone(phone) or "unknown", room)


def mark_call_answered(room: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE calls SET status = 'in_progress', answered_at = datetime('now') WHERE room = ?",
            (room,),
        )


def finish_call(room: str, status: str = "completed", detail: str = "", turns: int = 0) -> None:
    """Close out a call. Duration is measured from pickup, or 0 if never answered."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE calls
               SET status = ?, detail = ?, turns = ?, ended_at = datetime('now'),
                   duration_sec = CASE
                       WHEN answered_at IS NULL THEN 0
                       ELSE CAST(strftime('%s', 'now') - strftime('%s', answered_at) AS INTEGER)
                   END
             WHERE room = ? AND ended_at IS NULL
            """,
            (status, detail[:300], turns, room),
        )


def get_calls(limit: int = 50) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def call_stats() -> dict[str, int]:
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)                                         AS total,
                   COALESCE(SUM(direction = 'inbound'), 0)          AS inbound,
                   COALESCE(SUM(direction = 'outbound'), 0)         AS outbound,
                   COALESCE(SUM(status = 'completed'), 0)           AS completed,
                   COALESCE(SUM(status = 'failed'), 0)              AS failed,
                   COALESCE(SUM(duration_sec), 0)                   AS total_duration_sec
              FROM calls
            """
        ).fetchone()
        return dict(row)
