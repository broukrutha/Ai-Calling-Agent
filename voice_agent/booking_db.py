"""
booking_db.py – SQLite-backed persistent storage for bookings.

Stores every booking created through the system (voice, text, or UI)
alongside the Cal.com booking UID so we can cancel / query later.

The DB file lives next to the ChromaDB directory for consistency.
Schema is auto-created on first access (zero setup required).
"""

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from voice_agent.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

# DB lives alongside ChromaDB — keeps all persistent data in one place
_DB_DIR = os.path.dirname(os.path.abspath(CHROMA_PERSIST_DIR))
DB_PATH = os.path.join(_DB_DIR, "bookings.db")

# Thread-local storage so each thread gets its own connection
_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    cal_booking_uid   TEXT    UNIQUE,
    cal_booking_id    INTEGER,
    event_type_id     INTEGER,
    date              TEXT    NOT NULL,                 -- YYYY-MM-DD
    time_ist          TEXT    NOT NULL,                 -- HH:MM
    start_utc         TEXT,                             -- ISO-8601 UTC
    attendee_name     TEXT    NOT NULL,
    attendee_email    TEXT    NOT NULL,
    timezone          TEXT    DEFAULT 'Asia/Kolkata',
    status            TEXT    DEFAULT 'confirmed',      -- confirmed | cancelled | rescheduled
    source            TEXT    DEFAULT 'voice',          -- voice | text | ui | api
    cal_response      TEXT,                             -- raw Cal.com JSON (for debugging)
    created_at        TEXT    DEFAULT (datetime('now')),
    updated_at        TEXT    DEFAULT (datetime('now')),
    cancelled_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_bookings_email  ON bookings(attendee_email);
CREATE INDEX IF NOT EXISTS idx_bookings_date   ON bookings(date);
CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);
CREATE INDEX IF NOT EXISTS idx_bookings_uid    ON bookings(cal_booking_uid);
"""


@contextmanager
def _get_conn():
    """
    Get a thread-local SQLite connection with WAL mode for concurrent reads.
    Each thread gets its own connection to avoid SQLite threading issues.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        _local.conn = conn
        logger.info(f"BookingDB connection opened: {DB_PATH}")

    try:
        yield _local.conn
        _local.conn.commit()
    except Exception:
        _local.conn.rollback()
        raise


# ──────────────────────────────────────────────
# Write operations
# ──────────────────────────────────────────────

def save_booking(
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
) -> int:
    """
    Persist a new booking to the local database.
    Returns the auto-incremented local ID.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO bookings
                (cal_booking_uid, cal_booking_id, event_type_id,
                 date, time_ist, start_utc,
                 attendee_name, attendee_email, timezone,
                 source, cal_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cal_booking_uid, cal_booking_id, event_type_id,
                booking_date, time_ist, start_utc,
                attendee_name, attendee_email, timezone,
                source, cal_response,
            ),
        )
        local_id = cursor.lastrowid
        logger.info(
            f"Booking saved: local_id={local_id} uid={cal_booking_uid} "
            f"{booking_date} {time_ist} {attendee_name}"
        )
        return local_id


def mark_cancelled(cal_booking_uid: str) -> bool:
    """
    Mark a booking as cancelled in the local database.
    Returns True if a row was updated.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            UPDATE bookings
               SET status       = 'cancelled',
                   cancelled_at = datetime('now'),
                   updated_at   = datetime('now')
             WHERE cal_booking_uid = ?
               AND status = 'confirmed'
            """,
            (cal_booking_uid,),
        )
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Booking cancelled: uid={cal_booking_uid}")
        else:
            logger.warning(f"No confirmed booking found for uid={cal_booking_uid}")
        return updated


# ──────────────────────────────────────────────
# Read operations
# ──────────────────────────────────────────────

def get_bookings(
    email: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Retrieve bookings with optional filters.
    Returns list of booking dicts, newest first.
    """
    query = "SELECT * FROM bookings WHERE 1=1"
    params: list = []

    if email:
        query += " AND attendee_email = ?"
        params.append(email)
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
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_booking_by_uid(cal_booking_uid: str) -> Optional[dict]:
    """Retrieve a single booking by its Cal.com UID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM bookings WHERE cal_booking_uid = ?",
            (cal_booking_uid,),
        ).fetchone()
        return dict(row) if row else None


def get_bookings_count(status: Optional[str] = None) -> int:
    """Return total count of bookings, optionally filtered by status."""
    if status:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM bookings WHERE status = ?",
                (status,),
            ).fetchone()
    else:
        with _get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM bookings").fetchone()
    return row["cnt"] if row else 0


def get_upcoming_bookings(limit: int = 10) -> list[dict]:
    """Get confirmed bookings from today onwards."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM bookings
             WHERE status = 'confirmed'
               AND date >= ?
             ORDER BY date ASC, time_ist ASC
             LIMIT ?
            """,
            (today, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def booking_stats() -> dict:
    """Return summary stats for the admin panel."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM bookings").fetchone()["c"]
        confirmed = conn.execute(
            "SELECT COUNT(*) as c FROM bookings WHERE status='confirmed'"
        ).fetchone()["c"]
        cancelled = conn.execute(
            "SELECT COUNT(*) as c FROM bookings WHERE status='cancelled'"
        ).fetchone()["c"]
        today = datetime.now().strftime("%Y-%m-%d")
        upcoming = conn.execute(
            "SELECT COUNT(*) as c FROM bookings WHERE status='confirmed' AND date >= ?",
            (today,),
        ).fetchone()["c"]

    return {
        "total": total,
        "confirmed": confirmed,
        "cancelled": cancelled,
        "upcoming": upcoming,
        "db_path": DB_PATH,
    }
