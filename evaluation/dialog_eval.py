"""
Booking dialog evaluation: does a scripted conversation end in the right state?

Each scenario plays a list of caller turns against the real booking dialog,
backed by the mock calendar in a throw-away database, then checks the outcome:
the right appointment booked (or deliberately not booked), or cancelled.

Scenarios use weekday names ("next monday"), so the expected date is always a
future weekday and the suite gives the same result any day it runs.

    {
      "id": "book-step-by-step",
      "setup": [{"day": "next monday", "slot": "14:00"}],     optional existing bookings
      "turns": ["I'd like an appointment", "next monday", "2 pm", ...],
      "expect": {"booked": true, "day": "next monday", "slot": "14:00", "email": "..."}
    }

``{ref}`` in a turn is replaced with the reference number of the first setup booking.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

from evaluation.dataset import DATASETS_DIR

DEFAULT_SCENARIOS = DATASETS_DIR / "dialogs.json"


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        scenarios = json.load(handle)
    seen: set[str] = set()
    for scenario in scenarios:
        if not scenario.get("id") or not scenario.get("turns") or "expect" not in scenario:
            raise ValueError(f"scenario needs id, turns and expect: {scenario!r}")
        if scenario["id"] in seen:
            raise ValueError(f"duplicate scenario id {scenario['id']!r}")
        seen.add(scenario["id"])
    return scenarios


async def _run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    from voice_agent import clock
    from voice_agent.booking import calcom, nlu, store
    from voice_agent.booking.dialog import BookingSession, handle_booking_query

    today = clock.now_local().date()
    ref_code = ""
    for booking in scenario.get("setup", []):
        day = nlu.parse_date(booking["day"], today)
        made = await calcom.create_booking(booking["slot"], day, "Setup Person", "setup@example.com", source="eval")
        ref_code = ref_code or (made.get("ref_code") or "")

    session = BookingSession(source="eval")
    transcript = []
    completed_turn = None
    for number, text in enumerate(scenario["turns"], start=1):
        text = text.replace("{ref}", " ".join(ref_code))
        result = await handle_booking_query(text, session)
        transcript.append({"user": text, "agent": result.text})
        if result.completed and completed_turn is None:
            completed_turn = number

    expect = scenario["expect"]
    bookings = [row for row in store.get_bookings(limit=100) if row["source"] == "eval"]
    problems: list[str] = []
    if "booked" in expect:
        booked = [row for row in bookings if row["status"] == "confirmed"]
        if expect["booked"]:
            if not booked:
                problems.append("no appointment was booked")
            else:
                row = booked[0]
                if "day" in expect and row["date"] != nlu.parse_date(expect["day"], today).isoformat():
                    problems.append(f"booked {row['date']}, expected {expect['day']}")
                if "slot" in expect and row["time_ist"] != expect["slot"]:
                    problems.append(f"booked {row['time_ist']}, expected {expect['slot']}")
                if "email" in expect and row["attendee_email"] != expect["email"]:
                    problems.append(f"email {row['attendee_email']!r}, expected {expect['email']!r}")
                if "name" in expect and row["attendee_name"] != expect["name"]:
                    problems.append(f"name {row['attendee_name']!r}, expected {expect['name']!r}")
        elif booked:
            problems.append("an appointment was booked but should not have been")
    if "cancelled" in expect:
        setup_rows = [row for row in store.get_bookings(limit=100) if row["attendee_email"] == "setup@example.com"]
        cancelled = any(row["status"] == "cancelled" for row in setup_rows)
        if cancelled != expect["cancelled"]:
            problems.append("the appointment was not cancelled" if expect["cancelled"] else "cancelled unexpectedly")
    if expect.get("final_state") and session.state.value != expect["final_state"]:
        problems.append(f"ended in state {session.state.value}, expected {expect['final_state']}")

    return {
        "id": scenario["id"],
        "description": scenario.get("description", ""),
        "passed": not problems,
        "problems": problems,
        "turns": len(scenario["turns"]),
        "completed_turn": completed_turn,
        "transcript": transcript,
    }


async def evaluate_dialogs(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run every scenario in its own empty database against the mock calendar.
    Both are scoped to this task, so a live server keeps using its real ones.
    """
    from voice_agent.booking import calcom, store

    rows = []
    with tempfile.TemporaryDirectory() as workdir, calcom.mock_calendar():
        for position, scenario in enumerate(scenarios):
            with store.use_database(str(Path(workdir) / f"dialog-{position}.db")):
                rows.append(await _run_scenario(scenario))
                store.close()

    completed = [row["completed_turn"] for row in rows if row["completed_turn"]]
    return {
        "suite": "dialog",
        "cases": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "success_rate": round(sum(1 for row in rows if row["passed"]) / len(rows), 4) if rows else None,
        "avg_turns_to_complete": round(statistics.mean(completed), 2) if completed else None,
        "failed": [row["id"] for row in rows if not row["passed"]],
        "rows": rows,
    }
