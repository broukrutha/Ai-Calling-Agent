"""
Multi-turn booking and cancellation dialog.

A small explicit state machine. Each turn takes the user's words plus the
:class:`BookingSession` for that conversation and returns what to say next.

    idle ──► need_date ──► need_time ──► need_name ──► need_email ──► confirming ──► idle
                 ▲             │                                          │
                 └─ offer_date ┘ (no openings: suggest the next day)      └─► booked

    idle ──► cancel_need_ref ──► cancel_confirm ──► idle

Design rules:

* Nothing is ever booked or cancelled without an explicit "yes".
* The assistant never invents a date or a time; if it wasn't said, it asks.
* A refusal always beats an agreement in the same sentence.
* After three replies in a row that can't be understood, the flow resets
  instead of looping forever.
"""

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass, field
from datetime import date, timedelta
from datetime import time as dtime
from enum import Enum
from typing import NamedTuple

from voice_agent import clock, config
from voice_agent.booking import calcom, nlu, store

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class State(str, Enum):
    IDLE = "idle"
    NEED_DATE = "need_date"
    NEED_TIME = "need_time"
    OFFER_DATE = "offer_date"
    NEED_NAME = "need_name"
    NEED_EMAIL = "need_email"
    CONFIRMING = "confirming"
    CANCEL_NEED_REF = "cancel_need_ref"
    CANCEL_CONFIRM = "cancel_confirm"


class DialogResult(NamedTuple):
    """What to say, and whether a booking or cancellation was just completed."""

    text: str
    completed: bool = False
    booking_uid: str | None = None
    ref_code: str | None = None


@dataclass
class BookingSession:
    """Conversation state for one caller / chat connection."""

    state: State = State.IDLE
    req_date: date | None = None
    req_slot: str = ""                       # "HH:MM"
    period: str | None = None                # preferred part of day, if any
    offered_slots: list[str] = field(default_factory=list)
    suggested_date: date | None = None
    attendee_name: str = ""
    attendee_email: str = ""
    cancel_uid: str = ""
    retries: int = 0
    source: str = "voice"
    one_shot: bool = False                   # REST call with no follow-up turns
    touched_at: float = field(default_factory=time_module.monotonic)

    def reset(self) -> None:
        """Back to idle. Who the caller is and where they came from is kept."""
        keep = (self.source, self.one_shot)
        self.__init__()  # type: ignore[misc]
        self.source, self.one_shot = keep

    def is_active(self) -> bool:
        """True while a flow is in progress and has not gone stale."""
        if self.state is State.IDLE:
            return False
        if time_module.monotonic() - self.touched_at > config.BOOKING_SESSION_TTL_SEC:
            logger.info("Booking session expired in state %s", self.state.value)
            self.reset()
            return False
        return True


# ── Public entry point ───────────────────────────────────────────────────────

async def handle_booking_query(
    user_text: str,
    session: BookingSession | None = None,
    attendee_name: str = "",
    attendee_email: str = "",
) -> DialogResult:
    """
    Advance the booking dialog by one turn.

    Pass the conversation's :class:`BookingSession` for multi-turn use. With no
    session (REST), a throw-away one is created: the request must then carry
    everything needed, and a known name and email count as consent to book.
    """
    if session is None:
        session = BookingSession(source="api", one_shot=True)
    session.is_active()   # expires stale flows
    session.touched_at = time_module.monotonic()

    if attendee_name and not session.attendee_name:
        session.attendee_name = " ".join(attendee_name.split())[:100]
    if attendee_email and not session.attendee_email and nlu.is_valid_email(attendee_email.lower()):
        session.attendee_email = attendee_email.strip().lower()

    handler = _HANDLERS[session.state]
    result = await handler(session, user_text)
    logger.info("Booking dialog → state=%s completed=%s", session.state.value, result.completed)
    return result


# ── Shared helpers ───────────────────────────────────────────────────────────

def _retry(session: BookingSession, prompt: str) -> DialogResult:
    """Ask again, or give up politely after too many misunderstandings."""
    session.retries += 1
    if session.retries >= MAX_RETRIES:
        session.reset()
        return DialogResult(
            "I'm sorry, I'm having trouble understanding. Let's start over — "
            "just tell me when you'd like to book or cancel an appointment."
        )
    return DialogResult(prompt)


def _abort(session: BookingSession) -> DialogResult:
    session.reset()
    return DialogResult("No problem, I've left everything as it was. Anything else I can help with?")


def _office_hours() -> str:
    start = nlu.speak_time(f"{config.OFFICE_START_HOUR:02d}:00")
    end = nlu.speak_time(f"{config.OFFICE_END_HOUR % 24:02d}:00")
    return f"{start} to {end}"


def _nearest(slots: list[str], wanted: str) -> list[str]:
    """Up to two open slots closest to the requested time (next one first)."""
    later = [slot for slot in slots if slot > wanted]
    earlier = [slot for slot in slots if slot < wanted]
    return ([later[0]] if later else []) + ([earlier[-1]] if earlier else [])


def _slot_to_time(slot: str) -> dtime | None:
    if not slot:
        return None
    hour, minute = (int(part) for part in slot.split(":"))
    return dtime(hour, minute)


def _confirmation_prompt(session: BookingSession) -> str:
    assert session.req_date is not None
    return (
        f"Just to confirm: {nlu.speak_time(session.req_slot)} on {nlu.speak_date(session.req_date)}, "
        f"for {session.attendee_name}, with the confirmation going to {session.attendee_email}. "
        "Shall I book it?"
    )


# ── Booking flow ─────────────────────────────────────────────────────────────

async def _choose_date(session: BookingSession, wanted: date, parsed: nlu.ParsedRequest) -> DialogResult:
    """A date is known: validate it, look up openings and move the flow on."""
    today = clock.now_local().date()
    if wanted < today:
        session.state = State.NEED_DATE
        return _retry(session, f"{nlu.speak_date(wanted)} has already passed. What other day works for you?")
    if wanted > today + timedelta(days=config.BOOKING_MAX_DAYS_AHEAD):
        session.state = State.NEED_DATE
        return _retry(
            session,
            f"I can only book up to {config.BOOKING_MAX_DAYS_AHEAD} days ahead. "
            "Which earlier day would you like?",
        )

    session.period = parsed.period or session.period
    slots = await calcom.get_available_slots(wanted)

    if not slots:
        next_date = await calcom.find_next_available_date(wanted)
        if next_date is None:
            session.reset()
            return DialogResult(
                f"I'm sorry, there are no openings on {nlu.speak_date(wanted)} or in the "
                "following two weeks. Please try again later."
            )
        session.state = State.OFFER_DATE
        session.suggested_date = next_date
        if parsed.time:
            session.req_slot = parsed.time.strftime("%H:%M")
        return DialogResult(
            f"There are no openings on {nlu.speak_date(wanted)}. "
            f"The next available day is {nlu.speak_date(next_date)}. Shall I look at that day?"
        )

    session.req_date = wanted
    session.retries = 0
    if parsed.time is not None:
        return await _choose_time(session, parsed.time.strftime("%H:%M"), slots)

    preferred = nlu.slots_in_period(slots, session.period)
    offered = preferred or slots
    session.state = State.NEED_TIME
    session.offered_slots = offered[:4]
    if session.period and not preferred:
        lead = f"There's nothing in the {session.period} on {nlu.speak_date(wanted)}, but I have "
    else:
        lead = f"On {nlu.speak_date(wanted)} I have "
    return DialogResult(f"{lead}{nlu.speak_slot_list(offered)}. What time works for you?")


async def _choose_time(session: BookingSession, wanted: str, slots: list[str]) -> DialogResult:
    """A time is known: check it against the openings for the chosen date."""
    assert session.req_date is not None
    day = nlu.speak_date(session.req_date)
    hour = int(wanted[:2])

    if not config.OFFICE_START_HOUR <= hour < config.OFFICE_END_HOUR:
        session.state = State.NEED_TIME
        session.offered_slots = slots[:4]
        return DialogResult(
            f"We take appointments from {_office_hours()}. "
            f"On {day} I have {nlu.speak_slot_list(slots)}. What time works for you?"
        )

    if wanted not in slots:
        nearest = _nearest(slots, wanted)
        session.state = State.NEED_TIME
        session.offered_slots = nearest
        options = " or ".join(nlu.speak_time(slot) for slot in nearest)
        return DialogResult(
            f"{nlu.speak_time(wanted)} on {day} isn't available. "
            f"The closest open {'time is' if len(nearest) == 1 else 'times are'} {options}. "
            "Which would you like?"
        )

    session.req_slot = wanted
    session.offered_slots = []
    session.retries = 0
    when = f"{nlu.speak_time(wanted)} on {day}"

    if session.attendee_name and session.attendee_email:
        if session.one_shot:
            return await _do_booking(session)
        session.state = State.CONFIRMING
        return DialogResult(_confirmation_prompt(session))
    if session.attendee_name:
        session.state = State.NEED_EMAIL
        return DialogResult(f"{when} is available. What email should I send the confirmation to?")
    session.state = State.NEED_NAME
    return DialogResult(f"{when} is available. May I have your name, please?")


async def _handle_idle(session: BookingSession, text: str) -> DialogResult:
    if nlu.is_cancel_intent(text):
        return await _start_cancel(session, text)

    parsed = nlu.parse_booking_request(text)
    if parsed.date is None:
        session.state = State.NEED_DATE
        session.period = parsed.period
        if parsed.time:
            session.req_slot = parsed.time.strftime("%H:%M")
        return DialogResult("Sure, I can help you book an appointment. What day works for you?")
    return await _choose_date(session, parsed.date, parsed)


async def _handle_need_date(session: BookingSession, text: str) -> DialogResult:
    parsed = nlu.parse_booking_request(text)
    if parsed.date is None:
        if nlu.is_abort(text) or nlu.classify_confirmation(text) is False:
            return _abort(session)
        return _retry(
            session,
            "Sorry, I didn't catch the day. You can say something like tomorrow, "
            "next Monday, or the 5th of April.",
        )
    if parsed.time is None:
        # A time given earlier ("book me at 3 pm") still applies.
        parsed = nlu.ParsedRequest(parsed.date, _slot_to_time(session.req_slot), parsed.period)
    return await _choose_date(session, parsed.date, parsed)


async def _handle_need_time(session: BookingSession, text: str) -> DialogResult:
    assert session.req_date is not None

    slot = nlu.pick_offered_slot(text, session.offered_slots)       # "the first one"
    parsed = nlu.parse_booking_request(text)
    if slot is None and parsed.date is not None and parsed.date != session.req_date:
        return await _choose_date(session, parsed.date, parsed)     # "actually, make it Tuesday"

    if slot is None:
        chosen = parsed.time or nlu.parse_time(text, allow_bare_number=True)
        if chosen is not None:
            slot = chosen.strftime("%H:%M")
        elif len(session.offered_slots) == 1 and nlu.classify_confirmation(text):
            slot = session.offered_slots[0]

    if slot is None and (nlu.is_abort(text) or nlu.classify_confirmation(text) is False):
        return _abort(session)

    slots = await calcom.get_available_slots(session.req_date)
    if not slots:
        return await _choose_date(session, session.req_date, nlu.ParsedRequest())

    if slot is None:
        period = nlu.parse_period(text)
        if period:
            session.period = period
            preferred = nlu.slots_in_period(slots, period)
            if preferred:
                session.offered_slots = preferred[:4]
                return DialogResult(
                    f"In the {period} I have {nlu.speak_slot_list(preferred)}. Which time would you like?"
                )
        return _retry(
            session,
            f"Sorry, I didn't catch the time. On {nlu.speak_date(session.req_date)} I have "
            f"{nlu.speak_slot_list(slots)}. Which works for you?",
        )
    return await _choose_time(session, slot, slots)


async def _handle_offer_date(session: BookingSession, text: str) -> DialogResult:
    if session.suggested_date is None:
        session.reset()
        return DialogResult("Sorry, something went wrong. Please tell me again when you'd like to book.")

    parsed = nlu.parse_booking_request(text)
    carried_time = _slot_to_time(session.req_slot)

    if parsed.date is not None:   # "no, try the 7th instead"
        session.req_slot = ""
        return await _choose_date(
            session, parsed.date, nlu.ParsedRequest(parsed.date, parsed.time or carried_time, parsed.period)
        )

    answer = nlu.classify_confirmation(text)
    if answer:
        suggested = session.suggested_date
        session.suggested_date = None
        session.req_slot = ""
        return await _choose_date(
            session, suggested, nlu.ParsedRequest(suggested, parsed.time or carried_time, parsed.period)
        )
    if answer is False or nlu.is_abort(text):
        session.reset()
        return DialogResult("No problem. Let me know whenever you'd like to book.")
    return _retry(
        session,
        f"Sorry, I didn't catch that. Shall I look at {nlu.speak_date(session.suggested_date)}, "
        "or would you prefer a different day?",
    )


async def _handle_need_name(session: BookingSession, text: str) -> DialogResult:
    if nlu.is_abort(text):
        return _abort(session)
    name = nlu.extract_name(text)
    if not name:
        return _retry(session, "Sorry, I didn't catch your name. Could you say it again?")

    session.attendee_name = name
    session.retries = 0
    if session.attendee_email:
        session.state = State.CONFIRMING
        return DialogResult(_confirmation_prompt(session))
    session.state = State.NEED_EMAIL
    return DialogResult(f"Thank you, {name}. What email should I send the confirmation to?")


async def _handle_need_email(session: BookingSession, text: str) -> DialogResult:
    email = nlu.extract_email(text)
    if not email:
        if nlu.is_abort(text):
            return _abort(session)
        return _retry(
            session,
            "Sorry, I didn't catch a valid email address. Please say it again, "
            "for example: rahul at gmail dot com.",
        )
    session.attendee_email = email
    session.retries = 0
    session.state = State.CONFIRMING
    return DialogResult(_confirmation_prompt(session))


async def _handle_confirming(session: BookingSession, text: str) -> DialogResult:
    corrected_email = nlu.extract_email(text)
    if corrected_email and corrected_email != session.attendee_email:
        session.attendee_email = corrected_email     # "no, it's rahul.s@gmail.com"
        return DialogResult(_confirmation_prompt(session))

    answer = nlu.classify_confirmation(text)
    if answer:
        return await _do_booking(session)

    if answer is False:
        tokens = set(text.lower().replace(",", " ").split())
        if tokens & {"email", "e-mail", "mail", "address"}:
            session.state = State.NEED_EMAIL
            return DialogResult("No problem. What's the correct email address?")
        if tokens & {"name", "spelling", "spelled"}:
            session.state = State.NEED_NAME
            return DialogResult("No problem. What name should the appointment be under?")
        if tokens & {"time", "date", "day", "slot", "different", "another", "change"}:
            session.state = State.NEED_DATE
            session.req_slot = ""
            return DialogResult("No problem. What day would you like instead?")
        session.reset()
        return DialogResult("Okay, I haven't booked anything. Let me know if you'd like a different time.")

    return _retry(session, "Sorry, I didn't catch that. " + _confirmation_prompt(session))


async def _do_booking(session: BookingSession) -> DialogResult:
    assert session.req_date is not None
    booked_date, booked_slot, email = session.req_date, session.req_slot, session.attendee_email
    result = await calcom.create_booking(
        slot_time_ist=booked_slot,
        target_date=booked_date,
        attendee_name=session.attendee_name,
        attendee_email=email,
        source=session.source,
    )

    if not result["success"]:
        if result["error_code"] == "slot_unavailable" and not session.one_shot:
            session.state = State.NEED_TIME
            session.req_slot = ""
            slots = await calcom.get_available_slots(booked_date)
            session.offered_slots = slots[:4]
            if slots:
                return DialogResult(
                    f"I'm sorry, that time was just taken. On {nlu.speak_date(booked_date)} I still "
                    f"have {nlu.speak_slot_list(slots)}. Which would you like?"
                )
        session.reset()
        return DialogResult(f"I'm sorry, I couldn't complete the booking. {result['message']}")

    session.reset()
    reference = (
        f" Your reference number is {nlu.speak_digits(result['ref_code'])}." if result.get("ref_code") else ""
    )
    return DialogResult(
        f"You're all set. Your appointment is booked for {nlu.speak_time(booked_slot)} on "
        f"{nlu.speak_date(booked_date)}, and a confirmation is on its way to {email}.{reference} "
        "Is there anything else I can help you with?",
        completed=True,
        booking_uid=result["booking_uid"],
        ref_code=result.get("ref_code"),
    )


# ── Cancellation flow ────────────────────────────────────────────────────────

_ASK_FOR_REF = (
    "I can help with that. What is the six digit reference number "
    "you were given when you booked?"
)


def _resolve_cancel_target(session: BookingSession, text: str) -> DialogResult | None:
    """Turn a spoken reference into a concrete booking and ask for confirmation."""
    ref = nlu.extract_booking_ref(text)
    if ref is None:
        return None

    kind, value = ref
    booking = store.get_booking_by_ref(value) if kind == "code" else store.get_booking_by_uid(value)
    if kind == "code" and booking is None:
        return _retry(
            session,
            f"I couldn't find an upcoming appointment with reference {nlu.speak_digits(value)}. "
            "Could you repeat the number?",
        )

    session.state = State.CANCEL_CONFIRM
    session.retries = 0
    if booking is not None and booking["status"] == "confirmed":
        session.cancel_uid = booking["cal_booking_uid"]
        when = f"{nlu.speak_time(booking['time_ist'])} on {nlu.speak_date(date.fromisoformat(booking['date']))}"
        return DialogResult(f"I found your appointment at {when}. Shall I cancel it?")

    # A Cal.com UID we have no local record for: Cal.com decides if it exists.
    session.cancel_uid = value
    return DialogResult("I have that booking reference. Shall I go ahead and cancel it?")


async def _start_cancel(session: BookingSession, text: str) -> DialogResult:
    session.state = State.CANCEL_NEED_REF
    return _resolve_cancel_target(session, text) or DialogResult(_ASK_FOR_REF)


async def _handle_cancel_need_ref(session: BookingSession, text: str) -> DialogResult:
    resolved = _resolve_cancel_target(session, text)
    if resolved is not None:
        return resolved
    if nlu.is_abort(text) or nlu.classify_confirmation(text) is False:
        return _abort(session)
    return _retry(
        session,
        "Sorry, I didn't catch a reference number. Please say the six digits one at a time.",
    )


async def _handle_cancel_confirm(session: BookingSession, text: str) -> DialogResult:
    answer = nlu.classify_confirmation(text)
    if answer:
        uid = session.cancel_uid
        session.reset()
        result = await calcom.cancel_booking(uid)
        if result["success"]:
            return DialogResult(
                "Your appointment has been cancelled. Is there anything else I can help you with?",
                completed=True,
                booking_uid=uid,
            )
        return DialogResult(f"I'm sorry, I couldn't cancel that. {result['message']}")
    if answer is False:
        session.reset()
        return DialogResult("Okay, your appointment is still in place. Anything else I can help with?")
    return _retry(session, "Sorry, I didn't catch that. Should I cancel the appointment — yes or no?")


_HANDLERS = {
    State.IDLE: _handle_idle,
    State.NEED_DATE: _handle_need_date,
    State.NEED_TIME: _handle_need_time,
    State.OFFER_DATE: _handle_offer_date,
    State.NEED_NAME: _handle_need_name,
    State.NEED_EMAIL: _handle_need_email,
    State.CONFIRMING: _handle_confirming,
    State.CANCEL_NEED_REF: _handle_cancel_need_ref,
    State.CANCEL_CONFIRM: _handle_cancel_confirm,
}
