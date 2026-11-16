"""End-to-end dialog tests against the mock calendar (no network)."""

from datetime import date

from voice_agent.booking import store
from voice_agent.booking.dialog import BookingSession, State, handle_booking_query


async def say(session, *utterances):
    """Feed several turns; return the last reply."""
    result = None
    for text in utterances:
        result = await handle_booking_query(text, session)
    return result


async def test_full_booking_conversation():
    session = BookingSession()
    reply = await say(session, "I'd like to book an appointment")
    assert session.state is State.NEED_DATE and "What day" in reply.text

    reply = await say(session, "tomorrow")
    assert session.state is State.NEED_TIME and "9 AM" in reply.text

    reply = await say(session, "11")
    assert session.state is State.NEED_NAME and "11 AM on Thursday, April 2" in reply.text

    reply = await say(session, "My name is Rahul Sharma")
    assert session.state is State.NEED_EMAIL and "Rahul Sharma" in reply.text

    reply = await say(session, "my email is rahul at gmail dot com")
    assert session.state is State.CONFIRMING
    assert "rahul@gmail.com" in reply.text and not reply.completed
    assert store.get_bookings() == []                      # nothing booked before the "yes"

    reply = await say(session, "yes please")
    assert reply.completed and reply.ref_code and session.state is State.IDLE
    assert " ".join(reply.ref_code) in reply.text          # read out digit by digit

    [booking] = store.get_bookings()
    assert (booking["date"], booking["time_ist"]) == ("2026-04-02", "11:00")
    assert (booking["attendee_name"], booking["attendee_email"]) == ("Rahul Sharma", "rahul@gmail.com")
    assert booking["status"] == "confirmed" and booking["ref_code"] == reply.ref_code


async def test_everything_in_one_sentence_skips_straight_to_the_name():
    session = BookingSession()
    reply = await say(session, "Book an appointment for Friday at 3 pm")
    assert session.state is State.NEED_NAME
    assert (session.req_date, session.req_slot) == (date(2026, 4, 3), "15:00")
    assert "3 PM on Friday, April 3" in reply.text


async def test_refusal_at_confirmation_books_nothing():
    session = BookingSession()
    await say(session, "book tomorrow at 2 pm", "Priya", "priya@example.com")
    assert session.state is State.CONFIRMING

    reply = await say(session, "No, please don't")         # used to book because of "please"
    assert not reply.completed and session.state is State.IDLE
    assert store.get_bookings() == []


async def test_email_can_be_corrected_at_confirmation():
    session = BookingSession()
    await say(session, "book tomorrow at 2 pm", "Priya", "priya@example.com")
    reply = await say(session, "no, the email is wrong")
    assert session.state is State.NEED_EMAIL

    reply = await say(session, "priya dot k at example dot com")
    assert session.state is State.CONFIRMING and "priya.k@example.com" in reply.text

    reply = await say(session, "yes")
    assert reply.completed
    assert store.get_bookings()[0]["attendee_email"] == "priya.k@example.com"


async def test_taken_slot_offers_the_nearest_alternatives():
    first = BookingSession()
    await say(first, "book tomorrow at 2 pm", "Asha", "asha@example.com", "yes")

    second = BookingSession()
    reply = await say(second, "book tomorrow at 2 pm")
    assert second.state is State.NEED_TIME
    assert "isn't available" in reply.text and "3 PM or 1 PM" in reply.text

    reply = await say(second, "the first one")
    assert second.state is State.NEED_NAME and second.req_slot == "15:00"


async def test_day_without_openings_suggests_the_next_one():
    session = BookingSession()
    reply = await say(session, "book an appointment on saturday at 10 am")
    assert session.state is State.OFFER_DATE
    assert "no openings on Saturday, April 4" in reply.text and "Monday, April 6" in reply.text

    reply = await say(session, "yes")                      # the 10 am carries over
    assert session.state is State.NEED_NAME
    assert (session.req_date, session.req_slot) == (date(2026, 4, 6), "10:00")


async def test_time_outside_office_hours_is_not_silently_moved():
    session = BookingSession()
    reply = await say(session, "book tomorrow at 9 pm")
    assert session.state is State.NEED_TIME
    assert "9 AM to 7 PM" in reply.text and not session.req_slot


async def test_past_slots_today_are_not_offered():
    session = BookingSession()
    reply = await say(session, "book an appointment today")    # frozen clock: 10:00
    assert "9 AM" not in reply.text and "10 AM" not in reply.text and "11 AM" in reply.text


async def test_part_of_day_preference():
    session = BookingSession()
    reply = await say(session, "book me in tomorrow afternoon")
    assert session.state is State.NEED_TIME
    assert reply.text.startswith("On Thursday, April 2 I have 12 PM")


async def test_dates_too_far_ahead_and_in_the_past():
    session = BookingSession()
    reply = await say(session, "book an appointment on 2027-03-01")
    assert session.state is State.NEED_DATE and "90 days" in reply.text


async def test_gives_up_politely_after_three_misunderstandings():
    session = BookingSession()
    await say(session, "book an appointment")
    await say(session, "blah", "blah blah")
    assert session.state is State.NEED_DATE
    reply = await say(session, "still nonsense")
    assert session.state is State.IDLE and "start over" in reply.text


async def test_changing_your_mind_mid_flow():
    session = BookingSession()
    await say(session, "book an appointment")
    reply = await say(session, "no, never mind")
    assert session.state is State.IDLE and "left everything" in reply.text


async def test_stale_session_expires(monkeypatch):
    session = BookingSession()
    await say(session, "book an appointment")
    assert session.is_active()
    session.touched_at -= 10_000
    assert not session.is_active() and session.state is State.IDLE


async def test_one_shot_rest_booking_needs_no_follow_up():
    result = await handle_booking_query(
        "Book me an appointment tomorrow at 4 pm",
        session=None,
        attendee_name="Kiran Rao",
        attendee_email="Kiran@Example.com",
    )
    assert result.completed
    [booking] = store.get_bookings()
    assert booking["attendee_email"] == "kiran@example.com" and booking["source"] == "api"


async def test_one_shot_without_contact_details_does_not_book():
    result = await handle_booking_query("Book me an appointment tomorrow at 4 pm")
    assert not result.completed and store.get_bookings() == []


# ── Cancellation ─────────────────────────────────────────────────────────────

async def _book(slot="14:00"):
    session = BookingSession()
    return await say(session, f"book tomorrow at {slot}", "Asha", "asha@example.com", "yes")


async def test_cancel_with_spoken_reference_number():
    booked = await _book()
    session = BookingSession()

    reply = await say(session, "I want to cancel my appointment")
    assert session.state is State.CANCEL_NEED_REF and "six digit" in reply.text

    reply = await say(session, " ".join(booked.ref_code))
    assert session.state is State.CANCEL_CONFIRM
    assert "2 PM on Thursday, April 2" in reply.text
    assert "Asha" not in reply.text and "asha@" not in reply.text    # no PII read back

    reply = await say(session, "yes")
    assert reply.completed and store.get_bookings()[0]["status"] == "cancelled"


async def test_cancel_is_not_executed_without_a_yes():
    booked = await _book()
    session = BookingSession()
    reply = await say(session, f"cancel my booking {booked.ref_code}", "no, keep it")
    assert not reply.completed and session.state is State.IDLE
    assert store.get_bookings()[0]["status"] == "confirmed"


async def test_unknown_reference_numbers_lock_out_after_three_tries():
    await _book()
    session = BookingSession()
    await say(session, "cancel my appointment", "111111", "222222")
    assert session.state is State.CANCEL_NEED_REF
    reply = await say(session, "333333")
    assert session.state is State.IDLE and "start over" in reply.text
    assert store.get_bookings()[0]["status"] == "confirmed"


async def test_cancel_words_alone_are_not_taken_as_a_reference():
    session = BookingSession()
    reply = await say(session, "please cancel my booking for tomorrow")
    assert session.state is State.CANCEL_NEED_REF and "reference number" in reply.text
