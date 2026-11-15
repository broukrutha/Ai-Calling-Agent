from datetime import date, time

import pytest

from voice_agent.booking import nlu

TODAY = date(2026, 4, 1)   # Wednesday


# ── Intent ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "I want to book an appointment",
    "Can you schedule a meeting for tomorrow?",
    "Book me in for Friday at 3 pm",
    "Are you free tomorrow at 10 am?",
    "Do you have any slots on Monday",
    "Check availability for April 6th",
    "मुझे कल अपॉइंटमेंट बुक करना है",
])
def test_booking_intent_detected(text):
    assert nlu.is_booking_intent(text)


@pytest.mark.parametrize("text", [
    "What is your name?",                       # "am" inside "name" used to match
    "I am looking for a free program",          # "am" + "free" used to match
    "Do you have a facebook page?",             # "book" inside "facebook"
    "Is there a free trial available?",
    "What is your cancellation policy for appointments?",
    "How much does it cost?",
    "Tell me about your pricing",
])
def test_non_booking_questions_go_to_the_knowledge_base(text):
    assert not nlu.is_booking_intent(text)


@pytest.mark.parametrize("text,expected", [
    ("I want to cancel my appointment", True),
    ("please cancel the booking 482915", True),
    ("cancel it", True),
    ("I need to cancel", True),
    ("book an appointment", False),
    ("what is the cancellation fee", False),
])
def test_cancel_intent(text, expected):
    assert nlu.is_cancel_intent(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("yes", True),
    ("Yeah, go ahead", True),
    ("okay sounds good", True),
    ("no problem, go ahead", True),
    ("हाँ", True),
    ("no", False),
    ("No, please don't", False),               # "please" must not turn this into a yes
    ("that's not right", False),
    ("wait, cancel that", False),
    ("नहीं", False),
    ("I would like to know more", None),       # "no" inside "know" must not match
    ("book", None),                            # "ok" inside "book" must not match
    ("maybe", None),
])
def test_classify_confirmation(text, expected):
    assert nlu.classify_confirmation(text) is expected


# ── Dates ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("today", date(2026, 4, 1)),
    ("tomorrow at 3", date(2026, 4, 2)),
    ("day after tomorrow", date(2026, 4, 3)),
    ("friday", date(2026, 4, 3)),
    ("this friday", date(2026, 4, 3)),
    ("monday", date(2026, 4, 6)),
    ("next monday", date(2026, 4, 6)),          # Monday of next week, not 12 days out
    ("next friday", date(2026, 4, 10)),
    ("wednesday", date(2026, 4, 8)),            # said on a Wednesday → next week's
    ("April 6", date(2026, 4, 6)),
    ("6th of April", date(2026, 4, 6)),
    ("on the 5th", date(2026, 4, 5)),
    ("march 15", date(2027, 3, 15)),            # already passed this year
    ("06/04/2026", date(2026, 4, 6)),           # day first
    ("2026-04-20", date(2026, 4, 20)),
    ("next week", date(2026, 4, 6)),
    ("कल", date(2026, 4, 2)),
    ("రేపు", date(2026, 4, 2)),
])
def test_parse_date(text, expected):
    assert nlu.parse_date(text, TODAY) == expected


@pytest.mark.parametrize("text", ["hello there", "book an appointment", "at 3 pm", "february 30"])
def test_parse_date_returns_none_instead_of_guessing(text):
    assert nlu.parse_date(text, TODAY) is None


# ── Times ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("3 pm", time(15, 0)),
    ("at 3:30 PM", time(15, 30)),
    ("10 a.m.", time(10, 0)),
    ("12 pm", time(12, 0)),
    ("12 am", time(0, 0)),
    ("15:00", time(15, 0)),
    ("at 3", time(15, 0)),                      # bare afternoon hour → PM
    ("at 10", time(10, 0)),
    ("at three", time(15, 0)),
    ("three pm", time(15, 0)),
    ("11 o'clock", time(11, 0)),
    ("half past 2", time(14, 30)),
    ("noon", time(12, 0)),
    ("9 in the evening", None),                 # needs "at"/am/pm to count as a time
    ("at 5 in the evening", time(17, 0)),
])
def test_parse_time(text, expected):
    assert nlu.parse_time(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("11", time(11, 0)),
    ("two", time(14, 0)),
    ("the 11 please", time(11, 0)),
    ("I have 2 questions for you", None),
])
def test_bare_number_only_counts_as_a_time_reply(text, expected):
    assert nlu.parse_time(text, allow_bare_number=True) == expected
    if expected is not None and ":" not in text:
        assert nlu.parse_time(text) is None


def test_date_digits_are_not_mistaken_for_the_time():
    parsed = nlu.parse_booking_request("book April 5 at 3 pm", None)
    assert (parsed.date, parsed.time) == (date(2026, 4, 5), time(15, 0))

    parsed = nlu.parse_booking_request("book on the 6th of april")
    assert parsed.date == date(2026, 4, 6) and parsed.time is None


def test_period_is_reported_when_no_exact_time():
    parsed = nlu.parse_booking_request("tomorrow morning")
    assert parsed.date == date(2026, 4, 2) and parsed.time is None and parsed.period == "morning"


def test_nothing_is_invented():
    parsed = nlu.parse_booking_request("I'd like to book an appointment")
    assert parsed == nlu.ParsedRequest(None, None, None)


# ── Names ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("My name is Rahul Sharma", "Rahul Sharma"),
    ("I am priya", "Priya"),
    ("Rahul", "Rahul"),
    ("this is Anil Kumar, thanks", "Anil Kumar"),
    ("my name is Rahul and my email is rahul@x.com", "Rahul"),
    ("मेरा नाम राहुल शर्मा है", "राहुल शर्मा"),
    ("राहुल शर्मा", "राहुल शर्मा"),             # Indic names survive intact
    ("O'Brien", "O'Brien"),
])
def test_extract_name(text, expected):
    assert nlu.extract_name(text) == expected


@pytest.mark.parametrize("text", [
    "what did you say?", "I am not sure", "sorry?", "yes", "12345", "rahul@gmail.com",
    "I'm looking for the pricing page", "<script>alert(1)</script>",
])
def test_extract_name_rejects_non_names(text):
    assert nlu.extract_name(text) == ""


# ── Emails ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("rahul@gmail.com", "rahul@gmail.com"),
    ("My email is Rahul.Sharma@Example.co.in.", "rahul.sharma@example.co.in"),
    ("rahul at gmail dot com", "rahul@gmail.com"),
    ("my email is rahul at gmail dot com", "rahul@gmail.com"),   # lead-in used to be glued on
    ("it's rahul dot sharma at gmail dot com", "rahul.sharma@gmail.com"),
    ("r a h u l at gmail dot com", "rahul@gmail.com"),
    ("rahul underscore s at yahoo dot co dot in", "rahul_s@yahoo.co.in"),
    ("rahul one two three at gmail dot com please", "rahul123@gmail.com"),
    ("you can reach me at priya at outlook dot com", "priya@outlook.com"),
    ("rahul at the rate gmail dot com", "rahul@gmail.com"),
    ("rahul at gmail", "rahul@gmail.com"),
])
def test_extract_email(text, expected):
    assert nlu.extract_email(text) == expected


@pytest.mark.parametrize("text", ["I don't have one", "yes go ahead", "book it at 3", "rahul at"])
def test_extract_email_returns_none(text):
    assert nlu.extract_email(text) is None


# ── References ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("482915", ("code", "482915")),
    ("it's 4 8 2 9 1 5", ("code", "482915")),
    ("four eight two nine one five", ("code", "482915")),
    ("cancel booking bXyZ12abCDef34GhIj5kLm", ("uid", "bXyZ12abCDef34GhIj5kLm")),
    ("mock-1a2b3c4d5e6f", ("uid", "mock-1a2b3c4d5e6f")),
    ("please cancel my booking for tomorrow", None),   # "tomorrow" used to be taken as an ID
    ("cancel my appointment", None),
    ("call me on 9988776655", None),
])
def test_extract_booking_ref(text, expected):
    assert nlu.extract_booking_ref(text) == expected


@pytest.mark.parametrize("uid,valid", [
    ("bXyZ12abCDef34", True), ("mock-1a2b", True),
    ("../../admin", False), ("a/b", False), ("", False), ("x" * 65, False), ("a b", False),
])
def test_booking_uid_validation_blocks_path_tricks(uid, valid):
    assert nlu.is_valid_booking_uid(uid) is valid


# ── Speaking ─────────────────────────────────────────────────────────────────

def test_speaking_helpers():
    assert nlu.speak_date(date(2026, 4, 6)) == "Monday, April 6"
    assert nlu.speak_time("09:00") == "9 AM"
    assert nlu.speak_time("14:30") == "2:30 PM"
    assert nlu.speak_time("12:00") == "12 PM"
    assert nlu.speak_slot_list(["09:00", "11:00", "14:00"]) == "9 AM, 11 AM and 2 PM"
    assert nlu.speak_digits("482915") == "4 8 2 9 1 5"


def test_pick_offered_slot():
    offered = ["09:00", "11:00", "14:00"]
    assert nlu.pick_offered_slot("the first one", offered) == "09:00"
    assert nlu.pick_offered_slot("2nd one please", offered) == "11:00"
    assert nlu.pick_offered_slot("the last one", offered) == "14:00"
    assert nlu.pick_offered_slot("the 2nd of April", offered) is None
    assert nlu.pick_offered_slot("eleven", offered) is None
