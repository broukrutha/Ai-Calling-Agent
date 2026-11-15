"""
Rule-based language understanding for the booking flow.

Everything here is a pure function of its input (plus an injectable "now"),
which keeps it fast, predictable and easy to test. It covers:

* intent detection (book / cancel / yes / no / abort),
* date and time extraction from natural speech,
* name, email and booking-reference extraction from speech-to-text output,
* speaking dates and times back in a natural way.

Matching is done on whole tokens, never on substrings, so "ok" does not fire
inside "book" and "am" does not fire inside "name".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from voice_agent import clock, config

# ── Normalisation ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[^\s.,!?;:।॥\"“”()]+")


def _tokens(text: str) -> list[str]:
    cleaned = text.lower().replace("’", "'").replace("‘", "'")
    return _TOKEN_RE.findall(cleaned)


def _norm(text: str) -> str:
    """Lower-cased tokens joined by single spaces and padded, for phrase lookups."""
    return " " + " ".join(_tokens(text)) + " "


def _has_phrase(norm: str, phrases: tuple[str, ...] | frozenset[str]) -> bool:
    return any(f" {phrase} " in norm for phrase in phrases)


# ── Intent detection ─────────────────────────────────────────────────────────

_BOOK_WORDS = frozenset({
    "book", "booking", "bookings", "appointment", "appointments", "schedule",
    "reschedule", "reserve", "reservation", "slot", "slots", "meeting", "consultation",
    # Hindi / Telugu loanwords as transcribed in native script
    "बुक", "बुकिंग", "अपॉइंटमेंट", "బుక్", "బుకింగ్", "అపాయింట్మెంట్", "అపాయింట్‌మెంట్",
})
_AVAILABILITY_WORDS = frozenset({"available", "availability", "free", "open", "vacant", "check"})
_CANCEL_VERBS = frozenset({"cancel", "cancelling", "canceling", "cancellation", "delete", "remove"})
_CANCEL_OBJECTS = frozenset({
    "appointment", "appointments", "booking", "bookings", "meeting", "reservation", "slot", "it",
})
_CANCEL_PHRASES = (
    "want to cancel", "need to cancel", "like to cancel", "please cancel", "call off",
)


def is_cancel_intent(text: str) -> bool:
    """True when the user wants to cancel an existing appointment."""
    tokens = set(_tokens(text))
    if tokens & _CANCEL_VERBS and tokens & _CANCEL_OBJECTS:
        return True
    return _has_phrase(_norm(text), _CANCEL_PHRASES)


_QUESTION_STARTS = (
    "what", "why", "who", "where", "which", "how much", "how long", "how many",
    "tell me about", "explain", "describe",
)
_ACTION_WORDS = frozenset({
    "book", "schedule", "reschedule", "reserve", "make", "get", "need", "want", "like", "set", "fix",
})


def is_booking_intent(text: str) -> bool:
    """True when the user wants to book, or asks about appointment availability."""
    tokens = set(_tokens(text))
    parsed = parse_booking_request(text)
    has_when = parsed.date is not None or parsed.time is not None

    if tokens & _BOOK_WORDS:
        # "What is your cancellation policy for appointments?" is a question
        # for the knowledge base, not a request to book.
        informational = _norm(text).lstrip().startswith(_QUESTION_STARTS)
        return has_when or not informational or bool(tokens & _ACTION_WORDS)
    return bool(tokens & _AVAILABILITY_WORDS) and has_when


_NO_TOKENS = frozenset({
    "no", "nope", "nah", "not", "don't", "dont", "never", "stop", "wait", "cancel",
    "wrong", "incorrect", "nahi", "nahin", "mat", "नहीं", "नही", "मत",
    "vaddu", "kaadu", "kadu", "వద్దు", "కాదు",
})
_NO_PHRASES = ("do not", "never mind", "nevermind", "hold on", "not now")
_YES_TOKENS = frozenset({
    "yes", "yeah", "yep", "yup", "ya", "yah", "sure", "ok", "okay", "correct", "right",
    "confirm", "confirmed", "absolutely", "definitely", "alright", "please", "fine", "perfect",
    "haan", "han", "ji", "हाँ", "हां", "जी", "avunu", "sare", "అవును", "సరే",
})
_YES_PHRASES = (
    "go ahead", "do it", "book it", "that's right", "thats right", "all right",
    "sounds good", "of course", "go for it", "why not", "that works",
)
# Polite phrases that contain a "no" word but are not a refusal.
_NOT_A_NO = ("no problem", "no worries", "no issue", "no issues")


def classify_confirmation(text: str) -> bool | None:
    """
    Interpret a reply to a yes/no question.

    Returns True (yes), False (no) or None (unclear). A refusal always wins:
    "no, please don't" is a no even though it contains "please".
    """
    norm = _norm(text)
    for phrase in _NOT_A_NO:
        norm = norm.replace(f" {phrase} ", " ")
    tokens = set(norm.split())
    if tokens & _NO_TOKENS or _has_phrase(norm, _NO_PHRASES):
        return False
    if tokens & _YES_TOKENS or _has_phrase(norm, _YES_PHRASES):
        return True
    return None


_ABORT_TOKENS = frozenset({"cancel", "stop", "abort", "quit", "nevermind", "exit"})
_ABORT_PHRASES = ("never mind", "forget it", "no thanks", "no thank you", "leave it", "not now")


def is_abort(text: str) -> bool:
    """True when the user wants to drop whatever flow is in progress."""
    norm = _norm(text)
    return bool(set(norm.split()) & _ABORT_TOKENS) or _has_phrase(norm, _ABORT_PHRASES)


# ── Date parsing ─────────────────────────────────────────────────────────────

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_WEEKDAY_ALT = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_DAY_AFTER_TOMORROW = ("day after tomorrow", "parso", "parson", "परसों", "ellundi", "ఎల్లుండి")
_TOMORROW = ("tomorrow", "tmrw", "kal", "कल", "repu", "రేపు")
_TODAY = ("today", "tonight", "aaj", "आज", "ee roju", "ఈరోజు", "ఈ రోజు")

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b(?:,?\s+(\d{{4}}))?"
)
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*(?::\d|am\b|pm\b))(?:,?\s+(\d{{4}}))?"
)
_WEEKDAY_RE = re.compile(rf"\b(?:(next|this|coming)\s+)?({_WEEKDAY_ALT})\b")
_ORDINAL_DAY_RE = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b")
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b")


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _roll_forward(month: int, day: int, year: int | None, today: date) -> date | None:
    """A month/day with no year means the next time that date comes around."""
    if year is not None:
        return _safe_date(year, month, day)
    candidate = _safe_date(today.year, month, day)
    if candidate is not None and candidate < today:
        candidate = _safe_date(today.year + 1, month, day)
    return candidate


def _extract_date(text: str, today: date) -> tuple[date | None, str]:
    """
    Find a date in ``text``.

    Returns the date and the text with the date phrase blanked out, so the
    time parser never mistakes "5" in "April 5" for an hour.
    """
    lowered = text.lower().replace("’", "'")

    def cut(match: re.Match[str]) -> str:
        return lowered[: match.start()] + " " + lowered[match.end():]

    match = _ISO_DATE_RE.search(lowered)
    if match:
        found = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if found:
            return found, cut(match)

    match = _DAY_MONTH_RE.search(lowered)
    if match:
        year = int(match.group(3)) if match.group(3) else None
        found = _roll_forward(_MONTHS[match.group(2)], int(match.group(1)), year, today)
        if found:
            return found, cut(match)

    match = _MONTH_DAY_RE.search(lowered)
    if match:
        year = int(match.group(3)) if match.group(3) else None
        found = _roll_forward(_MONTHS[match.group(1)], int(match.group(2)), year, today)
        if found:
            return found, cut(match)

    match = _NUMERIC_DATE_RE.search(lowered)
    if match:  # day first, as written in India
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        if year is not None and year < 100:
            year += 2000
        found = _roll_forward(month, day, year, today) if 1 <= month <= 12 else None
        if found:
            return found, cut(match)

    for phrases, offset in ((_DAY_AFTER_TOMORROW, 2), (_TOMORROW, 1), (_TODAY, 0)):
        for phrase in phrases:
            match = re.search(rf"(?<!\S){re.escape(phrase)}(?![^\s.,!?])", lowered)
            if match:
                return today + timedelta(days=offset), cut(match)

    match = _WEEKDAY_RE.search(lowered)
    if match:
        target = _WEEKDAYS[match.group(2)]
        if match.group(1) == "next":
            # "next Friday" = Friday of the following calendar week.
            days_ahead = (7 - today.weekday()) + target
        else:
            days_ahead = (target - today.weekday()) % 7
            if days_ahead == 0 and match.group(1) != "this":
                days_ahead = 7
        return today + timedelta(days=days_ahead), cut(match)

    match = _ORDINAL_DAY_RE.search(lowered)
    if match:
        day = int(match.group(1))
        candidate = _safe_date(today.year, today.month, day)
        if candidate is None or candidate < today:
            year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
            candidate = _safe_date(year, month, day)
        if candidate:
            return candidate, cut(match)

    match = _NEXT_WEEK_RE.search(lowered)
    if match:
        return today + timedelta(days=7 - today.weekday()), cut(match)

    return None, lowered


def parse_date(text: str, today: date | None = None) -> date | None:
    return _extract_date(text, today or clock.now_local().date())[0]


# ── Time parsing ─────────────────────────────────────────────────────────────

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUMBER_ALT = "|".join(_NUMBER_WORDS)
_PERIODS = {
    "morning": "morning", "subah": "morning", "सुबह": "morning",
    "afternoon": "afternoon", "dopahar": "afternoon", "दोपहर": "afternoon",
    "evening": "evening", "shaam": "evening", "sham": "evening", "शाम": "evening",
    "night": "evening", "tonight": "evening",
}

_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b")
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_WORD_AMPM_RE = re.compile(rf"\b({_NUMBER_ALT})\s*(am|pm)\b")
_HALF_PAST_RE = re.compile(rf"\bhalf\s+past\s+(\d{{1,2}}|{_NUMBER_ALT})\b")
_AT_HOUR_RE = re.compile(
    rf"\b(?:at|around|by|for)\s+(\d{{1,2}}|{_NUMBER_ALT})\b(?!\s*(?:st|nd|rd|th|people|persons|minutes|days)\b)"
)
_OCLOCK_RE = re.compile(rf"\b(\d{{1,2}}|{_NUMBER_ALT})\s*(?:o'clock|oclock|o clock|baje|बजे)")
_BARE_HOUR_RE = re.compile(rf"^\W*(?:(?:at|the|about|around|maybe|say|um|uh)\s+)*(\d{{1,2}}|{_NUMBER_ALT})\W*(?:please|would be (?:good|great|fine))?\W*$")


def _to_hour(raw: str) -> int:
    return _NUMBER_WORDS[raw] if raw in _NUMBER_WORDS else int(raw)


def _resolve_bare_hour(hour: int, period: str | None) -> int:
    """
    Decide AM/PM for an hour said without either ("at 3").

    A spoken period wins ("5 in the evening"). Otherwise an hour before
    opening time is assumed to be the afternoon when that lands inside office
    hours, which is what a caller booking an appointment means.
    """
    if hour >= 13:
        return hour
    if period in {"afternoon", "evening"} and hour < 12:
        return hour + 12
    if period == "morning":
        return hour
    if hour < config.OFFICE_START_HOUR and hour + 12 < config.OFFICE_END_HOUR:
        return hour + 12
    return hour


def parse_period(text: str) -> str | None:
    """'morning' | 'afternoon' | 'evening' when the user gives a part of the day."""
    for token in _tokens(text):
        if token in _PERIODS:
            return _PERIODS[token]
    return None


def parse_time(text: str, *, allow_bare_number: bool = False) -> time | None:
    """
    Extract a clock time. ``allow_bare_number`` accepts a reply that is just a
    number ("11", "two"), which only makes sense right after asking for a time.
    """
    lowered = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", text.lower().replace("’", "'"))
    period = parse_period(lowered)

    def build(hour: int, minute: int = 0) -> time | None:
        return time(hour, minute) if 0 <= hour <= 23 and 0 <= minute <= 59 else None

    match = _TIME_AMPM_RE.search(lowered)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if match.group(3) == "pm" else 0)
        return build(hour, minute)

    match = _WORD_AMPM_RE.search(lowered)
    if match:
        hour = _NUMBER_WORDS[match.group(1)] % 12 + (12 if match.group(2) == "pm" else 0)
        return build(hour)

    match = _TIME_24H_RE.search(lowered)
    if match:
        return build(_resolve_bare_hour(int(match.group(1)), period), int(match.group(2)))

    match = _HALF_PAST_RE.search(lowered)
    if match:
        return build(_resolve_bare_hour(_to_hour(match.group(1)), period), 30)

    tokens = set(_tokens(lowered))
    if tokens & {"noon", "midday"}:
        return time(12, 0)

    for pattern in (_OCLOCK_RE, _AT_HOUR_RE):
        match = pattern.search(lowered)
        if match:
            return build(_resolve_bare_hour(_to_hour(match.group(1)), period))

    if allow_bare_number:
        match = _BARE_HOUR_RE.match(lowered)
        if match:
            return build(_resolve_bare_hour(_to_hour(match.group(1)), period))

    return None


@dataclass(frozen=True)
class ParsedRequest:
    """What a single utterance tells us about the requested appointment."""

    date: date | None = None
    time: time | None = None
    period: str | None = None      # part of the day, when no exact time was given


def parse_booking_request(text: str, now: datetime | None = None) -> ParsedRequest:
    """Extract whatever date / time information is present. Never guesses."""
    current = now or clock.now_local()
    found_date, remainder = _extract_date(text, current.date())
    found_time = parse_time(remainder)
    return ParsedRequest(
        date=found_date,
        time=found_time,
        period=None if found_time else parse_period(text),
    )


# ── Slot helpers ─────────────────────────────────────────────────────────────

PERIOD_WINDOWS = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24)}


def slots_in_period(slots: list[str], period: str | None) -> list[str]:
    if period not in PERIOD_WINDOWS:
        return slots
    start, end = PERIOD_WINDOWS[period]
    return [slot for slot in slots if start <= int(slot[:2]) < end]


_ORDINAL_WORDS = {"first": 0, "earliest": 0, "second": 1, "third": 2, "last": -1, "latest": -1}
_ORDINAL_NUMERALS = {"1st": 0, "2nd": 1, "3rd": 2}
_ORDINAL_NOUNS = frozenset({"one", "option", "slot", "time"})


def pick_offered_slot(text: str, offered: list[str]) -> str | None:
    """
    Resolve "the first one" / "the last one" against the slots just offered.

    "2nd" only counts when it is followed by "one", "option" and the like, so
    "the 2nd of April" is still read as a date.
    """
    if not offered:
        return None
    tokens = _tokens(text)
    for position, token in enumerate(tokens):
        following = tokens[position + 1] if position + 1 < len(tokens) else ""
        if token in _ORDINAL_WORDS and following != "of":
            index = _ORDINAL_WORDS[token]
        elif token in _ORDINAL_NUMERALS and following in _ORDINAL_NOUNS:
            index = _ORDINAL_NUMERALS[token]
        else:
            continue
        if -len(offered) <= index < len(offered):
            return offered[index]
    return None


# ── Name / email / reference extraction ──────────────────────────────────────

_NAME_LEAD_RE = re.compile(
    r"(?:my\s+(?:full\s+)?name\s+is|my\s+name's|name\s+is|i\s+am|i'm|this\s+is|it's|it\s+is|"
    r"call\s+me|myself|mera\s+naam|मेरा\s+नाम|నా\s+పేరు)\s+(.+)",
    re.IGNORECASE,
)
_NAME_STOP_RE = re.compile(
    r"[,.;!?।\n]|\s+(?:and|but|my|email|e-mail|mail|here|speaking|hai|hoon|hun)\b|\s+(?:है|हूँ|हूं)(?!\S)",
    re.IGNORECASE,
)
_NAME_TRAIL_RE = re.compile(r"\s+(?:please|thanks|thank\s+you|sir|ma'am|madam)$", re.IGNORECASE)
_NOT_NAME_WORDS = frozenset({
    "what", "when", "where", "who", "why", "how", "which", "can", "could", "would", "should",
    "is", "are", "do", "does", "did", "will", "yes", "yeah", "no", "ok", "okay", "sure",
    "hello", "hi", "hey", "sorry", "pardon", "not", "looking", "calling", "trying", "going",
    "want", "need", "asking", "interested", "just", "the", "a", "an", "to", "for", "of",
    "book", "booking", "appointment", "cancel", "name", "email", "please", "thanks",
})


def extract_name(text: str) -> str:
    """
    Pull a person's name out of a reply such as "My name is Rahul Sharma".

    Returns an empty string when the reply does not look like a name, so the
    caller can ask again instead of booking under "What Did You Say".
    """
    cleaned = text.strip().replace("’", "'")
    match = _NAME_LEAD_RE.search(cleaned)
    candidate = match.group(1) if match else cleaned
    candidate = _NAME_STOP_RE.split(candidate, maxsplit=1)[0]
    candidate = _NAME_TRAIL_RE.sub("", candidate.strip()).strip(" '\"-")

    words = candidate.split()
    if not 1 <= len(words) <= 4 or not 2 <= len(candidate) <= 60:
        return ""
    if any(ch.isdigit() or ch in "@/\\<>{}[]()=_#$%^&*+|~`" for ch in candidate):
        return ""
    if any(word.lower() in _NOT_NAME_WORDS for word in words):
        return ""
    return " ".join(word.capitalize() if word.islower() else word for word in words)


_EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
_EMAIL_RE = re.compile(_EMAIL_PATTERN)
_EMAIL_LEAD_RE = re.compile(
    r"^(?:(?:yes|yeah|yep|sure|ok|okay|so|well|um|uh|and)[\s,]+)*"
    r"(?:(?:my|the)\s+)?(?:(?:e-?mail|mail)(?:\s+(?:address|id))?\s*(?:is|:)?|"
    r"it's|its|it\s+is|that's|that\s+is|"
    r"(?:you\s+can\s+)?(?:reach|email|mail|contact)\s+me\s+(?:at|on)|send\s+it\s+to)\s+"
)
_EMAIL_TRAIL_RE = re.compile(r"[\s,]+(?:please|thanks|thank\s+you|ok|okay)$")
_COMMON_MAIL_DOMAINS = {"gmail", "yahoo", "outlook", "hotmail", "icloud", "rediffmail"}
_DIGIT_WORDS = {"zero": "0", **{word: str(num) for word, num in _NUMBER_WORDS.items() if num < 10}}


def is_valid_email(value: str) -> bool:
    return bool(re.fullmatch(_EMAIL_PATTERN, value or "")) and len(value) <= 254


def extract_email(text: str) -> str | None:
    """
    Pull an email address out of typed or spoken text.

    Handles the way people say addresses aloud — "rahul dot sharma at gmail dot
    com", spelled-out letters, "underscore" — without swallowing lead-ins such
    as "my email is".
    """
    match = _EMAIL_RE.search(text)
    if match:
        return match.group(0).lower().rstrip(".")

    spoken = text.lower().strip().replace("’", "'").rstrip(".!? ")
    spoken = _EMAIL_TRAIL_RE.sub("", spoken)
    spoken = _EMAIL_LEAD_RE.sub("", spoken)
    spoken = spoken.replace(",", " ")
    spoken = re.sub(r"\bat\s+the\s+rate(?:\s+of)?\b", "@", spoken)
    if "@" not in spoken:
        at_matches = list(re.finditer(r"\bat\b", spoken))
        if at_matches:   # the last "at" is the separator: "reach me at rahul at gmail"
            last = at_matches[-1]
            spoken = spoken[: last.start()] + "@" + spoken[last.end():]
    spoken = re.sub(r"\bdot\b", ".", spoken)
    spoken = re.sub(r"\bunderscore\b", "_", spoken)
    spoken = re.sub(r"\b(?:dash|hyphen|minus)\b", "-", spoken)
    spoken = re.sub(r"\bplus\b", "+", spoken)
    spoken = re.sub(
        r"\b(" + "|".join(_DIGIT_WORDS) + r")\b", lambda m: _DIGIT_WORDS[m.group(1)], spoken
    )
    if "@" not in spoken:
        return None

    local, _, domain = spoken.partition("@")
    local = re.sub(r"\s+", "", local)
    domain = re.sub(r"\s+", "", domain)
    if domain in _COMMON_MAIL_DOMAINS:
        domain += ".com"
    candidate = f"{local}@{domain}"
    return candidate if is_valid_email(candidate) else None


_UID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}")
_VALID_UID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def is_valid_booking_uid(value: str) -> bool:
    """Guards anything that ends up in a Cal.com URL path."""
    return bool(_VALID_UID_RE.fullmatch(value or ""))


def extract_booking_ref(text: str) -> tuple[str, str] | None:
    """
    Find a booking reference in what the user said.

    Returns ``("code", "482915")`` for the 6-digit reference we read out at
    booking time (digits may be spoken one by one), ``("uid", "...")`` for a
    Cal.com booking UID, or None.
    """
    spoken = re.sub(
        r"\b(" + "|".join(_DIGIT_WORDS) + r")\b",
        lambda m: _DIGIT_WORDS[m.group(1)],
        text.lower(),
    )
    for group in re.findall(r"\d(?:[\s-]*\d)*", spoken):
        digits = re.sub(r"\D", "", group)
        if len(digits) == 6:
            return "code", digits

    for token in _UID_RE.findall(text):
        has_digit = any(ch.isdigit() for ch in token)
        has_alpha = any(ch.isalpha() for ch in token)
        mixed_case = token != token.lower() and token != token.upper()
        if has_alpha and (has_digit or (mixed_case and len(token) >= 16)):
            return "uid", token
    return None


# ── Speaking dates and times ─────────────────────────────────────────────────

def speak_date(value: date) -> str:
    """``date(2026, 4, 6)`` → "Monday, April 6"."""
    return f"{value.strftime('%A, %B')} {value.day}"


def speak_time(slot: str) -> str:
    """``"09:00"`` → "9 AM", ``"14:30"`` → "2:30 PM"."""
    hour, minute = (int(part) for part in slot.split(":"))
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


def speak_slot_list(slots: list[str], limit: int = 4) -> str:
    """Up to ``limit`` slots as a spoken list: "9 AM, 11 AM and 2 PM"."""
    shown = [speak_time(slot) for slot in slots[:limit]]
    if not shown:
        return ""
    if len(shown) == 1:
        return shown[0]
    spoken = ", ".join(shown[:-1]) + " and " + shown[-1]
    return spoken + (", plus a few later ones" if len(slots) > limit else "")


def speak_digits(code: str) -> str:
    """``"482915"`` → "4 8 2 9 1 5" so text-to-speech reads it digit by digit."""
    return " ".join(code)
