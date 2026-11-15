"""Request bodies. Every field is bounded so oversized input is rejected early."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

from voice_agent import config
from voice_agent.booking import nlu

_LANGUAGE_PATTERN = r"^(unknown|[a-z]{2,3}-IN)$"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    language: str = Field(default="en-IN", pattern=_LANGUAGE_PATTERN)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=config.MAX_TEXT_INGEST_CHARS)
    name: str = Field(default="manual_input", min_length=1, max_length=200)


class IngestURLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


class BookingCreateRequest(BaseModel):
    date: date
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="HH:MM, 24-hour, booking timezone")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not nlu.is_valid_email(value):
            raise ValueError("not a valid email address")
        return value


class BookingIntentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    name: str = Field(default="", max_length=100)
    email: str = Field(default="", max_length=254)


class BookingCancelRequest(BaseModel):
    booking_uid: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    reason: str = Field(default="", max_length=200)


class OutboundCallRequest(BaseModel):
    phone_number: str = Field(pattern=r"^\+[1-9]\d{7,14}$", description="E.164, e.g. +919988776655")
