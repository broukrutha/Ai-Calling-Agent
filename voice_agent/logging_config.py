"""
Logging setup: one call configures the root logger for the whole process.

* ``LOG_FORMAT=text`` – human-readable lines for local development.
* ``LOG_FORMAT=json`` – one JSON object per line for log aggregators.

Every record carries the current request id (``-`` outside a request) so a
single call or HTTP request can be followed across modules.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone

from voice_agent.config import LOG_FORMAT, LOG_LEVEL

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_PHONE_RE = re.compile(r"\+?\d[\d\s-]{6,}\d")


def mask_email(value: str) -> str:
    """``rahul@example.com`` → ``r***@example.com``. Safe to put in logs."""
    return _EMAIL_RE.sub(r"\1***@\2", value or "")


def mask_phone(value: str) -> str:
    """``+919988776655`` → ``+91******6655``."""
    def _mask(match: re.Match[str]) -> str:
        digits = match.group(0)
        if len(digits) <= 6:
            return digits
        return digits[:3] + "*" * (len(digits) - 7) + digits[-4:]

    return _PHONE_RE.sub(_mask, value or "")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Idempotent: safe to call from every entry point."""
    root = logging.getLogger()
    if getattr(root, "_voice_agent_configured", False):
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RequestIdFilter())
    if LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(request_id)s | %(message)s")
        )

    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root._voice_agent_configured = True  # type: ignore[attr-defined]

    # Per-request chatter from HTTP clients drowns out the pipeline logs.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
