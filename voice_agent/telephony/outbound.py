"""
Start an outbound phone call.

Dispatching creates a LiveKit room and asks the telephony worker
(``telephony_vobiz.py``) to join it. The worker reads the phone number from
the dispatch metadata, dials it through the SIP trunk and starts talking once
the call is answered.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from voice_agent import config
from voice_agent.booking import store
from voice_agent.logging_config import mask_phone

logger = logging.getLogger(__name__)

_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")


class TelephonyNotConfiguredError(RuntimeError):
    """LiveKit / SIP trunk settings are missing."""


def normalize_phone(raw: str) -> str:
    """Strip spaces, dashes and brackets; require E.164 (``+`` and country code)."""
    cleaned = re.sub(r"[\s().-]", "", raw or "")
    if not _E164_RE.fullmatch(cleaned):
        raise ValueError("Phone numbers must be in international format, for example +919988776655.")
    return cleaned


async def dispatch_call(phone_number: str, livekit_api: Any = None) -> dict[str, str]:
    """
    Queue a call to ``phone_number``. Returns the room name and dispatch id.

    ``livekit_api`` lets tests pass a stand-in for the ``livekit.api`` module.
    """
    phone = normalize_phone(phone_number)
    if not config.telephony_configured():
        raise TelephonyNotConfiguredError(
            "Telephony is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET."
        )
    if not config.OUTBOUND_TRUNK_ID:
        raise TelephonyNotConfiguredError(
            "No outbound SIP trunk. Run setup_vobiz_trunk.py and set OUTBOUND_TRUNK_ID."
        )

    if livekit_api is None:
        try:
            from livekit import api as livekit_api
        except ImportError as exc:
            raise TelephonyNotConfiguredError("The LiveKit SDK is not installed (pip install livekit-api).") from exc

    # Random room name: phone numbers must not end up in room names or URLs.
    room = f"call-{uuid.uuid4().hex[:12]}"
    client = livekit_api.LiveKitAPI(
        url=config.LIVEKIT_URL, api_key=config.LIVEKIT_API_KEY, api_secret=config.LIVEKIT_API_SECRET
    )
    try:
        dispatch = await client.agent_dispatch.create_dispatch(
            livekit_api.CreateAgentDispatchRequest(
                agent_name=config.LIVEKIT_AGENT_NAME,
                room=room,
                metadata=json.dumps({"phone_number": phone}),
            )
        )
    finally:
        await client.aclose()

    store.record_call(room, "outbound", phone, status="queued")
    logger.info("Outbound call queued to %s (room %s)", mask_phone(phone), room)
    return {"room": room, "dispatch_id": str(getattr(dispatch, "id", ""))}
