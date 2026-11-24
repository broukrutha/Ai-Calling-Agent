"""
``/ws/voice`` — the real-time channel used by the web client.

Client → server
    {"type": "ping"}
    {"type": "query", "query": "...", "language": "en-IN"}
    {"type": "audio", "data": "<base64>", "format": "webm", "language": "unknown", "duration_ms": 2300}
    {"type": "cancel"}                                      stop the turn in progress (barge-in)

Server → client
    {"type": "pong", "ready": true}
    {"type": "processing"}                                  a new turn has started
    {"type": "transcript",  "text": "..."}
    {"type": "noise",       "message": "..."}
    {"type": "text_chunk",  "text": "..."}
    {"type": "audio_chunk", "audio": "<base64 wav>"}        one per sentence, in order
    {"type": "done",        "answer": ..., "latency": {...}, "cost_inr": ..., ...}
    {"type": "text_result", ...}                            reply to "query"
    {"type": "error",       "message": "..."}

Each connection has its own conversation memory and booking state.

Turns run in a task beside the receive loop, so the caller can interrupt: a
``cancel`` message, or a new utterance arriving mid-answer, stops the current
turn at once instead of paying for the rest of an answer nobody will hear.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import re
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_agent import config
from voice_agent.logging_config import request_id_var
from voice_agent.pipeline import ConversationSession
from voice_agent.sarvam_client import LLMUnavailableError, ProviderError
from voice_agent.server.deps import AgentUnavailableError, get_services_ws
from voice_agent.server.security import allow, websocket_origin_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

_LANGUAGE_RE = re.compile(r"^(unknown|[a-z]{2,3}-IN)$")
_FORMAT_RE = re.compile(r"^[a-z0-9/;=. -]{1,60}$")
# Base64 inflates by 4/3; allow a little room for the JSON envelope.
_MAX_MESSAGE_CHARS = config.MAX_AUDIO_BYTES * 4 // 3 + 4096


async def _send(websocket: WebSocket, **payload) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


async def _error(websocket: WebSocket, message: str) -> None:
    await _send(websocket, type="error", message=message)


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    if not websocket_origin_allowed(websocket):
        logger.warning("WebSocket refused: origin %r is not allowed", websocket.headers.get("origin"))
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    request_id_var.set(f"ws-{uuid.uuid4().hex[:8]}")
    services = get_services_ws(websocket)
    session = ConversationSession()
    current: asyncio.Task | None = None
    logger.info("WebSocket connected")

    async def stop_current() -> None:
        nonlocal current
        if current is not None and not current.done():
            current.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current
        current = None

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > _MAX_MESSAGE_CHARS:
                await _error(websocket, "That recording is too long. Please keep it under a minute.")
                continue
            try:
                message = json.loads(raw)
                kind = message.get("type") if isinstance(message, dict) else None
            except ValueError:
                kind = None
            if kind is None:
                await _error(websocket, "Malformed message.")
                continue

            if kind == "ping":
                await _send(websocket, type="pong", ready=services.agent_ready)
                continue
            if kind == "cancel":
                await stop_current()
                continue
            if kind not in ("query", "audio"):
                await _error(websocket, "Unknown message type.")
                continue

            if allow(websocket, "expensive"):
                await _error(websocket, "You're sending requests too quickly. Please wait a moment.")
                continue

            await stop_current()          # a new turn interrupts the previous one
            current = asyncio.create_task(_run_turn(websocket, services, session, kind, message))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await stop_current()


async def _run_turn(websocket: WebSocket, services, session: ConversationSession, kind: str, message: dict) -> None:
    try:
        agent = services.agent
        if kind == "query":
            await _handle_query(websocket, agent, session, message)
        else:
            await _handle_audio(websocket, agent, session, message)
    except asyncio.CancelledError:
        raise
    except AgentUnavailableError as exc:
        await _safe_error(websocket, str(exc))
    except LLMUnavailableError:
        logger.exception("LLM unavailable")
        await _safe_error(websocket, "The assistant is temporarily unavailable. Please try again shortly.")
    except ProviderError:
        logger.exception("Speech provider error")
        await _safe_error(websocket, "The speech service is temporarily unavailable. Please try again shortly.")
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unhandled error in voice WebSocket")
        await _safe_error(websocket, "Something went wrong on our side. Please try again.")


async def _safe_error(websocket: WebSocket, message: str) -> None:
    with contextlib.suppress(Exception):
        await _error(websocket, message)


async def _handle_query(websocket: WebSocket, agent, session: ConversationSession, message: dict) -> None:
    query = str(message.get("query") or "").strip()
    if not query:
        await _error(websocket, "Please type a question.")
        return
    if len(query) > config.MAX_QUERY_CHARS:
        await _error(websocket, f"Questions are limited to {config.MAX_QUERY_CHARS} characters.")
        return
    language = str(message.get("language") or "en-IN")
    await _send(websocket, type="processing")
    result = await agent.query_text(
        query, session=session, language=language if _LANGUAGE_RE.match(language) else "en-IN"
    )
    await _send(websocket, type="text_result", **result)


async def _handle_audio(websocket: WebSocket, agent, session: ConversationSession, message: dict) -> None:
    try:
        audio = base64.b64decode(str(message.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        await _error(websocket, "The recording could not be decoded.")
        return
    if not audio:
        await _error(websocket, "The recording was empty.")
        return
    if len(audio) > config.MAX_AUDIO_BYTES:
        await _error(websocket, "That recording is too long. Please keep it under a minute.")
        return

    language = str(message.get("language") or "unknown")
    audio_format = str(message.get("format") or "webm").lower()
    duration_ms = message.get("duration_ms")
    seconds = duration_ms / 1000 if isinstance(duration_ms, (int, float)) and 0 < duration_ms < 600_000 else None

    await _send(websocket, type="processing")
    async for event in agent.run_stream(
        audio_bytes=audio,
        language_code=language if _LANGUAGE_RE.match(language) else "unknown",
        audio_format=audio_format if _FORMAT_RE.match(audio_format) else "webm",
        session=session,
        audio_seconds=seconds,
    ):
        if event["type"] == "audio_chunk":
            event = {**event, "audio": base64.b64encode(event["audio"]).decode()}
        await _send(websocket, **event)
