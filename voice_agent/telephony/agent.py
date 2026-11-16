"""
LiveKit telephony worker: answers and places phone calls over a Vobiz SIP trunk.

    Inbound:   phone → Vobiz SIP → LiveKit room → this worker
    Outbound:  dispatch (CLI / API) → this worker → dials through the SIP trunk

Per call: Sarvam streaming STT → LLM with tools (knowledge base, booking,
cancellation, hang-up) → Sarvam TTS. Every call is recorded in the call log.

Run with:  python telephony_vobiz.py start      (or ``dev`` for hot reload)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.plugins import openai as lk_openai

from voice_agent import config
from voice_agent.booking import store
from voice_agent.logging_config import configure_logging, mask_phone
from voice_agent.telephony import tools

try:
    from livekit.plugins import sarvam as lk_sarvam
except ImportError:  # pragma: no cover - depends on the optional plugin
    lk_sarvam = None

logger = logging.getLogger("telephony")

_rag = None
_rag_lock = asyncio.Lock()


async def _get_rag():
    """Knowledge-base retriever, shared by all calls handled in this process."""
    global _rag
    async with _rag_lock:
        if _rag is None:
            from voice_agent.rag_pipeline import RAGPipeline
            from voice_agent.sarvam_client import SarvamClient

            def build():
                return RAGPipeline(sarvam_client=SarvamClient())

            _rag = await asyncio.to_thread(build)
        return _rag


class PhoneAssistant(Agent):
    """The receptionist. Tools give it real abilities instead of guesses."""

    def __init__(self, greeting: str) -> None:
        super().__init__(instructions=tools.build_instructions())
        self._greeting = greeting

    async def on_enter(self) -> None:
        await self.session.say(self._greeting, allow_interruptions=True)

    @function_tool
    async def search_knowledge_base(self, context: RunContext, query: str) -> str:
        """Look up facts about the business: services, prices, opening hours, location, policies.

        Args:
            query: What the caller wants to know, as a short search phrase.
        """
        return await tools.search_knowledge_base(await _get_rag(), query)

    @function_tool
    async def check_availability(self, context: RunContext, date: str) -> str:
        """List the open appointment times on a given day.

        Args:
            date: The day to check, as YYYY-MM-DD.
        """
        return await tools.check_availability(date)

    @function_tool
    async def book_appointment(self, context: RunContext, date: str, time: str, name: str, email: str) -> str:
        """Book an appointment. Call only after the caller has confirmed every detail.

        Args:
            date: Appointment day as YYYY-MM-DD.
            time: Start time as 24-hour HH:MM, for example 14:00.
            name: The caller's full name.
            email: The caller's email address, for example rahul@gmail.com.
        """
        return await tools.book_appointment(date, time, name, email)

    @function_tool
    async def cancel_appointment(self, context: RunContext, reference_number: str) -> str:
        """Cancel an appointment. Call only after the caller has confirmed they want to cancel.

        Args:
            reference_number: The six digit reference number given when the appointment was booked.
        """
        return await tools.cancel_appointment(reference_number)

    @function_tool
    async def end_call(self, context: RunContext) -> str:
        """Hang up. Call after saying goodbye, once the caller has nothing more to ask."""
        await context.wait_for_playout()
        await _hang_up(get_job_context())
        return "The call has ended."


async def _hang_up(ctx: JobContext) -> None:
    """Deleting the room disconnects the phone leg."""
    try:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception:
        logger.warning("Could not delete room %s", ctx.room.name, exc_info=True)


def _build_llm():
    if config.OPENAI_API_KEY:
        logger.info("Call LLM: OpenAI %s", config.OPENAI_MODEL)
        return lk_openai.LLM(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, max_completion_tokens=250)
    if config.GROQ_API_KEY:
        logger.info("Call LLM: Groq %s", config.GROQ_MODEL)
        return lk_openai.LLM(
            model=config.GROQ_MODEL, base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY,
            max_completion_tokens=250,
        )
    raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY: the phone agent needs an LLM.")


def _build_stt():
    if lk_sarvam is None:
        raise RuntimeError("Install livekit-plugins-sarvam: the phone agent needs it for speech recognition.")
    return lk_sarvam.STT(
        language="unknown",                      # auto-detect the caller's language
        model=config.TELEPHONY_STT_MODEL,
        mode=config.TELEPHONY_STT_MODE,
        flush_signal=True,
        sample_rate=16000,
    )


def _build_tts():
    if lk_sarvam is not None:
        return lk_sarvam.TTS(
            target_language_code=config.TTS_LANGUAGE,
            model=config.TTS_MODEL,
            speaker=config.TTS_SPEAKER,
            speech_sample_rate=config.TELEPHONY_TTS_SAMPLE_RATE,
        )
    if config.OPENAI_API_KEY:
        logger.warning("Sarvam plugin missing — using OpenAI TTS.")
        return lk_openai.TTS(model="tts-1", voice="alloy")
    raise RuntimeError("No text-to-speech available: install livekit-plugins-sarvam.")


def _dial_target(ctx: JobContext) -> str:
    """Phone number to call, from dispatch metadata. Empty for inbound calls."""
    try:
        return str(json.loads(ctx.job.metadata or "{}").get("phone_number", "")).strip()
    except (ValueError, AttributeError):
        return ""


def _caller_number(participant) -> str:
    attributes = participant.attributes or {}
    number = attributes.get("sip.phoneNumber") or attributes.get("phoneNumber") or ""
    if not number:
        match = re.search(r"\+\d{7,15}", participant.identity or "")
        number = match.group() if match else ""
    return number


async def entrypoint(ctx: JobContext) -> None:
    """Runs once per call."""
    await ctx.connect()
    room = ctx.room.name
    dial_to = _dial_target(ctx)
    direction = "outbound" if dial_to else "inbound"

    if dial_to:
        if not config.OUTBOUND_TRUNK_ID:
            logger.error("Outbound call requested but OUTBOUND_TRUNK_ID is not set.")
            store.record_call(room, direction, dial_to, status="failed")
            store.finish_call(room, "failed", detail="OUTBOUND_TRUNK_ID is not configured")
            ctx.shutdown("no outbound trunk")
            return

        identity = f"sip-{room}"
        store.record_call(room, direction, dial_to, status="ringing")
        logger.info("Dialling %s", mask_phone(dial_to))
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=room,
                    sip_trunk_id=config.OUTBOUND_TRUNK_ID,
                    sip_call_to=dial_to,
                    participant_identity=identity,
                    participant_name="Caller",
                    wait_until_answered=True,       # returns when the phone is picked up
                )
            )
        except api.TwirpError as exc:
            sip_status = (exc.metadata or {}).get("sip_status", "")
            detail = f"{exc.message} {sip_status}".strip()
            logger.warning("Call to %s was not answered: %s", mask_phone(dial_to), detail)
            store.finish_call(room, "failed", detail=detail or "not answered")
            await _hang_up(ctx)
            ctx.shutdown("call not answered")
            return
        participant = await ctx.wait_for_participant(identity=identity)
        phone = dial_to
    else:
        participant = await ctx.wait_for_participant()
        phone = _caller_number(participant)
        store.record_call(room, direction, phone, status="ringing")

    store.mark_call_answered(room)
    logger.info("Call connected: %s %s", direction, mask_phone(phone) or "unknown")

    session = AgentSession(
        stt=_build_stt(),
        llm=_build_llm(),
        tts=_build_tts(),
        turn_detection="stt",
        min_endpointing_delay=0.05,
        allow_interruptions=True,
    )
    turns = 0
    finished = False

    @session.on("user_input_transcribed")
    def _on_transcript(event) -> None:
        nonlocal turns
        if event.is_final and len(event.transcript.strip()) >= 3:
            turns += 1

    def _finish(status: str, detail: str = "") -> None:
        nonlocal finished
        if not finished:
            finished = True
            store.finish_call(room, status, detail=detail, turns=turns)
            logger.info("Call ended (%s): %d caller turns", detail or status, turns)

    @ctx.room.on("participant_disconnected")
    def _on_hangup(gone) -> None:
        if gone.identity == participant.identity:
            _finish("completed", "caller hung up")
            ctx.shutdown("caller hung up")

    async def _on_shutdown() -> None:
        _finish("completed")

    ctx.add_shutdown_callback(_on_shutdown)

    async def _enforce_max_duration() -> None:
        await asyncio.sleep(config.CALL_MAX_DURATION_SEC)
        logger.warning("Call reached the %d s limit — hanging up.", config.CALL_MAX_DURATION_SEC)
        _finish("completed", "maximum call duration reached")
        await _hang_up(ctx)

    watchdog = asyncio.create_task(_enforce_max_duration())
    ctx.add_shutdown_callback(lambda: _cancel(watchdog))

    await session.start(
        room=ctx.room,
        agent=PhoneAssistant(greeting=config.CALL_GREETING),
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()


def main() -> None:
    configure_logging()
    if not config.telephony_configured():
        raise SystemExit(
            "Telephony is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env "
            "(https://cloud.livekit.io)."
        )
    if not config.SARVAM_API_KEY:
        raise SystemExit("SARVAM_API_KEY is required for speech on phone calls.")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=config.LIVEKIT_AGENT_NAME))
