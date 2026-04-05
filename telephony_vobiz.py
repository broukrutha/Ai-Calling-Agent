"""
telephony_vobiz.py – LiveKit + Vobiz SIP telephony integration.

HOW IT WORKS:
  1. This agent registers with LiveKit Cloud as a worker
  2. When a call comes in via Vobiz SIP trunk → LiveKit routes it to this agent
  3. Agent joins the LiveKit room, receives audio from the caller
  4. Audio → STT (Sarvam) → RAG → LLM (OpenAI GPT-4o-mini) → TTS (Sarvam) → audio back
  5. For outbound calls: use make_vobiz_call.py to dispatch a call

SETUP:
  1. Sign up at cloud.livekit.io → get URL, API Key, Secret
  2. Sign up at Vobiz → get SIP Domain, Username, Password
  3. Purchase a Vobiz DID number (Indian number)
  4. Run setup_vobiz_trunk.py to create/update the SIP trunk in LiveKit
  5. Configure your Vobiz number to point to LiveKit's SIP endpoint
  6. Start this agent: python telephony_vobiz.py start

ARCHITECTURE:
  Phone Call → Vobiz SIP → LiveKit Cloud → This Agent (STT/RAG/LLM/TTS)
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("vobiz-agent")
logging.basicConfig(level=logging.INFO)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Check if LiveKit agents SDK is available ─────────────────────────────────
try:
    from livekit import api as lk_api
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        RoomInputOptions,
        WorkerOptions,
        cli,
        llm,
    )
    from livekit.plugins import openai as lk_openai
    from livekit.plugins import silero

    # Try importing Sarvam LiveKit plugin
    try:
        from livekit.plugins import sarvam as lk_sarvam
        _SARVAM_PLUGIN = True
        logger.info("LiveKit Sarvam plugin available")
    except ImportError:
        _SARVAM_PLUGIN = False
        logger.info("LiveKit Sarvam plugin not installed — will use custom pipeline")

    _LIVEKIT_AVAILABLE = True
    logger.info("LiveKit agents SDK available")
except ImportError:
    _LIVEKIT_AVAILABLE = False
    logger.warning(
        "LiveKit agents SDK not installed. Run:\n"
        "  pip install livekit-agents livekit-plugins-openai livekit-plugins-silero livekit-plugins-sarvam"
    )


# ── Import our voice pipeline ────────────────────────────────────────────────
from voice_agent.config import (
    SARVAM_API_KEY,
    LLM_SYSTEM_PROMPT,
    TTS_SPEAKER,
    TTS_LANGUAGE,
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    VOBIZ_SIP_DOMAIN,
    OUTBOUND_TRUNK_ID,
)


# ══════════════════════════════════════════════════════════════════════════════
# VOICE PIPELINE AGENT (works with or without LiveKit Sarvam plugin)
# ══════════════════════════════════════════════════════════════════════════════

if _LIVEKIT_AVAILABLE:

    class VoiceAssistant(Agent):
        """
        LiveKit Agent that handles voice calls via Vobiz SIP trunk.

        Uses your existing RAG pipeline + Sarvam STT/TTS + OpenAI GPT-4o-mini
        to provide intelligent voice responses.
        """

        def __init__(self, caller_phone: str = "unknown"):
            # Build the system instructions using our existing prompt
            instructions = (
                LLM_SYSTEM_PROMPT + "\n\n"
                "You are a helpful voice assistant answering phone calls. "
                "Keep responses short and conversational (1-2 sentences max). "
                "If you don't know the answer, say so briefly. "
                "Be warm and natural — you're on a phone call."
            )

            super().__init__(instructions=instructions)
            self.caller_phone = caller_phone
            self._rag = None
            self._sarvam = None

        async def _ensure_pipeline(self):
            """Lazy-init the RAG pipeline and Sarvam client."""
            if self._rag is None:
                from voice_agent.rag_pipeline import RAGPipeline
                from voice_agent.sarvam_client import SarvamClient
                self._sarvam = SarvamClient()
                self._rag = RAGPipeline(sarvam_client=self._sarvam)
                logger.info("[AGENT] RAG pipeline initialized")

        async def on_enter(self):
            """Called when the agent enters the session — greet the caller."""
            greeting = (
                "Hello! I am your voice assistant. How can I help you today?"
            )
            await self.session.generate_reply(
                instructions=f"Say exactly this phrase: '{greeting}'"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # ENTRYPOINT — called by LiveKit when a call comes in
    # ════════════════════════════════════════════════════════════════════════════

    async def vobiz_entrypoint(ctx: JobContext):
        """Main entrypoint for LiveKit agent — handles Vobiz SIP calls."""
        await ctx.connect()
        logger.info(f"[ROOM] Connected: {ctx.room.name}")

        # ── Extract caller info ───────────────────────────────────────────
        phone_number = "unknown"

        # Try metadata (outbound dispatch)
        metadata = ctx.job.metadata or ""
        if metadata:
            try:
                meta = json.loads(metadata)
                phone_number = meta.get("phone_number", "unknown")
            except Exception:
                pass

        # Try SIP participant attributes
        for identity, participant in ctx.room.remote_participants.items():
            attr = participant.attributes or {}
            sip_phone = attr.get("sip.phoneNumber") or attr.get("phoneNumber")
            if sip_phone:
                phone_number = sip_phone
                break
            # Try extracting from identity
            if "+" in identity:
                m = re.search(r"\+\d{7,15}", identity)
                if m:
                    phone_number = m.group()
                    break

        logger.info(f"[CALLER] Phone: {phone_number}")

        # ── Build STT ─────────────────────────────────────────────────────
        if _SARVAM_PLUGIN:
            agent_stt = lk_sarvam.STT(
                language="unknown",       # auto-detect
                model="saaras:v3",
                mode="translate",
                flush_signal=True,
                sample_rate=16000,
            )
            logger.info("[STT] Using Sarvam Saaras v3 (LiveKit plugin)")
        else:
            # Fallback: use Silero VAD + custom STT
            agent_stt = None
            logger.warning("[STT] Sarvam plugin not available — manual STT needed")

        # ── Build LLM ─────────────────────────────────────────────────────
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            agent_llm = lk_openai.LLM(
                model="gpt-4o-mini",
                max_completion_tokens=120,
            )
            logger.info("[LLM] Using OpenAI GPT-4o-mini")
        else:
            # Fallback to Groq via OpenAI-compatible API
            agent_llm = lk_openai.LLM.with_groq(
                model="llama-3.3-70b-versatile",
                max_completion_tokens=120,
            )
            logger.info("[LLM] Using Groq (OpenAI key not found)")

        # ── Build TTS ─────────────────────────────────────────────────────
        if _SARVAM_PLUGIN:
            agent_tts = lk_sarvam.TTS(
                target_language_code=TTS_LANGUAGE,
                model="bulbul:v3",
                speaker=TTS_SPEAKER,
                speech_sample_rate=24000,
            )
            logger.info(f"[TTS] Using Sarvam Bulbul v3 — voice: {TTS_SPEAKER}")
        else:
            # Fallback: OpenAI TTS
            agent_tts = lk_openai.TTS(
                model="tts-1",
                voice="alloy",
            )
            logger.info("[TTS] Using OpenAI TTS (Sarvam plugin not available)")

        # ── Build agent ───────────────────────────────────────────────────
        agent = VoiceAssistant(caller_phone=phone_number)

        # ── Build session ─────────────────────────────────────────────────
        session_kwargs = {
            "llm": agent_llm,
            "tts": agent_tts,
            "turn_detection": "stt",
            "min_endpointing_delay": 0.05,
            "allow_interruptions": True,
        }

        if agent_stt:
            session_kwargs["stt"] = agent_stt

        session = AgentSession(**session_kwargs)

        room_input = RoomInputOptions(close_on_disconnect=False)
        await session.start(room=ctx.room, agent=agent, room_input_options=room_input)

        # Try pre-warming TTS
        try:
            await session.tts.prewarm()
            logger.info("[TTS] Pre-warmed successfully")
        except Exception as e:
            logger.debug(f"[TTS] Pre-warm skipped: {e}")

        logger.info("[AGENT] Session live — waiting for caller audio.")

        # ── Track call events ─────────────────────────────────────────────
        call_start = datetime.now()
        turn_count = 0

        @session.on("user_speech_committed")
        def on_user_speech(ev):
            nonlocal turn_count
            transcript = ev.user_transcript.strip()
            if not transcript or len(transcript) < 3:
                return
            turn_count += 1
            logger.info(f"[TURN {turn_count}] Caller: '{transcript}'")

        @ctx.room.on("participant_disconnected")
        def on_disconnect(participant):
            duration = int((datetime.now() - call_start).total_seconds())
            logger.info(
                f"[HANGUP] Call ended | Duration: {duration}s | "
                f"Turns: {turn_count} | Caller: {phone_number}"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # CLI ENTRY POINT
    # ════════════════════════════════════════════════════════════════════════════

    def start_vobiz_agent():
        """Start the LiveKit agent worker for Vobiz calls."""
        if not _LIVEKIT_AVAILABLE:
            print("❌ LiveKit agents SDK not installed!")
            print("Run: pip install livekit-agents livekit-plugins-openai livekit-plugins-silero livekit-plugins-sarvam")
            return

        if not LIVEKIT_URL:
            print("❌ LIVEKIT_URL not set in .env!")
            print("Sign up at https://cloud.livekit.io and add your credentials.")
            return

        cli.run_app(WorkerOptions(
            entrypoint_fnc=vobiz_entrypoint,
            agent_name="voice-assistant",
        ))


if __name__ == "__main__":
    if _LIVEKIT_AVAILABLE:
        start_vobiz_agent()
    else:
        print("❌ Cannot start — LiveKit agents SDK not installed.")
        print("Run: pip install livekit-agents livekit-plugins-openai livekit-plugins-silero livekit-plugins-sarvam")
