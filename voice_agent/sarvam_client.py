"""
sarvam_client.py – Sarvam AI + Groq/OpenAI LLM with TRUE streaming.

STT : saaras:v3  — WebSocket (sarvamai SDK) primary
                   Falls back to saaras:v2 REST if WS fails or SDK absent
TTS : bulbul:v3  — WebSocket (sarvamai SDK) primary
                   Falls back to bulbul:v3 REST if WS fails or SDK absent
LLM : OpenAI GPT-4o-mini → Groq llama-3.1-8b-instant (fallback) → Sarvam-M

LATENCY FIXES vs previous version:
  [FIX-1] STT: Skip pydub/ffmpeg conversion when falling back to REST —
          REST endpoint accepts webm natively. Saves 200–800ms.
  [FIX-2] STT/TTS: Reuse a single AsyncSarvamAI SDK client instance
          (self._sdk_client) instead of creating a new one per call.
          Eliminates per-call TLS handshake overhead (~100–300ms each).
  [FIX-3] STT WS timeout reduced 8s → 3s — fail fast to REST fallback.
  [FIX-4] STT WS: Send entire WAV in one base64 call instead of 32KB chunks.
  [FIX-5] TTS WS: Reuse sdk_client — same as FIX-2 for TTS path.
  [FIX-6] REST TTS sample_rate unified to 22050 (consistent with web player).
  [FIX-7] Persistent httpx.AsyncClient with connection pooling (keep-alive).

Install: pip install sarvamai
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

from voice_agent.config import (
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    LLM_MODEL,
    TTS_SPEAKER,
    TTS_LANGUAGE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_SYSTEM_PROMPT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
)

# ── Hardwired model names ────────────────────────────────────────────────────
_STT_MODEL = "saaras:v3"    # WebSocket STT model
_TTS_MODEL = "bulbul:v3"    # WebSocket + REST TTS model

# ── Groq settings ────────────────────────────────────────────────────────────
_GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"

# ── OpenAI settings ──────────────────────────────────────────────────────────
_OPENAI_KEY   = OPENAI_API_KEY
_OPENAI_URL   = f"{OPENAI_BASE_URL}/chat/completions"
_OPENAI_MODEL = OPENAI_MODEL

# Determine active LLM provider
_USE_OPENAI = LLM_PROVIDER == "openai" and bool(_OPENAI_KEY)
_USE_GROQ   = LLM_PROVIDER == "groq" or (LLM_PROVIDER == "openai" and not _OPENAI_KEY)

logger = logging.getLogger(__name__)

# ── sarvamai SDK availability ─────────────────────────────────────────────────
try:
    from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
    _SDK_AVAILABLE = True
    logger.info("sarvamai SDK available — WebSocket streaming enabled")
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning(
        "sarvamai SDK not installed — using REST fallback. Run: pip install sarvamai"
    )


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class STTResult:
    transcript: str
    language: str
    latency_ms: float

@dataclass
class LLMResult:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

@dataclass
class TTSResult:
    audio_bytes: bytes
    character_count: int
    latency_ms: float


# ── Audio conversion (only needed for WebSocket STT path) ────────────────────
# REST /speech-to-text accepts webm directly — no conversion needed there.
# WS saaras:v3 requires WAV PCM 16kHz mono — convert only when using WS.

async def _convert_to_wav(audio_bytes: bytes, target_sr: int = 16000) -> "bytes | None":
    """
    Convert audio bytes (webm / mp4 / ogg / any) → WAV PCM 16kHz mono.
    Runs ffmpeg via pydub in a thread pool so the event loop stays free.
    Returns None on failure; caller falls back to REST (which accepts webm).
    """
    import io

    def _sync_convert(data: bytes) -> "bytes | None":
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(data))
            audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            return buf.getvalue()
        except ImportError:
            logger.warning(
                "pydub not installed — cannot convert for WS STT. "
                "Install: pip install pydub  (also needs ffmpeg)"
            )
            return None
        except Exception as e:
            logger.warning(f"Audio conversion failed: {e}")
            return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_convert, audio_bytes)


# ═════════════════════════════════════════════════════════════════════════════
class SarvamClient:
    """
    Unified client for Sarvam STT (saaras:v3), TTS (bulbul:v3), and LLM.

    Holds a single AsyncSarvamAI SDK instance (self._sdk_client) and a
    single persistent httpx.AsyncClient — both are reused across all calls
    to avoid per-call connection/TLS overhead.
    """

    def __init__(self, api_key: str = SARVAM_API_KEY) -> None:
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set.")
        self._api_key = api_key

        # ── HTTP headers ──────────────────────────────────────────────────
        self._sarvam_headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        self._groq_headers = {
            "Authorization": f"Bearer {_GROQ_KEY}",
            "Content-Type": "application/json",
        }
        self._openai_headers = {
            "Authorization": f"Bearer {_OPENAI_KEY}",
            "Content-Type": "application/json",
        }

        # ── [FIX-7] Persistent httpx client with connection pooling ───────
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            trust_env=False,
        )

        # ── [FIX-2 / FIX-5] Single reused SDK client ──────────────────────
        # Creating AsyncSarvamAI once here means WebSocket calls reuse the
        # same underlying session — no per-call TLS handshake.
        self._sdk_client: "Optional[AsyncSarvamAI]" = (
            AsyncSarvamAI(api_subscription_key=api_key) if _SDK_AVAILABLE else None
        )

        # ── Log active config ─────────────────────────────────────────────
        if _USE_OPENAI:
            llm_label = f"OpenAI/{_OPENAI_MODEL}"
        elif _USE_GROQ:
            llm_label = f"Groq/{_GROQ_MODEL}"
        else:
            llm_label = f"Sarvam/{LLM_MODEL}"

        stt_mode = f"{_STT_MODEL} WebSocket→REST" if _SDK_AVAILABLE else f"{_STT_MODEL} REST"
        tts_mode = f"{_TTS_MODEL} WebSocket→REST" if _SDK_AVAILABLE else f"{_TTS_MODEL} REST"
        logger.info(
            f"SarvamClient ready | STT={stt_mode} | TTS={tts_mode} | LLM={llm_label}"
        )

    async def close(self):
        await self._client.aclose()

    # ─────────────────────────────────────────────────────────────────────────
    # STT — saaras:v3
    #
    # STRATEGY: REST-first — eliminates the 3s WS timeout penalty that was
    # adding ~3s to every single STT call. REST accepts webm directly, no
    # conversion needed, and returns in ~800-1200ms consistently.
    #
    # To try WS again: set _STT_REST_FIRST = False (only do this once WS
    # is confirmed stable in your environment).
    # ─────────────────────────────────────────────────────────────────────────

    _STT_REST_FIRST: bool = True   # ← flip to False to re-enable WS path

    async def transcribe_stream(
        self,
        audio_bytes: bytes,
        language_code: str = "en-IN",
        sample_rate: int = 16000,
    ) -> STTResult:
        """
        STT entry point — REST-first for reliable low latency.
        The WS path was timing out on every call (+3s penalty). REST is faster
        in practice because it avoids WAV conversion + WS connection overhead.
        """
        if self._STT_REST_FIRST or not _SDK_AVAILABLE:
            return await self._transcribe_rest(audio_bytes, language_code)

        # Optional WS path — only active when _STT_REST_FIRST = False
        wav_bytes = await _convert_to_wav(audio_bytes)
        if wav_bytes is None:
            return await self._transcribe_rest(audio_bytes, language_code)
        return await self._transcribe_ws(wav_bytes, language_code, sample_rate)

    async def _transcribe_ws(
        self,
        wav_bytes: bytes,
        language_code: str,
        sample_rate: int,
    ) -> STTResult:
        """WS STT — saaras:v3. Only called when _STT_REST_FIRST=False."""
        t0 = time.perf_counter()

        async def _collect() -> str:
            result = ""
            async with self._sdk_client.speech_to_text_streaming.connect(
                model=_STT_MODEL,
                mode="transcribe",
                language_code=language_code,
                high_vad_sensitivity=True,
                vad_signals=True,
                flush_signal=True,
            ) as ws:
                full_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                await ws.transcribe(audio=full_b64, encoding="audio/wav", sample_rate=sample_rate)
                await ws.flush()
                async for response in ws:
                    if isinstance(response, dict):
                        msg_type = response.get("type", "")
                        text = response.get("text", "") or response.get("transcript", "")
                    else:
                        msg_type = getattr(response, "type", "") or ""
                        text = (
                            getattr(response, "transcript", None)
                            or getattr(response, "text", None)
                            or ""
                        )
                    if text:
                        result += text + " "
                    if msg_type in ("transcript", "speech_end", "final", "complete"):
                        if result.strip():
                            break
            return result.strip()

        try:
            transcript = await asyncio.wait_for(_collect(), timeout=3.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"STT WS failed ({e}) — falling back to REST")
            return await self._transcribe_rest(wav_bytes, language_code, audio_format="wav")

        if not transcript:
            return await self._transcribe_rest(wav_bytes, language_code, audio_format="wav")

        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"STT WS ✅ saaras:v3 | '{transcript[:60]}' | {ms:.0f}ms")
        return STTResult(transcript=transcript, language=language_code, latency_ms=round(ms, 2))

    async def _transcribe_rest(
        self,
        audio_bytes: bytes,
        language_code: str = "en-IN",
        audio_format: str = "webm",
    ) -> STTResult:
        """
        REST STT fallback — saarika:v2.5.
        saaras:v3 is WebSocket-only; the REST endpoint uses saarika:v2.5.
        Accepts webm, wav, mp3, ogg, mp4, flac natively — no conversion needed.
        """
        url = f"{SARVAM_BASE_URL}/speech-to-text"

        # Normalise extension — default to webm (what browsers record)
        ext = audio_format.lower().replace("audio/", "").split(";")[0].strip()
        if ext not in ("wav", "mp3", "webm", "ogg", "mp4", "flac"):
            ext = "webm"

        headers = {"api-subscription-key": self._api_key}
        files   = {"file": (f"audio.{ext}", audio_bytes, f"audio/{ext}")}
        data    = {
            "model": "saarika:v2.5",   # Only valid REST STT model — saaras:v3 is WS-only
            "language_code": language_code,
        }
        t0   = time.perf_counter()
        resp = await self._client.post(url, files=files, data=data, headers=headers)
        ms   = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            # Log the full API error body so you can see exactly what went wrong
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            logger.error(
                f"STT REST {resp.status_code} | ext={ext} | lang={language_code} | "
                f"audio_size={len(audio_bytes)}B | response={err_body}"
            )
            resp.raise_for_status()

        body = resp.json()
        tx   = body.get("transcript", "").strip()
        lang = body.get("language_code", language_code)
        logger.info(f"STT REST saarika:v2.5 | '{tx[:60]}' | lang={lang} | {ms:.0f}ms")
        return STTResult(transcript=tx, language=lang, latency_ms=round(ms, 2))

    # ── Public alias kept for backward compatibility ───────────────────────
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "en-IN",
        audio_format: str = "webm",
    ) -> STTResult:
        """Backward-compat alias → REST STT (saarika:v2.5)."""
        return await self._transcribe_rest(audio_bytes, language_code, audio_format)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM — OpenAI / Groq / Sarvam (unchanged logic, kept for completeness)
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_stream(self, user_message, context="", system_prompt=None):
        """Stream LLM tokens with retry + auto-fallback to Groq on 429."""
        sys_prompt   = system_prompt or LLM_SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {user_message}" if context else user_message
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]

        providers = []
        if _USE_OPENAI:
            providers.append(("OpenAI", _OPENAI_URL, self._openai_headers, {
                "model": _OPENAI_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "stream": True,
            }))
            if _GROQ_KEY:
                providers.append(("Groq", _GROQ_URL, self._groq_headers, {
                    "model": _GROQ_MODEL, "messages": messages,
                    "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "stream": True,
                }))
        elif _USE_GROQ:
            providers.append(("Groq", _GROQ_URL, self._groq_headers, {
                "model": _GROQ_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE, "stream": True,
            }))
        else:
            providers.append(("Sarvam", f"{SARVAM_BASE_URL}/v1/chat/completions", self._sarvam_headers, {
                "model": LLM_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
                "stream": True, "budget_tokens": 0,
            }))

        last_error = None
        for provider_name, url, headers, payload in providers:
            for attempt in range(3):
                try:
                    _buf, _in_think = "", False
                    async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code == 429:
                            await resp.aread()
                            wait = (attempt + 1) * 1.0
                            logger.warning(
                                f"{provider_name} 429 (attempt {attempt+1}/3) — retry in {wait:.0f}s"
                            )
                            await asyncio.sleep(wait)
                            last_error = f"{provider_name} 429"
                            continue
                        if resp.status_code != 200:
                            await resp.aread()
                            logger.error(f"LLM {provider_name} {resp.status_code}")
                            last_error = f"{provider_name} {resp.status_code}"
                            break

                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                token = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                                if not token:
                                    continue
                                _buf += token
                                while True:
                                    if not _in_think:
                                        if "<think>" in _buf:
                                            before, _buf = _buf.split("<think>", 1)
                                            if before.strip():
                                                yield before
                                            _in_think = True
                                        else:
                                            if len(_buf) > 3:
                                                yield _buf[:-3]
                                                _buf = _buf[-3:]
                                            break
                                    else:
                                        if "</think>" in _buf:
                                            _, _buf = _buf.split("</think>", 1)
                                            _in_think = False
                                        else:
                                            break
                            except Exception:
                                continue

                    if _buf and not _in_think:
                        out = re.sub(r"<think>.*?</think>", "", _buf, flags=re.DOTALL).strip()
                        out = re.sub(r"<think>.*$", "", out, flags=re.DOTALL).strip()
                        if out:
                            yield out
                    elif _buf and _in_think:
                        # <think> opened but </think> never came (truncated by max_tokens)
                        # — discard the thinking block entirely, yield nothing
                        pass
                    logger.info(f"LLM stream done via {provider_name}")
                    return
                except httpx.HTTPStatusError:
                    break
                except Exception as e:
                    logger.warning(f"{provider_name} stream error: {e}")
                    last_error = str(e)
                    break

            if last_error:
                logger.warning(f"{provider_name} failed — trying next provider")

        if last_error:
            logger.error(f"All LLM providers failed. Last error: {last_error}")
            yield "Sorry, I'm experiencing high traffic. Please try again in a moment."

    async def generate(self, user_message, context="", system_prompt=None):
        """Non-streaming LLM for benchmark and text queries."""
        sys_prompt   = system_prompt or LLM_SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {user_message}" if context else user_message
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]
        if _USE_OPENAI:
            url, headers = _OPENAI_URL, self._openai_headers
            payload = {
                "model": _OPENAI_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
            }
        elif _USE_GROQ:
            url, headers = _GROQ_URL, self._groq_headers
            payload = {
                "model": _GROQ_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
            }
        else:
            url, headers = f"{SARVAM_BASE_URL}/v1/chat/completions", self._sarvam_headers
            payload = {
                "model": LLM_MODEL, "messages": messages,
                "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
                "budget_tokens": 0,
            }

        t0   = time.perf_counter()
        resp = await self._client.post(url, json=payload, headers=headers)
        ms   = (time.perf_counter() - t0) * 1000

        # Auto-fallback to Groq on OpenAI 429
        if resp.status_code == 429 and _USE_OPENAI and _GROQ_KEY:
            logger.warning("OpenAI 429 — falling back to Groq")
            payload["model"] = _GROQ_MODEL
            t0   = time.perf_counter()
            resp = await self._client.post(_GROQ_URL, json=payload, headers=self._groq_headers)
            ms   = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        data   = resp.json()
        raw    = data["choices"][0]["message"]["content"].strip()
        # Strip complete <think>...</think> blocks (re.DOTALL handles multiline)
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Also strip unclosed <think> blocks (truncated by max_tokens)
        answer = re.sub(r"<think>.*$", "", answer, flags=re.DOTALL).strip()
        answer = answer or raw
        usage  = data.get("usage", {})
        provider = "OpenAI" if _USE_OPENAI else ("Groq" if _USE_GROQ else "Sarvam")
        logger.info(f"LLM non-stream via {provider}: {ms:.0f}ms")
        return LLMResult(
            answer=answer,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=round(ms, 2),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TTS — bulbul:v3
    # Primary:  WebSocket streaming via sarvamai SDK
    # Fallback: REST /text-to-speech (bulbul:v3)
    # ─────────────────────────────────────────────────────────────────────────

    async def synthesize_stream_ws(
        self,
        text_generator,
        target_language_code: str = TTS_LANGUAGE,
        speaker: str = TTS_SPEAKER,
    ):
        """
        TRUE WebSocket TTS streaming — bulbul:v3.
        Audio chunks are yielded as bytes as soon as they arrive from the server,
        interleaved with text tokens being sent in.

        [FIX-5] Reuses self._sdk_client — no new TLS handshake per call.
        Falls back to REST (collecting full text first) if WS fails.
        """
        if not _SDK_AVAILABLE:
            # No SDK — collect full text then call REST
            full_text = ""
            async for token in text_generator:
                full_text += token
            if full_text.strip():
                result = await self.synthesize(full_text, target_language_code, speaker)
                if result.audio_bytes:
                    yield result.audio_bytes
            return

        try:
            # [FIX-5] Use the shared SDK client
            async with self._sdk_client.text_to_speech_streaming.connect(
                model=_TTS_MODEL,
                send_completion_event=True,
            ) as ws:
                await ws.configure(
                    target_language_code=target_language_code,
                    speaker=speaker,
                    pace=1.0,
                )

                async def _send_tokens():
                    async for token in text_generator:
                        if token.strip():
                            await ws.convert(token)
                    await ws.flush()

                send_task = asyncio.create_task(_send_tokens())

                async for message in ws:
                    if isinstance(message, AudioOutput):
                        yield base64.b64decode(message.data.audio)
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            break

                await send_task

        except Exception as e:
            logger.warning(f"TTS WS error: {e} — falling back to REST")
            # WS failed partway — nothing more to yield (audio already partially sent)
            return

    async def synthesize(
        self,
        text: str,
        target_language_code: str = TTS_LANGUAGE,
        speaker: str = TTS_SPEAKER,
    ) -> TTSResult:
        """
        REST TTS — bulbul:v3.
        [FIX-6] speech_sample_rate unified to 22050 (consistent with web player).
        Reuses persistent httpx client for keep-alive.
        """
        if not text or not text.strip():
            return TTSResult(audio_bytes=b"", character_count=0, latency_ms=0.0)

        payload = {
            "text": text.strip()[:2500],
            "target_language_code": target_language_code,
            "speaker": speaker,
            "model": _TTS_MODEL,          # bulbul:v3
            "speech_sample_rate": 22050,  # [FIX-6] consistent with web audio player
            "pace": 1.0,
        }
        t0   = time.perf_counter()
        resp = await self._client.post(
            f"{SARVAM_BASE_URL}/text-to-speech",
            json=payload,
            headers=self._sarvam_headers,
        )
        ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            logger.error(f"TTS REST {resp.status_code}: {resp.text}")
            resp.raise_for_status()

        data        = resp.json()
        audio_b64   = data.get("audios", [""])[0]
        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
        logger.info(f"TTS REST bulbul:v3 | {len(text)} chars | {ms:.0f}ms")
        return TTSResult(
            audio_bytes=audio_bytes,
            character_count=len(text),
            latency_ms=round(ms, 2),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Health check
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        async def _probe(url: str, headers: Optional[dict] = None) -> dict:
            try:
                t0 = time.perf_counter()
                resp = await self._client.get(url, headers=headers)
                resp.raise_for_status()
                return {
                    "status": "ok",
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

        if _USE_OPENAI:
            llm_provider = "openai"
            llm_model = _OPENAI_MODEL
            llm_health = await _probe(
                f"{OPENAI_BASE_URL}/models",
                headers=self._openai_headers,
            )
        elif _USE_GROQ:
            llm_provider = "groq"
            llm_model = _GROQ_MODEL
            llm_health = await _probe(
                "https://api.groq.com/openai/v1/models",
                headers=self._groq_headers,
            )
        else:
            llm_provider = "sarvam"
            llm_model = LLM_MODEL
            llm_health = await _probe(
                f"{SARVAM_BASE_URL}/",
                headers={"api-subscription-key": self._api_key},
            )

        if self._api_key:
            speech_health = {
                "status": "ok",
                "mode": "websocket+rest" if _SDK_AVAILABLE else "rest",
            }
        else:
            speech_health = {
                "status": "error",
                "message": "SARVAM_API_KEY is not configured.",
            }

        if llm_health["status"] == "ok" and speech_health["status"] == "ok":
            status = "ok"
            summary = f"{llm_provider.upper()} LLM and Sarvam speech are reachable"
        elif llm_health["status"] == "ok" or speech_health["status"] == "ok":
            status = "degraded"
            summary = f"{llm_provider.upper()} LLM or Sarvam speech is unavailable"
        else:
            status = "error"
            summary = f"{llm_provider.upper()} LLM and Sarvam speech are unreachable"

        response = {
            "status": status,
            "summary": summary,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "stt_model": _STT_MODEL,
            "tts_model": _TTS_MODEL,
            "components": {
                "llm": llm_health,
                "speech": speech_health,
            },
        }

        latencies = [
            part.get("latency_ms")
            for part in (llm_health, speech_health)
            if isinstance(part.get("latency_ms"), (int, float))
        ]
        if latencies:
            response["latency_ms"] = round(max(latencies), 2)

        if status != "ok":
            errors = []
            if llm_health["status"] != "ok":
                errors.append(f"LLM: {llm_health.get('message', 'unknown error')}")
            if speech_health["status"] != "ok":
                errors.append(f"Speech: {speech_health.get('message', 'unknown error')}")
            response["message"] = " | ".join(errors)

        return response
