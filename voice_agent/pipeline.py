"""
The voice pipeline: audio → STT → (booking dialog | RAG + LLM) → TTS → audio.

Latency design
--------------
What a caller perceives is the time until they *hear* something, not the time
until the whole answer exists. So the LLM answer is streamed, cut into
sentences as it arrives, and each sentence is sent to TTS immediately while
the LLM keeps writing. Audio is released strictly in sentence order.

    LLM tokens ─► sentence 1 ─► TTS ─► audio 1   (playing)
               ─► sentence 2 ─► TTS ─► audio 2   (ready, queued)
               ─► sentence 3 ─► TTS …

``first_audio_ms`` in the latency report is that perceived delay.

Every turn — answered, booked, filtered as noise, or failed — is written to
the trace store (:mod:`voice_agent.tracing`) with its latency breakdown,
provider, retrieved passages and cost. Its id travels back to the client, so
the person can rate the answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import struct
import time
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from voice_agent import config, tracing
from voice_agent.booking import BookingSession, handle_booking_query, is_booking_intent, is_cancel_intent
from voice_agent.rag_pipeline import RAGPipeline, Retrieval
from voice_agent.sarvam_client import (
    LANGUAGE_NAMES,
    LLMStreamStats,
    LLMUnavailableError,
    ProviderError,
    SarvamClient,
)
from voice_agent.speech import clean_markdown, to_speech

logger = logging.getLogger(__name__)

MIN_TTS_SENTENCE_CHARS = 12
LLM_FAILURE_REPLY = "Sorry, I'm having trouble answering right now. Please try again in a moment."
BOOKING_SOURCE = "Appointment calendar"

_FILLER_WORDS = frozenset({"um", "uh", "ah", "hmm", "hm", "er", "err", "mm", "mhm"})
_ABBREVIATIONS = frozenset({"mr.", "mrs.", "ms.", "dr.", "rs.", "no.", "st.", "vs.", "e.g.", "i.e.", "etc."})
_SENTENCE_END = ".!?।॥"
_CLAUSE_END = ",;:—–"


# ── Small text helpers ───────────────────────────────────────────────────────

def is_noise(transcript: str, *, in_dialog: bool = False) -> bool:
    """
    True for transcripts that are not a real utterance (silence, "um", stray
    punctuation). Inside a dialog, short replies such as "no" or "11" are
    exactly what we asked for, so only empty and filler-only input is dropped.
    """
    text = transcript.strip().lower().strip("?.!,। ")
    if not text or not re.search(r"\w", text):
        return True
    words = text.split()
    if all(word.strip("?.!,") in _FILLER_WORDS for word in words):
        return True
    if in_dialog:
        return False
    return len(text) < 3 or text.replace(" ", "").isdigit()


_ROMAN_TELUGU = frozenset({
    "gurinchi", "cheppu", "ante", "enti", "naku", "meeru", "evaru", "emiti", "ekkada",
    "ledu", "undhi", "cheyyi", "cheppandi",
})
_ROMAN_HINDI = frozenset({
    "kya", "hai", "hain", "mujhe", "batao", "kaun", "kahan", "kaise", "aap", "mera",
})
_SCRIPT_RANGES = (
    ("te-IN", "ఀ", "౿"), ("hi-IN", "ऀ", "ॿ"), ("ta-IN", "஀", "௿"),
    ("kn-IN", "ಀ", "೿"), ("ml-IN", "ഀ", "ൿ"), ("bn-IN", "ঀ", "৿"),
    ("gu-IN", "઀", "૿"), ("pa-IN", "਀", "੿"),
)


def detect_language(text: str) -> str:
    """Guess the language from the script, then from common romanised words."""
    for code, low, high in _SCRIPT_RANGES:
        if sum(1 for ch in text if low <= ch <= high) > 2:
            return code
    words = set(text.lower().split())
    if words & _ROMAN_TELUGU:
        return "te-IN"
    if words & _ROMAN_HINDI:
        return "hi-IN"
    return "en-IN"


def choose_reply_language(transcript: str, stt_language: str) -> str:
    detected = detect_language(transcript)
    if detected != "en-IN":
        return detected
    return stt_language if stt_language in LANGUAGE_NAMES else "en-IN"


class SentenceChunker:
    """
    Cut a stream of text into speakable sentences.

    A sentence ends at ``. ! ?`` (or a danda) followed by whitespace. Decimals
    ("3.5"), common abbreviations ("Rs. 30") and fragments shorter than
    ``min_chars`` do not end a sentence — short pieces sound choppy when
    synthesised on their own.

    With ``first_clause_chars`` set, the *first* piece may also end at a clause
    (``, ; : —``) once it is that long, so speech can start before the model
    has finished its first sentence. Later pieces are whole sentences.
    """

    def __init__(self, min_chars: int = MIN_TTS_SENTENCE_CHARS, first_clause_chars: int = 0) -> None:
        self._buffer = ""
        self._min_chars = min_chars
        self._first_clause_chars = first_clause_chars
        self._emitted = False

    def _boundary(self) -> int | None:
        text = self._buffer
        for index, char in enumerate(text[:-1]):
            if char == "\n":
                if len(text[:index].strip()) >= self._min_chars:
                    return index + 1
                continue
            if (
                char in _CLAUSE_END and self._first_clause_chars and not self._emitted
                and text[index + 1].isspace() and len(text[: index + 1].strip()) >= self._first_clause_chars
            ):
                return index + 1
            if char not in _SENTENCE_END or not text[index + 1].isspace():
                continue
            candidate = text[: index + 1].strip()
            if len(candidate) < self._min_chars:
                continue
            if candidate.split()[-1].lower() in _ABBREVIATIONS:
                continue
            return index + 1
        return None

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        while (cut := self._boundary()) is not None:
            sentence, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:]
            if sentence:
                sentences.append(sentence)
                self._emitted = True
        return sentences

    def flush(self) -> list[str]:
        remaining, self._buffer = self._buffer.strip(), ""
        return [remaining] if remaining else []


def split_sentences(text: str) -> list[str]:
    chunker = SentenceChunker()
    return chunker.feed(text) + chunker.flush()


def estimate_audio_seconds(audio_bytes: bytes, audio_format: str) -> float:
    """
    Recording length, for cost estimation. Exact for WAV (read from the
    header); browser recordings are Opus/AAC at roughly 32 kbit/s.
    """
    if audio_bytes[:4] == b"RIFF" and len(audio_bytes) > 44:
        byte_rate = struct.unpack_from("<I", audio_bytes, 28)[0]
        if byte_rate:
            return (len(audio_bytes) - 44) / byte_rate
    return len(audio_bytes) / 4000


def estimate_cost_inr(
    audio_seconds: float = 0.0, tts_chars: int = 0, input_tokens: int = 0, output_tokens: int = 0
) -> float:
    return round(
        audio_seconds / 3600 * config.COST_STT_PER_HOUR_INR
        + tts_chars / 10_000 * config.COST_TTS_PER_10K_CHARS_INR
        + input_tokens / 1_000_000 * config.COST_LLM_INPUT_PER_1M_TOKENS_INR
        + output_tokens / 1_000_000 * config.COST_LLM_OUTPUT_PER_1M_TOKENS_INR,
        6,
    )


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class ConversationSession:
    """Everything remembered about one caller or chat connection."""

    session_id: str = field(default_factory=tracing.new_trace_id)
    booking: BookingSession = field(default_factory=BookingSession)
    history: deque[dict[str, str]] = field(
        default_factory=lambda: deque(maxlen=max(config.CONVERSATION_HISTORY_MESSAGES, 0))
    )

    def remember(self, user_text: str, assistant_text: str) -> None:
        if self.history.maxlen:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": assistant_text})


@dataclass
class Latency:
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    first_token_ms: float = 0.0
    tts_ms: float = 0.0
    first_audio_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        perceived = self.first_audio_ms or self.total_ms
        return {
            "stt_ms": round(self.stt_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "llm_ms": round(self.llm_ms, 2),
            "first_token_ms": round(self.first_token_ms, 2),
            "tts_ms": round(self.tts_ms, 2),
            "first_audio_ms": round(self.first_audio_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "within_budget": perceived <= config.LATENCY_TARGET_TOTAL_MS,
            "budget_ms": config.LATENCY_TARGET_TOTAL_MS,
        }


class LRUCache:
    def __init__(self, max_size: int) -> None:
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Any:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def set(self, key: str, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def _passages_for_trace(retrieval: Retrieval | None) -> dict[str, Any]:
    if retrieval is None:
        return {"passage_count": 0, "top_score": None, "passages": [], "retrieval_query": "", "followup": False}
    return {
        "passage_count": retrieval.chunks,
        "top_score": max((p.dense_score for p in retrieval.passages), default=None),
        "passages": [
            {"source": p.source, "section": p.section, "page": p.page, "score": round(p.dense_score, 3)}
            for p in retrieval.passages
        ],
        "retrieval_query": retrieval.query,
        "followup": retrieval.followup,
    }


# ═════════════════════════════════════════════════════════════════════════════
class VoiceAgent:
    """Orchestrates one turn of conversation, spoken or typed."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        sarvam_client: SarvamClient | None = None,
        tracer: Any = tracing.record,
    ) -> None:
        self.client = sarvam_client or SarvamClient()
        self.rag = rag_pipeline or RAGPipeline(sarvam_client=self.client)
        self._cache = LRUCache(config.CACHE_MAX_SIZE) if config.ENABLE_ANSWER_CACHE else None
        self._tracer = tracer
        logger.info("Voice agent ready | answer cache %s", "on" if self._cache is not None else "off")

    async def warmup(self) -> None:
        """Run one retrieval so the first real caller doesn't pay for model load."""
        try:
            await self.rag.retrieve("warm up")
        except Exception:
            logger.warning("Warm-up retrieval failed", exc_info=True)

    async def close(self) -> None:
        await self.client.close()

    async def _trace(self, **fields: Any) -> str:
        trace_id = fields.setdefault("id", tracing.new_trace_id())
        fields.setdefault("prompt_version", config.PROMPT_VERSION)
        if self._tracer is not None:
            await asyncio.to_thread(self._tracer, fields)
        return trace_id

    # ── Cache ────────────────────────────────────────────────────────────────

    def _cache_key(self, text: str, language: str) -> str:
        """
        Same question, same language, same knowledge base → same answer. The
        knowledge-base version is part of the key, so ingesting or clearing
        documents can never serve an out-of-date answer.
        """
        normalised = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
        raw = f"{self.rag.vs.version}|{config.PROMPT_VERSION}|{language}|{normalised}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _wants_booking(self, text: str, session: ConversationSession | None) -> bool:
        if session is not None and session.booking.is_active():
            return True
        return is_booking_intent(text) or is_cancel_intent(text)

    # ── Speech synthesis ─────────────────────────────────────────────────────

    async def _synthesize(self, text: str, language: str) -> bytes | None:
        spoken = to_speech(text)
        if not spoken:
            return None
        try:
            return (await self.client.synthesize(spoken, target_language_code=language)).audio_bytes or None
        except ProviderError as exc:
            logger.warning("TTS failed for a sentence (%s) — continuing with text only.", exc)
            return None

    async def _speak(
        self, tokens: AsyncIterator[str], language: str, started: float, latency: Latency, spoken: list[bytes]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Forward text as it arrives and synthesise each finished sentence in the
        background. Audio is yielded in sentence order as soon as it is ready.
        """
        chunker = SentenceChunker(first_clause_chars=config.FIRST_CHUNK_MIN_CHARS)
        pending: deque[asyncio.Task[bytes | None]] = deque()
        tts_started: float | None = None

        def schedule(sentences: list[str]) -> None:
            nonlocal tts_started
            for sentence in sentences:
                if tts_started is None:
                    tts_started = time.perf_counter()
                pending.append(asyncio.create_task(self._synthesize(sentence, language)))

        def emit(audio: bytes | None) -> dict[str, Any] | None:
            if not audio:
                return None
            if not latency.first_audio_ms:
                latency.first_audio_ms = (time.perf_counter() - started) * 1000
            spoken.append(audio)
            return {"type": "audio_chunk", "audio": audio}

        try:
            async for token in tokens:
                yield {"type": "text_chunk", "text": token}
                schedule(chunker.feed(token))
                while pending and pending[0].done():
                    if event := emit(pending.popleft().result()):
                        yield event
            schedule(chunker.flush())
            while pending:
                if event := emit(await pending.popleft()):
                    yield event
        finally:
            for task in pending:      # the listener went away mid-answer
                task.cancel()
            if tts_started is not None:
                latency.tts_ms = (time.perf_counter() - tts_started) * 1000

    # ── Voice turn ───────────────────────────────────────────────────────────

    async def run_stream(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        audio_format: str = "webm",
        session: ConversationSession | None = None,
        audio_seconds: float | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process one recorded utterance. Yields, in order::

            {"type": "transcript",  "text": "..."}
            {"type": "noise",       "message": "..."}          # and stops
            {"type": "text_chunk",  "text": "..."}             # many
            {"type": "audio_chunk", "audio": b"<wav>"}         # one per sentence
            {"type": "done", "answer": ..., "citations": [...], "latency": {...}, "trace_id": ..., ...}

        Provider failures surface as ``ProviderError`` (after being traced).
        """
        started = time.perf_counter()
        latency = Latency()
        channel = "voice"
        session_id = session.session_id if session is not None else ""
        if audio_seconds is None:
            audio_seconds = estimate_audio_seconds(audio_bytes, audio_format)
        transcript = ""

        try:
            stt = await self.client.transcribe(audio_bytes, language_code=language_code, audio_format=audio_format)
            latency.stt_ms = stt.latency_ms
            transcript = stt.transcript

            in_dialog = session is not None and session.booking.is_active()
            if is_noise(transcript, in_dialog=in_dialog):
                await self._trace(
                    channel=channel, session_id=session_id, kind="noise", question=transcript,
                    stt_ms=latency.stt_ms, total_ms=(time.perf_counter() - started) * 1000,
                    cost_inr=estimate_cost_inr(audio_seconds),
                )
                yield {"type": "noise", "text": transcript, "message": "I couldn't make that out — please try again."}
                return
            yield {"type": "transcript", "text": transcript}

            spoken: list[bytes] = []
            done: dict[str, Any] = {"type": "done", "transcript": transcript, "language": stt.language}

            # ── Booking / cancellation ──────────────────────────────────────
            if self._wants_booking(transcript, session):
                booking = session.booking if session is not None else None
                if booking is not None:
                    booking.source = "voice"
                result = await handle_booking_query(transcript, booking)

                async def _reply() -> AsyncIterator[str]:
                    yield result.text

                async for event in self._speak(_reply(), "en-IN", started, latency, spoken):
                    yield event
                if session is not None:
                    session.remember(transcript, result.text)
                latency.total_ms = (time.perf_counter() - started) * 1000
                cost = estimate_cost_inr(audio_seconds, len(result.text))
                trace_id = await self._trace(
                    channel=channel, session_id=session_id, kind="booking", question=transcript,
                    answer=result.text, language="en-IN", cost_inr=cost, **latency.to_dict(),
                )
                done.update(
                    answer=result.text, sources=[BOOKING_SOURCE], citations=[], context_chunks=0,
                    is_booking=True, booking_completed=result.completed, ref_code=result.ref_code,
                    cost_inr=cost, latency=latency.to_dict(), trace_id=trace_id,
                )
                yield done
                return

            # ── Knowledge-base answer ───────────────────────────────────────
            reply_language = choose_reply_language(transcript, stt.language)
            history = list(session.history) if session is not None else []
            cache_key = self._cache_key(transcript, reply_language)
            cached = self._cache.get(cache_key) if self._cache is not None and not history else None
            stats = LLMStreamStats()
            retrieval: Retrieval | None = None
            failed = False

            if cached:
                if cached["audio"]:
                    yield {"type": "text_chunk", "text": cached["answer"]}
                    for audio in cached["audio"]:
                        if not latency.first_audio_ms:
                            latency.first_audio_ms = (time.perf_counter() - started) * 1000
                        yield {"type": "audio_chunk", "audio": audio}
                    spoken_chars = 0
                else:
                    # Answer was cached by a typed question: it has no audio yet.
                    async def _cached_text() -> AsyncIterator[str]:
                        yield cached["answer"]

                    async for event in self._speak(_cached_text(), reply_language, started, latency, spoken):
                        yield event
                    cached["audio"] = spoken
                    spoken_chars = len(cached["answer"])
                answer, citations = cached["answer"], cached["citations"]
                sources, chunks = cached["sources"], cached["context_chunks"]
                cost = estimate_cost_inr(audio_seconds, spoken_chars)
                done["cached"] = True
            else:
                retrieval = await self.rag.retrieve(transcript, history=history)
                latency.retrieval_ms = retrieval.latency_ms
                parts: list[str] = []

                async def _tokens() -> AsyncIterator[str]:
                    nonlocal failed
                    try:
                        async for token in self.client.generate_stream(
                            transcript, context=retrieval.context, history=history,
                            language=reply_language, stats=stats,
                        ):
                            parts.append(token)
                            yield token
                    except LLMUnavailableError:
                        logger.error("LLM unavailable for a voice turn", exc_info=True)
                        failed = True
                        if not parts:
                            parts.append(LLM_FAILURE_REPLY)
                            yield LLM_FAILURE_REPLY

                async for event in self._speak(_tokens(), reply_language, started, latency, spoken):
                    yield event

                answer = clean_markdown("".join(parts))
                citations, sources, chunks = retrieval.citations, retrieval.sources, retrieval.chunks
                latency.llm_ms = stats.total_ms
                latency.first_token_ms = stats.first_token_ms
                cost = estimate_cost_inr(audio_seconds, len(answer), stats.input_tokens, stats.output_tokens)
                if failed:
                    done["degraded"] = True
                elif self._cache is not None and answer and not history:
                    self._cache.set(cache_key, {
                        "answer": answer, "audio": spoken, "sources": sources,
                        "context_chunks": chunks, "citations": citations,
                    })

            if session is not None and not failed:
                session.remember(transcript, answer)
            latency.total_ms = (time.perf_counter() - started) * 1000
            trace_id = await self._trace(
                channel=channel, session_id=session_id, kind="error" if failed else "answer",
                question=transcript, answer=answer, language=reply_language,
                provider=stats.provider, model=stats.model, fallback=stats.fallback, cached=bool(cached),
                input_tokens=stats.input_tokens, output_tokens=stats.output_tokens, cost_inr=cost,
                error="all LLM providers failed" if failed else "",
                **_passages_for_trace(retrieval), **latency.to_dict(),
            )
            done.update(
                answer=answer, sources=sources, citations=citations, context_chunks=chunks,
                cost_inr=cost, latency=latency.to_dict(), trace_id=trace_id,
                provider=stats.provider, model=stats.model,
            )
            logger.info(
                "Voice turn | stt=%.0f rag=%.0f llm=%.0f tts=%.0f first_audio=%.0f total=%.0f ms",
                latency.stt_ms, latency.retrieval_ms, latency.llm_ms, latency.tts_ms,
                latency.first_audio_ms, latency.total_ms,
            )
            yield done

        except ProviderError as exc:
            latency.total_ms = (time.perf_counter() - started) * 1000
            await self._trace(
                channel=channel, session_id=session_id, kind="error", question=transcript,
                error=str(exc)[:300], **latency.to_dict(),
            )
            raise

    # ── Text turn ────────────────────────────────────────────────────────────

    async def query_text(
        self,
        text: str,
        session: ConversationSession | None = None,
        language: str = "en-IN",
        attendee_name: str = "",
        attendee_email: str = "",
        channel: str | None = None,
    ) -> dict[str, Any]:
        """Answer a typed message (chat box, REST ``/query``, benchmark, evals)."""
        started = time.perf_counter()
        channel = channel or ("chat" if session is not None else "api")
        session_id = session.session_id if session is not None else ""

        if self._wants_booking(text, session):
            booking = session.booking if session is not None else None
            if booking is not None:
                booking.source = "chat"
            result = await handle_booking_query(text, booking, attendee_name, attendee_email)
            if session is not None:
                session.remember(text, result.text)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            trace_id = await self._trace(
                channel=channel, session_id=session_id, kind="booking", question=text,
                answer=result.text, language="en-IN", total_ms=elapsed,
            )
            return {
                "query": text, "answer": result.text, "sources": [BOOKING_SOURCE], "citations": [],
                "context_chunks": 0, "latency": {"retrieval_ms": 0, "llm_ms": 0, "total_ms": elapsed},
                "cost_inr": 0.0, "tokens": {"input": 0, "output": 0}, "trace_id": trace_id,
                "is_booking": True, "booking_completed": result.completed, "ref_code": result.ref_code,
            }

        reply_language = choose_reply_language(text, language)
        history = list(session.history) if session is not None else []
        cache_key = self._cache_key(text, reply_language)
        cached = self._cache.get(cache_key) if self._cache is not None and not history else None
        if cached:
            if session is not None:
                session.remember(text, cached["answer"])
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            trace_id = await self._trace(
                channel=channel, session_id=session_id, kind="answer", question=text,
                answer=cached["answer"], language=reply_language, cached=True, total_ms=elapsed,
                passage_count=cached["context_chunks"],
            )
            return {
                "query": text, "answer": cached["answer"], "sources": cached["sources"],
                "citations": cached["citations"], "context_chunks": cached["context_chunks"],
                "latency": {"retrieval_ms": 0, "llm_ms": 0, "total_ms": elapsed},
                "cost_inr": 0.0, "tokens": {"input": 0, "output": 0}, "cached": True, "trace_id": trace_id,
            }

        try:
            result = await self.rag.query(text, history=history, language=reply_language)
        except LLMUnavailableError as exc:
            await self._trace(
                channel=channel, session_id=session_id, kind="error", question=text, language=reply_language,
                error=str(exc)[:300], total_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        answer = clean_markdown(result.answer)
        if self._cache is not None and answer and not history:
            self._cache.set(cache_key, {
                "answer": answer, "audio": [], "sources": result.sources,
                "context_chunks": result.context_chunks, "citations": result.citations,
            })
        if session is not None:
            session.remember(text, answer)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        cost = estimate_cost_inr(input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        trace_id = await self._trace(
            channel=channel, session_id=session_id, kind="answer", question=text, answer=answer,
            language=reply_language, provider=result.provider, model=result.model,
            fallback=result.fallback, input_tokens=result.input_tokens,
            output_tokens=result.output_tokens, cost_inr=cost, retrieval_ms=result.retrieval_latency_ms,
            llm_ms=result.llm_latency_ms, total_ms=total_ms, **_passages_for_trace(result.retrieval),
        )
        return {
            "query": text, "answer": answer, "sources": result.sources, "citations": result.citations,
            "context_chunks": result.context_chunks,
            "latency": {
                "retrieval_ms": result.retrieval_latency_ms,
                "llm_ms": result.llm_latency_ms,
                "total_ms": total_ms,
            },
            "cost_inr": cost,
            "tokens": {"input": result.input_tokens, "output": result.output_tokens},
            "provider": result.provider,
            "trace_id": trace_id,
            "context": result.retrieval.context if result.retrieval else "",
        }

    async def health_check(self) -> dict[str, Any]:
        return await self.client.health_check()
