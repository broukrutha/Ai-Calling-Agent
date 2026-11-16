"""In-memory stand-ins for the paid providers and the embedding model."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re

from voice_agent.sarvam_client import LLMResult, LLMUnavailableError, STTResult, TTSResult


class FakeProviderClient:
    """Mimics SarvamClient: scripted transcript, canned answer, fake audio."""

    llm_provider = "fake"
    llm_model = "fake-1"

    def __init__(self, transcript: str = "", answer: str = "Our office is open nine to five. Visit any weekday."):
        self.transcript = transcript
        self.answer = answer
        self.language = "en-IN"
        self.llm_down = False
        self.tts_delays: dict[str, float] = {}
        self.tts_calls: list[tuple[str, str]] = []
        self.llm_calls: list[dict] = []
        self.closed = False

    async def transcribe(self, audio_bytes, language_code="unknown", audio_format="webm"):
        return STTResult(transcript=self.transcript, language=self.language, latency_ms=5.0)

    def _record(self, user_message, context, history, language):
        self.llm_calls.append(
            {"message": user_message, "context": context, "history": list(history or []), "language": language}
        )

    async def generate(self, user_message, context="", system_prompt=None, history=None, language="en-IN"):
        self._record(user_message, context, history, language)
        if self.llm_down:
            raise LLMUnavailableError("down")
        return LLMResult(answer=self.answer, input_tokens=100, output_tokens=20, latency_ms=7.0, provider="fake")

    async def generate_stream(self, user_message, context="", system_prompt=None, history=None,
                              language="en-IN", stats=None):
        self._record(user_message, context, history, language)
        if self.llm_down:
            raise LLMUnavailableError("down")
        for piece in re.findall(r"\S+\s*", self.answer):
            await asyncio.sleep(0)
            yield piece
        if stats is not None:
            stats.provider, stats.input_tokens, stats.output_tokens, stats.total_ms = "fake", 100, 20, 7.0

    async def synthesize(self, text, target_language_code=None, speaker=None):
        self.tts_calls.append((text, target_language_code))
        await asyncio.sleep(self.tts_delays.get(text, 0))
        return TTSResult(audio_bytes=f"AUDIO<{text}>".encode(), character_count=len(text), latency_ms=1.0)

    async def health_check(self, max_age_sec: float = 30.0):
        return {
            "status": "ok", "llm_provider": "fake", "llm_model": "fake-1", "latency_ms": 1.0,
            "stt_model": "fake-stt", "tts_model": "fake-tts",
        }

    async def close(self):
        self.closed = True


class HashingEmbedder:
    """
    Deterministic bag-of-words embedding, so the real ChromaDB code paths can
    be tested without downloading a model.
    """

    DIMENSIONS = 128

    def __call__(self, input):  # noqa: A002 - name required by Chroma's protocol
        vectors = []
        for text in input:
            vector = [0.0] * self.DIMENSIONS
            for word in re.findall(r"\w+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIMENSIONS] += 1.0  # noqa: S324
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    def embed_query(self, input):  # noqa: A002
        return self(input)

    def embed_documents(self, input):  # noqa: A002
        return self(input)

    @staticmethod
    def name() -> str:
        return "hashing-test-embedder"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbedder:
        return HashingEmbedder()

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        return False
