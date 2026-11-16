"""
Provider client: Sarvam AI speech (STT + TTS) and a multi-provider LLM.

* **STT** – Sarvam ``/speech-to-text`` (REST). Accepts the browser's webm/ogg
  recording directly, so no transcoding step is needed.
* **TTS** – Sarvam ``/text-to-speech`` (REST), one request per sentence so the
  first audio can be played while the rest is still being generated.
* **LLM** – OpenAI-compatible chat completions with an ordered fallback chain
  (OpenAI → Groq → Sarvam, starting from ``LLM_PROVIDER``). A rate limit or
  outage on one provider moves to the next instead of failing the call.

A single pooled ``httpx.AsyncClient`` is shared by every request.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx

from voice_agent import config

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en-IN": "English", "hi-IN": "Hindi", "te-IN": "Telugu", "ta-IN": "Tamil",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "bn-IN": "Bengali", "gu-IN": "Gujarati",
    "mr-IN": "Marathi", "pa-IN": "Punjabi", "od-IN": "Odia",
}
_AUDIO_FORMATS = ("wav", "mp3", "webm", "ogg", "mp4", "flac")
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Older REST model, tried automatically if the configured one is refused.
_STT_FALLBACK_MODEL = "saarika:v2.5"


class ProviderError(RuntimeError):
    """A speech or LLM provider could not serve the request."""


class LLMUnavailableError(ProviderError):
    """Every configured LLM provider failed."""


# ── Results ──────────────────────────────────────────────────────────────────

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
    provider: str = ""


@dataclass
class TTSResult:
    audio_bytes: bytes
    character_count: int
    latency_ms: float


@dataclass
class LLMStreamStats:
    """Filled in while a stream runs, so the caller can report usage afterwards."""

    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    first_token_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class LLMProvider:
    name: str
    url: str
    model: str
    headers: dict[str, str] = field(repr=False)
    extra_body: dict[str, Any] = field(default_factory=dict)


# ── <think> filtering ────────────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning blocks some models emit, including an unterminated one."""
    return _THINK_OPEN_RE.sub("", _THINK_BLOCK_RE.sub("", text)).strip()


class ThinkTagFilter:
    """
    Streaming version of :func:`strip_think`.

    Tokens arrive in arbitrary pieces, so a tag can be split across two of
    them ("<thi" + "nk>"). The filter holds back any tail that could still
    turn into a tag and releases it once it is known to be ordinary text.
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def feed(self, token: str) -> str:
        self._buffer += token
        output = []
        while True:
            if self._inside:
                end = self._buffer.find(self._CLOSE)
                if end == -1:
                    keep = self._partial_tag_length(self._buffer, self._CLOSE)
                    self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
                    break
                self._buffer = self._buffer[end + len(self._CLOSE):]
                self._inside = False
            else:
                start = self._buffer.find(self._OPEN)
                if start == -1:
                    keep = self._partial_tag_length(self._buffer, self._OPEN)
                    cut = len(self._buffer) - keep
                    output.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    break
                output.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self._OPEN):]
                self._inside = True
        return "".join(output)

    def flush(self) -> str:
        """Whatever is left at end of stream (nothing, if still inside a block)."""
        remaining = "" if self._inside else self._buffer
        self._buffer = ""
        return remaining


# ── Prompt assembly ──────────────────────────────────────────────────────────

def build_messages(
    user_message: str,
    context: str = "",
    history: list[dict[str, str]] | None = None,
    language: str = "en-IN",
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages for a grounded, spoken answer."""
    reference = context.strip() or "(no matching reference material was found)"
    language_name = LANGUAGE_NAMES.get(language, "English")
    user_content = (
        f"<context>\n{reference}\n</context>\n\n"
        f"[Reply language: {language_name}]\n"
        f"Question: {user_message}"
    )
    messages = [{"role": "system", "content": system_prompt or config.LLM_SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_content})
    return messages


def build_provider_chain() -> list[LLMProvider]:
    """Configured LLM providers, preferred one first."""
    available: dict[str, LLMProvider] = {}
    if config.OPENAI_API_KEY:
        available["openai"] = LLMProvider(
            name="openai",
            url=f"{config.OPENAI_BASE_URL}/chat/completions",
            model=config.OPENAI_MODEL,
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
        )
    if config.GROQ_API_KEY:
        available["groq"] = LLMProvider(
            name="groq",
            url=f"{config.GROQ_BASE_URL}/chat/completions",
            model=config.GROQ_MODEL,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        )
    if config.SARVAM_API_KEY:
        available["sarvam"] = LLMProvider(
            name="sarvam",
            url=f"{config.SARVAM_BASE_URL}/v1/chat/completions",
            model=config.SARVAM_LLM_MODEL,
            headers={"api-subscription-key": config.SARVAM_API_KEY},
            extra_body={"budget_tokens": 0},
        )
    order = [config.LLM_PROVIDER] + [name for name in ("openai", "groq", "sarvam") if name != config.LLM_PROVIDER]
    return [available[name] for name in order if name in available]


# ═════════════════════════════════════════════════════════════════════════════
class SarvamClient:
    """Speech-to-text, text-to-speech and LLM access behind one object."""

    def __init__(self, api_key: str | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key if api_key is not None else config.SARVAM_API_KEY
        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is not set.")

        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.LLM_TIMEOUT_SEC, connect=5.0),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            trust_env=config.HTTP_TRUST_ENV,
        )
        self.providers = build_provider_chain()
        self._stt_models = list(dict.fromkeys([config.STT_MODEL, _STT_FALLBACK_MODEL]))
        self._tts_semaphore = asyncio.Semaphore(config.TTS_MAX_CONCURRENCY)
        self._health_cache: tuple[float, dict[str, Any]] | None = None

        logger.info(
            "Provider client ready | STT=%s | TTS=%s/%s | LLM chain=%s",
            config.STT_MODEL, config.TTS_MODEL, config.TTS_SPEAKER,
            " → ".join(f"{p.name}/{p.model}" for p in self.providers) or "none",
        )

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def llm_provider(self) -> str:
        return self.providers[0].name if self.providers else "none"

    @property
    def llm_model(self) -> str:
        return self.providers[0].model if self.providers else "none"

    # ── Speech-to-text ───────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        audio_format: str = "webm",
    ) -> STTResult:
        """
        Transcribe a recording. ``language_code="unknown"`` lets Sarvam detect
        the spoken language, which is returned on the result.
        """
        ext = audio_format.lower().replace("audio/", "").split(";")[0].strip()
        if ext not in _AUDIO_FORMATS:
            ext = "webm"

        started = time.perf_counter()
        last_error = "no STT model configured"
        for model in list(self._stt_models):
            data = {"model": model, "language_code": language_code}
            if model.startswith("saaras"):
                data["mode"] = "transcribe"
            try:
                resp = await self._client.post(
                    f"{config.SARVAM_BASE_URL}/speech-to-text",
                    files={"file": (f"audio.{ext}", audio_bytes, f"audio/{ext}")},
                    data=data,
                    headers={"api-subscription-key": self._api_key},
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Speech-to-text request failed: {exc}") from exc

            if resp.status_code == 200:
                if model != self._stt_models[0]:
                    logger.warning("STT model %s was refused; using %s from now on.", self._stt_models[0], model)
                    self._stt_models = [model]
                body = resp.json()
                transcript = (body.get("transcript") or "").strip()
                language = body.get("language_code") or language_code
                elapsed = (time.perf_counter() - started) * 1000
                logger.info("STT %s | %d chars | lang=%s | %.0f ms", model, len(transcript), language, elapsed)
                return STTResult(transcript=transcript, language=language, latency_ms=round(elapsed, 2))

            last_error = f"{resp.status_code} {resp.text[:300]}"
            logger.error("STT %s failed | format=%s | bytes=%d | %s", model, ext, len(audio_bytes), last_error)
            if resp.status_code not in (400, 404, 422):
                break   # not a model/parameter problem: the fallback model won't help

        raise ProviderError(f"Speech-to-text failed: {last_error}")

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _payload(self, provider: LLMProvider, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            **provider.extra_body,
        }
        if stream:
            payload["stream"] = True
            if provider.name == "openai":
                payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(
        self,
        user_message: str,
        context: str = "",
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        language: str = "en-IN",
    ) -> LLMResult:
        """One complete answer. Falls through the provider chain on failure."""
        messages = build_messages(user_message, context, history, language, system_prompt)
        last_error = "no LLM provider is configured"

        for provider in self.providers:
            for attempt in range(2):
                started = time.perf_counter()
                try:
                    resp = await self._client.post(
                        provider.url, json=self._payload(provider, messages, stream=False), headers=provider.headers
                    )
                except httpx.HTTPError as exc:
                    last_error = f"{provider.name}: {type(exc).__name__}"
                    logger.warning("LLM %s request error: %s", provider.name, exc)
                    break

                if resp.status_code == 200:
                    data = resp.json()
                    raw = (data["choices"][0]["message"].get("content") or "").strip()
                    usage = data.get("usage") or {}
                    elapsed = (time.perf_counter() - started) * 1000
                    logger.info("LLM %s/%s | %.0f ms", provider.name, provider.model, elapsed)
                    return LLMResult(
                        answer=strip_think(raw) or raw,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        latency_ms=round(elapsed, 2),
                        provider=provider.name,
                    )

                last_error = f"{provider.name}: HTTP {resp.status_code}"
                logger.warning("LLM %s returned %s: %s", provider.name, resp.status_code, resp.text[:300])
                if resp.status_code in _RETRYABLE_STATUS and attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                break

        raise LLMUnavailableError(f"All LLM providers failed ({last_error}).")

    async def generate_stream(
        self,
        user_message: str,
        context: str = "",
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        language: str = "en-IN",
        stats: LLMStreamStats | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield the answer token by token.

        Providers are tried in order until one starts streaming. Once text has
        been yielded the stream is committed to that provider: a mid-stream
        failure ends the answer early rather than restarting it in another voice.
        """
        messages = build_messages(user_message, context, history, language, system_prompt)
        stats = stats if stats is not None else LLMStreamStats()
        last_error = "no LLM provider is configured"
        started = time.perf_counter()

        for provider in self.providers:
            think = ThinkTagFilter()
            yielded = False
            try:
                async with self._client.stream(
                    "POST", provider.url, json=self._payload(provider, messages, stream=True), headers=provider.headers
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
                        last_error = f"{provider.name}: HTTP {resp.status_code}"
                        logger.warning("LLM %s stream returned %s: %s", provider.name, resp.status_code, body)
                        continue

                    stats.provider = provider.name
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            continue
                        usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                        if usage:
                            stats.input_tokens = usage.get("prompt_tokens", 0)
                            stats.output_tokens = usage.get("completion_tokens", 0)
                        choices = chunk.get("choices") or []
                        token = (choices[0].get("delta") or {}).get("content") if choices else None
                        if not token:
                            continue
                        visible = think.feed(token)
                        if visible:
                            if not yielded:
                                stats.first_token_ms = round((time.perf_counter() - started) * 1000, 2)
                                yielded = True
                            yield visible

                tail = think.flush()
                if tail:
                    yielded = True
                    yield tail
            except httpx.HTTPError as exc:
                last_error = f"{provider.name}: {type(exc).__name__}"
                logger.warning("LLM %s stream error: %s", provider.name, exc)
                if not yielded:
                    continue

            if yielded:
                stats.total_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.info("LLM stream %s/%s | %.0f ms", provider.name, provider.model, stats.total_ms)
                return
            last_error = f"{provider.name}: empty response"

        raise LLMUnavailableError(f"All LLM providers failed ({last_error}).")

    # ── Text-to-speech ───────────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        target_language_code: str | None = None,
        speaker: str | None = None,
    ) -> TTSResult:
        """Turn text into WAV audio. Empty text returns empty audio."""
        text = (text or "").strip()
        if not text:
            return TTSResult(audio_bytes=b"", character_count=0, latency_ms=0.0)
        if len(text) > config.TTS_MAX_CHARS:
            logger.warning("TTS input truncated from %d to %d characters.", len(text), config.TTS_MAX_CHARS)
            text = text[: config.TTS_MAX_CHARS]

        payload = {
            "text": text,
            "target_language_code": target_language_code or config.TTS_LANGUAGE,
            "speaker": speaker or config.TTS_SPEAKER,
            "model": config.TTS_MODEL,
            "speech_sample_rate": config.TTS_SAMPLE_RATE,
            "pace": 1.0,
        }
        headers = {"api-subscription-key": self._api_key, "Content-Type": "application/json"}

        async with self._tts_semaphore:
            started = time.perf_counter()
            try:
                resp = await self._client.post(
                    f"{config.SARVAM_BASE_URL}/text-to-speech", json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Text-to-speech request failed: {exc}") from exc
            elapsed = (time.perf_counter() - started) * 1000

        if resp.status_code != 200:
            logger.error("TTS failed: %s %s", resp.status_code, resp.text[:300])
            raise ProviderError(f"Text-to-speech failed with HTTP {resp.status_code}.")

        audios = resp.json().get("audios") or [""]
        audio = base64.b64decode(audios[0]) if audios[0] else b""
        logger.info("TTS %s | %d chars | %.0f ms", config.TTS_MODEL, len(text), elapsed)
        return TTSResult(audio_bytes=audio, character_count=len(text), latency_ms=round(elapsed, 2))

    # ── Health ───────────────────────────────────────────────────────────────

    async def health_check(self, max_age_sec: float = 30.0) -> dict[str, Any]:
        """
        Reachability of the primary LLM provider, cached for ``max_age_sec`` so
        dashboards and probes cannot turn into a stream of provider requests.
        """
        if self._health_cache and time.monotonic() - self._health_cache[0] < max_age_sec:
            return self._health_cache[1]

        if not self.providers:
            llm = {"status": "error", "message": "No LLM provider is configured."}
        else:
            provider = self.providers[0]
            probe_url = (
                f"{config.SARVAM_BASE_URL}/"
                if provider.name == "sarvam"
                else provider.url.rsplit("/chat/completions", 1)[0] + "/models"
            )
            started = time.perf_counter()
            try:
                resp = await self._client.get(probe_url, headers=provider.headers, timeout=5.0)
                if resp.status_code in (401, 403):
                    llm = {"status": "error", "message": f"{provider.name} rejected the API key."}
                elif resp.status_code >= 500:
                    llm = {"status": "error", "message": f"{provider.name} returned HTTP {resp.status_code}."}
                else:
                    llm = {"status": "ok", "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            except httpx.HTTPError as exc:
                llm = {"status": "error", "message": f"{provider.name} is unreachable ({type(exc).__name__})."}

        result: dict[str, Any] = {
            "status": llm["status"],
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "fallback_providers": [p.name for p in self.providers[1:]],
            "stt_model": self._stt_models[0],
            "tts_model": config.TTS_MODEL,
        }
        if "latency_ms" in llm:
            result["latency_ms"] = llm["latency_ms"]
        if "message" in llm:
            result["message"] = llm["message"]
        self._health_cache = (time.monotonic(), result)
        return result
