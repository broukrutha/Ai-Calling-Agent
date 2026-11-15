"""SarvamClient against a fake HTTP transport: fallbacks, streaming, parsing."""

import base64
import json

import httpx
import pytest

from voice_agent import config
from voice_agent.sarvam_client import (
    LLMStreamStats,
    LLMUnavailableError,
    ProviderError,
    SarvamClient,
    ThinkTagFilter,
    build_messages,
    strip_think,
)


def make_client(monkeypatch, handler, *, openai="sk-test", groq="gsk-test", preferred="openai"):
    monkeypatch.setattr(config, "SARVAM_API_KEY", "sarvam-test")
    monkeypatch.setattr(config, "OPENAI_API_KEY", openai)
    monkeypatch.setattr(config, "GROQ_API_KEY", groq)
    monkeypatch.setattr(config, "LLM_PROVIDER", preferred)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    return SarvamClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _no_sleep(_seconds):
    return None


def chat_response(text, status=200):
    return httpx.Response(status, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    })


def sse(*tokens, usage=None):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in tokens]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    lines.append("data: [DONE]")
    return httpx.Response(200, text="\n\n".join(lines) + "\n\n", headers={"content-type": "text/event-stream"})


# ── <think> handling ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("pieces,expected", [
    (["Hello ", "world"], "Hello world"),
    (["<think>plan</think>Answer"], "Answer"),
    (["<thi", "nk>secret plan</th", "ink>The answer"], "The answer"),      # tags split across tokens
    (["Before <think>hidden", " more</think> after"], "Before  after"),
    (["Visible <think>never closed"], "Visible "),
    (["2 < 3 and <b>bold</b>"], "2 < 3 and <b>bold</b>"),                # ordinary "<" is untouched
])
def test_think_filter_streaming(pieces, expected):
    think = ThinkTagFilter()
    assert "".join(think.feed(piece) for piece in pieces) + think.flush() == expected


def test_strip_think():
    assert strip_think("<think>a\nb</think>Hi<think>unfinished") == "Hi"


def test_prompt_marks_context_as_data_and_names_the_language():
    messages = build_messages("What time?", context="Open 9-5", history=[{"role": "user", "content": "hi"}], language="te-IN")
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert "<context>\nOpen 9-5\n</context>" in messages[-1]["content"]
    assert "[Reply language: Telugu]" in messages[-1]["content"]
    assert "(no matching reference material" in build_messages("Hi")[-1]["content"]


# ── LLM fallback chain ───────────────────────────────────────────────────────

def test_provider_order_follows_preference_and_available_keys(monkeypatch):
    client = make_client(monkeypatch, lambda r: chat_response("x"), preferred="groq")
    assert [p.name for p in client.providers] == ["groq", "openai", "sarvam"]
    client = make_client(monkeypatch, lambda r: chat_response("x"), openai="", preferred="openai")
    assert [p.name for p in client.providers] == ["groq", "sarvam"]


async def test_generate_falls_back_when_the_first_provider_is_rate_limited(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "api.openai.com":
            return httpx.Response(429, text="slow down")
        return chat_response("<think>hmm</think>Groq says hi")

    client = make_client(monkeypatch, handler)
    result = await client.generate("hello")
    assert result.answer == "Groq says hi" and result.provider == "groq"
    assert (result.input_tokens, result.output_tokens) == (12, 5)
    assert seen == ["api.openai.com", "api.openai.com", "api.groq.com"]      # one retry, then fall through


async def test_generate_does_not_retry_a_bad_api_key(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(401, text="bad key") if request.url.host == "api.openai.com" else chat_response("ok")

    client = make_client(monkeypatch, handler)
    assert (await client.generate("hello")).answer == "ok"
    assert seen == ["api.openai.com", "api.groq.com"]


async def test_generate_raises_when_every_provider_fails(monkeypatch):
    client = make_client(monkeypatch, lambda r: httpx.Response(503, text="down"))
    with pytest.raises(LLMUnavailableError):
        await client.generate("hello")


async def test_stream_yields_tokens_and_reports_usage(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        return sse("Hel", "lo <thi", "nk>x</think>", "there", usage={"prompt_tokens": 9, "completion_tokens": 3})

    client = make_client(monkeypatch, handler)
    stats = LLMStreamStats()
    text = "".join([token async for token in client.generate_stream("hi", stats=stats)])
    assert text == "Hello there"
    assert (stats.provider, stats.input_tokens, stats.output_tokens) == ("openai", 9, 3)


async def test_stream_falls_back_before_any_text_was_sent(monkeypatch):
    def handler(request):
        return httpx.Response(500, text="boom") if request.url.host == "api.openai.com" else sse("From ", "Groq")

    client = make_client(monkeypatch, handler)
    assert "".join([t async for t in client.generate_stream("hi")]) == "From Groq"


async def test_stream_raises_when_nothing_could_be_generated(monkeypatch):
    client = make_client(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMUnavailableError):
        [t async for t in client.generate_stream("hi")]


# ── Speech ───────────────────────────────────────────────────────────────────

async def test_transcribe_sends_the_recording_and_returns_detected_language(monkeypatch):
    captured = {}

    def handler(request):
        captured["body"] = request.content
        captured["key"] = request.headers["api-subscription-key"]
        return httpx.Response(200, json={"transcript": " नमस्ते ", "language_code": "hi-IN"})

    client = make_client(monkeypatch, handler)
    result = await client.transcribe(b"WEBMDATA", audio_format="audio/webm;codecs=opus")
    assert (result.transcript, result.language) == ("नमस्ते", "hi-IN")
    assert captured["key"] == "sarvam-test"
    assert b'filename="audio.webm"' in captured["body"] and b"saaras:v3" in captured["body"]
    assert b"unknown" in captured["body"]


async def test_transcribe_falls_back_to_the_older_model_and_remembers_it(monkeypatch):
    models = []

    def handler(request):
        model = "saaras:v3" if b"saaras:v3" in request.content else "saarika:v2.5"
        models.append(model)
        if model == "saaras:v3":
            return httpx.Response(400, json={"error": {"message": "model not supported on this endpoint"}})
        return httpx.Response(200, json={"transcript": "hello", "language_code": "en-IN"})

    client = make_client(monkeypatch, handler)
    assert (await client.transcribe(b"x")).transcript == "hello"
    assert (await client.transcribe(b"x")).transcript == "hello"
    assert models == ["saaras:v3", "saarika:v2.5", "saarika:v2.5"]


async def test_transcribe_surfaces_provider_failure(monkeypatch):
    client = make_client(monkeypatch, lambda r: httpx.Response(500, text="oops"))
    with pytest.raises(ProviderError):
        await client.transcribe(b"x")


async def test_synthesize_decodes_audio_and_truncates_long_text(monkeypatch):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"audios": [base64.b64encode(b"RIFFwav").decode()]})

    client = make_client(monkeypatch, handler)
    result = await client.synthesize("x" * 3000, target_language_code="hi-IN")
    assert result.audio_bytes == b"RIFFwav" and result.character_count == 2500
    assert sent["target_language_code"] == "hi-IN" and sent["model"] == "bulbul:v3" and len(sent["text"]) == 2500
    assert (await client.synthesize("   ")).audio_bytes == b""


async def test_health_check_is_cached(monkeypatch):
    hits = []

    def handler(request):
        hits.append(str(request.url))
        return httpx.Response(200, json={"data": []})

    client = make_client(monkeypatch, handler)
    first = await client.health_check()
    second = await client.health_check()
    assert first["status"] == "ok" and first is second and hits == ["https://api.openai.com/v1/models"]
    assert first["fallback_providers"] == ["groq", "sarvam"]


def test_client_requires_a_sarvam_key(monkeypatch):
    monkeypatch.setattr(config, "SARVAM_API_KEY", "")
    with pytest.raises(ValueError):
        SarvamClient()
