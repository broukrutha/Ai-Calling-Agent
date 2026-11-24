"""HTTP + WebSocket API: auth, limits, validation, error hygiene, end-to-end flows."""

import base64
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from tests.fakes import FakeProviderClient, HashingEmbedder
from voice_agent import config
from voice_agent.booking import store
from voice_agent.pipeline import VoiceAgent
from voice_agent.rag_pipeline import RAGPipeline
from voice_agent.server import create_app, security
from voice_agent.server.deps import Services
from voice_agent.vector_store import VectorStore

ADMIN = {"X-Admin-Token": "s3cret-admin-token-for-tests-123"}


@pytest.fixture
def provider():
    return FakeProviderClient()


@pytest.fixture
def services(tmp_path, provider):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"), embedding_function=HashingEmbedder())
    vector_store.add_documents([
        Document(page_content="Our office is open nine to five on weekdays.", metadata={"source": "faq.txt"}),
    ])
    agent = VoiceAgent(rag_pipeline=RAGPipeline(vector_store=vector_store, sarvam_client=provider), sarvam_client=provider)
    return Services(vector_store=vector_store, agent=agent)


@pytest.fixture
def client(services):
    security.limiter.reset()
    with TestClient(create_app(services)) as test_client:
        yield test_client


@pytest.fixture
def locked(monkeypatch):
    """Server with an admin token configured."""
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", ADMIN["X-Admin-Token"])


# ── Health and headers ───────────────────────────────────────────────────────

def test_liveness_probe_is_cheap_and_public(client, provider):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_health_reports_status_without_leaking_paths(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["api"]["llm_provider"] == "fake"
    assert body["knowledge_base"]["total_chunks"] == 1 and body["bookings"]["mode"] == "mock"
    assert body["auth_required"] is False
    dumped = json.dumps(body)
    assert "chroma" not in dumped and "bookings.db" not in dumped and "/tmp" not in dumped


def test_security_headers_and_csp(client):
    response = client.get("/")
    assert response.status_code == 200 and "<title>" in response.text
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in response.headers["Content-Security-Policy"].split("style-src")[0]
    assert len(response.headers["X-Request-ID"]) == 12


def test_ui_has_no_inline_scripts_or_handlers_that_csp_would_block(client):
    import re

    html = client.get("/").text
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handlers are blocked by the CSP"
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline <script> is blocked by the CSP"


# ── Admin authentication ─────────────────────────────────────────────────────

ADMIN_ROUTES = [
    ("post", "/ingest/text", {"json": {"text": "hello world", "name": "n"}}),
    ("post", "/ingest/url", {"json": {"url": "https://example.com"}}),
    ("get", "/kb/sources", {}),
    ("get", "/kb/search?q=hours", {}),
    ("delete", "/kb/sources?source=faq.txt", {}),
    ("delete", "/kb/clear", {}),
    ("get", "/ops/summary", {}),
    ("get", "/ops/timeseries", {}),
    ("get", "/ops/traces", {}),
    ("get", "/ops/export", {}),
    ("post", "/evaluation/runs", {"json": {"suite": "dialog"}}),
    ("get", "/evaluation/runs", {}),
    ("get", "/evaluation/datasets", {}),
    ("post", "/evaluation/generate", {"json": {"limit": 10}}),
    ("post", "/booking/create", {"json": {"date": "2026-04-06", "time": "10:00", "name": "A", "email": "a@example.com"}}),
    ("post", "/booking/cancel", {"json": {"booking_uid": "abc12345"}}),
    ("get", "/booking/list", {}),
    ("get", "/booking/history", {}),
    ("get", "/booking/stats", {}),
    ("get", "/calls", {}),
    ("post", "/calls/outbound", {"json": {"phone_number": "+919988776655"}}),
]


@pytest.mark.parametrize("method,path,kwargs", ADMIN_ROUTES)
def test_admin_routes_reject_missing_and_wrong_tokens(client, locked, method, path, kwargs):
    assert getattr(client, method)(path, **kwargs).status_code == 401
    assert getattr(client, method)(path, headers={"X-Admin-Token": "wrong"}, **kwargs).status_code == 401
    assert store.get_bookings() == []


def test_admin_token_accepted_as_header_or_bearer(client, locked):
    assert client.get("/booking/stats", headers=ADMIN).status_code == 200
    bearer = {"Authorization": f"Bearer {ADMIN['X-Admin-Token']}"}
    assert client.get("/booking/stats", headers=bearer).status_code == 200
    assert client.get("/health").json()["auth_required"] is True


def test_production_never_falls_open_without_a_token(client, monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", "")
    assert client.get("/booking/stats").status_code == 401


def test_production_config_requires_a_strong_token(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "APP_ENV", "production")
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", "short")
    with pytest.raises(RuntimeError, match="ADMIN_API_TOKEN"):
        config.validate()
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", "x" * 32)
    config.validate()


def test_calendar_key_without_event_type_is_a_startup_error(monkeypatch):
    monkeypatch.setattr(config, "CAL_API_KEY", "cal_live_x")
    monkeypatch.setattr(config, "CAL_EVENT_TYPE_ID", 0)
    with pytest.raises(RuntimeError, match="CAL_EVENT_TYPE_ID"):
        config.validate()


# ── Rate limiting and size limits ────────────────────────────────────────────

def test_expensive_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_EXPENSIVE_PER_MINUTE", 3)
    statuses = [client.post("/query", json={"query": f"question {i}"}).status_code for i in range(5)]
    assert statuses == [200, 200, 200, 429, 429]
    assert client.post("/query", json={"query": "again"}).headers["Retry-After"]
    assert client.get("/healthz").status_code == 200           # other routes unaffected


def test_oversized_and_invalid_bodies_are_rejected(client):
    assert client.post("/query", json={"query": "x" * 5000}).status_code == 422
    assert client.post("/query", json={"query": "   "}).status_code == 422
    assert client.post("/query", json={"query": "hi", "language": "<script>"}).status_code == 422
    huge = client.post("/ingest/text", content=b"x", headers={"Content-Length": str(10**9), "Content-Type": "application/json"})
    assert huge.status_code == 413


def test_pdf_upload_limits(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_BYTES", 1000)
    too_big = client.post("/ingest/pdf", files={"file": ("a.pdf", b"%PDF" + b"0" * 2000, "application/pdf")})
    assert too_big.status_code == 413
    wrong_type = client.post("/ingest/pdf", files={"file": ("a.exe", b"MZ", "application/pdf")})
    assert wrong_type.status_code == 400
    fake_pdf = client.post("/ingest/pdf", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert fake_pdf.status_code == 400 and "not a valid PDF" in fake_pdf.json()["detail"]


# ── Knowledge base ───────────────────────────────────────────────────────────

def test_ingest_list_and_delete_sources(client):
    added = client.post("/ingest/text", json={"text": "Parking is free for visitors.", "name": "Parking"})
    assert added.json() == {"status": "ok", "chunks_added": 1, "source": "Parking"}
    assert client.get("/kb/info").json()["total_chunks"] == 2

    sources = client.get("/kb/sources").json()["sources"]
    assert {"source": "Parking", "chunks": 1} in sources

    assert client.delete("/kb/sources", params={"source": "Parking"}).json()["chunks_removed"] == 1
    assert client.delete("/kb/sources", params={"source": "Parking"}).status_code == 404
    assert client.delete("/kb/clear").status_code == 200
    assert client.get("/kb/info").json()["total_chunks"] == 0


def test_kb_search_shows_what_the_agent_would_read(client, services):
    response = client.get("/kb/search", params={"q": "When is the office open?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] and body["citations"][0]["source"] == "faq.txt"
    assert body["citations"][0]["n"] == 1 and "nine to five" in body["citations"][0]["snippet"]
    assert body["latency_ms"] >= 0
    # The search uses the agent's own retriever, not a second index.
    assert services.retriever is services.agent.rag.retriever
    assert client.get("/kb/search", params={"q": ""}).status_code == 422


def test_url_ingest_refuses_internal_addresses(client):
    response = client.post("/ingest/url", json={"url": "http://127.0.0.1:8001/booking/history"})
    assert response.status_code == 400 and "public website" in response.json()["detail"]


# ── Chat ─────────────────────────────────────────────────────────────────────

def test_query_returns_a_grounded_answer(client, provider):
    body = client.post("/query", json={"query": "What are your office hours?"}).json()
    assert body["answer"] == provider.answer and body["sources"] == ["faq.txt"]


def test_internal_errors_are_not_leaked(services, provider):
    async def explode(*args, **kwargs):
        raise RuntimeError("db password is hunter2 at /srv/secret/path")

    provider.generate = explode
    security.limiter.reset()
    with TestClient(create_app(services), raise_server_exceptions=False) as test_client:
        response = test_client.post("/query", json={"query": "anything at all"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error." and response.json()["request_id"]
    assert "hunter2" not in response.text and "/srv/secret" not in response.text


def test_llm_outage_is_a_502_with_a_friendly_message(client, provider):
    provider.llm_down = True
    response = client.post("/query", json={"query": "What are your office hours?"})
    assert response.status_code == 502 and "upstream AI provider" in response.json()["detail"]


def test_missing_speech_key_is_a_clear_503(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "c"), embedding_function=HashingEmbedder())
    security.limiter.reset()
    with TestClient(create_app(Services(vector_store=vector_store))) as test_client:
        assert test_client.get("/health").json()["api"]["status"] == "api_key_missing"
        response = test_client.post("/query", json={"query": "hello there"})
        assert response.status_code == 503 and "SARVAM_API_KEY" in response.json()["detail"]
        assert test_client.get("/booking/slots", params={"date": "2026-04-06"}).status_code == 200


def test_answers_carry_citations_and_a_trace_id_that_can_be_rated(client):
    body = client.post("/query", json={"query": "What are your office hours?"}).json()
    assert body["citations"][0]["source"] == "faq.txt" and "nine to five" in body["citations"][0]["snippet"]
    trace_id = body["trace_id"]

    assert client.post("/feedback", json={"trace_id": trace_id, "rating": -1, "comment": "too vague"}).status_code == 200
    assert client.post("/feedback", json={"trace_id": "0" * 24, "rating": 1}).status_code == 404
    assert client.post("/feedback", json={"trace_id": trace_id, "rating": 5}).status_code == 422
    assert client.post("/feedback", json={"trace_id": "../etc", "rating": 1}).status_code == 422

    trace = client.get(f"/ops/traces/{trace_id}").json()
    assert (trace["kind"], trace["channel"], trace["rating"], trace["comment"]) == ("answer", "api", -1, "too vague")
    assert trace["passages"][0]["source"] == "faq.txt" and trace["provider"] == "fake"

    summary = client.get("/ops/summary", params={"window": "24h"}).json()
    assert summary["turns"] == 1 and summary["feedback"]["not_helpful"] == 1
    assert summary["latency_ms"]["total_ms"]["count"] == 1

    export = client.get("/ops/export").text.strip().splitlines()
    assert len(export) == 1 and '"answer_given"' in export[0]
    assert client.get("/ops/timeseries", params={"window": "7d"}).json()["points"]
    assert client.get("/ops/summary", params={"window": "1y"}).status_code == 422


def test_evaluation_runs_in_the_background_and_is_stored(services, client):
    import time

    services.eval_embedding_function = HashingEmbedder()
    datasets = client.get("/evaluation/datasets").json()
    assert datasets["golden"]["cases"] >= 60 and datasets["generated"]["available"] is False

    started = client.post("/evaluation/runs", json={"suite": "dialog"})
    assert started.status_code == 202
    run_id = started.json()["id"]
    for _ in range(100):
        run = client.get(f"/evaluation/runs/{run_id}").json()
        if run["status"] != "running":
            break
        time.sleep(0.05)
    assert run["status"] == "completed", run.get("error")
    [suite] = run["suites"]
    assert suite["suite"] == "dialog" and suite["gates_passed"] and suite["headline"]["success_rate"] == 1.0
    assert client.get("/evaluation/runs").json()["runs"][0]["id"] == run_id
    assert store.get_bookings() == []                       # evaluation never touches real bookings

    live = client.post("/evaluation/runs", json={"suite": "retrieval", "kb": "live"}).json()["id"]
    for _ in range(100):
        run = client.get(f"/evaluation/runs/{live}").json()
        if run["status"] != "running":
            break
        time.sleep(0.05)
    assert run["status"] == "failed" and "Generate questions" in run["error"]
    assert client.post("/evaluation/runs", json={"suite": "nonsense"}).status_code == 422

    # A run that has not finished its first suite reports empty lists, not an object.
    with store.connection() as conn:
        conn.execute("INSERT INTO eval_runs (id, suite, dataset) VALUES ('pending01', 'all', 'sample/golden')")
    pending = client.get("/evaluation/runs/pending01").json()
    assert pending["status"] == "running" and pending["results"] == [] and pending["suites"] == []


# ── Booking ──────────────────────────────────────────────────────────────────

def test_slots_create_history_cancel_round_trip(client):
    slots = client.get("/booking/slots", params={"date": "2026-04-06"}).json()["available_slots"]
    assert slots[0] == "09:00" and "18:00" in slots

    created = client.post("/booking/create", json={
        "date": "2026-04-06", "time": "10:00", "name": "Rahul", "email": "Rahul@Example.com",
    }).json()
    assert created["status"] == "ok" and len(created["ref_code"]) == 6

    taken = client.post("/booking/create", json={
        "date": "2026-04-06", "time": "10:00", "name": "Other", "email": "o@example.com",
    })
    assert taken.status_code == 409

    [row] = client.get("/booking/history").json()["bookings"]
    assert row["attendee_email"] == "rahul@example.com" and "cal_response" not in row
    assert client.get("/booking/stats").json()["stats"]["upcoming"] == 1
    assert "10:00" not in client.get("/booking/slots", params={"date": "2026-04-06"}).json()["available_slots"]

    assert client.post("/booking/cancel", json={"booking_uid": created["booking_uid"]}).status_code == 200
    assert client.post("/booking/cancel", json={"booking_uid": created["booking_uid"]}).status_code == 404
    assert client.get("/booking/history", params={"status": "cancelled"}).json()["count"] == 1


@pytest.mark.parametrize("payload", [
    {"date": "06-04-2026", "time": "10:00", "name": "A", "email": "a@example.com"},
    {"date": "2026-04-06", "time": "25:00", "name": "A", "email": "a@example.com"},
    {"date": "2026-04-06", "time": "10:00", "name": "", "email": "a@example.com"},
    {"date": "2026-04-06", "time": "10:00", "name": "A", "email": "nope"},
])
def test_booking_create_validation(client, payload):
    assert client.post("/booking/create", json=payload).status_code == 422


def test_cancel_uid_cannot_carry_a_path(client):
    assert client.post("/booking/cancel", json={"booking_uid": "../../etc"}).status_code == 422


def test_natural_language_booking(client):
    body = client.post("/booking/intent", json={
        "query": "Book me an appointment tomorrow at 2 PM", "name": "Kiran", "email": "kiran@example.com",
    }).json()
    assert body["booking_completed"] is True and len(body["ref_code"]) == 6

    body = client.post("/booking/intent", json={"query": "Book me an appointment tomorrow at 3 PM"}).json()
    assert body["booking_completed"] is False and "name" in body["response"]


def test_bad_date_is_a_validation_error_not_a_500(client):
    assert client.get("/booking/slots", params={"date": "tomorrow"}).status_code == 422


# ── Calls ────────────────────────────────────────────────────────────────────

def test_calls_listing_and_unconfigured_outbound(client):
    store.record_call("room-1", "inbound", "+911234567890", status="in_progress")
    body = client.get("/calls").json()
    assert body["calls"][0]["room"] == "room-1" and body["telephony_configured"] is False

    response = client.post("/calls/outbound", json={"phone_number": "+919988776655"})
    assert response.status_code == 503 and "LIVEKIT_URL" in response.json()["detail"]
    assert client.post("/calls/outbound", json={"phone_number": "9988776655"}).status_code == 422


# ── WebSocket ────────────────────────────────────────────────────────────────

def _audio_message(provider, transcript, **extra):
    provider.transcript = transcript
    return {"type": "audio", "data": base64.b64encode(b"\x00" * 4000).decode(), "format": "webm", **extra}


def _until(ws, *terminal):
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["type"] in terminal:
            return events


def test_ws_voice_turn_streams_events_in_order(client, provider):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong", "ready": True}

        ws.send_json(_audio_message(provider, "What are your office hours?", duration_ms=2000))
        events = _until(ws, "done", "error")
        kinds = [e["type"] for e in events]
        assert kinds[:2] == ["processing", "transcript"] and kinds[-1] == "done"
        assert events[-1]["trace_id"] and events[-1]["citations"][0]["source"] == "faq.txt"
        audio = [base64.b64decode(e["audio"]) for e in events if e["type"] == "audio_chunk"]
        assert audio == [b"AUDIO<Our office is open nine to five.>", b"AUDIO<Visit any weekday.>"]
        assert events[-1]["answer"] == provider.answer


def test_ws_text_and_voice_share_one_booking_conversation(client, provider):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "query", "query": "book an appointment tomorrow at 2 pm"})
        assert "May I have your name" in _until(ws, "text_result")[-1]["answer"]

        ws.send_json(_audio_message(provider, "Rahul Sharma"))
        assert "email" in _until(ws, "done")[-1]["answer"]

        ws.send_json({"type": "query", "query": "rahul@gmail.com"})
        assert "Shall I book it" in _until(ws, "text_result")[-1]["answer"]

        ws.send_json(_audio_message(provider, "Yes"))
        done = _until(ws, "done")[-1]
        assert done["booking_completed"] is True
    assert store.get_bookings()[0]["attendee_name"] == "Rahul Sharma"


def test_ws_survives_garbage_input(client, provider):
    with client.websocket_connect("/ws/voice") as ws:
        for bad in ("not json", "[1,2,3]", json.dumps({"type": "launch-missiles"}),
                    json.dumps({"type": "audio", "data": "!!!not-base64!!!"}),
                    json.dumps({"type": "audio", "data": ""}),
                    json.dumps({"type": "query", "query": ""})):
            ws.send_text(bad)
            assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"            # still alive


def test_ws_recording_size_limit(client, provider, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUDIO_BYTES", 1000)
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "audio", "data": base64.b64encode(b"\x00" * 5000).decode()})
        assert "too long" in ws.receive_json()["message"]


def test_ws_provider_failure_becomes_a_friendly_error(client, provider):
    from voice_agent.sarvam_client import ProviderError

    async def broken(*args, **kwargs):
        raise ProviderError("sarvam 500: internal trace id abc")

    provider.transcribe = broken
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_audio_message(provider, "hello"))
        events = _until(ws, "error")
        assert "temporarily unavailable" in events[-1]["message"] and "trace id" not in events[-1]["message"]


def test_ws_rejects_cross_site_origins(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(
        "/ws/voice", headers={"origin": "https://evil.example"}
    ):
        pass
    assert refused.value.code == 1008

    with client.websocket_connect("/ws/voice", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_ws_rate_limit(client, provider, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_EXPENSIVE_PER_MINUTE", 2)
    with client.websocket_connect("/ws/voice") as ws:
        for _ in range(2):
            ws.send_json({"type": "query", "query": "What are your office hours?"})
            assert _until(ws, "text_result")[-1]["type"] == "text_result"
        ws.send_json({"type": "query", "query": "What are your office hours?"})
        assert "too quickly" in ws.receive_json()["message"]


def test_ws_new_utterance_interrupts_the_answer_in_progress(client, provider):
    import asyncio

    provider.answer = "First sentence that is slow to say. Second sentence follows here."
    provider.tts_delays = {"First sentence that is slow to say.": 0.5}
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(_audio_message(provider, "Tell me something long"))
        assert _until(ws, "transcript")[-1]["text"] == "Tell me something long"
        ws.send_json({"type": "cancel"})
        provider.answer = "Short."
        ws.send_json({"type": "query", "query": "Quick one"})
        events = _until(ws, "text_result")
        assert not any(e["type"] == "done" for e in events)     # the interrupted turn never finished
        assert events[-1]["answer"] == "Short."
    assert asyncio.iscoroutinefunction(provider.synthesize)
