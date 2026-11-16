"""Phone-agent tools and outbound call dispatch (no LiveKit connection needed)."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from tests.fakes import FakeProviderClient, HashingEmbedder
from voice_agent import config
from voice_agent.booking import store
from voice_agent.rag_pipeline import RAGPipeline
from voice_agent.telephony import outbound, tools
from voice_agent.vector_store import VectorStore


@pytest.fixture
def rag(tmp_path):
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"), embedding_function=HashingEmbedder())
    vector_store.add_documents([
        Document(page_content="A consultation costs five hundred rupees.", metadata={"source": "prices.txt"}),
    ])
    return RAGPipeline(vector_store=vector_store, sarvam_client=FakeProviderClient())


async def test_knowledge_tool_returns_grounding_text(rag):
    assert "five hundred rupees" in await tools.search_knowledge_base(rag, "consultation cost")
    rag.vs.clear()
    assert "No information" in await tools.search_knowledge_base(rag, "consultation cost")


async def test_availability_tool():
    assert (await tools.check_availability("2026-04-06")).startswith("Open times on Monday, April 6: 9 AM, 10 AM")
    assert "next day with openings is Monday, April 6" in await tools.check_availability("2026-04-04")
    assert "in the past" in await tools.check_availability("2026-03-01")
    assert "could not understand" in await tools.check_availability("sometime soon")
    assert "Thursday, April 2" in await tools.check_availability("tomorrow")      # lenient with a small LLM


async def test_book_and_cancel_tools_round_trip():
    reply = await tools.book_appointment("2026-04-06", "14:00", "Rahul Sharma", "rahul at gmail dot com")
    assert reply.startswith("BOOKED for 2 PM on Monday, April 6")
    [booking] = store.get_bookings()
    assert booking["attendee_email"] == "rahul@gmail.com" and booking["source"] == "phone"
    assert " ".join(booking["ref_code"]) in reply

    again = await tools.book_appointment("2026-04-06", "14:00", "Someone Else", "x@example.com")
    assert again.startswith("NOT BOOKED") and "Open times that day" in again

    assert (await tools.cancel_appointment("000000")).startswith("NOT CANCELLED")
    assert (await tools.cancel_appointment("no idea")).startswith("NOT CANCELLED")
    spoken_reference = " ".join(booking["ref_code"])
    assert (await tools.cancel_appointment(spoken_reference)).startswith("CANCELLED the appointment at 2 PM")
    assert store.get_bookings()[0]["status"] == "cancelled"


@pytest.mark.parametrize("day,time,name,email,reason", [
    ("2026-04-06", "14:00", "Rahul", "not an email", "email address is not valid"),
    ("2026-04-06", "whenever", "Rahul", "r@example.com", "date or time was not understood"),
    ("someday", "14:00", "Rahul", "r@example.com", "date or time was not understood"),
    ("2026-04-06", "14:00", "  ", "r@example.com", "name is missing"),
    ("2026-04-06", "22:00", "Rahul", "r@example.com", "is not available"),
])
async def test_book_tool_refuses_bad_input_and_books_nothing(day, time, name, email, reason):
    reply = await tools.book_appointment(day, time, name, email)
    assert reply.startswith("NOT BOOKED") and reason in reply
    assert store.get_bookings() == []


def test_phone_instructions_carry_today_and_the_guardrails():
    prompt = tools.build_instructions()
    assert "Wednesday, April 1, 2026" in prompt and "9 AM to 7 PM" in prompt
    assert 'Only say the appointment is booked if book_appointment returns "BOOKED"' in prompt


def test_phone_agent_exposes_its_tools_to_the_llm():
    pytest.importorskip("livekit.agents")
    from voice_agent.telephony.agent import PhoneAssistant

    agent = PhoneAssistant(greeting="Hello")
    names = {tool.info.name for tool in agent.tools}
    assert names == {"search_knowledge_base", "check_availability", "book_appointment", "cancel_appointment", "end_call"}


# ── Outbound dispatch ────────────────────────────────────────────────────────

class FakeLiveKit:
    def __init__(self):
        self.requests = []
        self.closed = False
        self.agent_dispatch = SimpleNamespace(create_dispatch=self._create)

    def LiveKitAPI(self, **credentials):  # noqa: N802 - mirrors the SDK name
        self.credentials = credentials
        return self

    def CreateAgentDispatchRequest(self, **fields):  # noqa: N802
        return fields

    async def _create(self, request):
        self.requests.append(request)
        return SimpleNamespace(id="AD_123")

    async def aclose(self):
        self.closed = True


@pytest.fixture
def telephony(monkeypatch):
    monkeypatch.setattr(config, "LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setattr(config, "LIVEKIT_API_KEY", "key")
    monkeypatch.setattr(config, "LIVEKIT_API_SECRET", "secret")
    monkeypatch.setattr(config, "OUTBOUND_TRUNK_ID", "ST_abc")


async def test_dispatch_creates_a_room_and_logs_the_call(telephony):
    livekit = FakeLiveKit()
    call = await outbound.dispatch_call("+91 99887-76655", livekit_api=livekit)

    [request] = livekit.requests
    assert request["agent_name"] == "voice-assistant"
    assert json.loads(request["metadata"]) == {"phone_number": "+919988776655"}
    assert request["room"] == call["room"] and "9988776655" not in call["room"]   # no PII in room names
    assert call["dispatch_id"] == "AD_123" and livekit.closed

    [logged] = store.get_calls()
    assert (logged["direction"], logged["phone"], logged["status"]) == ("outbound", "+919988776655", "queued")


@pytest.mark.parametrize("number", ["9988776655", "+0123456789", "+91abc", "", "+1", "+9198877665544332211"])
async def test_dispatch_rejects_numbers_that_are_not_e164(telephony, number):
    with pytest.raises(ValueError):
        await outbound.dispatch_call(number, livekit_api=FakeLiveKit())
    assert store.get_calls() == []


async def test_dispatch_explains_missing_configuration(monkeypatch, telephony):
    monkeypatch.setattr(config, "OUTBOUND_TRUNK_ID", "")
    with pytest.raises(outbound.TelephonyNotConfiguredError, match="OUTBOUND_TRUNK_ID"):
        await outbound.dispatch_call("+919988776655", livekit_api=FakeLiveKit())
    monkeypatch.setattr(config, "LIVEKIT_URL", "")
    with pytest.raises(outbound.TelephonyNotConfiguredError, match="LIVEKIT_URL"):
        await outbound.dispatch_call("+919988776655", livekit_api=FakeLiveKit())


def test_outbound_api_endpoint(telephony, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from voice_agent.server import create_app, security
    from voice_agent.server.deps import Services
    from voice_agent.server.routes import calls as calls_route

    async def fake_dispatch(number):
        store.record_call("call-test", "outbound", number)
        return {"room": "call-test", "dispatch_id": "AD_9"}

    monkeypatch.setattr(calls_route, "dispatch_call", fake_dispatch)
    security.limiter.reset()
    services = Services(vector_store=VectorStore(persist_dir=str(tmp_path / "c"), embedding_function=HashingEmbedder()))
    with TestClient(create_app(services)) as client:
        body = client.post("/calls/outbound", json={"phone_number": "+919988776655"}).json()
        assert body == {"status": "ok", "room": "call-test", "dispatch_id": "AD_9"}
        listing = client.get("/calls").json()
        assert listing["outbound_enabled"] is True and listing["stats"]["outbound"] == 1
