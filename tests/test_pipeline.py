"""Voice pipeline behaviour with fake providers: ordering, caching, memory, failure."""

import pytest
from langchain_core.documents import Document

from tests.fakes import FakeProviderClient, HashingEmbedder
from voice_agent import pipeline
from voice_agent.pipeline import ConversationSession, SentenceChunker, VoiceAgent
from voice_agent.rag_pipeline import RAGPipeline
from voice_agent.vector_store import VectorStore


@pytest.fixture
def client():
    return FakeProviderClient()


@pytest.fixture
def agent(tmp_path, client):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), embedding_function=HashingEmbedder())
    store.add_documents([
        Document(page_content="Our office is open nine to five on weekdays.", metadata={"source": "faq.txt"}),
    ])
    return VoiceAgent(rag_pipeline=RAGPipeline(vector_store=store, sarvam_client=client), sarvam_client=client)


async def collect(agent, client, transcript, session=None, **kwargs):
    client.transcript = transcript
    return [event async for event in agent.run_stream(b"\x00" * 8000, session=session, **kwargs)]


def of_type(events, kind):
    return [event for event in events if event["type"] == kind]


# ── Sentence chunking ────────────────────────────────────────────────────────

def test_sentences_are_cut_as_text_streams_in():
    chunker = SentenceChunker()
    out = []
    for token in ["Hello there, ", "welcome. ", "We open at 9.", "30 AM. Call", " us today!"]:
        out += chunker.feed(token)
    out += chunker.flush()
    assert out == ["Hello there, welcome.", "We open at 9.30 AM.", "Call us today!"]


def _chunks(chunker, tokens):
    out = []
    for token in tokens:
        out += chunker.feed(token)
    return out + chunker.flush()


def test_first_chunk_can_end_at_a_clause_to_start_speaking_sooner():
    tokens = ["Root canal treatment costs 6,500 rupees ", "for a front tooth, ", "and 9,000 rupees for a molar, ",
              "including the X-ray. ", "Shall I book a visit?"]
    assert _chunks(SentenceChunker(first_clause_chars=40), tokens) == [
        "Root canal treatment costs 6,500 rupees for a front tooth,",       # the "6,500" comma never cuts
        "and 9,000 rupees for a molar, including the X-ray.",              # later pieces are whole sentences
        "Shall I book a visit?",
    ]
    # Off by default, and a clause shorter than the minimum waits for the sentence.
    assert _chunks(SentenceChunker(), tokens)[0].endswith("including the X-ray.")
    assert _chunks(SentenceChunker(first_clause_chars=40), ["Yes, ", "we are open on Sunday mornings. ", "Come by!"]) == [
        "Yes, we are open on Sunday mornings.", "Come by!",
    ]


@pytest.mark.parametrize("text,expected", [
    ("It costs Rs. 30 per hour. Paid monthly.", ["It costs Rs. 30 per hour.", "Paid monthly."]),
    ("Version 3.5 is out now.", ["Version 3.5 is out now."]),
    ("Yes. No. Maybe so, we will see.", ["Yes. No. Maybe so, we will see."]),   # too short to stand alone
    ("नमस्ते, आप कैसे हैं। हम नौ बजे खुलते हैं।", ["नमस्ते, आप कैसे हैं।", "हम नौ बजे खुलते हैं।"]),
    ("తెలుగులో మమ్మల్ని సంప్రదించండి", ["తెలుగులో మమ్మల్ని సంప్రదించండి"]),      # used to split at every "మ"
    ("", []),
])
def test_split_sentences(text, expected):
    assert pipeline.split_sentences(text) == expected


@pytest.mark.parametrize("text,in_dialog,expected", [
    ("", False, True), ("...", False, True), ("um uh", False, True), ("Hmm.", True, True),
    ("ok", False, True), ("no", True, False), ("11", True, False), ("11", False, True),
    ("what are your hours", False, False), ("हाँ", True, False),
])
def test_noise_filter_keeps_short_answers_inside_a_dialog(text, in_dialog, expected):
    assert pipeline.is_noise(text, in_dialog=in_dialog) is expected


def test_language_detection():
    assert pipeline.detect_language("आपके कार्यालय का समय क्या है") == "hi-IN"
    assert pipeline.detect_language("మీ కార్యాలయ సమయం ఏమిటి") == "te-IN"
    assert pipeline.detect_language("office timings kya hai") == "hi-IN"
    assert pipeline.detect_language("what are your office hours") == "en-IN"
    assert pipeline.choose_reply_language("what are your hours", "ta-IN") == "ta-IN"
    assert pipeline.choose_reply_language("what are your hours", "unknown") == "en-IN"


def test_wav_duration_is_read_from_the_header():
    import struct

    pcm = b"\x00" * 32000                                       # 1 second of 16 kHz mono 16-bit
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    assert pipeline.estimate_audio_seconds(header + pcm, "wav") == pytest.approx(1.0)
    assert pipeline.estimate_audio_seconds(b"\x1aE\xdf\xa3" + b"\x00" * 7996, "webm") == pytest.approx(2.0)


# ── Voice turns ──────────────────────────────────────────────────────────────

async def test_voice_turn_streams_text_then_audio_per_sentence(agent, client):
    events = await collect(agent, client, "What are your office hours?")

    assert [e["type"] for e in events][0] == "transcript"
    assert events[-1]["type"] == "done"
    assert "".join(e["text"] for e in of_type(events, "text_chunk")).strip() == client.answer
    assert [e["audio"] for e in of_type(events, "audio_chunk")] == [
        b"AUDIO<Our office is open nine to five.>", b"AUDIO<Visit any weekday.>",
    ]

    done = events[-1]
    assert done["answer"] == client.answer and done["sources"] == ["faq.txt"] and done["context_chunks"] == 1
    assert done["latency"]["first_audio_ms"] > 0 and done["latency"]["within_budget"]
    assert done["cost_inr"] > 0
    assert "nine to five" in client.llm_calls[0]["context"]         # grounded in the retrieved chunk


async def test_audio_stays_in_sentence_order_even_when_later_sentences_finish_first(agent, client):
    client.answer = "First sentence is slow. Second sentence is fast. Third one is fast too."
    client.tts_delays = {"First sentence is slow.": 0.05}
    events = await collect(agent, client, "Tell me something long please")
    assert [e["audio"] for e in of_type(events, "audio_chunk")] == [
        b"AUDIO<First sentence is slow.>", b"AUDIO<Second sentence is fast.>", b"AUDIO<Third one is fast too.>",
    ]


async def test_first_audio_does_not_wait_for_the_whole_answer(agent, client):
    client.answer = "This is the opening sentence. " + "And here is some more detail for you. " * 5
    events = await collect(agent, client, "Give me the long version")
    kinds = [e["type"] for e in events]
    first_audio = kinds.index("audio_chunk")
    last_text = len(kinds) - 1 - kinds[::-1].index("text_chunk")
    assert first_audio < last_text                                   # audio started mid-stream


async def test_noise_is_dropped_without_calling_the_llm(agent, client):
    events = await collect(agent, client, "um")
    assert [e["type"] for e in events] == ["noise"] and client.llm_calls == []


async def test_reply_language_follows_the_caller(agent, client):
    client.language = "hi-IN"
    events = await collect(agent, client, "आपके कार्यालय का समय क्या है")
    assert client.llm_calls[0]["language"] == "hi-IN"
    assert {language for _, language in client.tts_calls} == {"hi-IN"}
    assert events[-1]["language"] == "hi-IN"


async def test_llm_outage_gives_a_spoken_apology_and_is_not_cached(agent, client):
    client.llm_down = True
    events = await collect(agent, client, "What are your office hours?")
    done = events[-1]
    assert done["answer"] == pipeline.LLM_FAILURE_REPLY and done["degraded"] is True
    assert of_type(events, "audio_chunk"), "the apology is spoken, not just shown"

    client.llm_down = False
    events = await collect(agent, client, "What are your office hours?")
    assert events[-1]["answer"] == client.answer and "cached" not in events[-1]


# ── Cache ────────────────────────────────────────────────────────────────────

async def test_repeat_question_is_served_from_cache_with_its_audio(agent, client):
    await collect(agent, client, "What are your office hours?")
    events = await collect(agent, client, "what are your office hours")       # punctuation/case differ
    assert events[-1]["cached"] is True and len(client.llm_calls) == 1
    assert len(of_type(events, "audio_chunk")) == 2 and len(client.tts_calls) == 2


async def test_cache_is_invalidated_when_the_knowledge_base_changes(agent, client):
    await agent.query_text("What are your office hours?")
    agent.rag.vs.add_documents([Document(page_content="We are now open on Saturdays too.", metadata={"source": "new.txt"})])
    client.answer = "We are open Saturdays as well."
    result = await agent.query_text("What are your office hours?")
    assert result["answer"] == "We are open Saturdays as well." and "cached" not in result


async def test_typed_answer_in_cache_is_still_spoken_on_a_voice_turn(agent, client):
    await agent.query_text("What are your office hours?")
    events = await collect(agent, client, "What are your office hours?")
    assert events[-1]["cached"] is True and len(of_type(events, "audio_chunk")) == 2


# ── Conversation memory ──────────────────────────────────────────────────────

async def test_follow_up_questions_carry_the_conversation(agent, client):
    session = ConversationSession()
    await agent.query_text("What are your office hours?", session=session)
    client.answer = "Yes, until one in the afternoon."
    await agent.query_text("And on Saturdays?", session=session)

    history = client.llm_calls[1]["history"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "What are your office hours?"


async def test_history_is_bounded(agent, client, monkeypatch):
    session = ConversationSession()
    for index in range(10):
        await agent.query_text(f"Question number {index}?", session=session)
    assert len(session.history) == 6


# ── Booking through the pipeline ─────────────────────────────────────────────

async def test_voice_booking_flow_end_to_end(agent, client):
    session = ConversationSession()
    await collect(agent, client, "Book an appointment tomorrow at 2 pm", session)
    await collect(agent, client, "Rahul Sharma", session)
    await collect(agent, client, "rahul at gmail dot com", session)
    events = await collect(agent, client, "Yes", session)            # a 3-letter reply must not be "noise"

    done = events[-1]
    assert done["is_booking"] and done["booking_completed"] and len(done["ref_code"]) == 6
    assert of_type(events, "audio_chunk") and client.llm_calls == []    # the LLM is never asked to "book"
    assert {language for _, language in client.tts_calls} == {"en-IN"}


async def test_booking_takes_priority_over_a_cached_answer(agent, client):
    session = ConversationSession()
    await agent.query_text("yes")                                    # caches an LLM answer for "yes"
    await agent.query_text("book tomorrow at 2 pm", session=session)
    await agent.query_text("Asha", session=session)
    await agent.query_text("asha@example.com", session=session)
    result = await agent.query_text("yes", session=session)
    assert result["is_booking"] and result["booking_completed"]


async def test_text_query_shape(agent, client):
    result = await agent.query_text("What are your office hours?")
    assert result["answer"] == client.answer and result["tokens"] == {"input": 100, "output": 20}
    assert set(result["latency"]) == {"retrieval_ms", "llm_ms", "total_ms"}


# ── Traces, citations, speech-safe output ────────────────────────────────────

async def test_every_voice_turn_is_traced_with_its_details(agent, client):
    from voice_agent import tracing

    events = await collect(agent, client, "What are your office hours?")
    done = events[-1]
    trace = tracing.get_trace(done["trace_id"])
    assert (trace["kind"], trace["channel"], trace["provider"]) == ("answer", "voice", "fake")
    assert trace["passage_count"] == 1 and trace["passages"][0]["source"] == "faq.txt"
    assert trace["first_audio_ms"] > 0 and trace["question"] == "What are your office hours?"
    assert done["citations"][0]["source"] == "faq.txt"

    await collect(agent, client, "um")
    await collect(agent, client, "Book an appointment tomorrow at 2 pm", ConversationSession())
    kinds = [row["kind"] for row in tracing.list_traces()]
    assert kinds == ["booking", "noise", "answer"]


async def test_provider_failures_are_traced_then_raised(agent, client):
    from voice_agent import tracing
    from voice_agent.sarvam_client import ProviderError

    async def broken(*args, **kwargs):
        raise ProviderError("stt down")

    client.transcribe = broken
    with pytest.raises(ProviderError):
        await collect(agent, client, "anything")
    [trace] = tracing.list_traces()
    assert trace["kind"] == "error" and "stt down" in trace["error"]


async def test_markdown_is_never_spoken_or_shown(agent, client):
    client.answer = "**Cleaning** costs ₹1,200 [1]. Book online at https://www.example.com/book today."
    events = await collect(agent, client, "How much is cleaning?")
    spoken = [text for text, _ in client.tts_calls]
    assert all("*" not in text and "http" not in text and "[1]" not in text for text in spoken)
    assert any("1,200 rupees" in text for text in spoken)
    assert events[-1]["answer"].startswith("Cleaning costs ₹1,200.")


async def test_typed_follow_up_searches_with_the_previous_turn(agent, client):
    session = ConversationSession()
    await agent.query_text("What are your office hours?", session=session)
    result = await agent.query_text("And on Saturdays?", session=session)
    from voice_agent import tracing

    trace = tracing.get_trace(result["trace_id"])
    assert trace["followup"] == 1 and "office hours" in trace["retrieval_query"]
