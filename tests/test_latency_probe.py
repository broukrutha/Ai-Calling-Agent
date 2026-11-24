"""scripts/latency_probe.py: runs spoken turns through the real pipeline and reports per-stage latency."""

import importlib.util
from pathlib import Path

from langchain_core.documents import Document

from tests.fakes import FakeProviderClient, HashingEmbedder
from voice_agent.pipeline import VoiceAgent
from voice_agent.rag_pipeline import RAGPipeline
from voice_agent.vector_store import VectorStore

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "latency_probe.py"
_spec = importlib.util.spec_from_file_location("latency_probe", _PATH)
latency_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(latency_probe)


async def test_probe_reports_every_stage(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "kb"), embedding_function=HashingEmbedder())
    store.add_documents([Document(page_content="We are open nine to five on weekdays.", metadata={"source": "faq.txt"})])
    client = FakeProviderClient(transcript="What are your opening hours?")
    agent = VoiceAgent(rag_pipeline=RAGPipeline(vector_store=store, sarvam_client=client), sarvam_client=client,
                       tracer=lambda trace: None)

    rows = await latency_probe.probe(agent, client, ["What are your opening hours?", "When do you open?"])

    assert len(rows) == 2 and all("error" not in row for row in rows), rows
    assert rows[0]["heard"] == "What are your opening hours?"
    assert all(row["first_audio_ms"] > 0 and row["total_ms"] >= row["first_audio_ms"] for row in rows)
    summary = latency_probe.summarize(rows)
    assert summary["first_audio_ms"]["n"] == 2
    text = latency_probe.report(rows, summary)
    assert "First audio" in text and "budget" in text


def test_summary_skips_failed_turns():
    rows = [{"question": "a", "first_audio_ms": 900.0, "total_ms": 1500.0}, {"question": "b", "error": "timeout"}]
    summary = latency_probe.summarize(rows)
    assert summary["first_audio_ms"] == {"p50": 900.0, "p95": 900.0, "n": 1}
    assert "failed: timeout" in latency_probe.report(rows, summary)
