"""Trace store: recording, masking, feedback, summaries, time series, export, retention."""

from datetime import datetime, timedelta, timezone

from voice_agent import config, tracing
from voice_agent.booking import store


def _turn(**extra):
    base = {
        "channel": "voice", "kind": "answer", "question": "What does cleaning cost?", "answer": "1,200 rupees.",
        "provider": "groq", "model": "gpt-oss", "passage_count": 2, "top_score": 0.61,
        "passages": [{"source": "faq.md", "section": "Prices", "score": 0.61}],
        "stt_ms": 300, "retrieval_ms": 40, "llm_ms": 700, "first_audio_ms": 1200, "total_ms": 1900,
        "input_tokens": 400, "output_tokens": 30, "cost_inr": 0.004,
    }
    base.update(extra)
    tracing.record(base)
    return base


def test_record_and_read_back_with_pii_masked():
    _turn(id="a" * 24, question="Book me in, my email is rahul@gmail.com and phone +919988776655")
    trace = tracing.get_trace("a" * 24)
    assert "rahul@gmail.com" not in trace["question"] and "r***@gmail.com" in trace["question"]
    assert "9988776655" not in trace["question"]
    assert trace["passages"] == [{"source": "faq.md", "section": "Prices", "score": 0.61}]
    assert trace["prompt_version"] == config.PROMPT_VERSION


def test_text_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "TRACE_STORE_TEXT", False)
    _turn(id="b" * 24)
    trace = tracing.get_trace("b" * 24)
    assert trace["question"] == "" and trace["answer"] == "" and trace["total_ms"] == 1900


def test_tracing_never_raises(monkeypatch):
    def broken():
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "connection", broken)
    tracing.record({"kind": "answer", "channel": "chat"})      # logged, not raised


def test_summary_numbers():
    _turn()
    _turn(cached=True, provider="", input_tokens=0, output_tokens=0, cost_inr=0)
    _turn(fallback=True, provider="sarvam", model="sarvam-105b")
    _turn(passage_count=0, passages=[], top_score=None)
    _turn(kind="booking", passage_count=0)
    _turn(kind="error", error="all LLM providers failed", total_ms=500)
    first = tracing.list_traces(limit=1)[0]["id"]
    tracing.set_feedback(first, 1)

    summary = tracing.summary("24h")
    assert summary["turns"] == 6 and summary["by_kind"] == {"answer": 4, "booking": 1, "error": 1}
    assert summary["error_rate"] == round(1 / 6, 4)
    assert summary["cache_hit_rate"] == 0.25
    assert summary["fallback_rate"] == round(1 / 3, 4)
    assert summary["no_context_rate"] == 0.25
    assert summary["providers"] == {"groq/gpt-oss": 2, "sarvam/sarvam-105b": 1}
    assert summary["latency_ms"]["first_audio_ms"] == {"p50": 1200.0, "p95": 1200.0, "count": 6}
    assert summary["feedback"] == {"rated": 1, "helpful": 1, "not_helpful": 0, "helpful_rate": 1.0}
    assert summary["cost_inr"] == round(0.004 * 5, 4)


def test_empty_summary_has_no_division_errors():
    summary = tracing.summary("1h")
    assert summary["turns"] == 0 and summary["error_rate"] is None and summary["latency_ms"]["total_ms"]["p50"] is None


def test_timeseries_buckets_cover_the_window():
    _turn()
    series = tracing.timeseries("24h")
    assert series["step_sec"] == 3600 and 24 <= len(series["points"]) <= 25
    assert sum(point["turns"] for point in series["points"]) == 1
    assert tracing.timeseries("7d")["step_sec"] == 86400
    assert tracing.timeseries("1h")["step_sec"] == 300


def test_export_turns_rated_answers_into_draft_cases():
    _turn(id="c" * 24)
    _turn(id="d" * 24)
    tracing.set_feedback("c" * 24, -1, "wrong price")
    [case] = tracing.export_cases(rating=-1)
    assert case["question"] == "What does cleaning cost?" and case["notes"]["comment"] == "wrong price"
    assert case["notes"]["sources"] == ["faq.md"] and case["evidence"] == []


def test_old_traces_are_purged(monkeypatch):
    _turn(id="e" * 24)
    old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S")
    with store.connection() as conn:
        conn.execute("UPDATE traces SET created_at = ? WHERE id = ?", (old, "e" * 24))
    monkeypatch.setattr(tracing, "_last_purge", 0.0)
    monkeypatch.setattr(tracing.time, "monotonic", lambda: 10_000.0)
    _turn(id="f" * 24)
    assert tracing.get_trace("e" * 24) is None and tracing.get_trace("f" * 24) is not None
