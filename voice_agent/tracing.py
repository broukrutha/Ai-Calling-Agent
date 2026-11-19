"""
Conversation traces: one record per turn, for monitoring and debugging.

Each answered turn records what was asked, what was retrieved (source, section,
score), which provider and model answered, the latency of every stage, token
use, cost, cache hits, fallbacks and errors. People can rate answers, and the
poorly rated ones can be exported straight into an evaluation dataset.

Personal data: emails and phone numbers in questions and answers are masked
before they are written. Set ``TRACE_STORE_TEXT=false`` to keep no text at
all. Traces older than ``TRACE_RETENTION_DAYS`` are deleted automatically.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from voice_agent import config
from voice_agent.booking import store
from voice_agent.logging_config import mask_email, mask_phone

logger = logging.getLogger(__name__)

WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
_TIMING_FIELDS = ("stt_ms", "retrieval_ms", "llm_ms", "first_token_ms", "tts_ms", "first_audio_ms", "total_ms")
_TEXT_FIELDS = ("question", "retrieval_query", "answer")
_COLUMNS = (
    "id", "channel", "session_id", "kind", "question", "retrieval_query", "answer", "language", "provider",
    "model", "prompt_version", "cached", "followup", "fallback", "passage_count", "top_score", "passages",
    *_TIMING_FIELDS, "input_tokens", "output_tokens", "cost_inr", "error",
)
_last_purge = 0.0


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(math.ceil(fraction * len(ordered)), 1)
    return ordered[min(rank, len(ordered)) - 1]


def new_trace_id() -> str:
    return secrets.token_hex(12)


def _scrub(text: str) -> str:
    if not config.TRACE_STORE_TEXT:
        return ""
    return mask_phone(mask_email(text or ""))[:2000]


def record(trace: dict[str, Any]) -> None:
    """Write one turn. Never raises: tracing must not break a conversation."""
    if not config.TRACING_ENABLED:
        return
    try:
        row = {column: trace.get(column) for column in _COLUMNS}
        row["id"] = row["id"] or new_trace_id()
        for name in _TEXT_FIELDS:
            row[name] = _scrub(str(row[name] or ""))
        row["passages"] = json.dumps(row["passages"] or [], ensure_ascii=False)[:4000]
        for name in ("cached", "followup", "fallback"):
            row[name] = 1 if row[name] else 0
        for name in ("channel", "session_id", "kind", "language", "provider", "model", "error"):
            row[name] = str(row[name] or "")[:300]
        row["prompt_version"] = row["prompt_version"] or config.PROMPT_VERSION
        for name in ("passage_count", "input_tokens", "output_tokens"):
            row[name] = int(row[name] or 0)
        row["cost_inr"] = float(row["cost_inr"] or 0.0)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with store.connection() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO traces ({', '.join(_COLUMNS)}) VALUES ({placeholders})",  # noqa: S608
                [row[column] for column in _COLUMNS],
            )
        _purge_old()
    except Exception:
        logger.exception("Could not record trace")


def _purge_old() -> None:
    global _last_purge
    if time.monotonic() - _last_purge < 3600:
        return
    _last_purge = time.monotonic()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=config.TRACE_RETENTION_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
    with store.connection() as conn:
        removed = conn.execute("DELETE FROM traces WHERE created_at < ?", (cutoff,)).rowcount
    if removed:
        logger.info("Purged %d traces older than %d days", removed, config.TRACE_RETENTION_DAYS)


def set_feedback(trace_id: str, rating: int, comment: str = "") -> bool:
    """Rate an answer: 1 helpful, -1 not helpful. Returns False for an unknown trace."""
    with store.connection() as conn:
        cursor = conn.execute(
            "UPDATE traces SET rating = ?, comment = ? WHERE id = ?", (rating, _scrub(comment)[:500], trace_id)
        )
        return cursor.rowcount > 0


def _decode(row: Any) -> dict[str, Any]:
    data = dict(row)
    try:
        data["passages"] = json.loads(data.get("passages") or "[]")
    except ValueError:
        data["passages"] = []
    return data


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with store.connection() as conn:
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
    return _decode(row) if row else None


def list_traces(
    limit: int = 50, kind: str | None = None, rating: int | None = None, since: str | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM traces WHERE 1=1"
    params: list[Any] = []
    if kind:
        query += " AND kind = ?"
        params.append(kind)
    if rating is not None:
        query += " AND rating = ?"
        params.append(rating)
    if since:
        query += " AND created_at >= ?"
        params.append(since)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with store.connection() as conn:
        return [_decode(row) for row in conn.execute(query, params).fetchall()]


def _window_start(window: str) -> datetime:
    return datetime.now(timezone.utc) - WINDOWS.get(window, WINDOWS["24h"])


def _rows_since(start: datetime, limit: int = 50_000) -> list[dict[str, Any]]:
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE created_at >= ? ORDER BY created_at ASC LIMIT ?",
            (start.strftime("%Y-%m-%dT%H:%M:%S"), limit),
        ).fetchall()
    return [dict(row) for row in rows]


def _latency_block(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [row[field] for row in rows if isinstance(row.get(field), (int, float)) and row[field] > 0]
    if not values:
        return {"p50": None, "p95": None, "count": 0}
    return {"p50": round(percentile(values, 0.5), 1), "p95": round(percentile(values, 0.95), 1), "count": len(values)}


def summary(window: str = "24h") -> dict[str, Any]:
    """Headline numbers for the monitoring dashboard."""
    rows = _rows_since(_window_start(window))
    answers = [row for row in rows if row["kind"] == "answer"]
    llm_answers = [row for row in answers if not row["cached"]]
    rated = [row for row in rows if row["rating"] in (1, -1)]
    turns = len(rows)

    def share(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    return {
        "window": window,
        "turns": turns,
        "by_kind": dict(Counter(row["kind"] for row in rows)),
        "by_channel": dict(Counter(row["channel"] for row in rows)),
        "error_rate": share(sum(1 for row in rows if row["kind"] == "error"), turns),
        "cache_hit_rate": share(sum(1 for row in answers if row["cached"]), len(answers)),
        "fallback_rate": share(sum(1 for row in llm_answers if row["fallback"]), len(llm_answers)),
        "no_context_rate": share(sum(1 for row in answers if not row["passage_count"]), len(answers)),
        "followup_rate": share(sum(1 for row in answers if row["followup"]), len(answers)),
        "providers": dict(Counter(f"{row['provider']}/{row['model']}" for row in llm_answers if row["provider"])),
        "latency_ms": {field: _latency_block(rows, field) for field in _TIMING_FIELDS},
        "tokens": {
            "input": sum(row["input_tokens"] for row in rows),
            "output": sum(row["output_tokens"] for row in rows),
        },
        "cost_inr": round(sum(row["cost_inr"] for row in rows), 4),
        "feedback": {
            "rated": len(rated),
            "helpful": sum(1 for row in rated if row["rating"] == 1),
            "not_helpful": sum(1 for row in rated if row["rating"] == -1),
            "helpful_rate": share(sum(1 for row in rated if row["rating"] == 1), len(rated)),
        },
        "prompt_versions": dict(Counter(row["prompt_version"] for row in rows)),
    }


def timeseries(window: str = "24h") -> dict[str, Any]:
    """Per-bucket volume, latency percentiles, errors and cost, for charts."""
    start = _window_start(window)
    step = timedelta(hours=1) if WINDOWS.get(window, WINDOWS["24h"]) <= timedelta(hours=24) else timedelta(days=1)
    if window == "1h":
        step = timedelta(minutes=5)
    buckets: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows_since(start):
        try:
            moment = datetime.strptime(row["created_at"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        index = int((moment - start) / step)
        buckets[start + index * step].append(row)

    points = []
    cursor = start
    now = datetime.now(timezone.utc)
    while cursor <= now:
        rows = buckets.get(cursor, [])
        first_audio = _latency_block(rows, "first_audio_ms")
        total = _latency_block(rows, "total_ms")
        points.append({
            "t": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "turns": len(rows),
            "errors": sum(1 for row in rows if row["kind"] == "error"),
            "first_audio_p50": first_audio["p50"],
            "first_audio_p95": first_audio["p95"],
            "total_p50": total["p50"],
            "total_p95": total["p95"],
            "cost_inr": round(sum(row["cost_inr"] for row in rows), 4),
        })
        cursor += step
    return {"window": window, "step_sec": int(step.total_seconds()), "points": points}


def export_cases(rating: int | None = -1, limit: int = 500) -> list[dict[str, Any]]:
    """
    Turn rated traces into draft evaluation cases (JSONL rows). The reference
    answer is left for a person to fill in: that is the point of reviewing them.
    """
    cases = []
    for trace in list_traces(limit=limit, kind="answer", rating=rating):
        if not trace["question"]:
            continue
        cases.append({
            "id": f"trace-{trace['id'][:10]}",
            "question": trace["question"],
            "type": "factoid",
            "answerable": True,
            "evidence": [],
            "reference": "",
            "answer_keywords": [],
            "notes": {
                "answer_given": trace["answer"],
                "rating": trace["rating"],
                "comment": trace["comment"],
                "sources": sorted({p.get("source", "") for p in trace["passages"]}),
                "created_at": trace["created_at"],
            },
        })
    return cases
