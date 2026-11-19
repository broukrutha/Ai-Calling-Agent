"""
Answer evaluation: is what the agent *says* right, grounded and speakable?

Every case is sent through the real agent (retrieval + LLM), then scored:

Deterministic checks (always run, free)
  * keywords        — the expected facts (prices, names) appear in the answer
  * refusal         — did the agent say it doesn't know?
  * voice format    — no Markdown, lists or URLs; short enough to listen to

LLM judge (when a model is configured)
  * correctness     — matches the reference answer (1, 0.5 or 0)
  * faithfulness    — every claim is supported by the retrieved passages
  * refusal         — the judge's own reading of whether it declined

A case passes when an answerable question gets a correct, faithful, non-refusing
answer, or an unanswerable one gets a refusal instead of an invented answer.

The judge sees the reference answer and the retrieved passages, never the
agent's internals, and must return strict JSON with a short reason. It uses the
same provider chain as the agent, so treat its scores as a strong signal, not
ground truth: review the failures it reports.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from typing import Any

from evaluation import metrics
from evaluation.dataset import EvalCase

JUDGE_PROMPT = """You grade answers given by a voice assistant that must answer only from reference passages.

You receive: the QUESTION, the REFERENCE answer written by a person (or a note that the
documents do not contain the answer), the PASSAGES the assistant retrieved, and the ANSWER.

Grade strictly:
- correctness: 1 if the answer gives the same facts as the reference (wording may differ,
  numbers must match), 0.5 if partly right or missing an important detail, 0 if wrong or
  if it fails to answer. For questions the documents cannot answer, 1 if the assistant
  says it does not have that information, 0 if it invents an answer.
- faithful: true if every factual claim in the answer is supported by the PASSAGES
  (saying "I don't have that information" is faithful), false if anything is invented.
- refused: true if the answer declines or says it does not have the information.

Reply with JSON only, no other text:
{"correctness": 1, "faithful": true, "refused": false, "reason": "one short sentence"}"""

_REFUSAL_PATTERNS = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont)\s+(?:have|know)|(?:not|isn't|is\s+not)\s+(?:mentioned|available|in\s+the)|"
    r"no\s+information|couldn't\s+find|could\s+not\s+find|unable\s+to\s+find|not\s+sure|"
    r"don't\s+have\s+(?:that|this|any)|i'm\s+afraid|cannot\s+help\s+with|can't\s+help\s+with|"
    r"(?:reference|provided)\s+(?:material|information)\s+does(?:n't|\s+not))",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*]\s|\[[^\]]+\]\(|https?://|\|)", re.MULTILINE)
_SENTENCE = re.compile(r"[.!?।]+(?:\s|$)")
MAX_SPOKEN_WORDS = 70


def detect_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(answer or ""))


def voice_format_ok(answer: str) -> bool:
    """Speakable: no Markdown or links, at most ~3 sentences / 70 words."""
    if _MARKDOWN.search(answer or ""):
        return False
    words = len((answer or "").split())
    sentences = max(len(_SENTENCE.findall(answer or "")), 1)
    return words <= MAX_SPOKEN_WORDS and sentences <= 4


def parse_judgement(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a judge reply; tolerant of code fences and prose."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    try:
        correctness = float(data.get("correctness", 0))
    except (TypeError, ValueError):
        return None
    return {
        "correctness": min(max(correctness, 0.0), 1.0),
        "faithful": bool(data.get("faithful", False)),
        "refused": bool(data.get("refused", False)),
        "reason": str(data.get("reason", ""))[:300],
    }


async def judge_answer(client: Any, case: EvalCase, answer: str, context: str) -> dict[str, Any] | None:
    reference = case.reference if case.answerable else (
        "The documents do not contain this information. The correct behaviour is to say so."
    )
    user = (
        f"QUESTION: {case.question}\n\nREFERENCE: {reference}\n\n"
        f"PASSAGES:\n{context or '(none were retrieved)'}\n\nANSWER: {answer}"
    )
    try:
        result = await client.complete(
            [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            max_tokens=200,
            temperature=0.0,
        )
    except Exception:
        return None
    return parse_judgement(result.answer)


async def evaluate_answers(cases: list[EvalCase], agent: Any, judge_client: Any | None = None) -> dict[str, Any]:
    """Run every case through ``agent.query_text`` and score the answers."""
    from voice_agent.pipeline import ConversationSession

    rows: list[dict[str, Any]] = []
    for case in cases:
        session = ConversationSession()
        for turn in case.history:
            session.history.append(dict(turn))
        try:
            result = await agent.query_text(case.question, session=session, language=case.language, channel="eval")
        except Exception as exc:
            rows.append({"id": case.id, "question": case.question, "type": case.type,
                         "answerable": case.answerable, "error": f"{type(exc).__name__}: {exc}"[:300],
                         "passed": False})
            continue

        answer = result["answer"]
        row: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "type": case.type,
            "answerable": case.answerable,
            "answer": answer,
            "reference": case.reference,
            "keywords": metrics.keyword_accuracy(answer, case.answer_keywords) if case.answerable else None,
            "refused": detect_refusal(answer),
            "voice_ok": voice_format_ok(answer),
            "latency_ms": result["latency"]["total_ms"],
            "cost_inr": result.get("cost_inr", 0.0),
            "tokens": result.get("tokens", {}),
            "sources": result.get("sources", []),
        }
        judgement = await judge_answer(judge_client, case, answer, result.get("context", "")) if judge_client else None
        if judgement:
            row["judge"] = judgement
            row["refused"] = judgement["refused"] or row["refused"]

        if case.answerable:
            correct = judgement["correctness"] >= 0.5 if judgement else (row["keywords"] or 0) >= 0.5
            faithful = judgement["faithful"] if judgement else True
            row["passed"] = correct and faithful and not row["refused"]
        else:
            faithful = judgement["faithful"] if judgement else True
            row["passed"] = row["refused"] and faithful
        rows.append(row)

    return summarize(rows, judged=judge_client is not None)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize(rows: list[dict[str, Any]], judged: bool) -> dict[str, Any]:
    ok = [row for row in rows if "error" not in row]
    answerable = [row for row in ok if row["answerable"]]
    unanswerable = [row for row in ok if not row["answerable"]]
    judged_rows = [row for row in ok if "judge" in row]

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        by_type[row["type"]].append(row)
    latencies = [row["latency_ms"] for row in ok]

    return {
        "suite": "answers",
        "judged": judged and bool(judged_rows),
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "passed": sum(1 for row in rows if row["passed"]),
        "pass_rate": _mean([1.0 if row["passed"] else 0.0 for row in rows]),
        "answer_accuracy": _mean([1.0 if row["passed"] else 0.0 for row in answerable]),
        "correctness": _mean([row["judge"]["correctness"] for row in judged_rows if row["answerable"]]),
        "faithfulness": _mean([1.0 if row["judge"]["faithful"] else 0.0 for row in judged_rows]),
        "keyword_recall": _mean([row["keywords"] for row in answerable if row["keywords"] is not None]),
        "refusal_accuracy": _mean([1.0 if row["refused"] else 0.0 for row in unanswerable]),
        "false_refusal_rate": _mean([1.0 if row["refused"] else 0.0 for row in answerable]),
        "voice_format_rate": _mean([1.0 if row["voice_ok"] else 0.0 for row in ok]),
        "by_type": {
            kind: _mean([1.0 if row["passed"] else 0.0 for row in subset]) for kind, subset in sorted(by_type.items())
        },
        "latency_ms": {
            "p50": round(metrics.percentile(latencies, 0.5), 1) if latencies else None,
            "p95": round(metrics.percentile(latencies, 0.95), 1) if latencies else None,
        },
        "cost_inr": round(sum(row["cost_inr"] for row in ok), 5),
        "failed": [row["id"] for row in rows if not row["passed"]],
        "rows": rows,
    }
