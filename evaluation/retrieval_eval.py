"""
Retrieval evaluation: does the context handed to the LLM contain the answer?

Runs every case through a retriever and scores the passages it returns:

* hit@1, hit@3 and context hit (answer present anywhere in the context)
* MRR and nDCG over the returned passages
* evidence recall (for answers that need several facts)
* context precision (how much irrelevant text reaches the LLM)
* abstention (informational): for questions the documents cannot answer, how
  often the retriever returned nothing. Refusing correctly is scored on the
  final answer by the answer suite.

No LLM is involved, so this suite is fast, free and deterministic: it is the
one that runs in CI.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from evaluation import metrics
from evaluation.dataset import EvalCase


class PassageLike(Protocol):
    text: str
    source: str
    score: float


Retriever = Callable[[str, list[dict[str, str]]], Awaitable[list[PassageLike]]]


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


async def evaluate_retrieval(cases: list[EvalCase], retrieve: Retriever) -> dict[str, Any]:
    """Score ``retrieve(question, history)`` on every case."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        passages = await retrieve(case.question, list(case.history))
        elapsed_ms = (time.perf_counter() - started) * 1000
        texts = [passage.text for passage in passages]
        row: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "type": case.type,
            "answerable": case.answerable,
            "latency_ms": round(elapsed_ms, 2),
            "returned": len(passages),
            "sources": [passage.source for passage in passages],
            "scores": [round(float(passage.score), 4) for passage in passages],
        }
        if case.answerable:
            relevance = metrics.relevance_vector(texts, case.evidence)
            row.update(
                relevance=relevance,
                hit_at_1=metrics.hit_at_k(relevance, 1),
                hit_at_3=metrics.hit_at_k(relevance, 3),
                context_hit=metrics.hit_at_k(relevance, len(relevance)),
                rr=metrics.reciprocal_rank(relevance),
                ndcg=metrics.ndcg_at_k(relevance, len(relevance)),
                evidence_recall=metrics.evidence_recall(texts, case.evidence),
                context_precision=metrics.context_precision(relevance),
            )
            row["passed"] = row["evidence_recall"] == 1.0
        else:
            # Informational: similarity alone cannot reliably tell "no answer here"
            # from a hard question, so refusing is judged on the answer, not here.
            row["abstained"] = len(passages) == 0
            row["passed"] = None
        rows.append(row)
    return summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def block(subset: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "cases": len(subset),
            "hit_at_1": _mean([row["hit_at_1"] for row in subset]),
            "hit_at_3": _mean([row["hit_at_3"] for row in subset]),
            "context_hit": _mean([row["context_hit"] for row in subset]),
            "mrr": _mean([row["rr"] for row in subset]),
            "ndcg": _mean([row["ndcg"] for row in subset]),
            "evidence_recall": _mean([row["evidence_recall"] for row in subset]),
            "context_precision": _mean([row["context_precision"] for row in subset]),
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answerable:
        by_type[row["type"]].append(row)

    latencies = [row["latency_ms"] for row in rows]
    return {
        "suite": "retrieval",
        "cases": len(rows),
        "passed": sum(1 for row in answerable if row["passed"]),
        "failed": [row["id"] for row in answerable if not row["passed"]],
        "answerable": block(answerable),
        "by_type": {kind: block(subset) for kind, subset in sorted(by_type.items())},
        "abstention": {
            "cases": len(unanswerable),
            "rate": _mean([1.0 if row["abstained"] else 0.0 for row in unanswerable]),
        },
        "avg_passages": _mean([float(row["returned"]) for row in rows]),
        "latency_ms": {
            "p50": round(metrics.percentile(latencies, 0.5), 2),
            "p95": round(metrics.percentile(latencies, 0.95), 2),
        },
        "rows": rows,
    }
