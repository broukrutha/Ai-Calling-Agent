"""
Metric functions. Pure, dependency-free and individually tested.

Retrieval metrics take the ranked passages the pipeline actually hands to the
LLM, and a relevance judgement per passage (does it contain the evidence?).
"""

from __future__ import annotations

import math
import re


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: ``percentile(xs, 0.95)`` is the P95 of ``xs``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(math.ceil(fraction * len(ordered)), 1)
    return ordered[min(rank, len(ordered)) - 1]


def normalize(text: str) -> str:
    """Lower-case, unify quotes and collapse whitespace, for containment checks."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def contains_evidence(passage: str, evidence: tuple[str, ...] | list[str]) -> bool:
    text = normalize(passage)
    return any(normalize(item) in text for item in evidence)


def relevance_vector(passages: list[str], evidence: tuple[str, ...] | list[str]) -> list[int]:
    return [1 if contains_evidence(passage, evidence) else 0 for passage in passages]


def hit_at_k(relevance: list[int], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[int]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """Binary-relevance nDCG@k."""
    gains = relevance[:k]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    # The ideal ranking puts every relevant passage the pipeline found at the top.
    return dcg / idcg


def evidence_recall(passages: list[str], evidence: tuple[str, ...] | list[str]) -> float:
    """Share of the evidence strings present anywhere in the retrieved passages."""
    if not evidence:
        return 1.0
    text = normalize(" ".join(passages))
    return sum(1 for item in evidence if normalize(item) in text) / len(evidence)


def context_precision(relevance: list[int]) -> float:
    """Share of retrieved passages that are relevant (how much noise reaches the LLM)."""
    return sum(relevance) / len(relevance) if relevance else 0.0


_NUMBER_SEPARATOR = re.compile(r"(?<=\d)[,\s](?=\d)")


def keyword_accuracy(answer: str, expected_keywords: list[str] | tuple[str, ...]) -> float:
    """
    Share of expected keywords found in the answer (0.0–1.0). Digit groups are
    compared without separators, so "1,200", "1 200" and "1200" all match "1200".
    """
    if not expected_keywords:
        return 1.0
    text = _NUMBER_SEPARATOR.sub("", normalize(answer))
    found = sum(1 for keyword in expected_keywords if _NUMBER_SEPARATOR.sub("", normalize(keyword)) in text)
    return round(found / len(expected_keywords), 3)
