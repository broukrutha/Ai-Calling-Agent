"""Small, dependency-free metric helpers used by the benchmark."""

from __future__ import annotations

import math


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: ``percentile(xs, 0.95)`` is the P95 of ``xs``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(math.ceil(fraction * len(ordered)), 1)
    return ordered[min(rank, len(ordered)) - 1]


def keyword_accuracy(answer: str, expected_keywords: list[str]) -> float:
    """
    Share of expected keywords that appear in the answer (0.0–1.0).

    A coarse but transparent proxy for answer quality: it catches answers that
    miss the point or refuse, without needing a second model to grade them.
    """
    if not expected_keywords:
        return 1.0
    text = answer.lower()
    return round(sum(1 for keyword in expected_keywords if keyword.lower() in text) / len(expected_keywords), 3)
