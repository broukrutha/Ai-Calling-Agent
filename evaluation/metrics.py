"""
metrics.py – Latency tracker, cost estimator, and accuracy evaluator.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics

from voice_agent.config import (
    COST_STT_PER_HOUR_INR,
    COST_TTS_PER_10K_CHARS_INR,
    LATENCY_TARGET_TOTAL_MS,
)


# ──────────────────────────────────────────────
# Latency Tracker
# ──────────────────────────────────────────────

class LatencyTracker:
    """Tracks per-step and aggregate latency across multiple queries."""

    def __init__(self):
        self.records: List[Dict] = []

    def record(self, stt_ms: float, retrieval_ms: float, llm_ms: float, tts_ms: float):
        total = stt_ms + retrieval_ms + llm_ms + tts_ms
        self.records.append({
            "stt_ms": stt_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total,
            "within_budget": total <= LATENCY_TARGET_TOTAL_MS,
        })

    def summary(self) -> dict:
        if not self.records:
            return {}
        totals = [r["total_ms"] for r in self.records]
        sorted_totals = sorted(totals)
        n = len(sorted_totals)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(totals), 2),
            "median_ms": round(statistics.median(totals), 2),
            "p95_ms": round(sorted_totals[int(n * 0.95)], 2),
            "min_ms": round(min(totals), 2),
            "max_ms": round(max(totals), 2),
            "within_budget_pct": round(
                sum(1 for r in self.records if r["within_budget"]) / n * 100, 1
            ),
            "component_means": {
                "stt_ms": round(statistics.mean(r["stt_ms"] for r in self.records), 2),
                "retrieval_ms": round(statistics.mean(r["retrieval_ms"] for r in self.records), 2),
                "llm_ms": round(statistics.mean(r["llm_ms"] for r in self.records), 2),
                "tts_ms": round(statistics.mean(r["tts_ms"] for r in self.records), 2),
            },
        }


# ──────────────────────────────────────────────
# Cost Estimator
# ──────────────────────────────────────────────

class CostEstimator:
    """Accumulates and calculates API costs in INR."""

    def __init__(self):
        self.total_audio_seconds: float = 0.0
        self.total_tts_chars: int = 0
        self.total_llm_tokens: int = 0
        self.query_count: int = 0

    def add_query(
        self,
        audio_duration_s: float = 0.0,
        tts_chars: int = 0,
        llm_tokens: int = 0,
    ):
        self.total_audio_seconds += audio_duration_s
        self.total_tts_chars += tts_chars
        self.total_llm_tokens += llm_tokens
        self.query_count += 1

    @property
    def stt_cost_inr(self) -> float:
        return round((self.total_audio_seconds / 3600) * COST_STT_PER_HOUR_INR, 6)

    @property
    def tts_cost_inr(self) -> float:
        return round((self.total_tts_chars / 10000) * COST_TTS_PER_10K_CHARS_INR, 6)

    @property
    def llm_cost_inr(self) -> float:
        return 0.0  # Sarvam-M is currently free

    @property
    def total_cost_inr(self) -> float:
        return round(self.stt_cost_inr + self.tts_cost_inr, 6)

    def summary(self) -> dict:
        return {
            "total_queries": self.query_count,
            "stt_cost_inr": self.stt_cost_inr,
            "tts_cost_inr": self.tts_cost_inr,
            "llm_cost_inr": self.llm_cost_inr,
            "total_cost_inr": self.total_cost_inr,
            "per_query_avg_inr": round(
                self.total_cost_inr / max(self.query_count, 1), 6
            ),
            "projected_100_queries_inr": round(
                (self.total_cost_inr / max(self.query_count, 1)) * 100, 3
            ),
            "projected_1000_queries_inr": round(
                (self.total_cost_inr / max(self.query_count, 1)) * 1000, 2
            ),
        }


# ──────────────────────────────────────────────
# Accuracy Evaluator
# ──────────────────────────────────────────────

class AccuracyEvaluator:
    """
    Simple keyword-based accuracy evaluator.
    For production, replace with BLEU/ROUGE or embedding similarity.
    """

    def __init__(self):
        self.scores: List[float] = []

    def evaluate(self, answer: str, expected_keywords: List[str]) -> float:
        """
        Score an answer against expected keywords.
        Returns a score between 0.0 and 1.0.
        """
        if not expected_keywords:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        score = hits / len(expected_keywords)
        self.scores.append(score)
        return round(score, 3)

    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return round(statistics.mean(self.scores), 3)

    def summary(self) -> dict:
        if not self.scores:
            return {"count": 0, "mean_score": 0.0}
        return {
            "count": len(self.scores),
            "mean_score": self.average_score(),
            "min_score": round(min(self.scores), 3),
            "max_score": round(max(self.scores), 3),
        }
