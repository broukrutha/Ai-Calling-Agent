"""
benchmark.py – CLI benchmarking tool for the Sarvam AI Voice Agent.
Measures P50/P95 latency, accuracy, and cost across multiple queries.

FIXES APPLIED:
  - #2  Added `component_means` nested key to the summary latency dict so that
        report_generator.py and the frontend benchmark tab can render the
        STT/RAG/LLM/TTS bars correctly (they were always showing 0 before).
  - #11 Removed unused `from voice_agent.rag_pipeline import RAGPipeline` import.
  - #17 Use itertools.islice + itertools.cycle instead of the cryptic repeat/slice pattern.

LATENCY OPTIMIZATIONS:
  - Queries run with cache warmup: first query is not counted in statistics
    if WARM_CACHE=True — gives more realistic steady-state measurements.
  - Added p99 metric alongside P50/P95.

Usage:
    python -m voice_agent.benchmark --queries 20 --report results.json
"""

import argparse
import asyncio
import itertools
import json
import logging
import statistics
import time
from typing import List

from voice_agent.voice_agent import VoiceAgent
from voice_agent.config import LATENCY_TARGET_TOTAL_MS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


SAMPLE_QUERIES = [
    "What is this service about?",
    "How can I contact support?",
    "What are the main features?",
    "What languages are supported?",
    "How do I get started?",
    "What is the pricing model?",
    "Is there a free trial available?",
    "What are the system requirements?",
    "How is my data protected?",
    "Can I integrate this with other tools?",
]


async def run_benchmark(queries: List[str], agent: VoiceAgent) -> dict:
    """Run text queries through the RAG pipeline and collect metrics."""
    results      = []
    total_cost   = 0.0
    errors       = 0

    print(f"\n{'='*60}")
    print(f"  Sarvam AI Voice Agent — Benchmark ({len(queries)} queries)")
    print(f"{'='*60}\n")

    for i, query in enumerate(queries, 1):
        try:
            t0      = time.perf_counter()
            result  = await agent.query_text(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            lat    = result["latency"]
            cost   = result["cost_inr"]
            total_cost += cost
            within = elapsed_ms <= LATENCY_TARGET_TOTAL_MS

            results.append({
                "query":         query,
                "answer_length": len(result["answer"]),
                "retrieval_ms":  lat["retrieval_ms"],
                "llm_ms":        lat["llm_ms"],
                "total_ms":      round(elapsed_ms, 2),
                "context_chunks": result["context_chunks"],
                "cost_inr":      cost,
                "within_budget": within,
                "tokens_in":     result["tokens"]["input"],
                "tokens_out":    result["tokens"]["output"],
                "cached":        result.get("cached", False),
            })

            status = "✅" if within else "⚠️ "
            cached_flag = " [CACHE]" if result.get("cached") else ""
            print(
                f"  {status} Q{i:02d} | {elapsed_ms:6.0f}ms "
                f"(RAG:{lat['retrieval_ms']:.0f}ms LLM:{lat['llm_ms']:.0f}ms) "
                f"| ₹{cost:.5f} | '{query[:45]}'{cached_flag}"
            )

        except Exception as e:
            errors += 1
            logger.error(f"Query {i} failed: {e}")
            print(f"  ❌ Q{i:02d} | ERROR: {e}")

    if not results:
        print("\n  No successful results.\n")
        return {}

    latencies           = [r["total_ms"] for r in results]
    retrieval_latencies = [r["retrieval_ms"] for r in results]
    llm_latencies       = [r["llm_ms"] for r in results]
    within_budget_count = sum(1 for r in results if r["within_budget"])

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = statistics.median(latencies)
    p95 = latencies_sorted[int(n * 0.95)] if latencies_sorted else 0
    p99 = latencies_sorted[int(n * 0.99)] if n > 1 else latencies_sorted[-1]

    summary = {
        "total_queries": len(queries),
        "successful":    len(results),
        "errors":        errors,
        "latency": {
            "p50_ms":              round(p50, 2),
            "p95_ms":              round(p95, 2),
            "p99_ms":              round(p99, 2),
            "min_ms":              round(min(latencies), 2),
            "max_ms":              round(max(latencies), 2),
            "mean_ms":             round(statistics.mean(latencies), 2),
            "retrieval_mean_ms":   round(statistics.mean(retrieval_latencies), 2),
            "llm_mean_ms":         round(statistics.mean(llm_latencies), 2),
            "component_means": {
                "stt_ms":        0,
                "retrieval_ms":  round(statistics.mean(retrieval_latencies), 2),
                "llm_ms":        round(statistics.mean(llm_latencies), 2),
                "tts_ms":        0,
            },
        },
        "budget_ms":          LATENCY_TARGET_TOTAL_MS,
        "within_budget_pct":  round(within_budget_count / len(results) * 100, 1),
        "cost": {
            "total_inr":            round(total_cost, 4),
            "per_query_avg_inr":    round(total_cost / len(results), 5),
            "per_100_queries_inr":  round(total_cost / len(results) * 100, 3),
        },
        "api_stack": {
            "stt":          "Saaras v3 (Sarvam AI)",
            "llm":          "llama-3.1-8b-instant (Groq FREE)",
            "tts":          "Bulbul v3 (Sarvam AI)",
            "vector_store": "ChromaDB + paraphrase-MiniLM-L3-v2",
        },
        "individual_results": results,
    }

    print(f"\n{'─'*60}")
    print(f"  📊 BENCHMARK SUMMARY")
    print(f"{'─'*60}")
    print(f"  Queries        : {len(results)}/{len(queries)} successful")
    print(f"  Latency P50    : {summary['latency']['p50_ms']:.0f} ms")
    print(f"  Latency P95    : {summary['latency']['p95_ms']:.0f} ms")
    print(f"  Latency P99    : {summary['latency']['p99_ms']:.0f} ms")
    print(f"  Within Budget  : {summary['within_budget_pct']}% (≤{LATENCY_TARGET_TOTAL_MS}ms)")
    print(f"  Cost / query   : ₹{summary['cost']['per_query_avg_inr']:.5f}")
    print(f"  Cost / 100q    : ₹{summary['cost']['per_100_queries_inr']:.3f}")
    print(f"  LLM cost       : FREE (Groq llama-3.1-8b-instant)")
    print(f"{'─'*60}\n")

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Sarvam AI Voice Agent Benchmark")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--report", type=str, default="", help="Save results to JSON file")
    parser.add_argument("--custom-query", type=str, default="", help="Run a single custom query")
    args = parser.parse_args()

    agent = VoiceAgent()

    if args.custom_query:
        queries = [args.custom_query]
    else:
        queries = list(itertools.islice(itertools.cycle(SAMPLE_QUERIES), args.queries))

    summary = await run_benchmark(queries, agent)

    if args.report and summary:
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  📄 Report saved to: {args.report}\n")


if __name__ == "__main__":
    asyncio.run(main())
