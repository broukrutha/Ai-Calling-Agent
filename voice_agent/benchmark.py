"""
Benchmark the text pipeline (retrieval + LLM): latency, cost and answer accuracy.

    python -m voice_agent.benchmark --queries 20 --report results.json --html report.html

Speech-to-text and text-to-speech are not exercised here; use the web client's
latency panel for end-to-end voice timings.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import statistics
import time
from typing import Any

from evaluation.metrics import keyword_accuracy, percentile
from voice_agent import config
from voice_agent.pipeline import VoiceAgent

logger = logging.getLogger(__name__)


async def run_benchmark(cases: list[dict[str, Any]], agent: VoiceAgent, verbose: bool = True) -> dict[str, Any]:
    """
    Run each case (``{"query": ..., "expected_keywords": [...]}``) through the
    agent and summarise the results.
    """
    results: list[dict[str, Any]] = []
    errors = 0

    for number, case in enumerate(cases, start=1):
        query = case["query"]
        started = time.perf_counter()
        try:
            answer = await agent.query_text(query)
        except Exception as exc:
            errors += 1
            logger.error("Benchmark query %d failed: %s", number, exc)
            if verbose:
                print(f"  FAIL Q{number:02d} | {query[:50]}")
            continue

        elapsed = (time.perf_counter() - started) * 1000
        row = {
            "query": query,
            "answer_length": len(answer["answer"]),
            "retrieval_ms": answer["latency"]["retrieval_ms"],
            "llm_ms": answer["latency"]["llm_ms"],
            "total_ms": round(elapsed, 2),
            "context_chunks": answer["context_chunks"],
            "cost_inr": answer["cost_inr"],
            "within_budget": elapsed <= config.LATENCY_TARGET_TOTAL_MS,
            "tokens_in": answer["tokens"]["input"],
            "tokens_out": answer["tokens"]["output"],
            "cached": answer.get("cached", False),
            "accuracy": keyword_accuracy(answer["answer"], case.get("expected_keywords", [])),
        }
        results.append(row)
        if verbose:
            flag = "ok  " if row["within_budget"] else "SLOW"
            print(f"  {flag} Q{number:02d} | {elapsed:6.0f} ms | acc {row['accuracy']:.0%} | {query[:50]}")

    summary: dict[str, Any] = {
        "total_queries": len(cases),
        "successful": len(results),
        "errors": errors,
        "budget_ms": config.LATENCY_TARGET_TOTAL_MS,
        "api_stack": {
            "llm": f"{agent.client.llm_provider}/{agent.client.llm_model}",
            "stt": config.STT_MODEL,
            "tts": config.TTS_MODEL,
            "embeddings": config.EMBEDDING_MODEL,
        },
        "individual_results": results,
    }
    if not results:
        summary.update(latency={}, cost={}, accuracy={"mean_score": 0.0}, within_budget_pct=0.0)
        return summary

    totals = [row["total_ms"] for row in results]
    retrieval_mean = statistics.mean(row["retrieval_ms"] for row in results)
    llm_mean = statistics.mean(row["llm_ms"] for row in results)
    total_cost = sum(row["cost_inr"] for row in results)
    scores = [row["accuracy"] for row in results]

    summary.update(
        latency={
            "p50_ms": round(statistics.median(totals), 2),
            "p95_ms": round(percentile(totals, 0.95), 2),
            "p99_ms": round(percentile(totals, 0.99), 2),
            "min_ms": round(min(totals), 2),
            "max_ms": round(max(totals), 2),
            "mean_ms": round(statistics.mean(totals), 2),
            "retrieval_mean_ms": round(retrieval_mean, 2),
            "llm_mean_ms": round(llm_mean, 2),
        },
        within_budget_pct=round(sum(row["within_budget"] for row in results) / len(results) * 100, 1),
        cost={
            "total_inr": round(total_cost, 5),
            "per_query_avg_inr": round(total_cost / len(results), 6),
            "per_100_queries_inr": round(total_cost / len(results) * 100, 4),
        },
        accuracy={
            "mean_score": round(statistics.mean(scores), 3),
            "min_score": min(scores),
            "max_score": max(scores),
        },
    )
    if verbose:
        latency = summary["latency"]
        print(
            f"\n  {len(results)}/{len(cases)} ok | P50 {latency['p50_ms']:.0f} ms | "
            f"P95 {latency['p95_ms']:.0f} ms | within budget {summary['within_budget_pct']}% | "
            f"accuracy {summary['accuracy']['mean_score']:.0%} | "
            f"INR {summary['cost']['per_query_avg_inr']:.5f}/query\n"
        )
    return summary


def main() -> None:
    from evaluation.report_generator import generate_html_report
    from evaluation.test_queries import TEST_QUERIES
    from voice_agent.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Benchmark the AI Calling Agent text pipeline.")
    parser.add_argument("--queries", type=int, default=10, help="number of queries to run")
    parser.add_argument("--custom-query", default="", help="run one query of your own instead")
    parser.add_argument("--report", default="", help="write the JSON summary to this file")
    parser.add_argument("--html", default="", help="write an HTML report to this file")
    args = parser.parse_args()

    configure_logging()
    logging.getLogger().setLevel(logging.WARNING)

    if args.custom_query:
        cases = [{"query": args.custom_query, "expected_keywords": []}]
    else:
        cases = list(itertools.islice(itertools.cycle(TEST_QUERIES), max(args.queries, 1)))

    async def run() -> dict[str, Any]:
        agent = VoiceAgent()
        try:
            return await run_benchmark(cases, agent)
        finally:
            await agent.close()

    summary = asyncio.run(run())
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"  JSON summary: {args.report}")
    if args.html:
        print(f"  HTML report:  {generate_html_report(summary, args.html)}")


if __name__ == "__main__":
    main()
