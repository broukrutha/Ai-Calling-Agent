"""
Evaluation command line.

    python -m evaluation run --suite retrieval            # fast, free, no LLM: what CI gates on
    python -m evaluation run --suite all --html report.html
    python -m evaluation run --suite answers --kb live --dataset data/eval/generated.jsonl
    python -m evaluation generate --limit 30              # questions from your own documents
    python -m evaluation export-feedback                  # poorly rated answers → draft cases

``run`` exits with status 1 when a regression threshold is missed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from evaluation.dataset import load_cases, save_cases
from evaluation.runner import SUITES, run_suite

GENERATED_DEFAULT = Path("data/eval/generated.jsonl")


def _headline(summary: dict) -> str:
    suite = summary["suite"]
    if suite == "retrieval":
        a = summary["answerable"]
        return (f"answer in context {a['context_hit']:.1%} · ranked first {a['hit_at_1']:.1%} · "
                f"MRR {a['mrr']:.3f} · precision {a['context_precision']:.1%} · "
                f"{summary['passed']}/{summary['answerable']['cases']} passed")
    if suite == "dialog":
        return f"task success {summary['success_rate']:.1%} ({summary['passed']}/{summary['cases']})"
    parts = [f"pass rate {summary['pass_rate']:.1%}"]
    for key, label in (("correctness", "correctness"), ("faithfulness", "faithfulness"),
                       ("refusal_accuracy", "refusals"), ("voice_format_rate", "speakable")):
        if summary.get(key) is not None:
            parts.append(f"{label} {summary[key]:.1%}")
    return " · ".join(parts)


async def _client():
    from voice_agent import config
    from voice_agent.sarvam_client import SarvamClient

    if not config.SARVAM_API_KEY:
        return None
    return SarvamClient()


async def _run(args: argparse.Namespace) -> int:
    from evaluation.report_generator import generate_html_report

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    cases = load_cases(args.dataset) if args.dataset else None
    client = await _client() if "answers" in suites or args.kb == "live" else None
    store = None
    if args.kb == "live":
        from voice_agent.vector_store import VectorStore

        store = VectorStore()

    results, failed = [], False
    try:
        for suite in suites:
            if suite == "answers" and client is None:
                print("  skip  answers: set SARVAM_API_KEY and an LLM key to evaluate answers")
                continue
            summary = await run_suite(suite, kb=args.kb, cases=cases, vector_store=store, client=client,
                                      judge=not args.no_judge)
            results.append(summary)
            status = "ok  " if summary["gates_passed"] else "FAIL"
            failed |= not summary["gates_passed"]
            print(f"  {status}  {suite:<9} {_headline(summary)}  ({summary['duration_sec']}s)")
            for gate in summary["gates"]:
                if not gate["passed"]:
                    print(f"        below threshold: {gate['metric']} = {gate['value']} (minimum {gate['minimum']})")
            if summary.get("failed"):
                print(f"        failing cases: {', '.join(summary['failed'][:12])}")
    finally:
        if client is not None:
            await client.close()

    if args.json:
        payload = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        await asyncio.to_thread(Path(args.json).write_text, payload, encoding="utf-8")
        print(f"  JSON report: {args.json}")
    if args.html:
        print(f"  HTML report: {await asyncio.to_thread(generate_html_report, results, args.html)}")
    return 1 if failed else 0


async def _generate(args: argparse.Namespace) -> int:
    from evaluation.generate import generate_cases
    from voice_agent.vector_store import VectorStore

    client = await _client()
    if client is None:
        print("Set SARVAM_API_KEY and an LLM key: questions are written by the LLM.")
        return 1
    try:
        cases = await generate_cases(VectorStore(), client, limit=args.limit)
    finally:
        await client.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_cases(cases, args.out)
    print(f"Wrote {len(cases)} cases to {args.out}. Review them, then run:")
    print(f"  python -m evaluation run --suite all --kb live --dataset {args.out}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from voice_agent import tracing

    rows = tracing.export_cases(rating=-1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} poorly rated answers to {args.out}. Add evidence and a reference answer to each.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description="Evaluate the AI Calling Agent.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run evaluation suites")
    run.add_argument("--suite", choices=[*SUITES, "all"], default="retrieval")
    run.add_argument("--kb", choices=["sample", "live"], default="sample")
    run.add_argument("--dataset", help="JSONL dataset (default: the golden set for the sample KB)")
    run.add_argument("--no-judge", action="store_true", help="skip the LLM judge in the answers suite")
    run.add_argument("--json", help="write the full results to this file")
    run.add_argument("--html", help="write an HTML report to this file")

    gen = commands.add_parser("generate", help="write questions from your own knowledge base")
    gen.add_argument("--limit", type=int, default=30)
    gen.add_argument("--out", default=str(GENERATED_DEFAULT))

    export = commands.add_parser("export-feedback", help="export poorly rated answers as draft cases")
    export.add_argument("--out", default="data/eval/feedback.jsonl")

    args = parser.parse_args()
    from voice_agent.logging_config import configure_logging

    configure_logging()
    logging.getLogger().setLevel(logging.WARNING)
    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "generate":
        return asyncio.run(_generate(args))
    return _export(args)


if __name__ == "__main__":
    sys.exit(main())
