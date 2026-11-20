"""
Run evaluation suites and check them against regression thresholds.

Knowledge bases:
  * ``sample`` – a temporary index of ``sample_doc/`` built for the run. The
    golden dataset is written for it, so results are comparable over time and
    across machines. This is what CI gates on.
  * ``live``   – your real knowledge base, scored with a dataset generated
    from your own documents (see :mod:`evaluation.generate`).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from evaluation.dataset import DEFAULT_DATASET, EvalCase, load_cases

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample_doc"
THRESHOLDS_FILE = Path(__file__).resolve().parent / "thresholds.json"
SUITES = ("retrieval", "dialog", "answers")


def build_sample_store(embedding_function: Any = None) -> Any:
    """Index ``sample_doc/`` into a throw-away vector store."""
    from voice_agent.document_processor import ingest
    from voice_agent.vector_store import VectorStore

    store = VectorStore(persist_dir=tempfile.mkdtemp(prefix="eval-kb-"), collection_name="eval_sample",
                        embedding_function=embedding_function)
    if not store.is_available():
        raise RuntimeError(f"The embedding model could not be loaded: {store.init_error}")
    for path in sorted(SAMPLE_DOCS.iterdir()):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            store.add_documents(ingest(str(path), "file"))
    return store


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict[str, dict[str, float]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {suite: values for suite, values in data.items() if not suite.startswith("_")}


def _lookup(summary: dict[str, Any], dotted: str) -> Any:
    value: Any = summary
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def check_thresholds(summary: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """One entry per gated metric: value, minimum and whether it passed."""
    checks = []
    for metric, minimum in thresholds.items():
        value = _lookup(summary, metric)
        checks.append({
            "metric": metric,
            "minimum": minimum,
            "value": value,
            "passed": value is not None and value >= minimum,
        })
    return checks


async def run_suite(
    suite: str,
    *,
    kb: str = "sample",
    cases: list[EvalCase] | None = None,
    vector_store: Any = None,
    client: Any = None,
    embedding_function: Any = None,
    judge: bool = True,
) -> dict[str, Any]:
    """
    Run one suite. ``vector_store`` is the live store (for ``kb="live"``),
    ``client`` the provider client (needed by the answers suite).
    """
    from evaluation.answer_eval import evaluate_answers
    from evaluation.dialog_eval import evaluate_dialogs, load_scenarios
    from evaluation.retrieval_eval import evaluate_retrieval
    from voice_agent.rag_pipeline import RAGPipeline

    started = time.perf_counter()
    if suite == "dialog":
        summary = await evaluate_dialogs(load_scenarios())
    else:
        if kb == "sample":
            store = build_sample_store(embedding_function)
            cases = cases if cases is not None else load_cases(DEFAULT_DATASET)
        else:
            if vector_store is None:
                raise ValueError("The live knowledge base is not available.")
            if not cases:
                raise ValueError("Generate a dataset from your documents before evaluating the live knowledge base.")
            store = vector_store
        rag = RAGPipeline(vector_store=store, sarvam_client=client or object())

        if suite == "retrieval":
            async def retrieve(question: str, history: list[dict[str, str]]) -> list[Any]:
                return (await rag.retrieve(question, history=history)).passages

            summary = await evaluate_retrieval(cases, retrieve)
        elif suite == "answers":
            if client is None:
                raise ValueError("The answers suite needs an LLM: set SARVAM_API_KEY and an LLM key.")
            from voice_agent.pipeline import VoiceAgent

            agent = VoiceAgent(rag_pipeline=rag, sarvam_client=client, tracer=None)
            summary = await evaluate_answers(cases, agent, judge_client=client if judge else None)
        else:
            raise ValueError(f"Unknown suite {suite!r}. Choose from {', '.join(SUITES)}.")

    summary["kb"] = "sample" if suite == "dialog" else kb
    summary["duration_sec"] = round(time.perf_counter() - started, 2)
    thresholds = load_thresholds().get(suite, {}) if summary["kb"] == "sample" else {}
    summary["gates"] = check_thresholds(summary, thresholds)
    summary["gates_passed"] = all(check["passed"] for check in summary["gates"])
    return summary
