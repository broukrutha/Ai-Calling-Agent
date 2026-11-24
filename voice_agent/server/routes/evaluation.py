"""
Evaluation runs from the console.

Runs execute in the background, one at a time, and are stored, so quality can
be compared across prompt, model and retrieval changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from voice_agent import config
from voice_agent.booking import store
from voice_agent.server.deps import Services, get_services
from voice_agent.server.schemas import EvalRunRequest, GenerateRequest
from voice_agent.server.security import rate_limit, require_admin

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation"], dependencies=[Depends(require_admin)])


def generated_dataset_path() -> Path:
    return Path(config.EVAL_DATA_DIR) / "generated.jsonl"


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    """The few numbers the run list shows."""
    keys = {
        "retrieval": ["answerable.context_hit", "answerable.hit_at_1", "answerable.mrr", "answerable.context_precision"],
        "dialog": ["success_rate"],
        "answers": ["pass_rate", "correctness", "faithfulness", "refusal_accuracy", "voice_format_rate"],
    }.get(summary.get("suite", ""), [])
    out: dict[str, Any] = {}
    for key in keys:
        value: Any = summary
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        out[key.split(".")[-1]] = value
    return out


def _save(run_id: str, **fields: Any) -> None:
    with store.connection() as conn:
        sets = ", ".join(f"{name} = ?" for name in fields)
        conn.execute(f"UPDATE eval_runs SET {sets} WHERE id = ?", [*fields.values(), run_id])  # noqa: S608


def _row(row: Any, full: bool) -> dict[str, Any]:
    data = dict(row)
    results = json.loads(data.pop("summary") or "[]")
    if not isinstance(results, list):      # a run that has not finished a suite yet
        results = []
    data["suites"] = [
        {"suite": item.get("suite"), "kb": item.get("kb"), "gates_passed": item.get("gates_passed"),
         "headline": _headline(item), "failed": item.get("failed", [])[:20]}
        for item in results
    ]
    if full:
        data["results"] = results
    return data


async def _execute(run_id: str, suites: list[str], kb: str, services: Services, embedder: Any) -> None:
    from evaluation.dataset import load_cases
    from evaluation.runner import run_suite

    results: list[dict[str, Any]] = []
    try:
        cases = None
        if kb == "live":
            path = generated_dataset_path()
            if not path.exists():
                raise ValueError("Generate questions from your documents first.")
            cases = await asyncio.to_thread(load_cases, path)
        client = services.agent.client if services.agent_ready else None
        for suite in suites:
            if suite == "answers" and client is None:
                continue
            results.append(await run_suite(
                suite, kb=kb, cases=cases, vector_store=services.vector_store, client=client,
                embedding_function=embedder,
            ))
            await asyncio.to_thread(_save, run_id, summary=json.dumps(results, ensure_ascii=False, default=str))
        await asyncio.to_thread(
            _save, run_id, status="completed", finished_at=_now(),
            summary=json.dumps(results, ensure_ascii=False, default=str),
        )
    except Exception as exc:
        logger.exception("Evaluation run %s failed", run_id)
        await asyncio.to_thread(_save, run_id, status="failed", finished_at=_now(), error=str(exc)[:300])


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


@router.post("/runs", status_code=202, dependencies=[Depends(rate_limit("expensive"))])
async def start_run(req: EvalRunRequest, request: Request, services: Services = Depends(get_services)) -> dict:
    running = getattr(request.app.state, "eval_task", None)
    if running is not None and not running.done():
        raise HTTPException(status_code=409, detail="An evaluation is already running.")
    if req.suite == "answers" and not services.agent_ready:
        raise HTTPException(status_code=503, detail="The answers suite needs SARVAM_API_KEY and an LLM key.")

    suites = ["retrieval", "dialog", "answers"] if req.suite == "all" else [req.suite]
    run_id = secrets.token_hex(8)
    dataset = "generated" if req.kb == "live" else "golden"
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs (id, suite, dataset) VALUES (?, ?, ?)", (run_id, req.suite, f"{req.kb}/{dataset}")
        )
    embedder = getattr(services, "eval_embedding_function", None)
    request.app.state.eval_task = asyncio.create_task(_execute(run_id, suites, req.kb, services, embedder))
    return {"status": "ok", "id": run_id}


@router.get("/runs")
async def list_runs(limit: int = 20) -> dict:
    def load() -> list[dict[str, Any]]:
        with store.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 100),)
            ).fetchall()
        return [_row(row, full=False) for row in rows]

    return {"status": "ok", "runs": await asyncio.to_thread(load)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    def load() -> dict[str, Any] | None:
        with store.connection() as conn:
            row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return _row(row, full=True) if row else None

    run = await asyncio.to_thread(load)
    if run is None:
        raise HTTPException(status_code=404, detail="No such evaluation run.")
    return run


@router.get("/datasets")
async def datasets() -> dict:
    from evaluation.dataset import DEFAULT_DATASET, load_cases

    def describe(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"available": False, "cases": 0}
        cases = load_cases(path)
        return {"available": True, "cases": len(cases), "answerable": sum(1 for case in cases if case.answerable)}

    return {
        "golden": await asyncio.to_thread(describe, DEFAULT_DATASET),
        "generated": await asyncio.to_thread(describe, generated_dataset_path()),
    }


@router.post("/generate", dependencies=[Depends(rate_limit("expensive"))])
async def generate(req: GenerateRequest, services: Services = Depends(get_services)) -> dict:
    """Write evaluation questions from the live knowledge base with the LLM."""
    from evaluation.dataset import save_cases
    from evaluation.generate import generate_cases

    if not services.agent_ready:
        raise HTTPException(status_code=503, detail="Generating questions needs SARVAM_API_KEY and an LLM key.")
    if services.vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="Add documents to the knowledge base first.")
    cases = await generate_cases(services.vector_store, services.agent.client, limit=req.limit)
    path = generated_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(save_cases, cases, path)
    return {"status": "ok", "cases": len(cases), "answerable": sum(1 for case in cases if case.answerable)}
