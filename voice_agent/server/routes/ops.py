"""Monitoring: per-turn traces, summaries, time series, feedback and export."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from voice_agent import tracing
from voice_agent.server.deps import Services, get_services
from voice_agent.server.schemas import FeedbackRequest
from voice_agent.server.security import rate_limit, require_admin

router = APIRouter(tags=["Monitoring"])
_WINDOW = Query("24h", pattern="^(1h|24h|7d|30d)$")


@router.get("/ops/summary", dependencies=[Depends(require_admin)])
async def summary(window: str = _WINDOW, services: Services = Depends(get_services)) -> dict:
    """Headline numbers: volume, latency percentiles per stage, errors, cost, feedback."""
    data = await asyncio.to_thread(tracing.summary, window)
    if services.agent_ready:
        try:
            data["providers_health"] = (await services.agent.health_check()).get("providers", [])
        except Exception:
            data["providers_health"] = []
    return data


@router.get("/ops/timeseries", dependencies=[Depends(require_admin)])
async def timeseries(window: str = _WINDOW) -> dict:
    return await asyncio.to_thread(tracing.timeseries, window)


@router.get("/ops/traces", dependencies=[Depends(require_admin)])
async def traces(
    limit: int = Query(50, ge=1, le=500),
    kind: str | None = Query(None, pattern="^(answer|booking|noise|error)$"),
    rating: int | None = Query(None, ge=-1, le=1),
) -> dict:
    rows = await asyncio.to_thread(tracing.list_traces, limit, kind, rating)
    return {"status": "ok", "traces": rows, "count": len(rows)}


@router.get("/ops/traces/{trace_id}", dependencies=[Depends(require_admin)])
async def trace(trace_id: str) -> dict:
    row = await asyncio.to_thread(tracing.get_trace, trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such trace.")
    return row


@router.get("/ops/export", dependencies=[Depends(require_admin)], response_class=PlainTextResponse)
async def export(rating: int = Query(-1, ge=-1, le=1)) -> PlainTextResponse:
    """Rated answers as JSONL draft evaluation cases."""
    rows = await asyncio.to_thread(tracing.export_cases, rating)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return PlainTextResponse(
        body, media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="feedback_cases.jsonl"'},
    )


@router.post("/feedback", dependencies=[Depends(rate_limit())])
async def feedback(req: FeedbackRequest) -> dict:
    """Rate an answer. The trace id comes back with every answer."""
    found = await asyncio.to_thread(tracing.set_feedback, req.trace_id, req.rating, req.comment)
    if not found:
        raise HTTPException(status_code=404, detail="That answer could not be found.")
    return {"status": "ok"}
