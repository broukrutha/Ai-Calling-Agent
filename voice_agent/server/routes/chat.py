from __future__ import annotations

import itertools

from fastapi import APIRouter, Depends, Query

from voice_agent.server.deps import Services, get_services
from voice_agent.server.schemas import QueryRequest
from voice_agent.server.security import rate_limit, require_admin

router = APIRouter()


@router.post("/query", tags=["Chat"], dependencies=[Depends(rate_limit("expensive"))])
async def text_query(req: QueryRequest, services: Services = Depends(get_services)) -> dict:
    """One-off question answered from the knowledge base (no conversation memory)."""
    return await services.agent.query_text(req.query, language=req.language)


@router.get(
    "/benchmark", tags=["Evaluation"],
    dependencies=[Depends(require_admin), Depends(rate_limit("expensive"))],
)
async def benchmark(n: int = Query(5, ge=1, le=20), services: Services = Depends(get_services)) -> dict:
    """Run ``n`` evaluation queries and report latency, cost and answer accuracy."""
    from evaluation.test_queries import TEST_QUERIES
    from voice_agent.benchmark import run_benchmark

    cases = list(itertools.islice(itertools.cycle(TEST_QUERIES), n))
    return await run_benchmark(cases, services.agent, verbose=False)
