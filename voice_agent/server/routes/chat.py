from __future__ import annotations

from fastapi import APIRouter, Depends

from voice_agent.server.deps import Services, get_services
from voice_agent.server.schemas import QueryRequest
from voice_agent.server.security import rate_limit

router = APIRouter()


@router.post("/query", tags=["Chat"], dependencies=[Depends(rate_limit("expensive"))])
async def text_query(req: QueryRequest, services: Services = Depends(get_services)) -> dict:
    """One-off question answered from the knowledge base (no conversation memory)."""
    return await services.agent.query_text(req.query, language=req.language)
