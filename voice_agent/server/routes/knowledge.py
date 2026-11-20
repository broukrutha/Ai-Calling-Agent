from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from voice_agent import config
from voice_agent.document_processor import ingest
from voice_agent.server.deps import Services, get_services
from voice_agent.server.schemas import IngestTextRequest, IngestURLRequest
from voice_agent.server.security import rate_limit, require_admin

logger = logging.getLogger(__name__)

router = APIRouter()
_admin_write = [Depends(require_admin), Depends(rate_limit("expensive"))]


async def _ingest(services: Services, source: str | bytes, kind: str, name: str) -> dict:
    """Parse, chunk and embed off the event loop so live calls are not stalled."""
    def work() -> int:
        return services.vector_store.add_documents(ingest(source, kind, name=name))

    chunks = await asyncio.to_thread(work)
    return {"status": "ok", "chunks_added": chunks, "source": name}


@router.post("/ingest/text", tags=["Knowledge Base"], dependencies=_admin_write)
async def ingest_text(req: IngestTextRequest, services: Services = Depends(get_services)) -> dict:
    return await _ingest(services, req.text, "text", req.name.strip())


@router.post("/ingest/url", tags=["Knowledge Base"], dependencies=_admin_write)
async def ingest_url(req: IngestURLRequest, services: Services = Depends(get_services)) -> dict:
    return await _ingest(services, req.url.strip(), "url", req.url.strip())


@router.post("/ingest/pdf", tags=["Knowledge Base"], dependencies=_admin_write)
async def ingest_pdf(file: UploadFile = File(...), services: Services = Depends(get_services)) -> dict:
    filename = (file.filename or "uploaded.pdf").replace("\\", "/").rsplit("/", 1)[-1][:200]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    data = await file.read(config.MAX_PDF_BYTES + 1)
    if len(data) > config.MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413, detail=f"PDFs are limited to {config.MAX_PDF_BYTES // (1024 * 1024)} MB."
        )
    return await _ingest(services, data, "pdf", filename)


@router.get("/kb/info", tags=["Knowledge Base"], dependencies=[Depends(rate_limit())])
async def kb_info(services: Services = Depends(get_services)) -> dict:
    return services.vector_store.collection_info()


@router.get("/kb/sources", tags=["Knowledge Base"], dependencies=[Depends(require_admin)])
async def kb_sources(services: Services = Depends(get_services)) -> dict:
    sources = await asyncio.to_thread(services.vector_store.sources)
    return {"status": "ok", "sources": sources}


@router.get("/kb/search", tags=["Knowledge Base"], dependencies=[Depends(require_admin), Depends(rate_limit())])
async def kb_search(
    q: str = Query(min_length=1, max_length=500),
    k: int | None = Query(default=None, ge=1, le=10),
    services: Services = Depends(get_services),
) -> dict:
    """Run the agent's retrieval for ``q`` and show the passages it would read."""
    store = services.vector_store
    if not store.is_available():
        raise HTTPException(status_code=503, detail="The knowledge base is unavailable: the embedding model could not be loaded.")
    started = time.perf_counter()
    passages, searched, _ = await asyncio.to_thread(services.retriever.search, q.strip(), None, k)
    return {
        "status": "ok",
        "query": searched,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "citations": [passage.citation(number) for number, passage in enumerate(passages, start=1)],
    }


@router.delete("/kb/sources", tags=["Knowledge Base"], dependencies=[Depends(require_admin)])
async def kb_delete_source(
    source: str = Query(min_length=1, max_length=2048), services: Services = Depends(get_services)
) -> dict:
    removed = await asyncio.to_thread(services.vector_store.delete_source, source)
    if not removed:
        raise HTTPException(status_code=404, detail="No such source in the knowledge base.")
    return {"status": "ok", "chunks_removed": removed}


@router.delete("/kb/clear", tags=["Knowledge Base"], dependencies=[Depends(require_admin)])
async def kb_clear(services: Services = Depends(get_services)) -> dict:
    await asyncio.to_thread(services.vector_store.clear)
    logger.warning("Knowledge base cleared by an admin.")
    return {"status": "ok", "message": "Knowledge base cleared."}
