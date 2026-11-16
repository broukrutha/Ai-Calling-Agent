from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from voice_agent import __version__, config
from voice_agent.booking import calcom, store
from voice_agent.server.deps import Services, get_services

router = APIRouter(tags=["Status"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict:
    """Cheap check for load balancers and container orchestrators. Calls nothing external."""
    return {"status": "ok"}


@router.get("/health", summary="Detailed status for the dashboard")
async def health(services: Services = Depends(get_services)) -> dict:
    knowledge_base = services.vector_store.collection_info()

    if not services.agent_ready:
        api = {"status": "api_key_missing", "message": "Set SARVAM_API_KEY in .env to enable voice and chat."}
    else:
        try:
            api = await services.agent.health_check()
        except Exception:
            api = {"status": "error", "message": "The voice agent failed to start. Check the server logs."}

    try:
        stats = await asyncio.to_thread(store.booking_stats)
        bookings = {"total": stats["total"], "upcoming": stats["upcoming"]}
    except Exception:
        bookings = {"total": 0, "upcoming": 0}
    bookings["mode"] = "mock" if calcom.is_mock_mode() else "cal.com"

    healthy = api.get("status") == "ok" and knowledge_base["available"]
    return {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "environment": config.APP_ENV,
        "api": api,
        "knowledge_base": knowledge_base,
        "bookings": bookings,
        "telephony": {"configured": config.telephony_configured(), "outbound": bool(config.OUTBOUND_TRUNK_ID)},
        "auth_required": bool(config.ADMIN_API_TOKEN),
        "latency_budget_ms": config.LATENCY_TARGET_TOTAL_MS,
    }
