from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from voice_agent import config
from voice_agent.booking import store
from voice_agent.server.schemas import OutboundCallRequest
from voice_agent.server.security import rate_limit, require_admin
from voice_agent.telephony.outbound import TelephonyNotConfiguredError, dispatch_call

router = APIRouter(prefix="/calls", tags=["Calls"], dependencies=[Depends(require_admin)])


@router.get("")
async def list_calls(limit: int = 50) -> dict:
    """Recent phone calls handled by the telephony worker."""
    calls = await asyncio.to_thread(store.get_calls, min(max(limit, 1), 200))
    return {
        "status": "ok",
        "calls": calls,
        "stats": await asyncio.to_thread(store.call_stats),
        "telephony_configured": config.telephony_configured(),
        "outbound_enabled": config.telephony_configured() and bool(config.OUTBOUND_TRUNK_ID),
    }


@router.post("/outbound", dependencies=[Depends(rate_limit("expensive"))])
async def outbound(req: OutboundCallRequest) -> dict:
    """Ask the telephony worker to phone a number. The worker must be running."""
    try:
        call = await dispatch_call(req.phone_number)
    except TelephonyNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", **call}
