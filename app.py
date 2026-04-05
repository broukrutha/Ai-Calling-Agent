"""
app.py – FastAPI server for the Sarvam AI Voice Agent.

LATENCY OPTIMIZATIONS:
  - VoiceAgent and VectorStore are pre-initialized at startup (not lazily on
    first request) — eliminates cold-start latency for the first real request.
  - Added GZip middleware for compressing large JSON responses (benchmark, kb/info).
  - uvicorn started with loop="uvloop" (if available) for faster async I/O.
  - SarvamClient persistent httpx connection pool is shared across all requests
    via the singleton VoiceAgent instance.

BOOKING SYSTEM:
  - Cal.com integration with slot checking, booking creation, cancellation.
  - Multi-turn voice booking with name/email collection.
  - Local SQLite persistence of all booking records.
  - REST endpoints for slot check, create, cancel, list, history.

Endpoints:
  GET  /              – Serve the web UI
  GET  /health        – Health check (API connectivity)
  POST /ingest        – Ingest documents (file upload, URL, or raw text)
  POST /query         – Text query with RAG response
  GET  /kb/info       – Knowledge base statistics
  DELETE /kb/clear    – Clear the knowledge base
  GET  /benchmark     – Run benchmark and return JSON results
  WS   /ws/voice      – Full voice pipeline over WebSocket
  GET  /booking/slots – Available slots for a date
  POST /booking/create – Create a booking
  POST /booking/intent – Natural language booking
  POST /booking/cancel – Cancel a booking
  GET  /booking/list   – List bookings (from Cal.com + local DB)
  GET  /booking/history – Local booking history from SQLite
  GET  /booking/stats   – Booking statistics
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import (
    Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voice_agent.document_processor import ingest as ingest_docs
from voice_agent.vector_store import VectorStore
from voice_agent.config import (
    ADMIN_API_TOKEN,
    ALLOWED_ORIGINS,
    LATENCY_TARGET_TOTAL_MS,
    SARVAM_API_KEY,
)
from voice_agent.cal_booking import (
    is_booking_intent,
    handle_booking_query,
    get_available_slots,
    create_booking,
    cancel_booking,
    list_bookings_from_cal,
    parse_booking_request,
    BookingSession,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler — replaces deprecated @app.on_event('startup')."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("Startup: initializing VectorStore…")
    _get_vs()

    # Initialize booking DB early
    try:
        from voice_agent.booking_db import booking_stats
        stats = booking_stats()
        logger.info(f"Startup: BookingDB ready — {stats['total']} bookings, {stats['upcoming']} upcoming")
    except Exception as e:
        logger.warning(f"Startup: BookingDB init warning: {e}")

    if SARVAM_API_KEY:
        try:
            logger.info("Startup: initializing VoiceAgent…")
            get_agent()
            logger.info("Startup: VoiceAgent ready.")
        except Exception as e:
            logger.warning(
                f"Startup: VoiceAgent init failed (will retry on first request): {e}"
            )
    else:
        logger.warning("Startup: SARVAM_API_KEY not set — VoiceAgent not pre-initialized.")

    yield  # application runs here

    # ── Shutdown (nothing to clean up yet) ───────────────────────────────


app = FastAPI(
    title="Sarvam AI Voice Agent",
    description="Low-latency RAG-powered voice assistant with Cal.com booking integration",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OPTIMIZED: GZip compress large responses (benchmark JSON, kb/info, etc.)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_ngrok_skip_header(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Singletons
_agent = None
_vector_store: Optional[VectorStore] = None


def _get_vs() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_agent():
    global _agent
    if _agent is None:
        if not SARVAM_API_KEY:
            raise ValueError(
                "SARVAM_API_KEY is not set. Please add it to your .env file. "
                "Get a free key at https://cloud.sarvam.ai"
            )
        from voice_agent.voice_agent import VoiceAgent
        _agent = VoiceAgent()
    return _agent


async def require_admin_token(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    """Protect sensitive endpoints when an admin token is configured."""
    if not ADMIN_API_TOKEN:
        logger.warning(
            "Sensitive endpoint accessed without ADMIN_API_TOKEN configured. "
            "Set ADMIN_API_TOKEN to enforce protection."
        )
        return
    if x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token.")


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    language: str = "en-IN"


class IngestTextRequest(BaseModel):
    text: str
    name: str = "manual_input"


class IngestURLRequest(BaseModel):
    url: str


class BookingCheckRequest(BaseModel):
    date: str           # "YYYY-MM-DD"

class BookingCreateRequest(BaseModel):
    date: str           # "YYYY-MM-DD"
    time: str           # "HH:MM" in IST (24h)
    name: str
    email: str

class BookingIntentRequest(BaseModel):
    query: str
    name: str  = "User"
    email: str = "user@example.com"

class BookingCancelRequest(BaseModel):
    booking_uid: str
    reason: str = ""


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Static files not found. Place index.html in ./static/</h1>")


@app.get("/health", tags=["Status"])
async def health():
    kb = _get_vs().collection_info()
    overall_status = "ok" if kb.get("available", False) else "degraded"

    # Include booking stats
    try:
        from voice_agent.booking_db import booking_stats
        bk_stats = booking_stats()
    except Exception:
        bk_stats = {"total": 0, "upcoming": 0}

    if not SARVAM_API_KEY:
        return {
            "status": overall_status,
            "api": {"status": "api_key_missing", "message": "Set SARVAM_API_KEY in .env"},
            "knowledge_base": kb,
            "bookings": bk_stats,
            "latency_budget_ms": LATENCY_TARGET_TOTAL_MS,
            "version": "2.0.0",
        }
    try:
        agent = get_agent()
        api_health = await agent.health_check()
    except Exception as e:
        api_health = {"status": "error", "message": str(e)}
    return {
        "status": overall_status,
        "api": api_health,
        "knowledge_base": kb,
        "bookings": bk_stats,
        "latency_budget_ms": LATENCY_TARGET_TOTAL_MS,
        "version": "2.0.0",
    }


# ── Knowledge Base ─────────────────────────────────────────────────────────────

@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text(req: IngestTextRequest):
    try:
        docs = ingest_docs(req.text, source_type="text", name=req.name)
        count = _get_vs().add_documents(docs)
        return {"status": "ok", "chunks_added": count, "source": req.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", tags=["Ingestion"])
async def ingest_url(req: IngestURLRequest):
    try:
        docs = ingest_docs(req.url, source_type="url")
        count = _get_vs().add_documents(docs)
        return {"status": "ok", "chunks_added": count, "source": req.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        pdf_bytes = await file.read()
        docs = ingest_docs(pdf_bytes, source_type="pdf", name=file.filename)
        count = _get_vs().add_documents(docs)
        return {"status": "ok", "chunks_added": count, "source": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/info", tags=["Knowledge Base"])
async def kb_info():
    return _get_vs().collection_info()


@app.delete("/kb/clear", tags=["Knowledge Base"])
async def kb_clear(_: None = Depends(require_admin_token)):
    _get_vs().clear()
    return {"status": "ok", "message": "Knowledge base cleared."}


# ── Query ──────────────────────────────────────────────────────────────────────

@app.post("/query", tags=["Query"])
async def text_query(req: QueryRequest):
    try:
        result = await get_agent().query_text(req.query, booking_session=None)
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Benchmark ──────────────────────────────────────────────────────────────────

@app.get("/benchmark", tags=["Evaluation"])
async def run_benchmark(n: int = 5):
    from voice_agent.benchmark import run_benchmark as _run_benchmark, SAMPLE_QUERIES
    n = min(max(n, 1), 20)
    queries = (SAMPLE_QUERIES * ((n // len(SAMPLE_QUERIES)) + 1))[:n]
    try:
        summary = await _run_benchmark(queries, get_agent())
        return summary
    except Exception as e:
        logger.exception("Benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket Voice Pipeline ──────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Full voice pipeline over WebSocket.

    Client sends:
        { "type": "audio", "data": "<base64 wav>", "language": "en-IN" }

    Server responds:
        { "type": "result", "transcript": "...", "answer": "...",
          "audio": "<base64 wav>", "latency": {...}, "cost_inr": 0.001,
          "sources": [...] }

    Or on error:
        { "type": "error", "message": "..." }
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    try:
        agent = get_agent()
    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "pong"}))
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))
        except WebSocketDisconnect:
            pass
        return

    booking_session = BookingSession()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg.get("type") == "query":
                query = msg.get("query", "").strip()
                if not query:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Empty query"}))
                    continue
                result = await agent.query_text(query, booking_session=booking_session)
                await websocket.send_text(json.dumps({"type": "text_result", **result}))
                continue

            if msg.get("type") == "audio":
                audio_b64 = msg.get("data", "")
                language = msg.get("language", "en-IN")

                if not audio_b64:
                    await websocket.send_text(json.dumps({"type": "error", "message": "No audio data"}))
                    continue

                await websocket.send_text(json.dumps({"type": "processing"}))

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_format = msg.get("format", "webm")

                    async for event in agent.run_stream(
                        audio_bytes=audio_bytes,
                        language_code=language,
                        audio_format=audio_format,
                        booking_session=booking_session,
                    ):
                        if event.get("type") == "audio_chunk" and isinstance(event.get("audio"), bytes):
                            event = {**event, "audio": base64.b64encode(event["audio"]).decode()}
                        await websocket.send_text(json.dumps(event))

                except Exception as e:
                    logger.exception("Voice pipeline error")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ── Booking (Cal.com) ──────────────────────────────────────────────────────────

@app.get("/booking/slots", tags=["Booking"])
async def booking_slots(date: str):
    """
    Get available slots for a given date.
    date: YYYY-MM-DD
    Returns list of available time strings in IST e.g. ["09:00", "10:00"]
    """
    try:
        from datetime import date as date_type
        d = date_type.fromisoformat(date)
        slots = await get_available_slots(d)
        return {"status": "ok", "date": date, "available_slots": slots}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/booking/create", tags=["Booking"])
async def booking_create(req: BookingCreateRequest, _: None = Depends(require_admin_token)):
    """
    Create a booking on Cal.com.
    time: HH:MM in IST (24h format)
    """
    try:
        from datetime import date as date_type
        d = date_type.fromisoformat(req.date)
        result = await create_booking(
            slot_time_ist=req.time,
            target_date=d,
            attendee_name=req.name,
            attendee_email=req.email,
            source="ui",
        )
        if result["success"]:
            return {"status": "ok", **result}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/booking/intent", tags=["Booking"])
async def booking_intent(req: BookingIntentRequest):
    """
    Natural language booking — parse query, check slots, book if available.
    Used by the voice agent when booking intent is detected.

    FIXED: Now creates a temporary BookingSession for REST use.
    The session parameter is Optional in handle_booking_query().
    """
    try:
        response_text, completed = await handle_booking_query(
            user_text=req.query,
            session=None,  # One-shot session — no multi-turn for REST
            attendee_name=req.name,
            attendee_email=req.email,
        )
        return {
            "status": "ok",
            "response": response_text,
            "booking_completed": completed,
        }
    except Exception as e:
        logger.exception("Booking intent failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/booking/cancel", tags=["Booking"])
async def booking_cancel_endpoint(req: BookingCancelRequest, _: None = Depends(require_admin_token)):
    """Cancel a booking by its Cal.com UID."""
    try:
        result = await cancel_booking(req.booking_uid)
        if result["success"]:
            return {"status": "ok", **result}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/booking/list", tags=["Booking"])
async def booking_list(
    status: str = Query("upcoming", description="Filter: upcoming, past, cancelled"),
    email: Optional[str] = Query(None, description="Filter by attendee email"),
    _: None = Depends(require_admin_token),
):
    """List bookings from Cal.com (with local DB fallback)."""
    try:
        bookings = await list_bookings_from_cal(
            status=status,
            attendee_email=email,
        )
        return {"status": "ok", "bookings": bookings, "count": len(bookings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/booking/history", tags=["Booking"])
async def booking_history(
    email: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: None = Depends(require_admin_token),
):
    """Get booking history from local SQLite database."""
    try:
        from voice_agent.booking_db import get_bookings
        bookings = get_bookings(
            email=email,
            status=status,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        return {"status": "ok", "bookings": bookings, "count": len(bookings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/booking/stats", tags=["Booking"])
async def booking_stats_endpoint(_: None = Depends(require_admin_token)):
    """Get booking statistics from local database."""
    try:
        from voice_agent.booking_db import booking_stats, get_upcoming_bookings
        stats = booking_stats()
        upcoming = get_upcoming_bookings(limit=5)
        return {
            "status": "ok",
            "stats": stats,
            "next_appointments": upcoming,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Server utilities
# ──────────────────────────────────────────────

def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is already bound to host:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False   # bind succeeded → port is free
        except OSError:
            return True    # bind failed → port is in use


def _force_free_port(port: int, host: str = "127.0.0.1", wait_secs: float = 2.0) -> None:
    """
    Kill every process listening on host:port, then wait until the OS confirms
    the port is free (or timeout). Works on Windows and Linux/Mac.
    """
    import platform
    import socket
    import subprocess
    import time

    if not _is_port_in_use(port, host):
        return  # nothing to do

    system = platform.system()
    print(f"  [WARN] Port {port} is in use - killing occupying process(es)...")

    try:
        if system == "Windows":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            killed = set()
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                    except ValueError:
                        continue
                    if pid in killed or pid == os.getpid():
                        continue
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True
                    )
                    killed.add(pid)
                    print(f"  [INFO] Killed PID {pid} (was holding port {port})")
        else:
            import signal as _signal
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True
            )
            for pid_str in result.stdout.strip().splitlines():
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == os.getpid():
                    continue
                os.kill(pid, _signal.SIGTERM)
                print(f"  [INFO] Killed PID {pid} (was holding port {port})")

    except Exception as exc:
        print(f"  [WARN] Auto-kill failed: {exc}")
        print(f"      Run manually:  netstat -ano | findstr :{port}  then  taskkill /PID <pid> /F")
        return

    # Wait for the OS to actually release the socket (up to wait_secs)
    deadline = time.time() + wait_secs
    while time.time() < deadline:
        if not _is_port_in_use(port, host):
            print(f"  [OK] Port {port} is now free.")
            return
        time.sleep(0.2)

    print(f"  [WARN] Port {port} still appears busy after {wait_secs}s. Trying to start anyway...")


if __name__ == "__main__":
    from voice_agent.config import SERVER_HOST, SERVER_PORT

    # Kill any leftover process on the port BEFORE uvicorn tries to bind.
    _force_free_port(SERVER_PORT, SERVER_HOST)

    # reload=False — required on Python 3.13 + Windows.
    uvicorn.run(
        "app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        loop="asyncio",
    )
