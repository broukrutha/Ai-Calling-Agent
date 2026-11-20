"""
Application factory.

``create_app()`` wires configuration, middleware, error handling and routes.
Tests call it with fake services; ``app.py`` at the repository root calls it
with none and serves the result.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voice_agent import __version__, config
from voice_agent.booking import calcom, store
from voice_agent.document_processor import IngestError
from voice_agent.logging_config import configure_logging, request_id_var
from voice_agent.sarvam_client import ProviderError
from voice_agent.server import security
from voice_agent.server.deps import AgentUnavailableError, Services
from voice_agent.server.routes import booking, calls, chat, evaluation, health, knowledge, ops, voice_ws

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
# Largest legitimate request body: a PDF upload plus multipart overhead.
_MAX_BODY_BYTES = config.MAX_PDF_BYTES + 1024 * 1024


@asynccontextmanager
async def _lifespan(app: FastAPI):
    services: Services = app.state.services
    await asyncio.to_thread(lambda: services.vector_store)       # loads the embedding model
    try:
        stats = await asyncio.to_thread(store.booking_stats)
        logger.info("Booking database ready: %d bookings, %d upcoming", stats["total"], stats["upcoming"])
    except Exception:
        logger.exception("Booking database could not be opened")

    if services.agent_ready:
        try:
            agent = await asyncio.to_thread(lambda: services.agent)
            await agent.warmup()
        except Exception:
            logger.exception("Voice agent failed to initialise; it will be retried on first use")

    logger.info("AI Calling Agent %s started (%s)", __version__, config.APP_ENV)
    yield

    task = getattr(app.state, "eval_task", None)
    if task is not None and not task.done():
        task.cancel()
    await services.close()
    await calcom.close()
    store.close()
    logger.info("Shutdown complete")


def create_app(services: Services | None = None) -> FastAPI:
    configure_logging()
    for warning in config.validate():     # raises ConfigError before anything is served
        logger.warning("Config: %s", warning)
    app = FastAPI(
        title="AI Calling Agent",
        description="Voice agent API: speech, knowledge-base answers, appointment booking and phone calls.",
        version=__version__,
        lifespan=_lifespan,
        # Interactive docs are a development aid; don't publish the API surface in production.
        docs_url=None if config.IS_PRODUCTION else "/docs",
        redoc_url=None,
        openapi_url=None if config.IS_PRODUCTION else "/openapi.json",
    )
    app.state.services = services or Services()

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Token", "Authorization"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
                response = JSONResponse(status_code=413, content={"detail": "Request body too large."})
            else:
                response = await call_next(request)
        finally:
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        for name, value in security.SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if not request.url.path.startswith(security.CSP_EXEMPT_PATHS):
            response.headers.setdefault("Content-Security-Policy", security.CONTENT_SECURITY_POLICY)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers.setdefault("Cache-Control", "no-cache")

        if not request.url.path.startswith(("/static", "/healthz")):
            logger.info(
                "%s %s → %d (%.0f ms) [%s]",
                request.method, request.url.path, response.status_code,
                (time.perf_counter() - started) * 1000, request_id,
            )
        return response

    _register_error_handlers(app)

    for module in (health, knowledge, chat, booking, calls, ops, evaluation, voice_ws):
        app.include_router(module.router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(str(page))
        return HTMLResponse("<h1>Web client not found</h1>", status_code=404)

    return app


def _register_error_handlers(app: FastAPI) -> None:
    """
    Errors we raise on purpose carry a message written for the user. Anything
    else is logged in full and answered with a generic message and a request
    id, so internals (paths, stack traces, provider responses) never leak.
    """

    @app.exception_handler(IngestError)
    async def _ingest_error(request: Request, exc: IngestError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(AgentUnavailableError)
    async def _agent_unavailable(request: Request, exc: AgentUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError):
        logger.error("Provider error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=502,
            content={"detail": "An upstream AI provider is unavailable. Please try again shortly."},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        problems = [
            {"field": ".".join(str(part) for part in error["loc"][1:]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": "Invalid request.", "errors": problems})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        request_id = request_id_var.get()
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )
