"""
Access control and abuse protection.

* **Admin token** – required for anything that changes data or spends money.
  Compared in constant time. In development with no token configured the
  check is skipped (a warning is logged at start-up); in production the server
  refuses to start without one.
* **Rate limiting** – a sliding one-minute window per client address, with a
  tighter limit for endpoints that call paid providers.
* **WebSocket origin check** – browsers do not apply CORS to WebSockets, so
  cross-site pages are refused explicitly.
* **Security headers** – including a Content-Security-Policy that forbids
  inline scripts.
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from fastapi import HTTPException, Request, WebSocket

from voice_agent import config

# ── Admin authentication ─────────────────────────────────────────────────────

def _presented_token(headers) -> str:
    token = headers.get("x-admin-token", "")
    if not token:
        scheme, _, value = headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer":
            token = value.strip()
    return token


def token_is_valid(token: str) -> bool:
    expected = config.ADMIN_API_TOKEN
    if not expected:
        return not config.IS_PRODUCTION
    return hmac.compare_digest(token.encode(), expected.encode())


async def require_admin(request: Request) -> None:
    """Dependency for admin-only routes."""
    if not token_is_valid(_presented_token(request.headers)):
        raise HTTPException(
            status_code=401,
            detail="A valid admin token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Rate limiting ────────────────────────────────────────────────────────────

class RateLimiter:
    """In-process sliding-window limiter. Use a shared store if you run several replicas."""

    _MAX_TRACKED_KEYS = 50_000

    def __init__(self, window_sec: float = 60.0) -> None:
        self._window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int) -> float:
        """Record a hit. Returns 0 if allowed, else the seconds until a retry."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= limit:
            return max(self._window - (now - hits[0]), 0.1)
        hits.append(now)
        if len(self._hits) > self._MAX_TRACKED_KEYS:
            self._prune(now)
        return 0.0

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self._window]:
            del self._hits[key]

    def reset(self) -> None:
        self._hits.clear()


limiter = RateLimiter()


def client_address(connection: Request | WebSocket) -> str:
    return connection.client.host if connection.client else "unknown"


def _limit_for(bucket: str) -> int:
    return config.RATE_LIMIT_EXPENSIVE_PER_MINUTE if bucket == "expensive" else config.RATE_LIMIT_PER_MINUTE


def allow(connection: Request | WebSocket, bucket: str = "default") -> float:
    """0 if this client may proceed, else seconds to wait."""
    if not config.RATE_LIMIT_ENABLED:
        return 0.0
    return limiter.check(f"{bucket}:{client_address(connection)}", _limit_for(bucket))


def rate_limit(bucket: str = "default"):
    """Dependency factory: ``Depends(rate_limit("expensive"))``."""

    async def dependency(request: Request) -> None:
        wait = allow(request, bucket)
        if wait:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down.",
                headers={"Retry-After": str(int(wait) + 1)},
            )

    return dependency


# ── WebSocket origin ─────────────────────────────────────────────────────────

def websocket_origin_allowed(websocket: WebSocket) -> bool:
    """
    Same-origin pages and configured origins may connect. Clients that send
    no Origin header (scripts, native apps) are not browsers and are allowed.
    """
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    if origin in config.ALLOWED_ORIGINS or "*" in config.ALLOWED_ORIGINS:
        return True
    return urlparse(origin).netloc == websocket.headers.get("host", "")


# ── Response headers ─────────────────────────────────────────────────────────

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "font-src 'self'",
    "img-src 'self' data:",
    "media-src 'self' blob:",
    "connect-src 'self' ws: wss:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
# The interactive API docs load their assets from a CDN and use inline scripts.
CSP_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json")
