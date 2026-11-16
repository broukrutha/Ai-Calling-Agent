"""
ASGI entry point.

    python app.py                 # serves on HOST:PORT from the environment
    uvicorn app:app --reload      # development with auto-reload

Run a single process: conversation state and rate limits are kept in memory.
"""

import uvicorn

from voice_agent import config
from voice_agent.server import create_app

try:
    app = create_app()
except config.ConfigError as exc:
    raise SystemExit(f"Cannot start: {exc}") from None


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_config=None,        # logging is configured by the application
        access_log=False,       # requests are logged once, with a request id
    )
