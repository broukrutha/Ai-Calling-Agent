"""
config.py – Central configuration for the Sarvam AI Voice Agent.

LATENCY OPTIMIZATIONS:
  - LLM_MAX_TOKENS reduced to 60 (was 80) — shorter answers = faster generation
  - TOP_K_RETRIEVAL = 2 (keep small for faster vector search)
  - CACHE_MAX_SIZE increased to 200 for better hit rate
  - LLM_TEMPERATURE lowered to 0.2 for more deterministic, faster responses
  - GROQ_MODEL switched to llama-3.1-8b-instant (already fast)
"""

import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


def _sanitize_dead_local_proxy_env() -> None:
    """
    Clear the common Windows placeholder proxy (127.0.0.1:9) when inherited
    into this app process. That proxy is unreachable and breaks Hugging Face,
    httpx, and provider API calls.
    """
    proxy_env_names = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for env_name in proxy_env_names:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        parsed = urlparse(raw_value)
        if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == 9:
            os.environ.pop(env_name, None)


_sanitize_dead_local_proxy_env()

# ──────────────────────────────────────────────
# Sarvam AI API
# ──────────────────────────────────────────────
SARVAM_API_KEY: str  = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL: str = "https://api.sarvam.ai"

# ── Models ─────────────────────────────────────
STT_MODEL: str   = "saarika:v2.5"
LLM_MODEL: str   = "sarvam-m"
TTS_MODEL: str   = "bulbul:v3"
TTS_SPEAKER: str = "shubh"
TTS_LANGUAGE: str = "en-IN"

# ──────────────────────────────────────────────
# LLM Provider Selection
# Options: "openai" (GPT-4o-mini, best quality)
#          "groq"   (llama-3.1-8b, fast & free)
#          "sarvam" (sarvam-m, fallback)
# ──────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "groq" | "sarvam"

# ── OpenAI (GPT-4o-mini — recommended for voice)
OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str     = "gpt-4o-mini"
OPENAI_BASE_URL: str  = "https://api.openai.com/v1"

# ── Groq (fast & free fallback)
USE_GROQ: bool      = LLM_PROVIDER == "groq"
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str  = "https://api.groq.com"
GROQ_MODEL: str     = "llama-3.1-8b-instant"
GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
USE_GROQ_STT: bool  = False   # Sarvam STT supports Telugu/Hindi — keep it

# ──────────────────────────────────────────────
# Latency Targets (milliseconds)
# ──────────────────────────────────────────────
LATENCY_TARGET_TOTAL_MS: int = 3000
LATENCY_TARGET_STT_MS: int   = 800
LATENCY_TARGET_RAG_MS: int   = 300
LATENCY_TARGET_LLM_MS: int   = 500
LATENCY_TARGET_TTS_MS: int   = 800

# ──────────────────────────────────────────────
# RAG / ChromaDB
# ──────────────────────────────────────────────
CHROMA_PERSIST_DIR: str     = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = "voice_agent_kb"

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")

CHUNK_SIZE: int           = 512
CHUNK_OVERLAP: int        = 50
TOP_K_RETRIEVAL: int      = 2    # Keep at 2 for minimal retrieval latency
MIN_RELEVANCE_SCORE: float = 0.15

# ──────────────────────────────────────────────
# LLM Generation
# ──────────────────────────────────────────────
LLM_MAX_TOKENS: int    = 512    # Increased: models with <think> blocks need headroom for reasoning + answer
LLM_TEMPERATURE: float = 0.2   # OPTIMIZED: lower temp = faster, more deterministic output

LLM_SYSTEM_PROMPT: str = (
    "You are a smart, friendly AI scheduling assistant — like a human receptionist. "
    "You help users book, check, and cancel appointments naturally. "
    "NEVER output raw dates or ISO timestamps (like 2026-04-03T15:00:00). "
    "ALWAYS speak like a human: say 'April 3rd at 3 PM', never '2026-04-03 15:00'. "
    "If the user's query is about a booking or appointment, handle it conversationally. "
    "For all other questions, answer ONLY using the provided context. "
    "Do NOT think aloud or use <think> tags. Reply directly. "
    "If the context does not contain the answer, say so briefly. "
    "CRITICAL: Reply in the EXACT language specified in [IMPORTANT: Reply in this language only: ...]. "
    "If it says te-IN reply in Telugu. If en-IN reply in English. If hi-IN reply in Hindi. "
    "Keep answers concise — 1-2 sentences maximum for non-booking queries."
)

# ──────────────────────────────────────────────
# Answer Cache
# ──────────────────────────────────────────────
ENABLE_ANSWER_CACHE: bool = True
CACHE_MAX_SIZE: int       = 200   # OPTIMIZED: doubled from 100 → more cache hits

# ──────────────────────────────────────────────
# Cost Constants (INR)
# ──────────────────────────────────────────────
COST_STT_PER_HOUR_INR: float      = 30.0
COST_TTS_PER_10K_CHARS_INR: float = 30.0
COST_LLM_PER_TOKEN_INR: float     = 0.0

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 8001
ALLOWED_ORIGINS_RAW: str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8001,http://127.0.0.1:8001,http://localhost:3000,http://127.0.0.1:3000",
)
ALLOWED_ORIGINS: list[str] = [
    origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
]
ADMIN_API_TOKEN: str = os.getenv("ADMIN_API_TOKEN", "")

# ──────────────────────────────────────────────
# LiveKit + Vobiz Telephony
# ──────────────────────────────────────────────
LIVEKIT_URL: str        = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY: str    = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")

VOBIZ_SIP_DOMAIN: str      = os.getenv("VOBIZ_SIP_DOMAIN", "")
VOBIZ_USERNAME: str        = os.getenv("VOBIZ_USERNAME", "")
VOBIZ_PASSWORD: str        = os.getenv("VOBIZ_PASSWORD", "")
VOBIZ_OUTBOUND_NUMBER: str = os.getenv("VOBIZ_OUTBOUND_NUMBER", "")
OUTBOUND_TRUNK_ID: str     = os.getenv("OUTBOUND_TRUNK_ID", "")

# ──────────────────────────────────────────────
# Cal.com Booking
# ──────────────────────────────────────────────
CAL_API_KEY: str       = os.getenv("CAL_API_KEY", "")
CAL_USERNAME: str      = os.getenv("CAL_USERNAME", "ediga-broukruth-goud-0bphb5")
CAL_EVENT_SLUG: str    = os.getenv("CAL_EVENT_SLUG", "Appointments")
CAL_EVENT_TYPE_ID: int = int(os.getenv("CAL_EVENT_TYPE_ID", "5244785"))
URL_INGEST_TIMEOUT_SEC: int = int(os.getenv("URL_INGEST_TIMEOUT_SEC", "15"))
URL_INGEST_MAX_BYTES: int = int(os.getenv("URL_INGEST_MAX_BYTES", str(2 * 1024 * 1024)))