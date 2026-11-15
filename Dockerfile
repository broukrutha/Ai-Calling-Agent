# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/models

WORKDIR /app

# CPU-only PyTorch keeps the image about 2 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app.py telephony_vobiz.py make_vobiz_call.py setup_vobiz_trunk.py ./
COPY voice_agent ./voice_agent
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY static ./static
COPY sample_doc ./sample_doc

# Run as an unprivileged user; all state lives on the /data volume.
RUN useradd --create-home --uid 10001 agent && mkdir -p /data && chown -R agent:agent /data /app
USER agent

ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8001 \
    CHROMA_PERSIST_DIR=/data/chroma_db \
    BOOKINGS_DB_PATH=/data/bookings.db

VOLUME ["/data"]
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"

# One process: conversation state and the rate limiter live in memory.
# Behind a reverse proxy, set FORWARDED_ALLOW_IPS so client addresses are trusted.
CMD ["python", "app.py"]
