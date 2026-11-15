# AI Calling Agent

A voice agent that answers and places phone calls, answers questions from your own documents, and books or cancels appointments — with a web console to run it all.

- **Speaks and listens** in English and Indian languages (Sarvam AI speech-to-text and text-to-speech).
- **Answers from your knowledge base** — PDFs, web pages and text you upload — instead of guessing.
- **Books real appointments** through Cal.com, with a confirmation step and a spoken reference number.
- **Works on the phone** through LiveKit and a Vobiz SIP trunk, inbound and outbound.
- **Runs without any accounts** for a first look: with no keys set, bookings use a local mock calendar.

## How it works

```text
                        ┌──────────────── Web console (browser) ────────────────┐
                        │ Overview · Playground · Knowledge base · Appointments │
                        │ Calls · Evaluation                                    │
                        └──────────────┬───────────────────────────┬────────────┘
                                 REST  │                 WebSocket │ /ws/voice
                        ┌──────────────▼───────────────────────────▼────────────┐
                        │                 FastAPI server (app.py)               │
                        │  auth · rate limits · request ids · security headers  │
                        └───┬───────────────┬────────────────┬──────────────┬───┘
                            │               │                │              │
                    Voice pipeline     Knowledge base    Booking dialog   Call log
                    STT → LLM → TTS    ChromaDB +        state machine    SQLite
                    (streamed)         local embeddings  → Cal.com
                            │                                  ▲              ▲
   Phone ── Vobiz SIP ── LiveKit ── Telephony worker ──────────┴──────────────┘
                                   (telephony_vobiz.py: same knowledge base,
                                    booking and call log, used as LLM tools)
```

**Latency.** The LLM answer is streamed, cut into sentences as it arrives, and each sentence is sent to text-to-speech while the model is still writing. The caller hears the first sentence without waiting for the whole answer. The Playground shows a per-turn latency waterfall, including time to first audio.

**Grounding.** Questions are answered only from retrieved passages. On phone calls the LLM has tools (`search_knowledge_base`, `check_availability`, `book_appointment`, `cancel_appointment`, `end_call`), so it can only claim a booking that the calendar actually accepted.

**Booking safety.** Nothing is booked or cancelled without an explicit "yes", a refusal always wins over an agreement in the same sentence, and the agent never invents a date or time it wasn't given.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/broukrutha/Ai-Calling-Agent.git
cd Ai-Calling-Agent
python -m venv .venv
```

Activate the environment — macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install and configure:

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Edit `.env`: set `SARVAM_API_KEY` and one of `OPENAI_API_KEY` / `GROQ_API_KEY`. Everything else is optional.

```bash
python scripts/seed_kb.py sample_doc/knowledge_base.txt
```

```bash
python app.py
```

Open <http://127.0.0.1:8001>. The first start downloads a small embedding model, so it takes a little longer.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example) for the full list).

| Variable | Purpose | Default |
|---|---|---|
| `APP_ENV` | `development` or `production`. Production requires an admin token and hides `/docs`. | `development` |
| `ADMIN_API_TOKEN` | Protects everything that changes data or spends money. | none |
| `ALLOWED_ORIGINS` | Browser origins allowed to call the API. | localhost |
| `SARVAM_API_KEY` | Speech-to-text and text-to-speech. | none |
| `LLM_PROVIDER` | Preferred LLM: `openai`, `groq` or `sarvam`. The others, if they have keys, are fallbacks. | `openai` |
| `OPENAI_API_KEY`, `GROQ_API_KEY` | LLM credentials. | none |
| `CAL_API_KEY`, `CAL_EVENT_TYPE_ID` | Cal.com booking. Without a key, bookings are local only (mock mode). | none |
| `BOOKING_TIMEZONE`, `OFFICE_START_HOUR`, `OFFICE_END_HOUR` | When appointments can be offered. | `Asia/Kolkata`, 9, 19 |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Telephony. | none |
| `VOBIZ_*`, `OUTBOUND_TRUNK_ID` | SIP trunk for phone calls. | none |
| `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_EXPENSIVE_PER_MINUTE` | Per-client request limits. | 120, 30 |

The server checks its configuration at start-up and refuses to run with unsafe production settings.

## Phone calls

1. Create a project at [LiveKit Cloud](https://cloud.livekit.io) and set the `LIVEKIT_*` values.
2. Set the `VOBIZ_*` values from your Vobiz console, then register the trunk:

   ```bash
   python setup_vobiz_trunk.py
   ```

   Put the printed `OUTBOUND_TRUNK_ID` in `.env`.
3. For inbound calls, point your Vobiz number at your LiveKit SIP endpoint and add a dispatch rule for the agent name `voice-assistant`.
4. Start the worker (a separate process from the web server):

   ```bash
   python telephony_vobiz.py start
   ```

5. Place a call from the console's **Calls** page, or from the command line:

   ```bash
   python make_vobiz_call.py --to +919988776655
   ```

Every call is recorded in the call log with its direction, duration, number of caller turns and outcome.

The worker loads the knowledge base when it starts: restart it after adding or removing documents.

## API

Interactive documentation is served at `/docs` in development.

| Method and path | What it does | Access |
|---|---|---|
| `GET /healthz` | Liveness probe; calls nothing external. | public |
| `GET /health` | Component status for the console. | public |
| `WS /ws/voice` | Voice and chat turns with conversation memory. | public, rate limited |
| `POST /query` | One-off question against the knowledge base. | public, rate limited |
| `GET /booking/slots?date=YYYY-MM-DD` | Open appointment times. | public |
| `POST /booking/intent` | Natural-language booking in one request. | public, rate limited |
| `POST /ingest/text`, `/ingest/url`, `/ingest/pdf` | Add knowledge-base sources. | admin |
| `GET /kb/sources`, `DELETE /kb/sources`, `DELETE /kb/clear` | Manage sources. | admin |
| `POST /booking/create`, `/booking/cancel` | Manage appointments. | admin |
| `GET /booking/history`, `/booking/stats`, `/booking/list` | Read appointments. | admin |
| `GET /calls`, `POST /calls/outbound` | Call log; place a call. | admin |
| `GET /benchmark?n=5` | Run the evaluation set. | admin |

Admin requests carry the token as `X-Admin-Token: <token>` or `Authorization: Bearer <token>`.

## Security

- **Admin token** on all write and paid operations, compared in constant time. In production the server will not start without one of at least 24 characters.
- **Rate limiting** per client address, with a tighter limit for endpoints that call paid providers (including WebSocket messages).
- **Input limits** on every request body, upload and recording.
- **URL ingestion** only fetches public http(s) hosts and re-checks every redirect, so it cannot be used to reach internal services.
- **Content-Security-Policy** with no inline scripts or styles; the console renders all server data as text and makes no third-party requests.
- **WebSocket origin check**, since browsers do not apply CORS to WebSockets.
- **Error hygiene**: internal errors return a request id, never stack traces or provider responses. Emails and phone numbers are masked in logs.
- **Data**: bookings and call records live in a local SQLite file that is excluded from git. Keep `.env`, `*.db` and `chroma_db/` out of version control and backups you share.

Known limits, by design for a single-node deployment: the rate limiter, conversation memory and answer cache are in process memory, so run one server process; put TLS and a body-size limit on your reverse proxy.

## Testing

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
```

```bash
ruff check .
```

The suite (300+ tests) runs offline in a few seconds: providers are replaced by fakes, the calendar by mock mode, and the clock is frozen. It covers the booking language understanding and dialog, the Cal.com client, the provider fallback chain, the streaming voice pipeline, knowledge-base storage, URL-fetch protections, every API route including authentication and limits, the WebSocket protocol, and the phone agent's tools.

To measure answer quality and speed against your own knowledge base:

```bash
python -m voice_agent.benchmark --queries 20 --html report.html
```

## Deployment

```bash
docker compose up --build
```

This starts the web server on port 8001 with data on a named volume. Add the phone worker with:

```bash
docker compose --profile telephony up --build
```

For production set `APP_ENV=production`, a strong `ADMIN_API_TOKEN` and your real `ALLOWED_ORIGINS`, and terminate TLS in front of the container. Use `/healthz` for health checks.

## Project layout

```text
app.py                      Web server entry point
telephony_vobiz.py          Phone worker entry point
make_vobiz_call.py          Place an outbound call from the command line
setup_vobiz_trunk.py        Register the SIP trunk with LiveKit
voice_agent/
  config.py                 Settings and start-up validation
  pipeline.py               Voice / chat turn orchestration (streamed STT → LLM → TTS)
  sarvam_client.py          Speech and multi-provider LLM client
  rag_pipeline.py           Retrieval and answer generation
  vector_store.py           ChromaDB knowledge base
  document_processor.py     PDF / URL / text ingestion
  booking/                  nlu.py · dialog.py · calcom.py · store.py
  telephony/                agent.py · tools.py · outbound.py
  server/                   app.py · security.py · schemas.py · routes/
evaluation/                 Evaluation question set, metrics and report
static/                     Web console (no build step)
scripts/                    seed_kb.py · check_llm.py · check_calcom.py
tests/                      Test suite
```

## Developer

**R Broukrutha** — B.Tech student, Gokaraju Rangaraju Institute of Engineering & Technology (GRIET)
