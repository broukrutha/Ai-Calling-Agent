# AI Calling Agent

A voice agent that answers and places phone calls, answers questions from your own documents, and books or cancels appointments — with a web console to run it all.

- **Speaks and listens** in English and Indian languages (Sarvam AI speech-to-text and text-to-speech).
- **Answers from your knowledge base** — PDFs, web pages and text you upload — instead of guessing, with numbered citations to the passages it used.
- **Books real appointments** through Cal.com, with a confirmation step and a spoken reference number.
- **Works on the phone** through LiveKit and a Vobiz SIP trunk, inbound and outbound.
- **Measured, not guessed**: every turn is traced (timings, model, passages, cost, caller rating), and a built-in evaluation suite scores retrieval, answers and booking dialogs, with quality gates in CI.
- **Runs without any accounts** for a first look: with no keys set, bookings use a local mock calendar.

## How it works

```text
                        ┌──────────────── Web console (browser) ────────────────┐
                        │ Overview · Playground · Knowledge base · Appointments │
                        │ Calls · Monitoring · Evaluation                       │
                        └──────────────┬───────────────────────────┬────────────┘
                                 REST  │                 WebSocket │ /ws/voice
                        ┌──────────────▼───────────────────────────▼────────────┐
                        │                 FastAPI server (app.py)               │
                        │  auth · rate limits · request ids · security headers  │
                        └───┬───────────────┬────────────────┬──────────────┬───┘
                            │               │                │              │
                    Voice pipeline     Knowledge base    Booking dialog   Call log,
                    STT → LLM → TTS    ChromaDB + BM25   state machine    traces, eval
                    (streamed)         hybrid search     → Cal.com        runs (SQLite)
                            │                                  ▲              ▲
   Phone ── Vobiz SIP ── LiveKit ── Telephony worker ──────────┴──────────────┘
                                   (telephony_vobiz.py: same knowledge base,
                                    booking and call log, used as LLM tools)
```

**Latency.** The model's answer is streamed and cut into speakable pieces as it arrives. Each piece goes to text-to-speech while the model is still writing. The first piece may end at a clause (a comma or semicolon) once it is 40 characters long, so the caller starts hearing the reply before the first sentence is finished (`FIRST_CHUNK_MIN_CHARS`). In the browser's hands-free mode, a 600 ms pause ends your turn; on the phone, the speech-to-text service decides.

Our own code adds about 40 ms per turn (p50, 81 ms p95): knowledge search plus the pipeline, measured with the real embedding model and instant stand-in services over the 63 golden questions. The rest is the speech and model services. To measure them with your keys:

```bash
python scripts/latency_probe.py
```

It speaks five questions, runs each through the full pipeline, and prints speech-to-text, search, model first token, text-to-speech and time to first audio at p50 and p95. In production, the Monitoring page shows the same numbers for every real call. The Playground's inspector shows them for each turn.

**Interruptions.** In hands-free mode the console listens continuously, sends what you said when you pause, and stops the agent mid-sentence if you start talking (barge-in): playback stops at once and the server cancels the rest of that answer.

**Reliability.** The LLM providers form a fallback chain. A provider that fails with a lasting error (bad key, exhausted quota, unknown model) is benched for a while by a circuit breaker, so later turns skip it instead of waiting for it to fail again; a provider that stalls mid-answer is abandoned for the next one.

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
python scripts/seed_kb.py sample_doc/knowledge_base.txt sample_doc/clinic_faq.md
```

```bash
python app.py
```

Open <http://127.0.0.1:8001>. The first start downloads a small embedding model, so it takes a little longer.

**Upgrading from an earlier version?** The knowledge base is now split by document structure and embedded with a stronger model. Rebuild it once:

```bash
python scripts/reindex_kb.py
```

If the server finds an index built with a different embedding model, it says so and asks you to run this, rather than returning poor matches.

## The console

| Page | What it is for |
|---|---|
| **Overview** | Conversations and latency over the last 24 hours, latest answer quality score, next appointments and the status of every component. |
| **Playground** | Talk (hands-free or push-to-talk, hold <kbd>Space</kbd>) or type to the agent. Each answer shows its sources; the inspector shows where the time went, the model used, cost, and the exact passages retrieved. Rate answers with 👍 / 👎. |
| **Knowledge base** | Add PDFs, web pages and text. *Try a search* shows exactly which passages the agent would read for a question. |
| **Appointments**, **Calls** | Book, cancel and review appointments; place calls and read the call log. |
| **Monitoring** | Time-to-first-audio p50/p95 against your budget, turns and errors over time, median time per stage, the fallback chain with benched providers, cost, caller ratings, and every turn with its full trace. Export poorly rated answers as draft test cases. |
| **Evaluation** | Run the retrieval, answer and booking-dialog suites and see scores, gates and failing cases. |

<kbd>Ctrl</kbd>+<kbd>K</kbd> opens a command palette. Light and dark themes follow the system and can be switched.

**Design.**
- **Colour:** two 12-step scales, a cool neutral and an indigo accent, generated in OKLCH. Like Linear's LCH themes and Stripe's Lab palette, equal steps look equal. Step roles follow Radix Colors.
- **Contrast:** every text/background pair meets WCAG AA (4.5:1) in both themes. [`scripts/generate_palette.py`](scripts/generate_palette.py) prints the tokens and the contrast report, and a test fails if `static/style.css` drifts from it or drops below AA.
- **Charts:** a categorical palette validated for colour-blind separation. Headline numbers show the change against the previous period.
- **Type:** Inter (with display optical sizes) and Geist Mono, both self-hosted under the OFL.

## Retrieval

Each document is split along its headings into passages of about 600 characters. Every passage carries its document and section path (*Clinic handbook › Payment and insurance*), which helps both search and citations. A question is then searched two ways:

- **by meaning**, with `all-MiniLM-L6-v2` embeddings, and
- **by keywords**, with BM25 (stemmed, so *cancelling* matches *cancellation*; Hinglish filler words are ignored).

The two scores are fused: meaning first, plus a bonus for passages that contain the question's informative words. Passages far below the best match are dropped so the model is not distracted. A **follow-up** such as *"And is it covered by insurance?"* is also searched together with the previous exchange, but matches that only come from that context are discounted, so the new topic ranks first. An optional cross-encoder re-ranker can be switched on with `ENABLE_RERANKER=true`.

Measured on the golden set (55 answerable questions about the sample documents, scored by whether the passage containing the answer's evidence was retrieved):

| | Before | Now |
|---|---|---|
| Answer in the passages given to the model | 90.7% | **98.2%** |
| Answer in the top passage | 85.2% | **96.4%** |
| Mean reciprocal rank | 0.880 | **0.973** |
| Share of retrieved passages that are relevant | 47% | **82%** |
| Follow-up questions | 75% | **100%** |
| Hinglish (code-mixed) questions | 50% | **100%** |

*Before* is the previous fixed-size splitter with dense search only, measured on the same questions (54 at the time; one follow-up case was added since).

## Monitoring and evaluation

**Tracing.** Every turn — phone, voice and text — is recorded: question, the query actually searched, passages and scores, model and whether it was a fallback, per-stage timings, tokens and estimated cost. Emails and phone numbers are masked; set `TRACE_STORE_TEXT=false` to keep timings without any text. Traces older than `TRACE_RETENTION_DAYS` are deleted.

**Evaluation.** Three suites, runnable from the console or the command line:

| Suite | What it checks | Cost |
|---|---|---|
| `retrieval` | Whether the passage with the answer is retrieved and ranked first; precision of what is retrieved. | free, seconds |
| `dialog` | 11 scripted booking conversations: confirmation, refusal, changes of mind, cancellation, past dates, closed hours. | free, instant |
| `answers` | Generates real answers and grades them with an LLM judge for correctness, faithfulness to the passages, refusing when the answer is not in the documents, and being speakable (no markdown, short). | uses your LLM |

```bash
python -m evaluation run --suite all --html eval-report.html
```

The command exits with an error when a score falls below its threshold in [`evaluation/thresholds.json`](evaluation/thresholds.json); CI runs the retrieval and dialog suites on every change. To evaluate **your own** documents, generate questions from them, review the file, then run against the live knowledge base:

```bash
python -m evaluation generate --limit 30
```

```bash
python -m evaluation run --suite retrieval --kb live
```

Answers rated 👎 in the console can be turned into draft test cases with `python -m evaluation export-feedback` (or *Export low-rated answers* on the Monitoring page).

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
| `EMBEDDING_MODEL` | Sentence-transformers model for the knowledge base. Changing it requires `scripts/reindex_kb.py`. | `all-MiniLM-L6-v2` |
| `RETRIEVAL_MODE`, `LEXICAL_WEIGHT` | `hybrid` (meaning + keywords) or `dense`; weight of the keyword match. | `hybrid`, 0.2 |
| `TOP_K_RETRIEVAL`, `RELEVANCE_MARGIN` | Passages given to the model; how far below the best match a passage may score. | 3, 0.15 |
| `ENABLE_RERANKER` | Cross-encoder re-ranking of the candidates. | `false` |
| `FIRST_CHUNK_MIN_CHARS` | Start speaking at the first clause once it is this long (0 waits for full sentences). | 40 |
| `LLM_STALL_TIMEOUT_SEC`, `LLM_REASONING_EFFORT` | Abandon a provider that stops streaming; reasoning effort for gpt-oss / o-series models. | 8, `low` |
| `TRACING_ENABLED`, `TRACE_STORE_TEXT`, `TRACE_RETENTION_DAYS` | Per-turn tracing for the Monitoring page. | on, on, 30 |

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
| `WS /ws/voice` | Voice and chat turns with conversation memory; `{"type": "cancel"}` stops the answer in progress. | public, rate limited |
| `POST /query` | One-off question against the knowledge base. | public, rate limited |
| `GET /booking/slots?date=YYYY-MM-DD` | Open appointment times. | public |
| `POST /booking/intent` | Natural-language booking in one request. | public, rate limited |
| `POST /ingest/text`, `/ingest/url`, `/ingest/pdf` | Add knowledge-base sources. | admin |
| `GET /kb/sources`, `DELETE /kb/sources`, `DELETE /kb/clear` | Manage sources. | admin |
| `GET /kb/search?q=` | The passages the agent would retrieve for a question, with scores. | admin |
| `POST /booking/create`, `/booking/cancel` | Manage appointments. | admin |
| `GET /booking/history`, `/booking/stats`, `/booking/list` | Read appointments. | admin |
| `GET /calls`, `POST /calls/outbound` | Call log; place a call. | admin |
| `POST /feedback` | Rate an answer (helpful / not helpful) by its trace id. | public, rate limited |
| `GET /ops/summary`, `/ops/timeseries`, `/ops/traces`, `/ops/traces/{id}`, `/ops/export` | Monitoring: percentiles, rates, cost, provider health, traces, low-rated answers as JSONL. | admin |
| `POST /evaluation/runs`, `GET /evaluation/runs`, `/evaluation/runs/{id}`, `/evaluation/datasets`, `POST /evaluation/generate` | Run and read evaluations; generate questions from your documents. | admin |

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

The suite (390+ tests) runs offline in a few seconds: providers are replaced by fakes, the calendar by mock mode, and the clock is frozen. It covers the booking language understanding and dialog, the Cal.com client, the provider fallback chain and circuit breaker, the streaming voice pipeline and barge-in, chunking, keyword and hybrid retrieval, tracing, the evaluation metrics and judge parsing, URL-fetch protections, every API route including authentication and limits, the WebSocket protocol, and the phone agent's tools.

Quality is measured separately by the evaluation suites (see [Monitoring and evaluation](#monitoring-and-evaluation)).

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
  rag_pipeline.py           Hybrid retrieval, follow-ups, citations, answer generation
  chunking.py               Structure-aware splitting with section headers
  lexical.py                BM25 keyword index
  vector_store.py           ChromaDB knowledge base
  document_processor.py     PDF / URL / text ingestion
  speech.py                 Makes answers speakable (no markdown, ₹ → rupees)
  tracing.py                Per-turn traces, summaries and time series
  booking/                  nlu.py · dialog.py · calcom.py · store.py
  telephony/                agent.py · tools.py · outbound.py
  server/                   app.py · security.py · schemas.py · routes/
evaluation/                 Golden questions, dialog scenarios, metrics, LLM judge, CLI, report
static/                     Web console: ES modules, no build step, self-hosted font
scripts/                    seed_kb.py · reindex_kb.py · latency_probe.py · generate_palette.py · check_llm.py · check_calcom.py
tests/                      Test suite
```

## Developer

**R Broukrutha** — B.Tech student, Gokaraju Rangaraju Institute of Engineering & Technology (GRIET)
