"""
Measure real voice latency with your own keys, stage by stage.

    python scripts/latency_probe.py
    python scripts/latency_probe.py --turns 8 --question "Do you open on Sunday?"

Each turn speaks a question with text-to-speech, then runs it through the
full voice pipeline exactly as a call would: speech-to-text, knowledge-base
search, the language model (streamed) and text-to-speech. It prints how long
each stage took and the time to first audio, the moment the caller would
start hearing the reply. The answer cache is switched off so every turn reaches
the model, and nothing is written to the Monitoring traces.

Not included: the silence the system waits for before it decides the caller
has finished speaking (on the phone this is decided by the speech-to-text
service; in the browser's hands-free mode it is 600 ms), and network time
between the caller and your server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent import config  # noqa: E402

DEFAULT_QUESTIONS = [
    "What are your opening hours?",
    "How can I book an appointment?",
    "How much does a consultation cost?",
    "Do you accept insurance?",
    "Where are you located?",
]
STAGES = [
    ("stt_ms", "Speech to text"),
    ("retrieval_ms", "Knowledge search"),
    ("first_token_ms", "Model, first token"),
    ("tts_ms", "Text to speech"),
    ("first_audio_ms", "First audio"),
    ("total_ms", "Whole turn"),
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


async def probe(agent: Any, client: Any, questions: list[str], language: str = "en-IN") -> list[dict[str, Any]]:
    """Run each question as a spoken turn. Returns one row per turn."""
    from voice_agent.pipeline import ConversationSession

    rows = []
    for number, question in enumerate(questions, start=1):
        spoken = await client.synthesize(question, target_language_code=language)
        row: dict[str, Any] = {"question": question}
        try:
            async for event in agent.run_stream(
                spoken.audio_bytes, language_code=language, audio_format="wav",
                session=ConversationSession(session_id=f"latency-probe-{number}"),
            ):
                if event["type"] == "done":
                    row.update(event["latency"])
                    row["heard"] = event.get("transcript", "")
                    row["model"] = "/".join(filter(None, (event.get("provider"), event.get("model"))))
                elif event["type"] == "noise":
                    row["error"] = "the recording was treated as noise"
        except Exception as exc:  # report and carry on with the next question
            row["error"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """p50 and p95 per stage over the turns that completed."""
    summary = {}
    for key, _ in STAGES:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and row[key] > 0]
        if values:
            summary[key] = {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "n": len(values)}
    return summary


def report(rows: list[dict[str, Any]], summary: dict[str, dict[str, float]]) -> str:
    lines = ["", "Turn by turn (ms)"]
    header = f"  {'question':38} " + " ".join(f"{label.split(',')[0][:11]:>11}" for _, label in STAGES)
    lines.append(header)
    for row in rows:
        if "error" in row:
            lines.append(f"  {row['question'][:38]:38}  failed: {row['error']}")
            continue
        cells = " ".join(f"{row.get(key, 0):>11.0f}" for key, _ in STAGES)
        lines.append(f"  {row['question'][:38]:38} {cells}")
    lines += ["", "Summary"]
    for key, label in STAGES:
        if key in summary:
            stats = summary[key]
            lines.append(f"  {label:20} p50 {stats['p50']:6.0f} ms   p95 {stats['p95']:6.0f} ms   ({stats['n']} turns)")
    first_audio = summary.get("first_audio_ms")
    if first_audio:
        verdict = "within" if first_audio["p95"] <= config.LATENCY_TARGET_TOTAL_MS else "over"
        lines.append(f"\n  Time to first audio at p95 is {verdict} the {config.LATENCY_TARGET_TOTAL_MS} ms budget.")
    models = sorted({row["model"] for row in rows if row.get("model")})
    if models:
        lines.append(f"  Answered by: {', '.join(models)}")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--turns", type=int, default=5, help="number of spoken questions (default 5)")
    parser.add_argument("--question", action="append", help="a question to ask; repeat for several")
    parser.add_argument("--language", default="en-IN")
    args = parser.parse_args()

    if not config.SARVAM_API_KEY:
        print("SARVAM_API_KEY is not set: it is needed for speech to text and text to speech.")
        return 1
    config.ENABLE_ANSWER_CACHE = False

    from voice_agent.pipeline import VoiceAgent
    from voice_agent.rag_pipeline import RAGPipeline
    from voice_agent.sarvam_client import SarvamClient

    client = SarvamClient()
    agent = VoiceAgent(rag_pipeline=RAGPipeline(sarvam_client=client), sarvam_client=client, tracer=lambda trace: None)
    if not agent.rag.vs.count():
        print("Note: the knowledge base is empty, so search is instant and answers say they have no information.")
    questions = args.question or [DEFAULT_QUESTIONS[i % len(DEFAULT_QUESTIONS)] for i in range(max(args.turns, 1))]

    print("Warming up (model load, connections)…")
    await agent.warmup()
    await probe(agent, client, questions[:1], args.language)      # not counted: first connections are slower
    print(f"Measuring {len(questions)} turns…")
    rows = await probe(agent, client, questions, args.language)
    print(report(rows, summarize(rows)))
    await agent.close()
    return 0 if any("error" not in row for row in rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
