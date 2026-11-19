"""
Write evaluation questions from your own knowledge base.

The golden dataset only fits the sample documents. To evaluate retrieval and
answers on *your* documents, this samples chunks from the live knowledge base
and asks the LLM to write, for each, a question a caller might ask, a short
reference answer and the exact phrase from the chunk that answers it.

A generated case is kept only if its evidence phrase really occurs in the chunk,
so retrieval can be scored automatically. Review the file before relying on it:
generated questions are easier than real callers' questions.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from evaluation.dataset import EvalCase
from evaluation.metrics import normalize

logger = logging.getLogger(__name__)

GENERATOR_PROMPT = """You write test questions for a phone assistant that answers from a company's documents.

Given one PASSAGE, write one question a caller could realistically ask that this passage answers.
Ask it the way people speak, not with the passage's wording. Then give:
- "reference": the correct answer in one short sentence,
- "evidence": a phrase copied EXACTLY from the passage (4 to 12 words) that contains the answer.

Reply with JSON only:
{"question": "...", "reference": "...", "evidence": "..."}"""

UNANSWERABLE = [
    "What is the Wi-Fi password?",
    "What's the weather like today?",
    "Who won the cricket match yesterday?",
    "Can you recommend a good restaurant nearby?",
    "What is your company's stock price?",
]


def _parse(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not all(isinstance(data.get(key), str) and data[key].strip() for key in ("question", "reference", "evidence")):
        return None
    return data


async def generate_cases(vector_store: Any, client: Any, limit: int = 30, seed: int = 7) -> list[EvalCase]:
    chunks = [chunk for chunk in vector_store.all_chunks() if len(chunk.text) > 120]
    random.Random(seed).shuffle(chunks)  # noqa: S311 - sampling, not security
    # Spread questions across sources rather than over-sampling the largest document.
    by_source: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_source.setdefault(str(chunk.metadata.get("source", "")), []).append(chunk)
    ordered: list[Any] = []
    while any(by_source.values()) and len(ordered) < limit * 2:
        for items in by_source.values():
            if items:
                ordered.append(items.pop())

    cases: list[EvalCase] = []
    for chunk in ordered:
        if len(cases) >= limit:
            break
        try:
            result = await client.complete(
                [{"role": "system", "content": GENERATOR_PROMPT}, {"role": "user", "content": f"PASSAGE:\n{chunk.text}"}],
                max_tokens=250,
                temperature=0.4,
            )
        except Exception:
            logger.warning("Question generation failed for one passage", exc_info=True)
            continue
        data = _parse(result.answer)
        if not data or normalize(data["evidence"]) not in normalize(chunk.text):
            continue
        cases.append(EvalCase(
            id=f"gen-{len(cases) + 1:03d}",
            question=data["question"].strip(),
            type="paraphrase",
            evidence=(data["evidence"].strip(),),
            reference=data["reference"].strip(),
            source=str(chunk.metadata.get("source", "")),
        ))
    for number, question in enumerate(UNANSWERABLE, start=1):
        cases.append(EvalCase(
            id=f"gen-none-{number}", question=question, type="unanswerable", answerable=False,
            reference="Not in the documents; the assistant should say it does not have that information.",
        ))
    return cases
