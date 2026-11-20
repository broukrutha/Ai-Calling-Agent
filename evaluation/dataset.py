"""
Evaluation datasets: one JSON object per line (JSONL).

    {
      "id": "clinic-price-cleaning",          unique, stable across versions
      "question": "How much does a teeth cleaning cost?",
      "type": "factoid",                      factoid | paraphrase | keyword | code_mixed |
                                              followup | topic_switch | unanswerable
      "answerable": true,                     false = the documents do not contain the answer
      "evidence": ["1,200 rupees"],           text a retrieved passage must contain
      "reference": "Teeth cleaning costs 1,200 rupees.",
      "answer_keywords": ["1200"],            cheap deterministic check on the answer
      "source": "clinic_faq.md",              optional
      "history": [{"role": "user", ...}],     optional, for follow-up questions
      "language": "en-IN"                     optional
    }

Evidence strings make retrieval scoring independent of how documents are
chunked: a passage is relevant when it contains one of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "golden.jsonl"
CASE_TYPES = {"factoid", "paraphrase", "keyword", "code_mixed", "followup", "topic_switch", "unanswerable"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    type: str = "factoid"
    answerable: bool = True
    evidence: tuple[str, ...] = ()
    reference: str = ""
    answer_keywords: tuple[str, ...] = ()
    source: str = ""
    history: tuple[dict, ...] = field(default_factory=tuple)
    language: str = "en-IN"

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        case = cls(
            id=str(data["id"]),
            question=str(data["question"]).strip(),
            type=str(data.get("type", "factoid")),
            answerable=bool(data.get("answerable", True)),
            evidence=tuple(str(item) for item in data.get("evidence", [])),
            reference=str(data.get("reference", "")),
            answer_keywords=tuple(str(item) for item in data.get("answer_keywords", [])),
            source=str(data.get("source", "")),
            history=tuple(
                {"role": str(turn["role"]), "content": str(turn["content"])}
                for turn in data.get("history", [])
            ),
            language=str(data.get("language", "en-IN")),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if not self.id or not self.question:
            raise ValueError("every case needs an id and a question")
        if self.type not in CASE_TYPES:
            raise ValueError(f"{self.id}: unknown type {self.type!r}")
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable cases need at least one evidence string")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable cases must not list evidence")
        for turn in self.history:
            if turn["role"] not in {"user", "assistant"}:
                raise ValueError(f"{self.id}: history roles must be user or assistant")

    def to_dict(self) -> dict:
        data = {
            "id": self.id, "question": self.question, "type": self.type, "answerable": self.answerable,
            "evidence": list(self.evidence), "reference": self.reference,
            "answer_keywords": list(self.answer_keywords),
        }
        if self.source:
            data["source"] = self.source
        if self.history:
            data["history"] = list(self.history)
        if self.language != "en-IN":
            data["language"] = self.language
        return data


def load_cases(path: str | Path = DEFAULT_DATASET) -> list[EvalCase]:
    """Read and validate a JSONL dataset. Duplicate ids are an error."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                case = EvalCase.from_dict(json.loads(line))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path}:{number}: duplicate id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


def save_cases(cases: list[EvalCase], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
