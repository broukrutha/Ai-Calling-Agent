"""
Retrieval-augmented generation.

Retrieval is hybrid:

    question ──┬─► dense search (embeddings) ───┐
               └─► keyword search (BM25)  ──────┼─► score fusion ─► filter ─► [rerank] ─► top K
    follow-up ───► dense search on the question ┘
    context        plus the previous turn

* **Dense** search finds passages that mean the same thing in other words.
* **BM25** finds exact terms embeddings blur: prices, product codes, names,
  romanised Hindi or Telugu words.
* **Follow-ups** ("Which days is he in?") are searched together with the
  previous exchange, so the pronoun has something to resolve to.
* **Score fusion** ranks by meaning (cosine similarity) plus a bonus for how
  much of the question's informative wording a passage matches. Weak,
  incidental keyword matches add almost nothing, so they cannot outrank the
  passage the embeddings found.
* An optional **cross-encoder** reranks the fused shortlist (``ENABLE_RERANKER``).

Each run returns :class:`Passage` objects carrying everything needed to cite
them. Embedding and search are CPU-bound and run in a worker thread so they
never stall calls being served on the event loop.

Every change here is measured with ``python -m evaluation run --suite retrieval``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from voice_agent import config
from voice_agent.lexical import BM25Index
from voice_agent.sarvam_client import LLMResult, SarvamClient
from voice_agent.vector_store import Candidate, VectorStore

logger = logging.getLogger(__name__)

_BROAD_QUERY_PHRASES = (
    "summarize", "summarise", "summary", "overview", "what is this", "what does it say",
    "tell me about", "give me a summary", "what is the document", "everything about",
)
_ANAPHORA = frozenset({
    "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "there", "that",
    "those", "this", "these", "one", "ones", "same", "such", "wo", "woh", "vo", "unka", "uska",
})
_FOLLOWUP_OPENERS = ("and ", "what about", "how about", "also ", "but ", "then ", "so ", "aur ")
_WORD = re.compile(r"[\w']+")

_reranker: Any = None
_reranker_lock = threading.Lock()


def _get_reranker() -> Any:
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
            logger.info("Reranker loaded: %s", config.RERANKER_MODEL)
        return _reranker


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass
class Passage:
    """One retrieved chunk, with what the UI and the eval suite need to cite it."""

    id: str
    text: str
    source: str
    section: str = ""
    page: int | None = None
    score: float = 0.0             # final ranking score (fused, or reranker)
    dense_score: float = 0.0       # cosine similarity, 0–1
    lexical_score: float = 0.0     # keyword coverage of the question, 0–1

    def citation(self, number: int) -> dict[str, Any]:
        body = self.text.split("\n", 1)[-1] if "\n" in self.text else self.text
        snippet = " ".join(body.split())
        return {
            "n": number,
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "snippet": snippet[:280] + ("…" if len(snippet) > 280 else ""),
            "score": round(self.dense_score, 3),
        }


@dataclass
class Retrieval:
    passages: list[Passage] = field(default_factory=list)
    query: str = ""                 # the text actually searched
    latency_ms: float = 0.0
    followup: bool = False

    @property
    def context(self) -> str:
        return "\n\n".join(f"[{number}] {passage.text}" for number, passage in enumerate(self.passages, start=1))

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(passage.source for passage in self.passages))

    @property
    def chunks(self) -> int:
        return len(self.passages)

    @property
    def citations(self) -> list[dict[str, Any]]:
        return [passage.citation(number) for number, passage in enumerate(self.passages, start=1)]


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    context_chunks: int
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_rag_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    fallback: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval: Retrieval | None = None


# ── Query understanding ──────────────────────────────────────────────────────

def is_followup(question: str, history: list[dict[str, str]] | None) -> bool:
    """Does the question lean on the previous turn ("And the clear ones?")?"""
    if not history or not any(turn.get("role") == "user" for turn in history):
        return False
    lowered = question.lower().strip()
    words = _WORD.findall(lowered)
    return bool(set(words) & _ANAPHORA) or lowered.startswith(_FOLLOWUP_OPENERS) or len(words) <= 3


def contextual_query(question: str, history: list[dict[str, str]] | None) -> str:
    """The question plus the previous exchange, for searching a follow-up."""
    last_user = next((t["content"] for t in reversed(history or []) if t.get("role") == "user"), "")
    last_answer = next((t["content"] for t in reversed(history or []) if t.get("role") == "assistant"), "")
    return " ".join(part for part in (last_user, last_answer[:240], question) if part)


def is_broad(question: str) -> bool:
    lowered = question.lower()
    return any(phrase in lowered for phrase in _BROAD_QUERY_PHRASES)


# ── Retriever ────────────────────────────────────────────────────────────────

FOLLOWUP_CONTEXT_WEIGHT = 0.75  # stable from 0.6 to 0.8 on the golden set; 0.85+ lets the old topic win


class HybridRetriever:
    """Dense + BM25 retrieval with rank fusion. Thread-safe; one per vector store."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self._index = BM25Index()
        self._chunks: dict[str, Candidate] = {}
        self._signature: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def _keyword_index(self) -> BM25Index:
        """Rebuild the BM25 index whenever the stored chunks change."""
        signature = self.vs.signature()
        with self._lock:
            if signature != self._signature:
                chunks = self.vs.all_chunks()
                self._chunks = {chunk.id: chunk for chunk in chunks}
                self._index = BM25Index().build([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])
                self._signature = signature
                logger.info("Keyword index built over %d chunks", len(chunks))
            return self._index

    def search(
        self, question: str, history: list[dict[str, str]] | None = None, k: int | None = None
    ) -> tuple[list[Passage], str, bool]:
        """Blocking search. Returns (passages, query searched, was follow-up)."""
        k = k or config.TOP_K_RETRIEVAL
        if is_broad(question):
            k = max(k * 2, 5)
        followup = is_followup(question, history)
        queries = [question] + ([contextual_query(question, history)] if followup else [])
        depth = max(config.RETRIEVAL_CANDIDATES, k)

        dense_scores: dict[str, float] = {}
        texts: dict[str, Candidate] = {}
        # A follow-up also searches with the previous turn folded in. Matches found
        # only that way are discounted, so the new topic ranks ahead of the old one.
        weights = [1.0] + [FOLLOWUP_CONTEXT_WEIGHT] * (len(queries) - 1)
        for weight, hits in zip(weights, self.vs.dense_search(queries, depth), strict=True):
            for hit in hits:
                dense_scores[hit.id] = max(dense_scores.get(hit.id, 0.0), hit.score * weight)
                texts[hit.id] = hit

        coverage: dict[str, float] = {}
        if config.RETRIEVAL_MODE == "hybrid" and config.LEXICAL_WEIGHT > 0:
            # Keywords are matched on the question as asked: run on a follow-up's
            # combined text they drag the ranking back to the previous topic.
            index = self._keyword_index()
            for hit in index.search(question, depth):
                coverage[hit.id] = hit.coverage
                if hit.id not in texts and hit.id in self._chunks:
                    texts[hit.id] = self._chunks[hit.id]
            # Strong keyword matches the embedding search missed get a real dense score.
            missing = [chunk_id for chunk_id in coverage if chunk_id not in dense_scores and chunk_id in texts]
            for weight, query in zip(weights, queries, strict=True):
                for chunk_id, score in self.vs.similarity(query, missing).items():
                    dense_scores[chunk_id] = max(dense_scores.get(chunk_id, 0.0), score * weight)

        # Score fusion: meaning first, plus a bonus for matching the question's
        # informative words. A weak, incidental keyword match adds almost nothing.
        fused = {
            chunk_id: dense_scores.get(chunk_id, 0.0) + config.LEXICAL_WEIGHT * coverage.get(chunk_id, 0.0) ** 2
            for chunk_id in texts
        }
        best = max(fused.values(), default=0.0)
        ordered = [
            chunk_id for chunk_id in sorted(fused, key=fused.get, reverse=True)
            if (dense_scores.get(chunk_id, 0.0) >= config.MIN_RELEVANCE_SCORE or coverage.get(chunk_id, 0.0) >= 0.5)
            # Passages far behind the best one are mostly noise for the LLM.
            and fused[chunk_id] >= best - config.RELEVANCE_MARGIN
        ]
        passages = [self._passage(texts[cid], fused[cid], dense_scores, coverage) for cid in ordered]

        if config.ENABLE_RERANKER and len(passages) > 1:
            passages = self._rerank(queries[-1], passages[: config.RERANK_CANDIDATES])
        return passages[:k], queries[-1], followup

    @staticmethod
    def _passage(chunk: Candidate, score: float, dense: dict[str, float], lexical: dict[str, float]) -> Passage:
        metadata = chunk.metadata
        page = metadata.get("page")
        return Passage(
            id=chunk.id,
            text=chunk.text,
            source=str(metadata.get("source", "unknown")),
            section=str(metadata.get("section", "")),
            page=int(page) if isinstance(page, (int, float)) else None,
            score=round(score, 5),
            dense_score=dense.get(chunk.id, 0.0),
            lexical_score=lexical.get(chunk.id, 0.0),
        )

    @staticmethod
    def _rerank(query: str, passages: list[Passage]) -> list[Passage]:
        try:
            scores = _get_reranker().predict([(query, passage.text) for passage in passages])
        except Exception:
            logger.exception("Reranking failed — keeping the fused order.")
            return passages
        for passage, score in zip(passages, scores, strict=True):
            passage.score = round(float(score), 4)
        return sorted(passages, key=lambda passage: passage.score, reverse=True)


# ── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, vector_store: VectorStore | None = None, sarvam_client: SarvamClient | None = None) -> None:
        self.vs = vector_store or VectorStore()
        self.client = sarvam_client or SarvamClient()
        self.retriever = HybridRetriever(self.vs)

    async def retrieve(
        self, query: str, history: list[dict[str, str]] | None = None, k: int | None = None
    ) -> Retrieval:
        """Passages relevant to ``query`` (and, for follow-ups, the previous turn)."""
        if not self.vs.is_available():
            return Retrieval(query=query)
        started = time.perf_counter()
        passages, searched, followup = await asyncio.to_thread(self.retriever.search, query, history, k)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("Retrieval | %d passages | follow-up=%s | %.0f ms", len(passages), followup, elapsed)
        return Retrieval(passages=passages, query=searched, latency_ms=round(elapsed, 2), followup=followup)

    async def query(
        self,
        user_query: str,
        k: int | None = None,
        history: list[dict[str, str]] | None = None,
        language: str = "en-IN",
    ) -> RAGResult:
        """Retrieve, then generate a complete (non-streamed) answer."""
        started = time.perf_counter()
        retrieval = await self.retrieve(user_query, history=history, k=k)
        llm: LLMResult = await self.client.generate(
            user_message=user_query, context=retrieval.context, history=history, language=language
        )
        return RAGResult(
            answer=llm.answer,
            sources=retrieval.sources,
            context_chunks=retrieval.chunks,
            retrieval_latency_ms=retrieval.latency_ms,
            llm_latency_ms=llm.latency_ms,
            total_rag_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            provider=llm.provider,
            model=llm.model,
            fallback=llm.fallback,
            citations=retrieval.citations,
            retrieval=retrieval,
        )
