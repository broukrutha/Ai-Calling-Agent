"""
Retrieval-augmented generation.

``retrieve()`` finds the relevant knowledge-base chunks; ``query()`` adds the
LLM call for a complete answer. The voice pipeline calls ``retrieve()`` on its
own so it can stream the LLM answer sentence by sentence.

Embedding and vector search are CPU-bound, so they run in a worker thread and
never stall other calls being served on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from voice_agent import config
from voice_agent.sarvam_client import LLMResult, SarvamClient
from voice_agent.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Queries that need a broader view of the documents than a single fact lookup.
_BROAD_QUERY_PHRASES = (
    "summarize", "summarise", "summary", "overview", "what is this", "what does it say",
    "tell me about", "give me a summary", "what is the document",
)

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


@dataclass
class Retrieval:
    context: str = ""
    sources: list[str] = field(default_factory=list)
    chunks: int = 0
    latency_ms: float = 0.0


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


class RAGPipeline:
    def __init__(self, vector_store: VectorStore | None = None, sarvam_client: SarvamClient | None = None) -> None:
        self.vs = vector_store or VectorStore()
        self.client = sarvam_client or SarvamClient()

    def _retrieve_sync(self, query: str, k: int) -> list[tuple[Any, float]]:
        if not config.ENABLE_RERANKER:
            return self.vs.retrieve(query, k=k)

        candidates = self.vs.retrieve(query, k=k * 3)   # over-fetch, then keep the best k
        if len(candidates) <= 1:
            return candidates
        try:
            scores = _get_reranker().predict([(query, doc.page_content) for doc, _ in candidates])
            ranked = sorted(zip(scores, candidates, strict=True), key=lambda pair: pair[0], reverse=True)
            return [hit for _, hit in ranked[:k]]
        except Exception:
            logger.exception("Reranking failed — using vector order.")
            return candidates[:k]

    async def retrieve(self, query: str, k: int | None = None) -> Retrieval:
        """Relevant context for ``query``, formatted for the prompt."""
        k = k or config.TOP_K_RETRIEVAL
        if any(phrase in query.lower() for phrase in _BROAD_QUERY_PHRASES):
            k = max(k * 2, 4)

        started = time.perf_counter()
        hits = await asyncio.to_thread(self._retrieve_sync, query, k)
        elapsed = (time.perf_counter() - started) * 1000

        sources: list[str] = []
        parts = []
        for doc, _score in hits:
            source = str(doc.metadata.get("source", "unknown"))
            if source not in sources:
                sources.append(source)
            parts.append(f"[Source: {source}]\n{doc.page_content}")

        logger.info("Retrieval | %d chunks | %.0f ms", len(hits), elapsed)
        return Retrieval(
            context="\n\n---\n\n".join(parts), sources=sources, chunks=len(hits), latency_ms=round(elapsed, 2)
        )

    async def query(
        self,
        user_query: str,
        k: int | None = None,
        history: list[dict[str, str]] | None = None,
        language: str = "en-IN",
    ) -> RAGResult:
        """Retrieve, then generate a complete (non-streamed) answer."""
        started = time.perf_counter()
        retrieval = await self.retrieve(user_query, k)
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
        )
