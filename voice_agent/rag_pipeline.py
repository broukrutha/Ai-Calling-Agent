"""
rag_pipeline.py – Retrieval-Augmented Generation pipeline.

LATENCY OPTIMIZATIONS APPLIED:
  - Reranker DISABLED by default: cross-encoder adds ~50–200ms per query with
    minimal accuracy gain for 2-chunk retrieval. Only load if explicitly enabled.
  - Adaptive k simplified: removes over-fetching that caused reranker overhead.
  - Simple questions (<=5 words) fetch exactly TOP_K_RETRIEVAL (2 chunks), not 4.
  - Summarization queries fetch at most 4 (was 8) — retrieval time scales with k.
  - context stored on RAGResult so streaming pipeline reuses it without re-querying.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from voice_agent.config import TOP_K_RETRIEVAL, MIN_RELEVANCE_SCORE
from voice_agent.sarvam_client import SarvamClient, LLMResult
from voice_agent.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Reranker — DISABLED by default for latency (adds 50-200ms) ───────────────
# Set ENABLE_RERANKER = True only if accuracy matters more than speed.
ENABLE_RERANKER: bool = False
_reranker = None


def get_reranker():
    """Lazy load cross-encoder reranker — only loads once."""
    global _reranker
    if not ENABLE_RERANKER:
        return None
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.info("Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    except ImportError:
        logger.warning("sentence-transformers not installed — reranking disabled.")
        _reranker = None
    except Exception as e:
        logger.warning(f"Reranker load failed: {e} — reranking disabled")
        _reranker = None
    return _reranker


def rerank(query: str, hits: list, top_k: int = 2) -> list:
    """
    Rerank retrieved chunks using cross-encoder.
    Only called when ENABLE_RERANKER is True.
    """
    if not hits:
        return hits
    reranker = get_reranker()
    if reranker is None:
        return hits[:top_k]

    try:
        t0 = time.perf_counter()
        pairs = [(query, doc.page_content) for doc, score in hits]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        result = [hit for _, hit in ranked[:top_k]]
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Reranked {len(hits)} → {len(result)} chunks | {ms:.0f}ms")
        return result
    except Exception as e:
        logger.warning(f"Reranking failed: {e} — using original order")
        return hits[:top_k]


# Keywords that mean the user wants a full document summary
SUMMARIZE_KEYWORDS = [
    "summarize", "summary", "overview", "explain", "describe",
    "what is the document", "what does it say", "tell me about",
    "what is this", "give me a summary",
]


@dataclass
class RAGResult:
    answer: str
    sources: List[str]
    context_chunks: int
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_rag_latency_ms: float
    input_tokens: int  = 0
    output_tokens: int = 0
    context: str       = ""     # stored so streaming pipeline reuses without re-retrieving


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:
        query → VectorStore.retrieve() → format context →
        SarvamClient.generate() → RAGResult
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        sarvam_client: Optional[SarvamClient] = None,
    ) -> None:
        self.vs     = vector_store or VectorStore()
        self.client = sarvam_client or SarvamClient()
        logger.info("RAGPipeline initialized.")

    async def query(self, user_query: str, k: int = TOP_K_RETRIEVAL) -> RAGResult:
        """
        Run the full RAG pipeline.

        OPTIMIZED adaptive k logic:
          - Summarization queries  → fetch up to 4 chunks (was 8, halved for speed)
          - All other queries      → fetch exactly TOP_K_RETRIEVAL (2 chunks)
          Removed the over-fetch-for-reranker pattern since reranker is disabled.
        """
        pipeline_start = time.perf_counter()

        # ── Adaptive chunk count — simplified ─────────────────────────
        query_lower = user_query.lower()
        if any(kw in query_lower for kw in SUMMARIZE_KEYWORDS):
            k = min(k * 2, 4)   # OPTIMIZED: was k*3 up to 8 — now max 4
            logger.info(f"Summarization query → fetching {k} chunks")
        # All other queries use exactly k=TOP_K_RETRIEVAL (2) — no over-fetching

        # ── Retrieval ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        hits = self.vs.retrieve(user_query, k=k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── Optional rerank (disabled by default for latency) ──────────
        if ENABLE_RERANKER and len(hits) > TOP_K_RETRIEVAL:
            hits = rerank(user_query, hits, top_k=TOP_K_RETRIEVAL)
        elif len(hits) > TOP_K_RETRIEVAL:
            hits = hits[:TOP_K_RETRIEVAL]

        # ── Build context string ───────────────────────────────────────
        sources: List[str] = []
        context_parts: List[str] = []
        for doc, score in hits:
            src = doc.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)
            context_parts.append(
                f"[Source: {src} | Relevance: {score:.2f}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        if not context:
            logger.warning("No relevant context found — answering from LLM knowledge only.")

        # ── LLM generation ─────────────────────────────────────────────
        llm_result: LLMResult = await self.client.generate(
            user_message=user_query,
            context=context,
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            f"RAG | retrieval={retrieval_ms:.0f}ms | llm={llm_result.latency_ms:.0f}ms | "
            f"total={total_ms:.0f}ms | chunks={len(hits)}"
        )

        return RAGResult(
            answer=llm_result.answer,
            sources=sources,
            context_chunks=len(hits),
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=llm_result.latency_ms,
            total_rag_latency_ms=round(total_ms, 2),
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
            context=context,
        )

    async def query_with_llm(self, user_query: str, k: int = TOP_K_RETRIEVAL) -> RAGResult:
        """Full RAG + LLM — used by /query endpoint and benchmark."""
        return await self.query(user_query, k)

    def ingest_documents(self, docs) -> int:
        """Add pre-chunked documents to the vector store."""
        return self.vs.add_documents(docs)

    def knowledge_base_info(self) -> dict:
        return self.vs.collection_info()
