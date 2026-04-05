"""
vector_store.py – ChromaDB-backed vector store with local embeddings.
Uses sentence-transformers/paraphrase-MiniLM-L3-v2 (free, fast, ~80 MB).

LATENCY OPTIMIZATIONS:
  - _ef (embedding function) is shared as a module-level singleton so the
    model is loaded ONCE even if VectorStore is instantiated multiple times.
  - retrieve() skips _collection.count() guard — ChromaDB handles empty
    collections gracefully (returns empty lists). No extra round-trip.
  - n_results capped at min(k, existing_count) only if we know count cheaply;
    otherwise ChromaDB handles it internally.
  - include list trimmed: only request what we need (no "embeddings").

FIX:
  - _get_ef() now CLEARS dead local proxy env vars instead of raising.
    Previously a Windows system proxy (127.0.0.1:9) would crash the entire
    VectorStore init with a RuntimeError, putting the KB in degraded mode.
    Now it silently removes the bad proxy and proceeds with the download,
    matching the same pattern already used in config.py.
"""

import logging
import os
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document

from voice_agent.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    MIN_RELEVANCE_SCORE,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton embedding function ─────────────────────────────────
# Loading the SentenceTransformer model takes ~1–3 s on first call.
# By holding the EF at module scope, all VectorStore instances share one model.
_SHARED_EF: SentenceTransformerEmbeddingFunction | None = None

# All proxy env var names to check (covers uppercase + lowercase variants)
_PROXY_ENV_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


def _get_ef() -> SentenceTransformerEmbeddingFunction:
    global _SHARED_EF
    if _SHARED_EF is None:
        # ── Clear dead local proxy before model download ───────────────────
        # Windows sometimes inherits a system proxy like http://127.0.0.1:9
        # that is unreachable. This blocks HuggingFace model downloads and
        # every outbound API call. We clear it here (same as config.py does)
        # instead of raising — the model will then download directly.
        for env_name in _PROXY_ENV_NAMES:
            raw_proxy = os.getenv(env_name, "").strip()
            if not raw_proxy:
                continue
            parsed = urlparse(raw_proxy)
            if parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == 9:
                os.environ.pop(env_name, None)
                logger.warning(
                    f"Cleared dead local proxy from {env_name} ({raw_proxy}). "
                    "Embedding model will be downloaded directly."
                )

        _SHARED_EF = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL, device="cpu"
        )
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return _SHARED_EF


class VectorStore:
    """
    Wraps ChromaDB with a SentenceTransformer embedding function.
    Provides add / retrieve / clear operations.
    """

    def __init__(self) -> None:
        self._ef: Optional[SentenceTransformerEmbeddingFunction] = None
        self._client = None
        self._collection = None
        self._available = False
        self._init_error: Optional[str] = None

        try:
            self._ef = _get_ef()   # shared singleton — no reload cost
            self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info(
                f"VectorStore ready – collection '{CHROMA_COLLECTION_NAME}' "
                f"({self._collection.count()} docs)."
            )
        except Exception as exc:
            self._init_error = str(exc)
            logger.exception(
                "VectorStore unavailable. Running in degraded mode without knowledge-base embeddings."
            )

    def is_available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                "Knowledge base is unavailable because the embedding model could not be loaded. "
                f"Details: {self._init_error or 'unknown error'}"
            )

    # ── Write ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> int:
        """Embed and store a list of LangChain Document objects. Returns count added."""
        self._require_available()
        if not docs:
            logger.warning("add_documents called with empty list – skipping.")
            return 0

        import uuid
        ids, texts, metas = [], [], []

        for doc in docs:
            uid = str(uuid.uuid4())
            ids.append(uid)
            texts.append(doc.page_content)
            metas.append(doc.metadata or {})

        self._collection.add(ids=ids, documents=texts, metadatas=metas)
        logger.info(f"Added {len(docs)} chunks → total {self._collection.count()}.")
        return len(docs)

    # ── Read ───────────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, k: int = TOP_K_RETRIEVAL
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most relevant chunks.

        Skips _collection.count() guard — ChromaDB returns empty results
        naturally when the collection is empty, so no extra round-trip needed.
        Only requests documents/metadatas/distances (no embeddings) to reduce
        data transfer from ChromaDB.
        """
        if not self._available or self._collection is None:
            logger.warning("Vector store unavailable – returning no retrieval results.")
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=max(k, 1),
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Tuple[Document, float]] = []

        docs_list  = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        if not docs_list:
            logger.warning("Vector store is empty or no results returned.")
            return []

        for text, meta, dist in zip(docs_list, metas_list, dists_list):
            score = round(1.0 - float(dist), 4)
            if score >= MIN_RELEVANCE_SCORE:
                hits.append((Document(page_content=text, metadata=meta), score))

        logger.info(f"Retrieved {len(hits)} relevant chunks (query: '{query[:60]}…').")
        return hits

    def retrieve_context_string(self, query: str, k: int = TOP_K_RETRIEVAL) -> str:
        """Return retrieved chunks as a single formatted string for the LLM prompt."""
        hits = self.retrieve(query, k)
        if not hits:
            return "No relevant documents found in the knowledge base."
        parts = []
        for i, (doc, score) in enumerate(hits, 1):
            src = doc.metadata.get("source", "unknown")
            parts.append(
                f"[Chunk {i} | Source: {src} | Relevance: {score:.2f}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Admin ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        if not self._available or self._collection is None:
            return 0
        return self._collection.count()

    def clear(self) -> None:
        self._require_available()
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")

    def collection_info(self) -> dict:
        return {
            "collection":      CHROMA_COLLECTION_NAME,
            "total_chunks":    self.count(),
            "embedding_model": EMBEDDING_MODEL,
            "persist_dir":     CHROMA_PERSIST_DIR,
            "status":          "ready" if self._available else "degraded",
            "available":       self._available,
            "error":           self._init_error,
        }