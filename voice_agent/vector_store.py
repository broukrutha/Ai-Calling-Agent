"""
Knowledge-base storage: ChromaDB with local sentence-transformer embeddings.

* Chunk ids are derived from the content, so ingesting the same document
  twice updates it instead of duplicating every chunk.
* ``version`` increases on every change; the answer cache uses it to drop
  answers that were generated from an older knowledge base.
* If the embedding model cannot be loaded the store reports itself as
  unavailable and the rest of the application keeps running.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter
from typing import Any

from langchain_core.documents import Document

from voice_agent import config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500
_shared_embedder: Any = None
_embedder_lock = threading.Lock()


def _default_embedder() -> Any:
    """Load the embedding model once per process (it takes 1–3 seconds)."""
    global _shared_embedder
    with _embedder_lock:
        if _shared_embedder is None:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            _shared_embedder = SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL, device="cpu"
            )
            logger.info("Embedding model loaded: %s", config.EMBEDDING_MODEL)
        return _shared_embedder


def _chunk_id(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:32]


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma accepts only scalar metadata values and rejects empty dicts."""
    cleaned = {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in (metadata or {}).items()
        if value is not None
    }
    cleaned.setdefault("source", "unknown")
    return cleaned


class VectorStore:
    """Add, search and manage knowledge-base chunks."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_function: Any = None,
    ) -> None:
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self._name = collection_name or config.CHROMA_COLLECTION_NAME
        self._embedder = embedding_function
        self._client: Any = None
        self._collection: Any = None
        self._available = False
        self._init_error: str | None = None
        self._lock = threading.Lock()
        self.version = 0

        try:
            import chromadb

            if self._embedder is None:
                self._embedder = _default_embedder()
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._open_collection()
            self._available = True
            logger.info("Knowledge base ready: '%s' (%d chunks)", self._name, self._collection.count())
        except Exception as exc:
            self._init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Knowledge base unavailable — continuing without document retrieval.")

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def is_available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError("The knowledge base is unavailable: the embedding model could not be loaded.")

    # ── Write ────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> int:
        """Embed and store chunks. Returns how many distinct chunks were written."""
        self._require_available()
        unique: dict[str, tuple[str, dict[str, Any]]] = {}
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            metadata = _clean_metadata(doc.metadata)
            unique[_chunk_id(str(metadata["source"]), text)] = (text, metadata)
        if not unique:
            return 0

        ids = list(unique)
        with self._lock:
            for start in range(0, len(ids), _BATCH_SIZE):
                batch = ids[start:start + _BATCH_SIZE]
                self._collection.upsert(
                    ids=batch,
                    documents=[unique[i][0] for i in batch],
                    metadatas=[unique[i][1] for i in batch],
                )
            self.version += 1
        logger.info("Stored %d chunks (total %d).", len(ids), self._collection.count())
        return len(ids)

    def delete_source(self, source: str) -> int:
        """Remove every chunk that came from ``source``. Returns the number removed."""
        self._require_available()
        with self._lock:
            ids = self._collection.get(where={"source": source}, include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
                self.version += 1
        logger.info("Removed %d chunks from source '%s'.", len(ids), source)
        return len(ids)

    def clear(self) -> None:
        self._require_available()
        with self._lock:
            self._client.delete_collection(self._name)
            self._collection = self._open_collection()
            self.version += 1
        logger.info("Knowledge base cleared.")

    # ── Read ─────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """The ``k`` most relevant chunks with a 0–1 relevance score, best first."""
        if not self._available:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=max(k or config.TOP_K_RETRIEVAL, 1),
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            score = round(1.0 - float(distance), 4)
            if score >= config.MIN_RELEVANCE_SCORE:
                hits.append((Document(page_content=text, metadata=metadata or {}), score))
        return hits

    def count(self) -> int:
        return self._collection.count() if self._available else 0

    def sources(self) -> list[dict[str, Any]]:
        """Every ingested source with its chunk count, largest first."""
        if not self._available:
            return []
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        counts = Counter(str((metadata or {}).get("source", "unknown")) for metadata in metadatas)
        return [{"source": source, "chunks": chunks} for source, chunks in counts.most_common()]

    def collection_info(self) -> dict[str, Any]:
        return {
            "collection": self._name,
            "total_chunks": self.count(),
            "embedding_model": config.EMBEDDING_MODEL,
            "status": "ready" if self._available else "degraded",
            "available": self._available,
        }

    @property
    def init_error(self) -> str | None:
        return self._init_error
