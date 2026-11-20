"""
Knowledge-base storage: ChromaDB with local sentence-transformer embeddings.

* Ingesting a source replaces whatever was stored for it before, so updating a
  document never leaves stale chunks behind. Chunk ids are derived from the
  content.
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
from dataclasses import dataclass, field
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


@dataclass
class Candidate:
    """A stored chunk returned by a search."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0          # cosine similarity, 0–1 (dense search only)


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
        # The configured model is checked against the one the index was built
        # with; an injected embedding function (tests, evaluation) is trusted.
        self._model_name = config.EMBEDDING_MODEL if embedding_function is None else ""
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
            self._check_model()
            self._available = True
            logger.info("Knowledge base ready: '%s' (%d chunks)", self._name, self._collection.count())
        except Exception as exc:
            self._init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Knowledge base unavailable — continuing without document retrieval.")

    def _open_collection(self) -> Any:
        existing = {getattr(item, "name", item) for item in self._client.list_collections()}
        if self._name in existing:
            try:
                return self._client.get_collection(name=self._name, embedding_function=self._embedder)
            except ValueError as exc:
                if "embedding function" in str(exc).lower():
                    raise RuntimeError(
                        "The knowledge base was built with a different embedding model. Set EMBEDDING_MODEL "
                        "back to it, or rebuild the index with: python scripts/reindex_kb.py"
                    ) from exc
                raise
        metadata: dict[str, Any] = {"hnsw:space": "cosine"}
        if self._model_name:
            metadata["embedding_model"] = self._model_name
        return self._client.create_collection(name=self._name, embedding_function=self._embedder, metadata=metadata)

    def stored_model(self) -> str:
        """The embedding model the index was built with, when it can be told."""
        name = (self._collection.metadata or {}).get("embedding_model", "")
        if not name:
            try:
                settings = getattr(self._collection, "configuration_json", None) or {}
                name = ((settings.get("embedding_function") or {}).get("config") or {}).get("model_name", "")
            except Exception:
                name = ""
        return str(name or "")

    def _check_model(self) -> None:
        """Querying vectors built by another model returns confident nonsense: refuse instead."""
        if not self._model_name:
            return
        stored = self.stored_model()

        def short(name: str) -> str:
            return name.strip().lower().removeprefix("sentence-transformers/")

        if stored and short(stored) != short(self._model_name):
            raise RuntimeError(
                f"The knowledge base was built with the embedding model '{stored}' but EMBEDDING_MODEL is "
                f"'{self._model_name}'. Set EMBEDDING_MODEL={stored}, or rebuild the index with: "
                "python scripts/reindex_kb.py"
            )

    def is_available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError("The knowledge base is unavailable: the embedding model could not be loaded.")

    # ── Write ────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document], replace: bool = True) -> int:
        """
        Embed and store chunks. Returns how many distinct chunks were written.

        With ``replace`` (the default) every source in ``docs`` is first removed,
        so a re-ingested document fully supersedes its previous version.
        """
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
        sources = {str(metadata["source"]) for _, metadata in unique.values()}
        with self._lock:
            if replace:
                for source in sources:
                    stale = self._collection.get(where={"source": source}, include=[])["ids"]
                    stale = [chunk_id for chunk_id in stale if chunk_id not in unique]
                    if stale:
                        self._collection.delete(ids=stale)
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
            self._collection = self._open_collection()  # recreated with the configured model
            self.version += 1
        logger.info("Knowledge base cleared.")

    # ── Read ─────────────────────────────────────────────────────────────────

    def dense_search(self, queries: list[str], n: int) -> list[list[Candidate]]:
        """Nearest chunks for each query, best first, with no score cut-off."""
        if not self._available or not queries:
            return [[] for _ in queries]
        results = self._collection.query(
            query_texts=queries,
            n_results=max(n, 1),
            include=["documents", "metadatas", "distances"],
        )
        out: list[list[Candidate]] = []
        for position in range(len(queries)):
            ids = (results.get("ids") or [[]])[position]
            documents = (results.get("documents") or [[]])[position]
            metadatas = (results.get("metadatas") or [[]])[position]
            distances = (results.get("distances") or [[]])[position]
            out.append([
                Candidate(id=chunk_id, text=text, metadata=dict(metadata or {}), score=round(1.0 - float(distance), 4))
                for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False)
            ])
        return out

    def similarity(self, query: str, ids: list[str]) -> dict[str, float]:
        """Cosine similarity between ``query`` and specific stored chunks."""
        if not self._available or not ids:
            return {}
        import numpy as np

        stored = self._collection.get(ids=ids, include=["embeddings"])
        vectors = stored.get("embeddings")
        if vectors is None or len(vectors) == 0:
            return {}
        query_vector = np.asarray(self._embedder([query])[0], dtype=float)
        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vector) or 1.0)
        similarities = matrix @ query_vector / np.where(norms == 0, 1.0, norms)
        return {chunk_id: round(float(value), 4) for chunk_id, value in zip(stored["ids"], similarities, strict=True)}

    def all_chunks(self) -> list[Candidate]:
        """Every stored chunk (for building the keyword index)."""
        if not self._available:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        return [
            Candidate(id=chunk_id, text=text or "", metadata=dict(metadata or {}))
            for chunk_id, text, metadata in zip(
                data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or [], strict=False
            )
        ]

    def signature(self) -> tuple[int, int]:
        """Changes whenever the stored chunks change (used to invalidate caches)."""
        return (self.version, self.count())

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
            "retrieval": config.RETRIEVAL_MODE,
            "error": self._init_error if self._init_error and "embedding model" in self._init_error else None,
        }

    @property
    def init_error(self) -> str | None:
        return self._init_error
