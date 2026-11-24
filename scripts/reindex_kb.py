"""
Rebuild the knowledge-base index with the current settings.

    python scripts/reindex_kb.py

Use it after changing EMBEDDING_MODEL. Every stored passage is re-embedded
with the configured model; nothing has to be uploaded again. To also apply a
new chunking strategy, ingest the original files again instead
(python scripts/seed_kb.py <files>): re-ingesting a source replaces it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from voice_agent import config  # noqa: E402
from voice_agent.logging_config import configure_logging  # noqa: E402


def main() -> int:
    import chromadb

    configure_logging()
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    names = {getattr(item, "name", item) for item in client.list_collections()}
    if config.CHROMA_COLLECTION_NAME not in names:
        print("There is no knowledge base to rebuild yet.")
        return 0

    # Reading stored text needs no embedding model, so open the collection raw.
    old = client.get_collection(config.CHROMA_COLLECTION_NAME)
    data = old.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=dict(metadata or {}))
        for text, metadata in zip(data.get("documents") or [], data.get("metadatas") or [], strict=False)
        if text
    ]
    print(f"Re-embedding {len(documents)} passages with {config.EMBEDDING_MODEL} …")

    backup = Path(config.CHROMA_PERSIST_DIR) / "reindex_backup.jsonl"
    import json

    with open(backup, "w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps({"text": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")

    client.delete_collection(config.CHROMA_COLLECTION_NAME)
    from voice_agent.vector_store import VectorStore

    store = VectorStore()
    if not store.is_available():
        print(f"Could not open the new index: {store.init_error}")
        print(f"Your passages are saved in {backup}.")
        return 1
    written = store.add_documents(documents, replace=False)
    print(f"Done: {written} passages indexed. A copy of the text was kept in {backup}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
