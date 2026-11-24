"""
Load documents into the knowledge base from the command line.

    python scripts/seed_kb.py sample_doc/knowledge_base.txt sample_doc/clinic_faq.md
    python scripts/seed_kb.py brochure.pdf https://example.com/faq

Useful for first-time setup and for seeding a container. Re-running with the
same file replaces its previous passages, so it also applies chunking changes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent.document_processor import IngestError, ingest  # noqa: E402
from voice_agent.logging_config import configure_logging  # noqa: E402
from voice_agent.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Add files or web pages to the knowledge base.")
    parser.add_argument("sources", nargs="+", help="paths to .pdf / .txt / .md files, or http(s) URLs")
    args = parser.parse_args()

    configure_logging()
    store = VectorStore()
    if not store.is_available():
        print(f"The knowledge base is unavailable: {store.init_error}")
        return 1

    failed = 0
    for source in args.sources:
        kind = "url" if source.startswith(("http://", "https://")) else "file"
        if kind == "file":
            source = str(Path(source))
        try:
            chunks = store.add_documents(ingest(source, kind))
        except IngestError as exc:
            print(f"  FAIL  {source}: {exc}")
            failed += 1
            continue
        print(f"  ok    {source}: {chunks} passages")
    print(f"Knowledge base now holds {store.count()} passages.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
