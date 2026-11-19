"""Hybrid retrieval with real ChromaDB and a deterministic test embedder."""

import pytest
from langchain_core.documents import Document

from tests.fakes import HashingEmbedder
from voice_agent import config
from voice_agent.chunking import chunk_text
from voice_agent.rag_pipeline import HybridRetriever, contextual_query, is_followup
from voice_agent.vector_store import VectorStore

CLINIC = """# Clinic

## Prices
- Teeth cleaning: 1,200 rupees.
- Full-mouth OPG X-ray: 800 rupees.
- Metal braces: 35,000 rupees. Clear aligners: from 1,20,000 rupees.

## Dentists
- Dr. Arjun Rao, endodontist: root canals. Gachibowli, Monday, Wednesday and Friday.
- Dr. Meera Iyer, orthodontist: braces. Banjara Hills, Monday to Thursday.

## Parking
Free basement parking for two hours.
"""


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"), embedding_function=HashingEmbedder())
    vs.add_documents(chunk_text(CLINIC, source="clinic.md"))
    return vs


@pytest.fixture
def retriever(store, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 600)
    return HybridRetriever(store)


def test_followup_detection():
    history = [{"role": "user", "content": "Who does root canals?"}, {"role": "assistant", "content": "Dr. Arjun Rao."}]
    assert is_followup("Which days is he in?", history)
    assert is_followup("And the clear ones?", history)
    assert not is_followup("Which days is he in?", [])
    assert not is_followup("What does a teeth cleaning cost at your clinic?", history)
    assert contextual_query("Which days is he in?", history) == "Who does root canals? Dr. Arjun Rao. Which days is he in?"


def test_passages_carry_citation_details(retriever):
    passages, searched, followup = retriever.search("How much is a teeth cleaning?")
    assert passages and "1,200 rupees" in passages[0].text
    citation = passages[0].citation(1)
    assert citation["source"] == "clinic.md" and citation["section"] == "Prices" and citation["n"] == 1
    assert "1,200 rupees" in citation["snippet"] and not citation["snippet"].startswith("Clinic ›")
    assert searched == "How much is a teeth cleaning?" and followup is False


def test_keyword_signal_finds_rare_codes(retriever, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    passages, _, _ = retriever.search("OPG price")
    assert "OPG" in passages[0].text and passages[0].lexical_score > 0.5


def test_followup_uses_the_previous_turn(retriever):
    history = [{"role": "user", "content": "Who does root canals?"},
               {"role": "assistant", "content": "Dr. Arjun Rao, our endodontist."}]
    passages, searched, followup = retriever.search("Which days is he in?", history)
    assert followup and "Arjun Rao" in searched
    assert "Monday, Wednesday and Friday" in passages[0].text


def test_relevance_margin_trims_weak_passages(retriever, monkeypatch):
    monkeypatch.setattr(config, "RELEVANCE_MARGIN", 1.0)
    wide, _, _ = retriever.search("parking")
    monkeypatch.setattr(config, "RELEVANCE_MARGIN", 0.05)
    narrow, _, _ = retriever.search("parking")
    assert len(narrow) <= len(wide) and "parking" in narrow[0].text.lower()


def test_dense_mode_ignores_keywords(retriever, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "dense")
    passages, _, _ = retriever.search("OPG price")
    assert all(p.lexical_score == 0 for p in passages)


def test_keyword_index_follows_knowledge_base_changes(retriever, store):
    assert not any("veneer" in p.text.lower() for p in retriever.search("porcelain veneers")[0])
    store.add_documents([Document(page_content="Porcelain veneers cost 12,000 rupees.", metadata={"source": "new.md"})])
    passages, _, _ = retriever.search("porcelain veneers")
    assert "12,000" in passages[0].text


def test_reingesting_a_source_replaces_its_old_chunks(store):
    before = store.count()
    store.add_documents([Document(page_content="Only one passage now.", metadata={"source": "clinic.md"})])
    assert store.count() == 1 and before > 1
    store.add_documents([Document(page_content="Another file.", metadata={"source": "other.md"})], replace=True)
    assert store.count() == 2


def test_broad_questions_get_more_passages(retriever, monkeypatch):
    monkeypatch.setattr(config, "RELEVANCE_MARGIN", 1.0)
    narrow, _, _ = retriever.search("braces")
    broad, _, _ = retriever.search("tell me about braces")
    assert len(broad) >= len(narrow)


def test_mismatched_embedding_model_is_refused(tmp_path, monkeypatch):
    import voice_agent.vector_store as vs_module

    monkeypatch.setattr(vs_module, "_default_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    built = VectorStore(persist_dir=str(tmp_path / "kb"))
    assert built.is_available() and built.stored_model() == "all-MiniLM-L6-v2"

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    assert VectorStore(persist_dir=str(tmp_path / "kb")).is_available()      # same model, other spelling

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
    wrong = VectorStore(persist_dir=str(tmp_path / "kb"))
    assert not wrong.is_available() and "reindex_kb.py" in wrong.init_error
