"""Vector store (real ChromaDB, fake embeddings), ingestion and SSRF guards."""

import socket

import pytest
from langchain_core.documents import Document

from tests.fakes import HashingEmbedder
from voice_agent import document_processor as dp
from voice_agent.document_processor import IngestError
from voice_agent.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"), embedding_function=HashingEmbedder())


def _doc(text, source="faq.txt"):
    return Document(page_content=text, metadata={"source": source})


def test_retrieval_finds_the_relevant_chunk(store):
    store.add_documents([
        _doc("Our office hours are nine to five on weekdays."),
        _doc("Refunds are processed within seven business days.", source="policy.pdf"),
    ])
    [(doc, score), *_] = store.retrieve("when are refunds processed", k=1)
    assert "Refunds" in doc.page_content and doc.metadata["source"] == "policy.pdf" and score > 0.3


def test_reingesting_the_same_document_does_not_duplicate_chunks(store):
    docs = [_doc("Alpha beta gamma."), _doc("Delta epsilon zeta.")]
    assert store.add_documents(docs) == 2
    assert store.add_documents(docs) == 2
    assert store.count() == 2


def test_version_changes_on_every_write(store):
    versions = [store.version]
    store.add_documents([_doc("Alpha beta gamma.")])
    versions.append(store.version)
    store.delete_source("faq.txt")
    versions.append(store.version)
    store.clear()
    versions.append(store.version)
    assert len(set(versions)) == 4


def test_sources_listing_and_delete(store):
    store.add_documents([_doc("one two three"), _doc("four five six"), _doc("seven", source="b.pdf")])
    assert store.sources() == [{"source": "faq.txt", "chunks": 2}, {"source": "b.pdf", "chunks": 1}]
    assert store.delete_source("faq.txt") == 2
    assert store.sources() == [{"source": "b.pdf", "chunks": 1}]
    assert store.delete_source("missing") == 0


def test_blank_chunks_and_odd_metadata_are_handled(store):
    added = store.add_documents([
        Document(page_content="   ", metadata={"source": "x"}),
        Document(page_content="Real content here", metadata={}),
        Document(page_content="More content", metadata={"source": "y", "page": 3, "tags": ["a", "b"], "n": None}),
    ])
    assert added == 2
    assert {s["source"] for s in store.sources()} == {"unknown", "y"}


def test_store_degrades_instead_of_crashing_when_embeddings_fail(tmp_path, monkeypatch):
    import voice_agent.vector_store as vs_module

    def boom():
        raise OSError("model download blocked")

    monkeypatch.setattr(vs_module, "_default_embedder", boom)
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    assert not store.is_available() and store.retrieve("anything") == [] and store.count() == 0
    assert store.collection_info()["status"] == "degraded"
    with pytest.raises(RuntimeError):
        store.add_documents([_doc("text")])


# ── Ingestion ────────────────────────────────────────────────────────────────

def test_text_is_chunked_with_its_source():
    docs = dp.ingest("word " * 600, "text", name="Company FAQ")
    assert len(docs) > 1 and all(d.metadata["source"] == "Company FAQ" for d in docs)
    assert all(len(d.page_content) <= 512 for d in docs)


@pytest.mark.parametrize("source,kind", [("", "text"), ("   ", "text"), ("x", "carrier-pigeon")])
def test_bad_input_raises_a_user_safe_error(source, kind):
    with pytest.raises(IngestError):
        dp.ingest(source, kind)


def test_pdf_round_trip_and_rejection_of_non_pdfs():
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Consultations cost five hundred rupees.")
    data = pdf.tobytes()
    pdf.close()

    [doc] = dp.ingest(data, "pdf", name="prices.pdf")
    assert "five hundred rupees" in doc.page_content
    assert doc.metadata == {"source": "prices.pdf", "page": 1}

    with pytest.raises(IngestError, match="not a valid PDF"):
        dp.ingest(b"MZ\x90\x00 definitely an exe", "pdf", name="evil.pdf")


# ── SSRF ─────────────────────────────────────────────────────────────────────

def _resolve_to(monkeypatch, mapping):
    def fake_getaddrinfo(host, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "http://localhost/admin",
    "http://127.0.0.1:8001/booking/history",
    "http://169.254.169.254/latest/meta-data/",      # cloud metadata service
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://[::1]/",
    "http://internal.corp/",                          # public-looking name, private address
    "http://user:pass@example.com/",
    "http://",
])
def test_non_public_urls_are_rejected(monkeypatch, url):
    _resolve_to(monkeypatch, {
        "localhost": "127.0.0.1", "127.0.0.1": "127.0.0.1", "169.254.169.254": "169.254.169.254",
        "10.0.0.5": "10.0.0.5", "192.168.1.1": "192.168.1.1", "::1": "::1",
        "internal.corp": "10.1.2.3", "example.com": "93.184.216.34",
    })
    with pytest.raises(IngestError):
        dp.validate_public_url(url)


def test_public_url_is_accepted(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    dp.validate_public_url("https://example.com/about")


class _FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.encoding = "utf-8"
        self.is_redirect = status in (301, 302, 303, 307, 308) and "Location" in self.headers
        self.is_permanent_redirect = False

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, pages):
    fetched = []

    def fake_get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False     # redirects must be followed by hand
        fetched.append(url)
        return pages[url]

    monkeypatch.setattr("requests.Session.get", fake_get)
    return fetched


def test_redirect_to_an_internal_address_is_blocked(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "93.184.216.34", "169.254.169.254": "169.254.169.254"})
    fetched = _serve(monkeypatch, {
        "https://example.com/start": _FakeResponse(302, {"Location": "http://169.254.169.254/latest/meta-data/"}),
    })
    with pytest.raises(IngestError, match="not a public website"):
        dp.load_url("https://example.com/start")
    assert fetched == ["https://example.com/start"]   # the internal URL was never requested


def test_page_is_fetched_cleaned_and_chunked(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    html = b"<html><head><script>track()</script></head><body><nav>Menu</nav><p>We open at nine.</p></body></html>"
    _serve(monkeypatch, {
        "https://example.com/": _FakeResponse(301, {"Location": "/about"}),
        "https://example.com/about": _FakeResponse(200, {"Content-Type": "text/html; charset=utf-8"}, html),
    })
    [doc] = dp.load_url("https://example.com/")
    assert doc.page_content == "We open at nine." and doc.metadata["source"] == "https://example.com/"


def test_oversized_and_binary_pages_are_rejected(monkeypatch):
    _resolve_to(monkeypatch, {"example.com": "93.184.216.34"})
    monkeypatch.setattr(dp.config, "URL_INGEST_MAX_BYTES", 1000)
    _serve(monkeypatch, {
        "https://example.com/big": _FakeResponse(200, {"Content-Type": "text/html"}, b"x" * 5000),
        "https://example.com/zip": _FakeResponse(200, {"Content-Type": "application/zip"}, b"PK"),
    })
    with pytest.raises(IngestError, match="larger than"):
        dp.load_url("https://example.com/big")
    with pytest.raises(IngestError, match="Only HTML"):
        dp.load_url("https://example.com/zip")
