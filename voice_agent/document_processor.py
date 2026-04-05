"""
document_processor.py – Load and chunk documents from PDF, URL, or raw text.
Produces LangChain Document objects ready for embedding.
"""

import ipaddress
import logging
import socket
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from voice_agent.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    URL_INGEST_MAX_BYTES,
    URL_INGEST_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("URL must include a valid hostname.")

    hostname = parsed.hostname.strip().lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Localhost URLs are not allowed.")

    try:
        addrinfo = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{hostname}'.") from exc

    for entry in addrinfo:
        ip_text = entry[4][0]
        ip_obj = ipaddress.ip_address(ip_text)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise ValueError("Private or non-public IP addresses are not allowed.")


# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────

def load_pdf(file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
    """Parse a PDF from raw bytes and return chunked Documents."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1},
                )
            )
    doc.close()
    logger.info(f"PDF '{filename}': {len(pages)} pages loaded.")
    return _make_splitter().split_documents(pages)


def load_url(url: str, timeout: int = URL_INGEST_TIMEOUT_SEC) -> List[Document]:
    """Scrape a URL, strip HTML, and return chunked Documents."""
    _validate_public_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (VoiceAgent/1.0)"}
    with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
            raise ValueError(f"Unsupported URL content type: {content_type or 'unknown'}")

        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > URL_INGEST_MAX_BYTES:
            raise ValueError(
                f"URL response too large. Maximum allowed size is {URL_INGEST_MAX_BYTES} bytes."
            )

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > URL_INGEST_MAX_BYTES:
                raise ValueError(
                    f"URL response too large. Maximum allowed size is {URL_INGEST_MAX_BYTES} bytes."
                )
            chunks.append(chunk)
        raw_html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    raw = " ".join(soup.get_text(separator="\n").split())
    logger.info(f"URL '{url}': {len(raw)} characters scraped.")
    docs = [Document(page_content=raw, metadata={"source": url})]
    return _make_splitter().split_documents(docs)


def load_text(text: str, source_name: str = "manual_input") -> List[Document]:
    """Chunk raw plain text."""
    docs = [Document(page_content=text.strip(), metadata={"source": source_name})]
    return _make_splitter().split_documents(docs)


def load_file(path: str) -> List[Document]:
    """Auto-detect and load a local file (PDF or .txt)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".pdf":
        return load_pdf(p.read_bytes(), filename=p.name)
    else:
        return load_text(p.read_text(encoding="utf-8"), source_name=p.name)


# ──────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────

def ingest(source: str | bytes, source_type: str, name: str = "") -> List[Document]:
    """
    Unified ingestion entry point.

    Args:
        source: URL string, file path string, text string, or PDF bytes.
        source_type: One of 'url', 'pdf', 'text', 'file'.
        name: Optional display name / filename.

    Returns:
        List of chunked LangChain Document objects.
    """
    source_type = source_type.lower().strip()
    if source_type == "url":
        return load_url(str(source))
    elif source_type == "pdf":
        if isinstance(source, bytes):
            return load_pdf(source, filename=name or "uploaded.pdf")
        return load_pdf(Path(str(source)).read_bytes(), filename=name or str(source))
    elif source_type == "text":
        return load_text(str(source), source_name=name or "text_input")
    elif source_type == "file":
        return load_file(str(source))
    else:
        raise ValueError(f"Unknown source_type: '{source_type}'. Use url/pdf/text/file.")
