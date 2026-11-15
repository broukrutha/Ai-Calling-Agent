"""
Turn PDFs, web pages and raw text into chunks ready for embedding.

URL ingestion fetches attacker-influenced addresses from inside your network,
so it is guarded against server-side request forgery: only public http(s)
hosts are allowed, and that check is repeated for every redirect hop.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from voice_agent import config

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5
_MAX_PDF_PAGES = 500
_ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
_USER_AGENT = "Mozilla/5.0 (compatible; VoiceAgentBot/1.0)"


class IngestError(ValueError):
    """The source was rejected or could not be read. The message is user-safe."""


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "। ", " ", ""],
    )


# ── URL safety ───────────────────────────────────────────────────────────────

def validate_public_url(url: str) -> None:
    """Raise :class:`IngestError` unless ``url`` points at a public web host."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise IngestError("Only http:// and https:// URLs are allowed.")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise IngestError("The URL must include a hostname.")
    if parsed.username or parsed.password:
        raise IngestError("URLs with embedded credentials are not allowed.")

    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)}
    except socket.gaierror as exc:
        raise IngestError(f"Could not resolve host '{hostname}'.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise IngestError("That address is not a public website.")


def _fetch_public(url: str, timeout: int) -> requests.Response:
    """GET with redirects followed by hand so every hop is validated."""
    session = requests.Session()
    session.trust_env = config.HTTP_TRUST_ENV
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validate_public_url(current)
        resp = session.get(
            current, headers={"User-Agent": _USER_AGENT}, timeout=timeout, stream=True, allow_redirects=False
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location", "")
            resp.close()
            if not location:
                raise IngestError("The site sent a redirect without a destination.")
            current = urljoin(current, location)
            continue
        return resp
    raise IngestError("Too many redirects.")


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_pdf(file_bytes: bytes, filename: str = "document.pdf") -> list[Document]:
    """Extract text from a PDF (bytes) and chunk it."""
    if not file_bytes.startswith(b"%PDF"):
        raise IngestError("That file is not a valid PDF.")
    import pymupdf

    try:
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise IngestError("The PDF could not be opened. It may be corrupted.") from exc

    try:
        if pdf.needs_pass:
            raise IngestError("Password-protected PDFs are not supported.")
        if pdf.page_count > _MAX_PDF_PAGES:
            raise IngestError(f"PDFs are limited to {_MAX_PDF_PAGES} pages.")
        pages = [
            Document(page_content=text, metadata={"source": filename, "page": number})
            for number, page in enumerate(pdf, start=1)
            if (text := page.get_text("text").strip())
        ]
    finally:
        pdf.close()

    if not pages:
        raise IngestError("No text was found in the PDF. Scanned documents need OCR first.")
    logger.info("PDF '%s': %d pages with text.", filename, len(pages))
    return _splitter().split_documents(pages)


def load_url(url: str, timeout: int | None = None) -> list[Document]:
    """Download a public web page, strip the markup and chunk the text."""
    limit = config.URL_INGEST_MAX_BYTES
    try:
        resp = _fetch_public(url, timeout or config.URL_INGEST_TIMEOUT_SEC)
    except requests.RequestException as exc:
        raise IngestError("The page could not be downloaded.") from exc

    with resp:
        if resp.status_code >= 400:
            raise IngestError(f"The site responded with HTTP {resp.status_code}.")
        content_type = resp.headers.get("Content-Type", "").lower()
        if not any(kind in content_type for kind in _ALLOWED_CONTENT_TYPES):
            raise IngestError("Only HTML and plain-text pages can be ingested.")

        body = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                body.extend(chunk)
                if len(body) > limit:
                    raise IngestError(f"The page is larger than the {limit // 1024} KB limit.")
        except requests.RequestException as exc:
            raise IngestError("The download was interrupted.") from exc
        html = bytes(body).decode(resp.encoding or "utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator="\n").split())
    if not text:
        raise IngestError("No readable text was found on that page.")
    logger.info("URL ingested: %d characters.", len(text))
    return _splitter().split_documents([Document(page_content=text, metadata={"source": url})])


def load_text(text: str, source_name: str = "manual_input") -> list[Document]:
    text = (text or "").strip()
    if not text:
        raise IngestError("There is no text to ingest.")
    return _splitter().split_documents([Document(page_content=text, metadata={"source": source_name})])


def load_file(path: str) -> list[Document]:
    """Load a local ``.pdf`` or text file (used by the seeding script)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"File not found: {path}")
    if file_path.suffix.lower() == ".pdf":
        return load_pdf(file_path.read_bytes(), filename=file_path.name)
    return load_text(file_path.read_text(encoding="utf-8"), source_name=file_path.name)


def ingest(source: str | bytes, source_type: str, name: str = "") -> list[Document]:
    """
    Single entry point: ``source_type`` is ``url``, ``pdf``, ``text`` or ``file``.

    Raises :class:`IngestError` with a message that is safe to show the user.
    """
    kind = source_type.lower().strip()
    if kind == "url":
        return load_url(str(source))
    if kind == "pdf":
        data = source if isinstance(source, bytes) else Path(str(source)).read_bytes()
        return load_pdf(data, filename=name or "uploaded.pdf")
    if kind == "text":
        return load_text(str(source), source_name=name or "text_input")
    if kind == "file":
        return load_file(str(source))
    raise IngestError(f"Unknown source type '{source_type}'.")
