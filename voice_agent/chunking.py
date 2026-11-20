"""
Structure-aware chunking.

Generic fixed-size splitting cuts list items and sentences in half and strips
every chunk of the context it came from ("40 rupees per hour" — parking where?).
This chunker instead:

1. Splits the document into sections by its headings: Markdown ``#`` headings,
   or plain-text heading lines (a short title line that opens a paragraph).
2. Breaks each section into units — paragraphs and list items — that are never
   split unless a single unit is larger than a whole chunk.
3. Packs whole units into chunks of up to ``CHUNK_SIZE`` characters.
4. Prefixes every chunk with its location, e.g.
   ``Sunrise Dental Care › Branches and opening hours › Gachibowli branch``,
   so both the embedding and the keyword index know what the chunk is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from voice_agent import config

HEADER_SEPARATOR = " › "

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,3}[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
_TERMINAL_PUNCTUATION = tuple(".,;:!?।")


@dataclass
class Section:
    path: list[str]
    blocks: list[list[str]] = field(default_factory=list)


def _looks_like_heading(line: str) -> bool:
    """A plain-text heading: short, title-like, no sentence punctuation at the end."""
    text = line.strip()
    if not text or len(text) > 70 or text.endswith(_TERMINAL_PUNCTUATION):
        return False
    if _BULLET.match(text) or len(text.split()) > 9:
        return False
    return text[0].isalpha() and text[0].isupper()


def _blocks(lines: list[str]) -> list[list[str]]:
    """Group lines into blank-line separated blocks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_sections(text: str, default_title: str = "") -> tuple[str, list[Section]]:
    """Return the document title and its sections, in order."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = default_title
    sections: list[Section] = [Section(path=[])]
    md_stack: list[tuple[int, str]] = []

    has_markdown = any(_MD_HEADING.match(line) for line in lines)
    if has_markdown:
        body: list[str] = []
        for line in lines:
            match = _MD_HEADING.match(line)
            if not match:
                body.append(line)
                continue
            sections[-1].blocks.extend(_blocks(body))
            body = []
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1 and not md_stack and not any(s.blocks for s in sections):
                title = heading
                continue
            md_stack = [(lvl, name) for lvl, name in md_stack if lvl < level] + [(level, heading)]
            sections.append(Section(path=[name for _, name in md_stack]))
        sections[-1].blocks.extend(_blocks(body))
    else:
        blocks = _blocks(lines)
        # A one-line opening block is the document's title.
        if blocks and len(blocks[0]) == 1 and _looks_like_heading(blocks[0][0].split(" — ")[0]):
            title = blocks.pop(0)[0].strip()
        for block in blocks:
            if len(block) > 1 and _looks_like_heading(block[0]):
                sections.append(Section(path=[block[0].strip()], blocks=[block[1:]]))
            else:
                sections[-1].blocks.append(block)

    return title, [section for section in sections if section.blocks]


def _units(blocks: list[list[str]]) -> list[str]:
    """Paragraphs and list items. A list item keeps its continuation lines."""
    units: list[str] = []
    for block in blocks:
        current: list[str] = []
        in_item = False
        for line in block:
            if _BULLET.match(line):
                if current:
                    units.append((" " if in_item else "\n").join(current))
                current, in_item = [line.strip()], True
            else:
                current.append(line.strip())
        if current:
            units.append((" " if in_item else "\n").join(current))
    return [unit for unit in units if unit.strip()]


def _split_oversized(unit: str, limit: int) -> list[str]:
    """Break a unit longer than a chunk at sentence boundaries, then by characters."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(unit):
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            splitter = RecursiveCharacterTextSplitter(chunk_size=limit, chunk_overlap=min(config.CHUNK_OVERLAP, limit // 4))
            pieces.extend(splitter.split_text(sentence))
        elif len(current) + len(sentence) + 1 > limit and current:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _pack(units: list[str], limit: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        for piece in ([unit] if len(unit) <= limit else _split_oversized(unit, limit)):
            if current and size + len(piece) + 1 > limit:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_text(text: str, source: str, base_metadata: dict | None = None, title: str = "") -> list[Document]:
    """Split a document into section-aware chunks with a location header."""
    limit = max(config.CHUNK_SIZE, 200)
    doc_title, sections = parse_sections(text, default_title=title)
    documents: list[Document] = []
    for section in sections:
        header_parts = [part for part in [doc_title, *section.path] if part]
        header = HEADER_SEPARATOR.join(header_parts)
        for body in _pack(_units(section.blocks), limit):
            metadata = {**(base_metadata or {}), "source": source}
            if doc_title:
                metadata["title"] = doc_title
            if section.path:
                metadata["section"] = HEADER_SEPARATOR.join(section.path)
            metadata["chunk"] = len(documents)
            content = f"{header}\n{body}" if header else body
            documents.append(Document(page_content=content, metadata=metadata))
    return documents
