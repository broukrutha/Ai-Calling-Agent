"""Structure-aware chunking."""

from voice_agent import config
from voice_agent.chunking import HEADER_SEPARATOR, chunk_text, parse_sections

MARKDOWN = """# Sunrise Dental

Intro line.

## Branches

### Gachibowli
Open Sunday 10 to 2.
Paid parking: 40 rupees per hour.

## Prices

- Cleaning: 1,200 rupees.
- Root canal: 6,500 rupees for a front tooth
  and 9,000 rupees for a molar.
"""

PLAIN = """Acme Voice Agent — Handbook

About This System
This is a voice agent.
It answers questions.

Pricing Information
STT costs Rs 30 per hour.
"""


def test_markdown_sections_and_title():
    title, sections = parse_sections(MARKDOWN)
    assert title == "Sunrise Dental"
    assert [s.path for s in sections] == [[], ["Branches", "Gachibowli"], ["Prices"]]


def test_plain_text_headings_are_detected():
    title, sections = parse_sections(PLAIN)
    assert title == "Acme Voice Agent — Handbook"
    assert [s.path for s in sections] == [["About This System"], ["Pricing Information"]]


def test_every_chunk_carries_its_location():
    docs = chunk_text(MARKDOWN, source="clinic.md")
    parking = next(d for d in docs if "40 rupees" in d.page_content)
    assert parking.page_content.startswith(f"Sunrise Dental{HEADER_SEPARATOR}Branches{HEADER_SEPARATOR}Gachibowli\n")
    assert parking.metadata == {"source": "clinic.md", "title": "Sunrise Dental", "section": f"Branches{HEADER_SEPARATOR}Gachibowli", "chunk": 1}


def test_list_items_keep_their_continuation_lines(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    docs = chunk_text(MARKDOWN, source="clinic.md")
    canal = [d for d in docs if "Root canal" in d.page_content]
    assert len(canal) == 1 and "9,000 rupees for a molar" in canal[0].page_content


def test_list_items_are_never_cut_in_half(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    items = "\n".join(f"- Item {n}: costs {n * 100} rupees and includes a free follow-up visit." for n in range(1, 12))
    docs = chunk_text(f"# Prices\n\n## List\n\n{items}\n", source="p.md")
    assert len(docs) > 1
    for n in range(1, 12):
        assert sum(f"Item {n}:" in d.page_content for d in docs) == 1
        assert any(f"Item {n}: costs {n * 100} rupees and includes a free follow-up visit." in d.page_content for d in docs)


def test_oversized_paragraph_splits_at_sentences(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    paragraph = " ".join(f"Sentence number {n} is here." for n in range(40))
    docs = chunk_text(paragraph, source="long.txt")
    assert len(docs) > 3
    assert all(len(d.page_content) <= 200 for d in docs)
    assert all(d.page_content.rstrip().endswith(".") for d in docs)


def test_short_sentence_lines_are_not_mistaken_for_headings():
    title, sections = parse_sections("Intro\n\nPython 3.10 or higher.\n4GB RAM minimum.\n")
    assert all(not s.path for s in sections)
