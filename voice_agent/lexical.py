"""
BM25 keyword search over the knowledge-base chunks.

Embeddings capture meaning but blur exact terms: product codes ("OPG"),
prices, names, and words in romanised Hindi or Telugu that the embedding model
never learned. A keyword index catches exactly those, and the two rankings are
fused in :mod:`voice_agent.rag_pipeline`.

Besides the BM25 score, every hit reports its **coverage**: the share of the
query's informative weight (IDF) that the chunk matches. Matching the rare word
"OPG" covers most of "OPG x-ray price?"; matching the common word "need" covers
almost nothing of "How much memory does the server need?". Fusion uses coverage
so that weak, incidental keyword matches cannot outvote the embedding ranking.

Pure Python. Indexing 10,000 chunks takes well under a second; a query is a
dictionary walk over the query's terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("english")
except ImportError:  # pragma: no cover - fallback when the optional stemmer is missing
    _STEMMER = None

_DIGIT_GROUP = re.compile(r"(?<=\d)[,](?=\d)")
_TOKEN = re.compile(r"[a-z0-9ऀ-෿]+")

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it
its me my no not of on or our please should so than that the their them then there these they this
those to was we were what when where which who whom why will with would you your yours about any
just also tell know want like get much many some more most very really
ka ki ke kya hai hain ho ko se mein me kaise kitna kitni kitne kab kahan aur bhi toh ya nahi hota
hoti karna karo chahiye tha thi wala wali hum aap ap ji
""".split())


@lru_cache(maxsize=50_000)
def _stem(token: str) -> str:
    """retrieval/retrieves → retriev, aligners → align, braces → brace."""
    if token.isdigit():
        return token
    if _STEMMER is not None:
        return _STEMMER.stemWord(token)
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = _DIGIT_GROUP.sub("", text.lower().replace("₹", " rupees "))
    return [_stem(token) for token in _TOKEN.findall(text) if token not in STOPWORDS]


@dataclass(frozen=True)
class LexicalHit:
    id: str
    score: float
    coverage: float = 0.0     # share of the query's IDF weight this chunk matches, 0–1


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over a fixed set of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[int]] = {}
        self._avg_length = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, ids: list[str], texts: list[str]) -> BM25Index:
        self._ids = list(ids)
        self._term_freqs = [Counter(tokenize(text)) for text in texts]
        self._lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        document_freq: Counter[str] = Counter()
        self._postings = {}
        for position, freqs in enumerate(self._term_freqs):
            for term in freqs:
                document_freq[term] += 1
                self._postings.setdefault(term, []).append(position)
        total = len(self._ids)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in document_freq.items()
        }
        return self

    def search(self, query: str, limit: int = 20) -> list[LexicalHit]:
        if not self._ids:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        # A query word the corpus never uses is maximally informative: it counts
        # in the denominator, so chunks cannot claim full coverage without it.
        rarest = max(self._idf.values(), default=1.0)
        total_weight = sum(self._idf.get(term, rarest) for term in terms)

        scores: dict[int, float] = {}
        matched: dict[int, float] = {}
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for position in self._postings[term]:
                freq = self._term_freqs[position][term]
                norm = 1 - self.b + self.b * self._lengths[position] / (self._avg_length or 1)
                scores[position] = scores.get(position, 0.0) + idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
                matched[position] = matched.get(position, 0.0) + idf
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            LexicalHit(
                id=self._ids[position],
                score=round(score, 4),
                coverage=round(matched[position] / total_weight, 4) if total_weight else 0.0,
            )
            for position, score in ranked
        ]
