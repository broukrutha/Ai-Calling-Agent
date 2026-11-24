"""BM25 keyword search."""

from voice_agent.lexical import BM25Index, tokenize


def test_tokenize_folds_numbers_currency_stopwords_and_suffixes():
    assert tokenize("Cleaning costs ₹1,200!") == ["clean", "cost", "rupe", "1200"]
    assert tokenize("cleaning ka price kya hai?") == ["clean", "price"]
    assert tokenize("What are the braces and aligners?") == ["brace", "align"]
    assert tokenize("retrieval retrieves") == ["retriev", "retriev"]


def _index():
    return BM25Index().build(
        ["opg", "cleaning", "server"],
        [
            "Full-mouth OPG X-ray: 800 rupees.",
            "Teeth cleaning (scaling and polishing): 1,200 rupees.",
            "Start the server with python app.py. Audio is not stored on the server.",
        ],
    )


def test_rare_terms_rank_first():
    hits = _index().search("OPG x-ray price?")
    assert hits[0].id == "opg"
    assert hits[0].coverage > 0.6


def test_coverage_is_low_for_incidental_matches():
    [hit] = _index().search("How much memory does the server need?")
    assert hit.id == "server"
    assert hit.coverage < 0.5          # "memory" is unknown to the corpus and counts against it


def test_code_mixed_query_matches():
    assert _index().search("cleaning ka price kya hai")[0].id == "cleaning"


def test_empty_index_and_query():
    assert BM25Index().search("anything") == []
    assert _index().search("the and of") == []
