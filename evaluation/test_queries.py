"""
test_queries.py – Sample test query suite with expected answer keywords.
Used by the benchmark and accuracy evaluator.
"""

from typing import List, Dict, Any

# Structure: {query, expected_keywords, category}
TEST_QUERIES: List[Dict[str, Any]] = [
    # General
    {
        "query": "What is this service about?",
        "expected_keywords": ["service", "assistant", "ai", "help"],
        "category": "general",
    },
    {
        "query": "How can I contact customer support?",
        "expected_keywords": ["contact", "support", "email", "help"],
        "category": "support",
    },
    {
        "query": "What languages are supported?",
        "expected_keywords": ["language", "english", "hindi", "indian"],
        "category": "features",
    },
    {
        "query": "How do I get started with this service?",
        "expected_keywords": ["start", "sign", "register", "account", "begin"],
        "category": "onboarding",
    },
    {
        "query": "What are the main features of this product?",
        "expected_keywords": ["feature", "capability", "function"],
        "category": "features",
    },
    # Pricing
    {
        "query": "How much does it cost?",
        "expected_keywords": ["cost", "price", "free", "plan", "rupee"],
        "category": "pricing",
    },
    {
        "query": "Is there a free trial available?",
        "expected_keywords": ["trial", "free", "credit", "test"],
        "category": "pricing",
    },
    {
        "query": "What subscription plans are available?",
        "expected_keywords": ["plan", "subscription", "starter", "pro", "business"],
        "category": "pricing",
    },
    # Technical
    {
        "query": "What audio formats are supported?",
        "expected_keywords": ["wav", "mp3", "audio", "format"],
        "category": "technical",
    },
    {
        "query": "What is the response latency of the system?",
        "expected_keywords": ["latency", "second", "millisecond", "real-time", "fast"],
        "category": "technical",
    },
    {
        "query": "How is user data protected?",
        "expected_keywords": ["data", "privacy", "secure", "protect", "encrypt"],
        "category": "security",
    },
    {
        "query": "Can I integrate this API with my application?",
        "expected_keywords": ["api", "integrate", "sdk", "endpoint", "application"],
        "category": "integration",
    },
    # Voice-specific
    {
        "query": "What speech-to-text model is used?",
        "expected_keywords": ["saaras", "stt", "speech", "transcription", "model"],
        "category": "technical",
    },
    {
        "query": "What text-to-speech voices are available?",
        "expected_keywords": ["bulbul", "tts", "voice", "speaker"],
        "category": "technical",
    },
    {
        "query": "Does the assistant support multiple Indian languages?",
        "expected_keywords": ["hindi", "tamil", "telugu", "bengali", "indian", "language"],
        "category": "features",
    },
    # RAG-specific
    {
        "query": "How does the knowledge base work?",
        "expected_keywords": ["knowledge", "document", "rag", "retrieval", "pdf", "database"],
        "category": "rag",
    },
    {
        "query": "Can I upload PDF documents?",
        "expected_keywords": ["pdf", "upload", "document", "file"],
        "category": "rag",
    },
    {
        "query": "How accurate are the responses?",
        "expected_keywords": ["accurate", "accuracy", "reliable", "correct"],
        "category": "accuracy",
    },
    {
        "query": "What is Sarvam AI?",
        "expected_keywords": ["sarvam", "ai", "india", "language", "model"],
        "category": "general",
    },
    {
        "query": "How do I add new documents to the system?",
        "expected_keywords": ["ingest", "upload", "add", "document", "pdf", "url"],
        "category": "rag",
    },
]


def get_queries_by_category(category: str) -> List[Dict[str, Any]]:
    return [q for q in TEST_QUERIES if q["category"] == category]


def get_all_query_texts() -> List[str]:
    return [q["query"] for q in TEST_QUERIES]
