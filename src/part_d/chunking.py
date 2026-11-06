"""Deterministic word-based policy chunking."""
from typing import Any, Dict, List

CHUNK_SIZE: int = 40
CHUNK_OVERLAP: int = 10

EXPECTED_CHUNK_COUNTS: Dict[str, int] = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2,
}


def chunk_document(
    text: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Chunk a single document using deterministic word windowing.
    
    Algorithm:
      - Split on whitespace into words.
      - First chunk spans words 0 to min(chunk_size, len(words)).
      - Subsequent chunks advance by step = chunk_size - chunk_overlap.
      - Stop after the first chunk whose end reaches or passes the last word.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap if chunk_size > chunk_overlap else chunk_size
    chunks: List[Dict[str, Any]] = []
    start = 0

    while True:
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "source": filename,
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })
        if end >= len(words):
            break
        start += step

    return chunks


def chunk_policy_corpus(
    documents: Dict[str, str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Chunk an entire dictionary of policy documents."""
    all_chunks: List[Dict[str, Any]] = []
    for filename, text in documents.items():
        doc_chunks = chunk_document(text, filename, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks
