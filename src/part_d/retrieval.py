"""Retrieval interface for policy documents."""
from typing import Any, Callable, Dict, List, Optional
from src.part_d.document_loader import load_policy_documents
from src.part_d.chunking import chunk_policy_corpus
from src.part_d.vector_store import PolicyVectorStore

_DEFAULT_VECTOR_STORE: Optional[PolicyVectorStore] = None


def get_default_vector_store(policies_dir: str = "data/policies") -> PolicyVectorStore:
    """Get or initialize the standard overlapping (overlap=10) PolicyVectorStore."""
    global _DEFAULT_VECTOR_STORE
    if _DEFAULT_VECTOR_STORE is None:
        docs = load_policy_documents(policies_dir)
        chunks = chunk_policy_corpus(docs, chunk_size=40, chunk_overlap=10)
        _DEFAULT_VECTOR_STORE = PolicyVectorStore(chunks)
    return _DEFAULT_VECTOR_STORE


def get_zero_overlap_vector_store(policies_dir: str = "data/policies") -> PolicyVectorStore:
    """Initialize a zero-overlap (overlap=0) PolicyVectorStore for comparative evaluation."""
    docs = load_policy_documents(policies_dir)
    chunks = chunk_policy_corpus(docs, chunk_size=40, chunk_overlap=0)
    return PolicyVectorStore(chunks)


def retrieve(query: str, k: int = 3, policies_dir: str = "data/policies") -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from the default policy vector store."""
    store = get_default_vector_store(policies_dir)
    return store.retrieve(query, k=k)
