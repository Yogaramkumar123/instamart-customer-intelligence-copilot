"""TF-IDF vector store and cosine similarity retrieval for policy chunks."""
from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PolicyVectorStore:
    """Manages TF-IDF vectorization and cosine similarity retrieval over policy chunks."""

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        chunk_texts = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar policy chunks for a query using cosine similarity.
        
        Returns list of dicts:
          [{"source": filename, "score": float, "text": chunk_text}]
        """
        if not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Rank indices in descending order
        ranked_indices = similarities.argsort()[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(similarities[idx]),
                "text": self.chunks[idx]["text"],
            })

        return results
