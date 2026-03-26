# =============================================================================
# src/reranker/reranker.py
#
# PURPOSE:
#   Re-scores a list of candidate chunks retrieved by FAISS using a
#   cross-encoder model. A cross-encoder reads the query and each chunk
#   together (as a pair), producing a more accurate relevance score than
#   the embedding-based cosine similarity used in the first-stage retrieval.
#
#   This is the second stage of the two-stage retrieval pipeline:
#     Stage 1 — FAISS (fast, approximate)  →  top_k candidates
#     Stage 2 — Reranker (slow, precise)   →  top_n final chunks for LLM
#
# INPUT:
#   - query:   str                    the user's question
#   - chunks:  list[dict]             candidate chunks from FAISS, each with
#                                     at minimum a "text" field
#   - top_n:   int                    how many chunks to keep after reranking
#
# OUTPUT:
#   - list[dict]: the top_n chunks sorted by descending reranker score,
#     each chunk dict extended with a "rerank_score" field
# =============================================================================

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        print(f"[Reranker] Loading model: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]
