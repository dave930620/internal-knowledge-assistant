# =============================================================================
# src/retrieval/faiss_index.py
#
# PURPOSE:
#   Builds a FAISS index from pre-computed embedding vectors and provides a
#   search interface to retrieve the top-K most similar chunks for a query
#   vector. Because vectors are L2-normalized during embedding, inner-product
#   search (FlatIP) is equivalent to cosine similarity.
#
# INPUT (build):
#   - vectors: numpy array of shape (N, D), float32, L2-normalized
#   - index_type: "FlatIP" (exact search; suitable for our corpus size)
#
# OUTPUT (build):
#   - data/embeddings/faiss.index   (serialized FAISS index file)
#
# INPUT (search):
#   - query_vector: numpy array of shape (1, D), float32, L2-normalized
#   - top_k: number of results to return
#
# OUTPUT (search):
#   - List of (chunk_id, score) tuples sorted by descending similarity
# =============================================================================

import json
import numpy as np
import faiss
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = BASE_DIR / "data/embeddings/faiss.index"
IDS_PATH   = BASE_DIR / "data/embeddings/chunk_ids.json"


def build_index(vectors: np.ndarray, index_type: str = "FlatIP") -> faiss.Index:
    dim = vectors.shape[1]
    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")
    index.add(vectors)
    print(f"[FaissIndex] Built {index_type} index  |  {index.ntotal} vectors  |  dim={dim}")
    return index


def save_index(index: faiss.Index, path: Path = INDEX_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"[FaissIndex] Saved index -> {path}")


def load_index(path: Path = INDEX_PATH) -> faiss.Index:
    index = faiss.read_index(str(path))
    print(f"[FaissIndex] Loaded index  |  {index.ntotal} vectors")
    return index


def load_chunk_ids(path: Path = IDS_PATH) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search(
    index: faiss.Index,
    chunk_ids: list[str],
    query_vector: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Returns a list of (chunk_id, score) sorted by descending similarity.
    query_vector must be shape (1, D) and L2-normalized.
    """
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunk_ids[idx], float(score)))
    return results
