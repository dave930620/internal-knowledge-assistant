# =============================================================================
# src/embedding/embedder.py
#
# PURPOSE:
#   Loads a sentence-transformer model and encodes a list of text strings into
#   dense vector embeddings. Used by build_embeddings.py during the embedding
#   step of the pipeline.
#
# INPUT:
#   - texts: list of strings (chunk texts)
#   - Config values: model_name, batch_size, device, normalize_embeddings
#
# OUTPUT:
#   - A numpy array of shape (N, D) where:
#       N = number of input texts
#       D = embedding dimension (384 for all-MiniLM-L6-v2)
# =============================================================================

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize_embeddings
        print(f"[Embedder] Loading model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return np.array(vectors, dtype=np.float32)
