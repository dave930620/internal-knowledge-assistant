# =============================================================================
# scripts/build_faiss_index.py
#
# PURPOSE:
#   Entry-point script for the FAISS index build step. Loads the pre-computed
#   embedding vectors, constructs a FAISS index, and writes it to disk.
#   After this step the retrieval layer can perform fast similarity search.
#
# INPUT:
#   data/embeddings/vectors.npy
#     Float32 numpy array of shape (N, D). Produced by build_embeddings.py.
#
#   configs/retrieval.yaml
#     Controls index_type and embedding_dim.
#
# OUTPUT:
#   data/embeddings/faiss.index
#     Serialized FAISS index. Loaded at query time by src/retrieval/faiss_index.py.
#
# USAGE:
#   python scripts/build_faiss_index.py
# =============================================================================

import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.faiss_index import build_index, save_index

BASE_DIR     = Path(__file__).resolve().parent.parent
VECTORS_PATH = BASE_DIR / "data/embeddings/vectors.npy"
CONFIG_PATH  = BASE_DIR / "configs/retrieval.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("FAISS Index Build")
    print("=" * 60)

    config = load_config(CONFIG_PATH)
    index_type = config["index_type"]
    expected_dim = config["embedding_dim"]

    vectors = np.load(VECTORS_PATH)
    print(f"[INFO]   Loaded vectors: shape={vectors.shape}, dtype={vectors.dtype}")

    if vectors.shape[1] != expected_dim:
        print(f"[ERROR]  Dimension mismatch: got {vectors.shape[1]}, expected {expected_dim}")
        sys.exit(1)

    index = build_index(vectors, index_type=index_type)
    save_index(index)

    print(f"\n[DONE]   FAISS index ready  |  {index.ntotal} vectors indexed")


if __name__ == "__main__":
    main()
