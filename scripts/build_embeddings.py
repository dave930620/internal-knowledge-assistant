# =============================================================================
# scripts/build_embeddings.py
#
# PURPOSE:
#   Entry-point script for the embedding step. Reads all chunks, encodes each
#   chunk's text into a dense vector using a sentence-transformer model, and
#   saves the vectors and their corresponding chunk IDs to disk.
#
# INPUT:
#   data/processed/chunks.jsonl
#     Produced by scripts/build_chunks.py. One chunk JSON per line.
#
#   configs/embedding.yaml
#     Controls model_name, batch_size, device, normalize_embeddings.
#
# OUTPUT:
#   data/embeddings/vectors.npy
#     Float32 numpy array of shape (N, D).
#     N = total number of chunks, D = embedding dimension (384).
#
#   data/embeddings/chunk_ids.json
#     Ordered list of chunk_id strings. Position i in this list corresponds
#     to row i in vectors.npy. Used by the FAISS index to map hits back to
#     chunk metadata.
#
# USAGE:
#   python scripts/build_embeddings.py
# =============================================================================

import json
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embedder import Embedder

BASE_DIR       = Path(__file__).resolve().parent.parent
CHUNKS_PATH    = BASE_DIR / "data/processed/chunks.jsonl"
EMBEDDINGS_DIR = BASE_DIR / "data/embeddings"
VECTORS_PATH   = EMBEDDINGS_DIR / "vectors.npy"
IDS_PATH       = EMBEDDINGS_DIR / "chunk_ids.json"
CONFIG_PATH    = BASE_DIR / "configs/embedding.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    print("=" * 60)
    print("Embedding Pipeline")
    print("=" * 60)

    config = load_config(CONFIG_PATH)
    print(f"[CONFIG] model={config['model_name']}, batch={config['batch_size']}, device={config['device']}")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"[INFO]   Loaded {len(chunks)} chunks")

    texts     = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    embedder = Embedder(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        device=config["device"],
        normalize_embeddings=config["normalize_embeddings"],
    )

    vectors = embedder.encode(texts)
    print(f"[INFO]   Vectors shape: {vectors.shape}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    print(f"[INFO]   Saved vectors -> {VECTORS_PATH}")

    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)
    print(f"[INFO]   Saved chunk IDs -> {IDS_PATH}")

    print(f"\n[DONE]   {len(chunks)} chunks embedded  |  shape={vectors.shape}")


if __name__ == "__main__":
    main()
