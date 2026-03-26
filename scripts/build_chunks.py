# =============================================================================
# scripts/build_chunks.py
#
# PURPOSE:
#   Entry-point script for the chunking step. Reads all parsed Document objects
#   from the ingestion output, splits them into text chunks using chunking.py,
#   and writes the result to disk.
#
# INPUT:
#   data/processed/documents.jsonl
#     Produced by scripts/ingest_corpus.py. One Document JSON per line.
#
#   configs/chunking.yaml
#     Controls chunk_size, chunk_overlap, and min_chunk_chars.
#
# OUTPUT:
#   data/processed/chunks.jsonl
#     One JSON line per chunk. Each line contains chunk_id, document_id,
#     source metadata, section info, and the chunk text itself.
#
# USAGE:
#   python scripts/build_chunks.py
# =============================================================================

import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.document import Document
from src.ingestion.chunking import chunk_document

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = BASE_DIR / "data/processed/documents.jsonl"
CHUNKS_PATH    = BASE_DIR / "data/processed/chunks.jsonl"
CONFIG_PATH    = BASE_DIR / "configs/chunking.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_documents(path: Path) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.model_validate_json(line))
    return docs


def main():
    print("=" * 60)
    print("Chunking Pipeline")
    print("=" * 60)

    config = load_config(CONFIG_PATH)
    chunk_size    = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    min_chars     = config["min_chunk_chars"]

    print(f"[CONFIG] chunk_size={chunk_size}, overlap={chunk_overlap}, min_chars={min_chars}")

    documents = load_documents(DOCUMENTS_PATH)
    print(f"[INFO]   Loaded {len(documents)} documents")

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap, min_chars)
        all_chunks.extend(chunks)

    print(f"[INFO]   Total chunks: {len(all_chunks)}")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[DONE]   Saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
