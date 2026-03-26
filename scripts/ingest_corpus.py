# =============================================================================
# scripts/ingest_corpus.py
#
# PURPOSE:
#   Entry-point script for the ingestion pipeline. Calls build_documents.py
#   to parse all raw HTML files and write structured Document objects to disk.
#   Run this script after download_raw.py has populated the raw data folder.
#
# INPUT:
#   Reads indirectly via build_documents.py:
#     data/raw/manifests/*.jsonl          (manifest records)
#     data/raw/**/*.html                  (raw HTML files)
#
# OUTPUT:
#   data/processed/documents.jsonl
#     One JSON line per document (see build_documents.py for full schema).
#
# USAGE:
#   python scripts/ingest_corpus.py
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.build_documents import build_all_documents, save_documents, OUTPUT_PATH


def main():
    print("=" * 60)
    print("Ingestion Pipeline")
    print("=" * 60)

    documents = build_all_documents()

    if not documents:
        print("[ERROR] No documents built. Check manifest paths and raw files.")
        sys.exit(1)

    save_documents(documents, OUTPUT_PATH)

    print()
    print(f"[DONE] {len(documents)} documents written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
