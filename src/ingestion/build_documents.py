# =============================================================================
# src/ingestion/build_documents.py
#
# PURPOSE:
#   Core ingestion logic. Reads all manifest files to discover downloaded HTML
#   files, parses each one using parser.py, and constructs validated Document
#   objects (defined in src/schema/document.py). Saves all documents as a
#   single JSONL file for use by the chunking step.
#
# INPUT:
#   Manifest JSONL files (produced by download_raw.py):
#     data/raw/manifests/engineering_blogs_manifest.jsonl
#     data/raw/manifests/official_docs_manifest.jsonl
#     data/raw/manifests/ops_manifest.jsonl
#
#   Raw HTML files referenced inside each manifest record, e.g.:
#     data/raw/engineering_blogs/aws/eng_aws_001.html
#
# OUTPUT:
#   data/processed/documents.jsonl
#     One JSON line per document. Each line is a serialized Document object:
#     {
#       "document_id": "eng_aws_001",
#       "source_type": "engineering_blog",
#       "doc_type":    "html",
#       "title":       str,
#       "url":         str,
#       "file_path":   str,
#       "updated_at":  str,
#       "version":     null,
#       "raw_text":    str,
#       "sections":    [ { section_id, section_title, section_level, content } ],
#       "metadata":    { "provider": str, "source_type": str }
#     }
# =============================================================================

import json
from pathlib import Path
from datetime import datetime, timezone

from src.schema.document import Document, Section
from src.ingestion.parser import parse_html

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MANIFEST_PATHS = [
    BASE_DIR / "data/raw/manifests/engineering_blogs_manifest.jsonl",
    BASE_DIR / "data/raw/manifests/official_docs_manifest.jsonl",
    BASE_DIR / "data/raw/manifests/ops_manifest.jsonl",
]

OUTPUT_PATH = BASE_DIR / "data/processed/documents.jsonl"


def load_manifest(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        print(f"[WARN] Manifest not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_document(record: dict) -> Document | None:
    raw_path = BASE_DIR / record["raw_path"]

    if not raw_path.exists():
        print(f"[WARN] File not found: {raw_path}")
        return None

    parsed = parse_html(raw_path)

    title = parsed["title"] or record.get("title") or record["doc_id"]

    sections = [
        Section(
            section_id=s["section_id"],
            section_title=s["section_title"],
            section_level=s["section_level"],
            content=s["content"],
        )
        for s in parsed["sections"]
    ]

    doc = Document(
        document_id=record["doc_id"],
        source_type=record["source_type"],
        doc_type=record.get("content_format", "html"),
        title=title,
        url=record.get("source_url"),
        file_path=record["raw_path"],
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        version=None,
        raw_text=parsed["raw_text"],
        sections=sections,
        metadata={
            "provider": record.get("provider", "unknown"),
            "source_type": record["source_type"],
        },
    )

    return doc


def build_all_documents() -> list[Document]:
    all_records = []
    for manifest_path in MANIFEST_PATHS:
        records = load_manifest(manifest_path)
        all_records.extend(records)

    print(f"[INFO] Total manifest records: {len(all_records)}")

    documents = []
    for record in all_records:
        doc = build_document(record)
        if doc:
            documents.append(doc)

    print(f"[INFO] Successfully built: {len(documents)} documents")
    return documents


def save_documents(documents: list[Document], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
    print(f"[INFO] Saved to {output_path}")
