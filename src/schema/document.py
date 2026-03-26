# =============================================================================
# src/schema/document.py
#
# PURPOSE:
#   Defines the core Pydantic data models used throughout the pipeline.
#   All ingestion, chunking, and retrieval steps pass data using these models
#   to ensure consistent structure and type validation.
#
# MODELS:
#   Section   — one heading-delimited block of content within a document
#   Document  — a full parsed document with metadata, raw text, and sections
#
# INPUT / OUTPUT:
#   This file has no I/O of its own. It is imported by other modules:
#     - src/ingestion/build_documents.py  (creates Document objects)
#     - src/ingestion/chunking.py         (reads Document to produce chunks)
#     - src/retrieval/                    (uses metadata fields for filtering)
# =============================================================================

from typing import List, Optional
from pydantic import BaseModel


class Section(BaseModel):
    section_id: str
    section_title: str
    section_level: int
    content: str


class Document(BaseModel):
    document_id: str
    source_type: str
    doc_type: str

    title: str
    url: Optional[str]
    file_path: str

    updated_at: Optional[str]
    version: Optional[str]

    raw_text: str
    sections: List[Section]

    metadata: dict