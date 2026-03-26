# =============================================================================
# src/ingestion/chunking.py
#
# PURPOSE:
#   Splits Document objects into small, overlapping text chunks suitable for
#   embedding and retrieval. Each chunk preserves metadata (doc_id, section
#   title, source URL, provider) so results can be traced back to their source.
#
#   Strategy:
#     1. Iterate over each section of a document.
#     2. If the section text fits within chunk_size tokens, emit it as one chunk.
#     3. If it is too long, apply a sliding window with overlap to split it.
#     4. Discard chunks shorter than min_chunk_chars (navigation noise, etc.).
#
# INPUT:
#   - A Document object (from src/schema/document.py)
#   - Config values: chunk_size (tokens), chunk_overlap (tokens),
#                    min_chunk_chars (characters)
#
# OUTPUT:
#   A list of dicts, each representing one chunk:
#     {
#       "chunk_id":      "eng_aws_001_sec0_c0",
#       "document_id":   "eng_aws_001",
#       "source_type":   "engineering_blog",
#       "provider":      "aws",
#       "source_url":    str,
#       "title":         str,   # document title
#       "section_title": str,
#       "section_level": int,
#       "text":          str,   # the chunk text
#       "chunk_index":   int    # position within the section
#     }
# =============================================================================

import re
from src.schema.document import Document


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace tokens (word-level approximation)."""
    return text.split()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sliding-window split of text into token-bounded chunks."""
    tokens = _tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap
    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_chars: int = 80,
) -> list[dict]:
    chunks = []

    for sec in doc.sections:
        text = sec.content.strip()
        if not text:
            continue

        raw_chunks = _chunk_text(text, chunk_size, chunk_overlap)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < min_chunk_chars:
                continue

            chunk_id = f"{doc.document_id}_sec{sec.section_id.split('_')[-1]}_c{idx}"

            chunks.append({
                "chunk_id":      chunk_id,
                "document_id":   doc.document_id,
                "source_type":   doc.source_type,
                "provider":      doc.metadata.get("provider", "unknown"),
                "source_url":    doc.url,
                "title":         doc.title,
                "section_title": sec.section_title.strip(),
                "section_level": sec.section_level,
                "text":          chunk_text,
                "chunk_index":   idx,
            })

    return chunks
