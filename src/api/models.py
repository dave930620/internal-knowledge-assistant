# =============================================================================
# src/api/models.py
#
# PURPOSE:
#   Pydantic request/response models for the FastAPI layer. Defines the exact
#   shape of data coming in from the client and going back in the response.
# =============================================================================

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    chunk_id:      str
    title:         str
    section_title: str
    source_url:    str
    rerank_score:  float


class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceItem]
