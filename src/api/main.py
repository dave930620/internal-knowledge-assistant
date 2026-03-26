# =============================================================================
# src/api/main.py
#
# PURPOSE:
#   FastAPI application entry point. Exposes a single POST /query endpoint
#   that accepts a natural language question and returns a grounded answer
#   with source citations from the internal knowledge corpus.
#
# INPUT  (POST /query):
#   JSON body: { "question": "..." }
#
# OUTPUT (POST /query):
#   JSON body:
#     {
#       "answer": str,
#       "sources": [
#         {
#           "chunk_id":      str,
#           "title":         str,
#           "section_title": str,
#           "source_url":    str,
#           "rerank_score":  float
#         }, ...
#       ]
#     }
#
# USAGE:
#   uvicorn src.api.main:app --reload
# =============================================================================

from fastapi import FastAPI, HTTPException
from src.api.models import QueryRequest, QueryResponse, SourceItem
from src.api import pipeline

app = FastAPI(
    title="Internal Knowledge Assistant",
    description="RAG-based semantic search and grounded QA over engineering documents.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = pipeline.run(request.question)

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )
