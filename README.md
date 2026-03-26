# Internal Knowledge Assistant

**RAG-based semantic search, reranking, and grounded QA over engineering documents**

---

## Overview

This project builds an internal knowledge assistant for engineering teams. Knowledge in real organizations is scattered across documentation, hard to find with keyword search, and difficult to synthesize into actionable answers.

This system solves that with a **multi-stage retrieval pipeline**:

1. User submits a question
2. Query is embedded into a dense vector
3. FAISS retrieves top-K candidate chunks by cosine similarity
4. A cross-encoder reranker re-scores and selects top-N chunks
5. An LLM generates a grounded answer with source citations

This is not a chatbot. It is a **retrieval + reasoning system** designed for real-world engineering workflows.

---

## Problem Statement

Engineering teams struggle with:

- Finding the right documentation quickly
- Understanding past design decisions
- Debugging incidents using historical knowledge
- Navigating large, heterogeneous knowledge bases

Traditional keyword search fails because semantic relevance is not the same as lexical overlap.

---

## Architecture

```
[Document Sources]
        |
[download_raw.py]        fetch HTML from engineering blogs and official docs
        |
[ingest_corpus.py]       parse HTML → structured Document objects
        |
[build_chunks.py]        section-aware sliding-window chunking
        |
[build_embeddings.py]    sentence-transformer → float32 vectors
        |
[build_faiss_index.py]   FAISS FlatIP index (cosine via normalized vectors)
        |
   [POST /query]
        |
   [Embedder]            encode query
        |
   [FAISS Retriever]     top_k=10 candidates
        |
   [Cross-Encoder]       rerank → top_n=5
        |
   [LLM Generator]       grounded answer + citations
        |
   [JSON Response]       { answer, sources[] }
```

---

## Data Sources

Real engineering documentation scraped from:

| Source | Type | Documents |
|---|---|---|
| AWS Architecture Blog | Engineering blog | 2 |
| Cloudflare Blog | Engineering blog | 3 |
| Kubernetes Docs | Official docs | 4 |
| Stripe Docs | Official docs | 2 |
| Kubernetes Troubleshooting | Ops runbook | 3 |

Organized into three categories: `engineering_blog`, `official_doc`, `ops_troubleshooting`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | BeautifulSoup, lxml |
| Data models | Pydantic v2 |
| Embedding | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector index | FAISS (`IndexFlatIP`) |
| Reranking | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | OpenAI API (`gpt-4o-mini`) |
| API | FastAPI + Uvicorn |
| Config | YAML per component |

---

## Project Structure

```
internal-knowledge-assistant/
│
├── configs/
│   ├── chunking.yaml          chunk_size=512, overlap=64
│   ├── embedding.yaml         model, batch_size, device
│   ├── retrieval.yaml         top_k=10, index_type=FlatIP
│   ├── reranker.yaml          model, top_n=5
│   └── generation.yaml        model, max_tokens, system_prompt
│
├── data/
│   ├── raw/                   downloaded HTML files + manifests
│   ├── processed/             documents.jsonl, chunks.jsonl
│   └── embeddings/            vectors.npy, chunk_ids.json, faiss.index
│
├── src/
│   ├── schema/
│   │   └── document.py        Document and Section Pydantic models
│   ├── ingestion/
│   │   ├── parser.py          HTML → {title, raw_text, sections[]}
│   │   ├── build_documents.py manifests + HTML → Document objects
│   │   └── chunking.py        Document → overlapping text chunks
│   ├── embedding/
│   │   └── embedder.py        SentenceTransformer wrapper
│   ├── retrieval/
│   │   └── faiss_index.py     build / save / load / search
│   ├── reranker/
│   │   └── reranker.py        CrossEncoder wrapper
│   ├── generation/
│   │   └── generator.py       prompt builder + OpenAI call
│   └── api/
│       ├── main.py            FastAPI app (GET /health, POST /query)
│       ├── pipeline.py        singleton components + run()
│       └── models.py          QueryRequest, QueryResponse, SourceItem
│
├── eval/
│   ├── queries.json           15 ground-truth queries (4 categories)
│   └── metrics.py             recall@k, MRR, nDCG@k
│
├── scripts/
│   ├── download_raw.py        fetch and save raw HTML
│   ├── ingest_corpus.py       parse HTML → documents.jsonl
│   ├── build_chunks.py        documents.jsonl → chunks.jsonl
│   ├── build_embeddings.py    chunks.jsonl → vectors.npy
│   ├── build_faiss_index.py   vectors.npy → faiss.index
│   └── run_eval.py            retrieval evaluation report
│
├── .env.example               API key template
├── .gitignore
└── requirements.txt
```

---

## Setup

```bash
git clone <repo>
cd internal-knowledge-assistant

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your OpenAI key:
# OPENAI_API_KEY=sk-...
```

---

## Running the Offline Pipeline

Run these once to build the knowledge index:

```bash
# 1. fetch raw HTML from documentation sources
python scripts/download_raw.py

# 2. parse HTML into structured Document objects
python scripts/ingest_corpus.py

# 3. split documents into overlapping text chunks
python scripts/build_chunks.py

# 4. embed all chunks with sentence-transformers
python scripts/build_embeddings.py

# 5. build the FAISS index
python scripts/build_faiss_index.py
```

Pipeline output at each stage:

| Step | Output | Size |
|---|---|---|
| ingest | `data/processed/documents.jsonl` | 14 documents |
| chunk | `data/processed/chunks.jsonl` | 182 chunks |
| embed | `data/embeddings/vectors.npy` | shape (182, 384) |
| index | `data/embeddings/faiss.index` | 182 vectors, dim=384 |

---

## Running the API

```bash
uvicorn src.api.main:app --reload
```

**Health check:**

```
GET /health
→ { "status": "ok" }
```

**Query endpoint:**

```
POST /query
Content-Type: application/json

{ "question": "How does Kubernetes schedule pods onto nodes?" }
```

**Response:**

```json
{
  "answer": "Kubernetes scheduling is handled by the kube-scheduler...",
  "sources": [
    {
      "chunk_id": "doc_k8s_003_sec10_c0",
      "title": "Nodes | Kubernetes",
      "section_title": "Resource capacity tracking",
      "source_url": "https://kubernetes.io/docs/concepts/architecture/nodes/",
      "rerank_score": 7.76
    }
  ]
}
```

---

## Evaluation

Run the retrieval evaluation against 15 manually written queries:

```bash
python scripts/run_eval.py
```

**Results (K=5):**

| Category | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| factual | 0.812 | 0.792 | 0.763 |
| design_reasoning | 0.667 | 0.750 | 0.668 |
| troubleshooting | 0.389 | 0.778 | 0.439 |
| **overall** | **0.689** | **0.778** | **0.673** |

Metrics used:
- **Recall@K** — fraction of relevant chunks that appear in the top-K results
- **MRR** — Mean Reciprocal Rank, 1/rank of the first relevant hit
- **nDCG@K** — normalized Discounted Cumulative Gain, accounts for rank position

Lower troubleshooting scores indicate areas for improvement: larger corpus coverage and finer-grained chunking of multi-step runbooks.

---

## Key Design Decisions

**Two-stage retrieval** — FAISS is fast but scores chunks independently of the query. The cross-encoder reranker processes (query, chunk) together for higher precision at the cost of latency. This is the standard production pattern.

**Section-aware chunking** — chunks are split within sections rather than across them, preserving semantic coherence and keeping section titles as metadata for citations.

**Normalized vectors + FlatIP** — L2-normalized embeddings make inner-product search equivalent to cosine similarity. `IndexFlatIP` performs exact search, appropriate for corpora under ~1M chunks.

**Temperature 0 generation** — grounded QA should be deterministic. The system prompt instructs the model to refuse to answer if the context is insufficient, preventing hallucination.

---

## Potential Improvements

- Hybrid search (BM25 + dense) for better lexical recall
- Query rewriting / HyDE for ambiguous questions
- Incremental indexing for new documents
- Caching frequent queries
- RAGAS-based answer faithfulness evaluation
- Larger corpus (Uber/Netflix engineering blogs, GitHub RFCs)

---

## License

MIT
