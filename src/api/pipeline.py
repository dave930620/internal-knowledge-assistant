# =============================================================================
# src/api/pipeline.py
#
# PURPOSE:
#   Loads all pipeline components once at server startup and exposes a single
#   run() function that executes the full query pipeline:
#
#     query → embed → FAISS retrieve → rerank → generate → response
#
#   Components are loaded as module-level singletons so the model weights are
#   only read from disk once, not on every request.
#
# INPUT:
#   - query: str    the user's question
#
# OUTPUT:
#   - dict { "answer": str, "sources": list[dict] }
# =============================================================================

import json
import yaml
import numpy as np
from pathlib import Path

from src.embedding.embedder import Embedder
from src.retrieval.faiss_index import load_index, load_chunk_ids, search
from src.reranker.reranker import Reranker
from src.generation.generator import Generator

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── load configs ──────────────────────────────────────────────────────────────

def _load_yaml(name: str) -> dict:
    with open(BASE_DIR / "configs" / name) as f:
        return yaml.safe_load(f)

retrieval_cfg  = _load_yaml("retrieval.yaml")
embedding_cfg  = _load_yaml("embedding.yaml")
reranker_cfg   = _load_yaml("reranker.yaml")
generation_cfg = _load_yaml("generation.yaml")

# ── load chunk lookup (chunk_id → chunk dict) ─────────────────────────────────

_chunks_by_id: dict[str, dict] = {}
with open(BASE_DIR / "data/processed/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        _chunks_by_id[c["chunk_id"]] = c

# ── initialise components (runs once on import) ───────────────────────────────

print("[Pipeline] Loading embedder ...")
_embedder = Embedder(
    model_name=embedding_cfg["model_name"],
    batch_size=embedding_cfg["batch_size"],
    device=embedding_cfg["device"],
    normalize_embeddings=embedding_cfg["normalize_embeddings"],
)

print("[Pipeline] Loading FAISS index ...")
_index     = load_index()
_chunk_ids = load_chunk_ids()

print("[Pipeline] Loading reranker ...")
_reranker = Reranker(
    model_name=reranker_cfg["model_name"],
    device=reranker_cfg["device"],
)

print("[Pipeline] Loading generator ...")
_generator = Generator(
    model=generation_cfg["model"],
    max_tokens=generation_cfg["max_tokens"],
    temperature=generation_cfg["temperature"],
    system_prompt=generation_cfg["system_prompt"],
)

print("[Pipeline] All components ready.")

# ── public interface ──────────────────────────────────────────────────────────

def run(query: str) -> dict:
    # 1. embed query
    qvec = _embedder.encode([query])

    # 2. FAISS retrieval
    hits = search(_index, _chunk_ids, qvec, top_k=retrieval_cfg["top_k"])
    candidates = [
        dict(_chunks_by_id[cid], faiss_score=score)
        for cid, score in hits
        if cid in _chunks_by_id
    ]

    # 3. rerank
    reranked = _reranker.rerank(query, candidates, top_n=reranker_cfg["top_n"])

    # 4. generate
    result = _generator.generate(query, reranked)

    return result
