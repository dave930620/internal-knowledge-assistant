# =============================================================================
# scripts/run_eval.py
#
# PURPOSE:
#   Runs the retrieval evaluation pipeline. For each query in eval/queries.json,
#   retrieves top-K chunks using the full two-stage pipeline (FAISS + reranker)
#   and computes Recall@K, MRR, and nDCG@10 against ground-truth chunk IDs.
#   Prints per-query results and aggregate (macro-average) scores.
#
# INPUT:
#   eval/queries.json
#     List of queries with ground-truth relevant_chunk_ids.
#
#   data/embeddings/faiss.index, data/embeddings/chunk_ids.json
#     FAISS index and ID mapping built by build_faiss_index.py.
#
#   data/processed/chunks.jsonl
#     Chunk metadata for building candidate dicts passed to the reranker.
#
# OUTPUT:
#   Printed evaluation report — per-query scores + macro averages.
#   No files written.
#
# USAGE:
#   python scripts/run_eval.py
# =============================================================================

import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embedder import Embedder
from src.retrieval.faiss_index import load_index, load_chunk_ids, search
from src.reranker.reranker import Reranker
from eval.metrics import recall_at_k, mrr, ndcg_at_k

BASE_DIR    = Path(__file__).resolve().parent.parent
QUERIES_PATH = BASE_DIR / "eval/queries.json"
CHUNKS_PATH  = BASE_DIR / "data/processed/chunks.jsonl"

K_FAISS    = 10   # FAISS retrieval pool
K_RERANK   = 5    # reranker output size
K_EVAL     = 5    # cutoff for Recall@K and nDCG@K


def load_yaml(name: str) -> dict:
    with open(BASE_DIR / "configs" / name) as f:
        return yaml.safe_load(f)


def main():
    print("=" * 65)
    print("Retrieval Evaluation")
    print("=" * 65)

    # ── load components ───────────────────────────────────────────────
    emb_cfg      = load_yaml("embedding.yaml")
    reranker_cfg = load_yaml("reranker.yaml")

    embedder = Embedder(
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg["batch_size"],
        device=emb_cfg["device"],
        normalize_embeddings=emb_cfg["normalize_embeddings"],
    )
    index     = load_index()
    chunk_ids = load_chunk_ids()
    reranker  = Reranker(model_name=reranker_cfg["model_name"], device=reranker_cfg["device"])

    chunks_by_id: dict[str, dict] = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_id[c["chunk_id"]] = c

    queries = json.loads(QUERIES_PATH.read_text())
    print(f"\n[INFO] Evaluating {len(queries)} queries  |  K_eval={K_EVAL}\n")

    # ── per-query eval ────────────────────────────────────────────────
    scores_by_category: dict[str, list] = defaultdict(list)
    all_recall, all_mrr, all_ndcg = [], [], []

    header = f"{'ID':>4}  {'Category':20s}  {'R@'+str(K_EVAL):>6}  {'MRR':>6}  {'nDCG@'+str(K_EVAL):>7}  Question"
    print(header)
    print("-" * 95)

    for q in queries:
        qvec     = embedder.encode([q["question"]])
        hits     = search(index, chunk_ids, qvec, top_k=K_FAISS)
        candidates = [
            dict(chunks_by_id[cid], faiss_score=score)
            for cid, score in hits
            if cid in chunks_by_id
        ]
        reranked     = reranker.rerank(q["question"], candidates, top_n=K_RERANK)
        retrieved_ids = [r["chunk_id"] for r in reranked]
        relevant      = set(q["relevant_chunk_ids"])

        r_k  = recall_at_k(retrieved_ids, relevant, K_EVAL)
        m    = mrr(retrieved_ids, relevant)
        n    = ndcg_at_k(retrieved_ids, relevant, K_EVAL)

        all_recall.append(r_k)
        all_mrr.append(m)
        all_ndcg.append(n)
        scores_by_category[q["category"]].append((r_k, m, n))

        print(f"{q['query_id']:>4}  {q['category']:20s}  {r_k:>6.3f}  {m:>6.3f}  {n:>7.3f}  {q['question'][:50]}")

    # ── aggregate ─────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "=" * 65)
    print(f"{'MACRO AVERAGE':30s}  R@{K_EVAL}={avg(all_recall):.3f}  MRR={avg(all_mrr):.3f}  nDCG@{K_EVAL}={avg(all_ndcg):.3f}")
    print("=" * 65)

    print("\nScores by category:")
    for cat, vals in sorted(scores_by_category.items()):
        rs = [v[0] for v in vals]
        ms = [v[1] for v in vals]
        ns = [v[2] for v in vals]
        print(f"  {cat:20s}  R@{K_EVAL}={avg(rs):.3f}  MRR={avg(ms):.3f}  nDCG@{K_EVAL}={avg(ns):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
