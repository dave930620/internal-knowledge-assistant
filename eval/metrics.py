# =============================================================================
# eval/metrics.py
#
# PURPOSE:
#   Computes standard information retrieval evaluation metrics given a list of
#   retrieved chunk IDs and a set of relevant (ground-truth) chunk IDs.
#
#   Metrics implemented:
#     - Recall@K   : fraction of relevant chunks that appear in the top-K results
#     - MRR        : Mean Reciprocal Rank — 1 / rank of the first relevant hit
#     - nDCG@K     : normalized Discounted Cumulative Gain at K
#
# INPUT:
#   retrieved    : ordered list of chunk_id strings (model output, best first)
#   relevant_ids : set of chunk_id strings that are ground-truth relevant
#   k            : cutoff (e.g. 5, 10)
#
# OUTPUT:
#   float in [0, 1] for each metric
# =============================================================================

import math


def recall_at_k(retrieved: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant_ids: set[str], k: int) -> float:
    def dcg(ids: list[str], cutoff: int) -> float:
        score = 0.0
        for i, cid in enumerate(ids[:cutoff], start=1):
            if cid in relevant_ids:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved, k)
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg  = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
