"""Information retrieval metrics: NDCG@K, MRR, Precision@K.

Pure functions with no maasv dependency.
"""

from __future__ import annotations

import math


def ndcg_at_k(
    ranked_ids: list[str],
    relevance_map: dict[str, float],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        ranked_ids: Memory IDs in the order returned by the system.
        relevance_map: Mapping of memory ID -> graded relevance (0.0 to 1.0).
        k: Cutoff rank.

    Returns:
        NDCG@K in [0.0, 1.0]. Returns 0.0 if there are no relevant documents.
    """
    if not relevance_map:
        return 0.0

    # DCG of the system ranking
    dcg = 0.0
    for i, mid in enumerate(ranked_ids[:k]):
        rel = relevance_map.get(mid, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: sort all relevance grades descending
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank.

    Args:
        ranked_ids: Memory IDs in the order returned by the system.
        relevant_ids: Set of memory IDs considered relevant (binary).

    Returns:
        Reciprocal rank of the first relevant result, or 0.0 if none found.
    """
    for i, mid in enumerate(ranked_ids):
        if mid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Precision at K.

    Args:
        ranked_ids: Memory IDs in the order returned by the system.
        relevant_ids: Set of memory IDs considered relevant (binary).
        k: Cutoff rank.

    Returns:
        Fraction of top-K results that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(top_k)
