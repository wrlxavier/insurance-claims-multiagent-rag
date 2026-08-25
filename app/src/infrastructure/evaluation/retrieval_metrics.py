"""Retrieval quality metrics against a golden question's reference clause ids [M2-06].

Pure functions, no I/O: each takes a ranked list of retrieved clause ids and
the exhaustive set of reference clause ids for one question (see
[infrastructure.evaluation.golden_set_schema.GoldenQuestion.reference_clause_ids])
and returns a single score. ``unanswerable`` questions carry an empty
``reference_clause_ids`` by schema construction, for which Recall/MRR/nDCG
are mathematically undefined -- every function here raises on an empty
``reference`` rather than silently returning 0.0, so a caller is forced to
exclude those questions explicitly instead of corrupting an aggregate with a
meaningless zero.
"""

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], reference: Sequence[str], k: int) -> float:
    """Fraction of ``reference`` present in the top-k of ``retrieved``."""
    if not reference:
        raise ValueError("recall_at_k is undefined for an empty reference set")
    reference_set = set(reference)
    hits = reference_set & set(retrieved[:k])
    return len(hits) / len(reference_set)


def reciprocal_rank(retrieved: Sequence[str], reference: Sequence[str]) -> float:
    """1/rank of the first retrieved id found in ``reference``; 0.0 if none."""
    if not reference:
        raise ValueError("reciprocal_rank is undefined for an empty reference set")
    reference_set = set(reference)
    for rank, clause_id in enumerate(retrieved, start=1):
        if clause_id in reference_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], reference: Sequence[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k: DCG/IDCG, 0.0 if no reference id fits within k."""
    if not reference:
        raise ValueError("ndcg_at_k is undefined for an empty reference set")
    reference_set = set(reference)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, clause_id in enumerate(retrieved[:k], start=1)
        if clause_id in reference_set
    )
    ideal_hits = min(len(reference_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
