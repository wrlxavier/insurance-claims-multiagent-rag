"""Rank / score fusion of the lexical and dense retrieval legs -- [M3-04].

Two strategies, compared on the golden set in ``docs/HYBRID_RETRIEVAL.md``:

* :func:`reciprocal_rank_fusion` -- rank-only, one parameter (``k``), the method
  named in the [M3-04] DoD.
* :func:`weighted_score_fusion` -- min-max normalises each leg's raw scores to
  ``[0, 1]`` and takes a weighted sum, so a leg that is confident (a top score
  well clear of its runners-up) pulls harder than its rank alone would say.

Both are pure functions over clause-id rankings and return a single ranked
``list[str]``. Ties break on the clause id ascending -- the same reproducibility
tie-break [infrastructure.rag.bm25.top_n] uses, so a committed Recall@k number
is stable run to run. Each input list is expected to already be deduplicated to
clause-id granularity (both legs roll chunk hits up before returning); a
repeated id within one list is counted once, at its best position.
"""

from collections.abc import Mapping, Sequence


def _rank_by_descending_score(scores: Mapping[str, float]) -> list[str]:
    """Clause ids ordered by score desc, ties broken by id asc."""
    return [
        clause_id
        for clause_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int
) -> list[str]:
    """Fuse ranked clause-id lists by summed reciprocal rank ``1 / (k + rank)``.

    ``rank`` is 1-based within each list. ``k`` dampens how much the very top of
    one list dominates (the standard RRF constant, ~60). A clause absent from a
    list contributes nothing from it.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, clause_id in enumerate(ranked, start=1):
            if clause_id in seen:
                continue
            seen.add(clause_id)
            scores[clause_id] = scores.get(clause_id, 0.0) + 1.0 / (k + rank)
    return _rank_by_descending_score(scores)


def weighted_score_fusion(
    scored_lists: Sequence[Sequence[tuple[str, float]]],
    *,
    weights: Sequence[float],
) -> list[str]:
    """Fuse ``(clause_id, score)`` lists by weighted sum of per-list min-max norms.

    Each list's raw scores are normalised to ``[0, 1]`` independently
    (``(s - min) / (max - min)``; a list whose scores are all equal normalises
    to all-``1.0``), scaled by that list's weight, and summed. ``weights`` is
    positional, one per list.
    """
    if len(scored_lists) != len(weights):
        raise ValueError(f"got {len(scored_lists)} lists but {len(weights)} weights")
    scores: dict[str, float] = {}
    for scored, weight in zip(scored_lists, weights, strict=True):
        best: dict[str, float] = {}
        for clause_id, value in scored:
            best[clause_id] = max(best.get(clause_id, value), value)
        for clause_id, value in _min_max(best).items():
            scores[clause_id] = scores.get(clause_id, 0.0) + weight * value
    return _rank_by_descending_score(scores)


def _min_max(scores: Mapping[str, float]) -> dict[str, float]:
    """Scale ``scores`` to ``[0, 1]``; all-equal (incl. singletons) -> all ``1.0``."""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return dict.fromkeys(scores, 1.0)
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}
