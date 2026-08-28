"""Hybrid retrieval: fuse the lexical and dense legs behind one interface -- [M3-04].

The single retrieval interface [M3-04]'s DoD asks for, so M4's graph node stays
unaware of the implementation: one ``retrieve(question, *, k, metadata_filter)``
that runs BM25 ([infrastructure.rag.lexical_retriever.LexicalRetriever]) and
dense cosine search ([infrastructure.rag.dense_retriever.DenseRetriever]) over
the same [infrastructure.rag.retrieval_filter.RetrievalFilter], then fuses the
two rankings.

Two fusion strategies, selected by [infrastructure.rag.hybrid_config.
FusionStrategy] and compared on the golden set in ``docs/HYBRID_RETRIEVAL.md``:
Reciprocal Rank Fusion (rank-only) and weighted score fusion (min-max
normalised, weighted). Each leg contributes ``CANDIDATE_DEPTH`` clause ids
before fusion so a clause outside either leg's top 10 can still be rescued by
cross-leg agreement.

Satisfies the [infrastructure.evaluation.retriever.Retriever] contract when
called as ``retrieve(question, k=k)``.
"""

from typing import Protocol

from infrastructure.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from infrastructure.rag.hybrid_config import (
    CANDIDATE_DEPTH,
    DEFAULT_FUSION_STRATEGY,
    FUSION_WEIGHTS,
    RRF_K,
    FusionStrategy,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter


class _ScoredLeg(Protocol):
    """One fusion input: [LexicalRetriever] / [DenseRetriever]'s scored method."""

    def retrieve_scored(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Up to ``k`` ``(clause_id, score)`` pairs, best match first."""
        ...


class HybridRetriever:
    """Fuses a BM25 leg and a dense leg over a shared metadata pre-filter."""

    def __init__(
        self,
        lexical: _ScoredLeg,
        dense: _ScoredLeg,
        *,
        fusion: FusionStrategy = DEFAULT_FUSION_STRATEGY,
    ) -> None:
        """Compose the two legs; ``fusion`` picks the combine strategy."""
        self._lexical = lexical
        self._dense = dense
        self._fusion = fusion

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids, best match first. May return fewer; never pads.

        ``metadata_filter`` reaches both legs unchanged -- the dense leg pushes
        it into SQL, the lexical leg applies it in memory before its roll-up.
        """
        if k <= 0:
            return []
        lexical_scored = self._lexical.retrieve_scored(
            question, k=CANDIDATE_DEPTH, metadata_filter=metadata_filter
        )
        dense_scored = self._dense.retrieve_scored(
            question, k=CANDIDATE_DEPTH, metadata_filter=metadata_filter
        )
        if self._fusion is FusionStrategy.RRF:
            fused = reciprocal_rank_fusion(
                [
                    [clause_id for clause_id, _ in lexical_scored],
                    [clause_id for clause_id, _ in dense_scored],
                ],
                k=RRF_K,
            )
        else:
            fused = weighted_score_fusion(
                [lexical_scored, dense_scored], weights=FUSION_WEIGHTS
            )
        return fused[:k]
