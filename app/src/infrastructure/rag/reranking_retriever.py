"""Cross-encoder reranking of a base retriever's candidates -- [M3-05].

Wraps any filtered retriever (in practice
[infrastructure.rag.hybrid_retriever.HybridRetriever], per
``docs/HYBRID_RETRIEVAL.md``'s reasoning that the higher-recall RRF candidate
set is the right thing to feed a reranker) behind the **same**
``retrieve(question, *, k, metadata_filter)`` interface, so M4's graph node and
the [M2-06] eval harness stay unaware of it.

Flow: ask the base retriever for ``candidate_depth`` clause ids over the given
filter, look up each candidate's passage text, score every ``(question,
passage)`` pair with the [infrastructure.rag.reranker.Reranker], then return the
``k`` highest-scoring clause ids. Equal scores keep the base retriever's order
(a stable sort), so the output is deterministic. The reranker can only reorder
the candidate set -- it never introduces a clause the base retriever did not
return, and it never pads.
"""

from collections.abc import Mapping
from typing import Protocol

from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_config import RERANK_CANDIDATE_DEPTH
from infrastructure.rag.retrieval_filter import RetrievalFilter


class _FilterableBase(Protocol):
    """The base retriever contract: a filtered, clause-id-ranking ``retrieve``.

    Structurally [infrastructure.evaluation.retriever.FilterableRetriever];
    re-declared here so ``infrastructure.rag`` imports nothing from
    ``infrastructure.evaluation`` (the same split ``hybrid_retriever._ScoredLeg``
    keeps).
    """

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids, ranked best-match first, filtered."""
        ...


class RerankingRetriever:
    """Reorders a base retriever's top candidates with a cross-encoder."""

    def __init__(
        self,
        base: _FilterableBase,
        reranker: Reranker,
        clause_text: Mapping[str, str],
        *,
        candidate_depth: int = RERANK_CANDIDATE_DEPTH,
    ) -> None:
        """Compose the base retriever, the reranker, and the clause-id -> text map.

        ``clause_text`` supplies the passage each candidate clause id is scored
        as; a candidate missing from it is scored as the empty string (it sinks
        to the bottom rather than raising).
        """
        self._base = base
        self._reranker = reranker
        self._clause_text = clause_text
        self._candidate_depth = candidate_depth

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids, reranked best match first. May return fewer."""
        if k <= 0:
            return []
        candidates = self._base.retrieve(
            question, k=self._candidate_depth, metadata_filter=metadata_filter
        )
        if not candidates:
            return []
        passages = [self._clause_text.get(clause_id, "") for clause_id in candidates]
        scores = self._reranker.rerank(question, passages)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order][:k]
