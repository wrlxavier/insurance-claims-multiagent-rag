"""Wrap the M3 retrieval pipeline as the agent graph's ``RetrievalPort`` -- [M4-04].

[M4-04]'s DoD: "wrap the M3 pipeline as a graph node, without leaking retrieval
internals into the graph". The graph node depends only on
[infrastructure.graph.context.RetrievalPort]; this module is the single adapter
that satisfies it from the live pipeline, and the one place that knows both the
graph's port shape ([RetrievedClause]) and the ``infrastructure.rag`` retrievers.

Flow, per ``retrieve`` call:

1. hybrid RRF over the metadata pre-filter -> ``candidate_depth`` clause ids;
2. cross-encoder rerank -- scored by hand rather than through
   [infrastructure.rag.reranking_retriever.RerankingRetriever], which discards
   the scores the [M3-07] gate needs (the same reason
   ``scripts/eval_insufficient_context_gate.py`` re-implements this);
3. keep the top ``k``;
4. optional exclusion co-retrieval -- reuse the real [infrastructure.rag.
   exclusion_co_retrieval.ExclusionCoRetrievalRetriever] over the reranked
   ranking; an injected exclusion clause carries ``score = 0.0`` (co-retrieval
   never re-scores, so rank-1 -- the gate's signal -- is unchanged);
5. hydrate each surviving id into a [RetrievedClause] from the chunk corpus.

Needs a live [infrastructure.rag.hybrid_retriever.HybridRetriever] (Postgres +
the ``embed`` group's models); a unit test drives it with fakes.

[M5-07] records steps 1-4 as a ``retrieval.rerank`` span. The retrieval node's
own span shows the ranking the assessment nodes then reason over; this one shows
how it was arrived at -- each candidate's hybrid rank beside its cross-encoder
rank and score, plus anything co-retrieval injected. ``RERANK_CANDIDATE_DEPTH``
is 10, the same as the node's ``k``, so the reranker re-orders rather than
prunes: the question this span answers is not "what was dropped" but "did the
hybrid legs surface the right clause at all, and where did the cross-encoder
move it" -- two different bugs with two different fixes.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol

from domain.clause_classification import ClauseType
from infrastructure.rag.exclusion_co_retrieval import ClauseGraph
from infrastructure.rag.exclusion_co_retrieval import (
    ExclusionCoRetrievalRetriever as _ExclusionCoRetrieval,
)
from infrastructure.rag.exclusion_co_retrieval_config import RESERVED_EXCLUSION_SLOTS
from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_config import RERANK_CANDIDATE_DEPTH
from infrastructure.rag.retrieval_filter import RetrievalFilter
from infrastructure.rag.retrieved_clause import RetrievedClause

_INJECTED_CLAUSE_SCORE = 0.0


class _FilterableBase(Protocol):
    """A clause-id-ranking ``retrieve`` -- structurally ``HybridRetriever``.

    Re-declared here rather than imported so this module keeps the
    ``infrastructure.rag`` -> ``infrastructure.evaluation`` non-dependency the
    other rag wrappers keep.
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


@dataclass(frozen=True)
class IndexedClause:
    """One clause's two text representations plus the provenance a citation needs.

    ``embed_text`` is the breadcrumb-prefixed string the reranker scores (the
    same representation BM25 indexes); ``excerpt`` is the human-readable
    ``display_text``. Both are rejoined in ``chunk_index`` order when a clause
    spans several chunks.
    """

    embed_text: str
    excerpt: str
    document_id: str
    susep_process: str
    clause_type: ClauseType


class _ChunkRow(Protocol):
    """The subset of ``infrastructure.rag.chunk_schema.ChunkRecord`` used here."""

    @property
    def source_clause_ids(self) -> list[str]: ...
    @property
    def chunk_index(self) -> int: ...
    @property
    def text(self) -> str: ...
    @property
    def display_text(self) -> str: ...
    @property
    def document_id(self) -> str: ...
    @property
    def susep_process(self) -> str: ...
    @property
    def clause_type(self) -> ClauseType: ...


def build_clause_index(chunks: Sequence[_ChunkRow]) -> dict[str, IndexedClause]:
    """``clause_id`` -> [IndexedClause], keyed by every ``source_clause_ids`` entry.

    The metadata (document, SUSEP process, clause type) is constant across a
    clause's chunks; the two texts are rejoined in ``chunk_index`` order. Mirrors
    ``scripts/eval_retrieval.py::build_clause_text_map`` but keeps ``display_text``
    and the provenance the citation type needs.
    """
    pieces: dict[str, list[tuple[int, str, str]]] = {}
    meta: dict[str, _ChunkRow] = {}
    for chunk in chunks:
        for clause_id in chunk.source_clause_ids:
            pieces.setdefault(clause_id, []).append(
                (chunk.chunk_index, chunk.text, chunk.display_text)
            )
            meta.setdefault(clause_id, chunk)
    index: dict[str, IndexedClause] = {}
    for clause_id, parts in pieces.items():
        ordered = sorted(parts, key=lambda part: part[0])
        source = meta[clause_id]
        index[clause_id] = IndexedClause(
            embed_text="\n\n".join(text for _, text, _ in ordered),
            excerpt="\n\n".join(display for _, _, display in ordered),
            document_id=source.document_id,
            susep_process=source.susep_process,
            clause_type=source.clause_type,
        )
    return index


class SpanRecorder(Protocol):
    """A ``span`` context manager -- structurally the graph's ``TracePort``.

    Re-declared rather than imported for the same reason ``_FilterableBase``
    above is: ``infrastructure.rag`` does not depend on
    ``infrastructure.graph``, and this module is not going to be the first to
    reverse that.
    """

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Open a span; fill the yielded mapping to set its output."""
        ...


class _NoSpans:
    """The no-op recorder this adapter uses unless the composition root traces."""

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Yield a throwaway mapping and record nothing."""
        return nullcontext({})


_NO_SPANS: SpanRecorder = _NoSpans()


class _FixedRanking:
    """A one-shot base that hands ``ExclusionCoRetrievalRetriever`` a set ranking."""

    def __init__(self, clause_ids: Sequence[str]) -> None:
        self._clause_ids = list(clause_ids)

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        return self._clause_ids[:k]


class GraphRetrievalAdapter:
    """The M3 hybrid + rerank (+ co-retrieval) pipeline behind the graph's port."""

    def __init__(
        self,
        hybrid: _FilterableBase,
        reranker: Reranker,
        clause_index: Mapping[str, IndexedClause],
        *,
        candidate_depth: int = RERANK_CANDIDATE_DEPTH,
        co_retrieval: ClauseGraph | None = None,
        reserved_exclusion_slots: int = RESERVED_EXCLUSION_SLOTS,
        tracer: SpanRecorder | None = None,
    ) -> None:
        """Compose the hybrid retriever, the reranker and the clause index.

        ``co_retrieval`` set turns on [M3-06] exclusion co-retrieval over the
        reranked ranking. A candidate missing from ``clause_index`` is scored as
        the empty string (it sinks) and hydrated as a bare id. ``tracer``
        defaults to the no-op ([M5-07]).
        """
        self._hybrid = hybrid
        self._reranker = reranker
        self._clause_index = clause_index
        self._candidate_depth = candidate_depth
        self._co_retrieval = co_retrieval
        self._reserved_exclusion_slots = reserved_exclusion_slots
        self._tracer = tracer if tracer is not None else _NO_SPANS

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedClause]:
        """Up to ``k`` clauses (more only when co-retrieval injects), best first."""
        if k <= 0:
            return []
        with self._tracer.span(
            "retrieval.rerank",
            input={"k": k, "candidate_depth": self._candidate_depth},
        ) as traced:
            candidates = self._hybrid.retrieve(
                question, k=self._candidate_depth, metadata_filter=metadata_filter
            )
            traced["n_hybrid_candidates"] = len(candidates)
            if not candidates:
                return []

            passages = [
                self._clause_index[cid].embed_text if cid in self._clause_index else ""
                for cid in candidates
            ]
            scores = self._reranker.rerank(question, passages)
            order = sorted(
                range(len(candidates)), key=lambda i: scores[i], reverse=True
            )
            ranked = [(candidates[i], float(scores[i])) for i in order]
            top = ranked[:k]

            score_by_id = dict(ranked)
            final_ids = [cid for cid, _ in top]
            if self._co_retrieval is not None:
                final_ids = _ExclusionCoRetrieval(
                    _FixedRanking(final_ids),
                    self._co_retrieval,
                    reserved_slots=self._reserved_exclusion_slots,
                ).retrieve(question, k=k, metadata_filter=metadata_filter)

            # Both ranks per candidate: where the hybrid legs put it, and where
            # the cross-encoder moved it to. A large gap on the clause that
            # mattered is the whole diagnosis.
            hybrid_rank = {cid: rank for rank, cid in enumerate(candidates, start=1)}
            traced.update(
                {
                    "candidates": [
                        {
                            "clause_id": clause_id,
                            "hybrid_rank": hybrid_rank[clause_id],
                            "rerank_rank": rank,
                            "rerank_score": score,
                        }
                        for rank, (clause_id, score) in enumerate(ranked, start=1)
                    ],
                    "kept": final_ids,
                    "co_retrieval_injected": [
                        clause_id
                        for clause_id in final_ids
                        if clause_id not in score_by_id
                    ],
                }
            )

            return [
                self._hydrate(cid, score_by_id.get(cid, _INJECTED_CLAUSE_SCORE))
                for cid in final_ids
                if cid in self._clause_index
            ]

    def _hydrate(self, clause_id: str, score: float) -> RetrievedClause:
        """Build a [RetrievedClause] from the clause index (caller guards presence)."""
        indexed = self._clause_index[clause_id]
        return RetrievedClause(
            clause_id=clause_id,
            document_id=indexed.document_id,
            susep_process=indexed.susep_process,
            clause_type=indexed.clause_type,
            excerpt=indexed.excerpt,
            score=score,
        )
