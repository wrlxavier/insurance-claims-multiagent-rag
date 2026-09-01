"""The retrieval port's return type: one ranked, scored, hydrated clause -- [M4-04].

The M3 retrievers rank *clause ids* ([infrastructure.evaluation.retriever.
FilterableRetriever] / [infrastructure.rag.hybrid_retriever.HybridRetriever]).
The agent graph's retrieval node ([M4-04]) needs more than an id: it builds a
typed ``Citation`` (clause + document + SUSEP process + clause type + quoted
excerpt + relevance score) and assembles the [infrastructure.rag.
insufficient_context_gate.GateSignals] the [M3-07] gate decides on. Both need
the reranker score and the clause's provenance, which ``retrieve(...) ->
list[str]`` does not carry.

``RetrievedClause`` is that richer row. It lives here in ``infrastructure.rag``
-- not in the graph package -- so the graph's ``RetrievalPort`` references it the
same way it already references [infrastructure.rag.retrieval_filter.
RetrievalFilter]: a ``TYPE_CHECKING`` import, no ``graph -> rag`` runtime edge
added and no ``rag -> graph`` edge either. The one module that produces these
from the live pipeline is [infrastructure.rag.graph_retrieval_adapter.
GraphRetrievalAdapter].
"""

from dataclasses import dataclass

from domain.clause_classification import ClauseType


@dataclass(frozen=True)
class RetrievedClause:
    """One clause a retriever returned, ranked best-first, with its provenance.

    ``score`` is the reranker's score for this clause against the query -- a
    sigmoid in [0, 1] for the pinned cross-encoder
    (``Alibaba-NLP/gte-multilingual-reranker-base``); ``0.0`` for a clause a
    structural post-process (exclusion co-retrieval) injected without re-scoring.
    ``excerpt`` is the clause as a human reads it (the chunk ``display_text``),
    never the breadcrumb-prefixed string the embedding model saw.
    """

    clause_id: str
    document_id: str
    susep_process: str
    clause_type: ClauseType
    excerpt: str
    score: float
