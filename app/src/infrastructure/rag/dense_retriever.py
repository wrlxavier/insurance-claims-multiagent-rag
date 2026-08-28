"""Dense (embedding) retrieval over the pgvector chunk table -- [M3-04].

The query side M3-02 deferred here: [infrastructure.rag.embedding_config.
format_query] the question, embed it through the same [Embedder] contract the
index side uses, then exact ``<=>`` cosine search over the metadata-filtered
partition ([infrastructure.database.chunk_repository.search_chunks_by_vector]),
rolled up from chunk hits to clause-id granularity the same way
[infrastructure.rag.lexical_retriever.LexicalRetriever] does -- so both legs
satisfy the clause-granularity [infrastructure.evaluation.retriever.Retriever]
contract the [M2-06] harness scores and [M3-04]'s fusion can combine them.

No HNSW / ``apply_ann_search_gucs``: ``docs/EMBEDDINGS.md``'s verdict is that
the index does not earn its place at this corpus size and the planner reads the
``(susep_process, cnpj)`` partition by btree + exact sort anyway.

The embedder is constructor-injected. Production wires the real
[infrastructure.rag.sentence_transformer_embedder.SentenceTransformerEmbedder]
(optionally wrapped in [infrastructure.rag.embedding_cache.CachingEmbedder]);
the test suite uses a fake, per the [M1-05b]/[M1-04d] no-live-calls precedent.
"""

from collections.abc import Sequence

from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import search_chunks_by_vector
from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_config import format_query
from infrastructure.rag.retrieval_filter import RetrievalFilter

# Chunks fetched per requested clause. A chunk carries >= 1 clause id and an
# over-long clause splits into a few chunks that share one, so k chunks can roll
# up to fewer than k clauses. Splits are rare and only ever 2-way ([M3-01]
# report), so 4x is comfortably enough; exact search over the small filtered
# partition makes the wider fetch free. Not a tuning knob -- chosen so the
# roll-up never under-fills, not to move a number.
_CLAUSE_ROLLUP_OVERSAMPLE = 4


class DenseRetriever:
    """Ranks clause ids for a question by cosine similarity of dense vectors."""

    def __init__(self, session: Session, embedder: Embedder) -> None:
        """Build over an open session and a query embedder."""
        self._session = session
        self._embedder = embedder

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids, best match first. May return fewer; never pads."""
        return [
            clause_id
            for clause_id, _score in self.retrieve_scored(
                question, k=k, metadata_filter=metadata_filter
            )
        ]

    def retrieve_scored(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Up to ``k`` ``(clause_id, cosine similarity)`` pairs, best first.

        Similarity is ``1 - cosine_distance`` (both stored and query vectors are
        L2-normalised, per [infrastructure.rag.embedding_config]). A clause's
        score is its best chunk's -- and since hits arrive in ascending distance
        order, that is simply the first chunk it appears in. The scores are what
        [M3-04]'s weighted-score fusion needs and :meth:`retrieve` discards.
        """
        if k <= 0:
            return []
        query_vector = self._embed_query(question)
        hits = search_chunks_by_vector(
            self._session,
            query_vector,
            k=k * _CLAUSE_ROLLUP_OVERSAMPLE,
            metadata_filter=metadata_filter,
        )
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()
        for _chunk_id, source_clause_ids, distance in hits:
            similarity = 1.0 - distance
            for clause_id in source_clause_ids:
                if clause_id not in seen:
                    seen.add(clause_id)
                    scored.append((clause_id, similarity))
                    if len(scored) == k:
                        return scored
        return scored

    def _embed_query(self, question: str) -> Sequence[float]:
        """Embed one query string through the pinned contract."""
        vectors = self._embedder.embed([format_query(question)])
        if len(vectors) != 1:
            raise ValueError(f"embedder returned {len(vectors)} vectors for 1 query")
        return vectors[0]
