"""BM25 lexical retriever over the chunk corpus -- [M3-03].

Chunk-level BM25, rolled up to clause-id granularity via each chunk's
``source_clause_ids`` (per [domain.chunk]: the anchor clause plus every short
clause merged into the chunk), so it satisfies the clause-granularity
[infrastructure.evaluation.retriever.Retriever] contract the [M2-06] eval
harness scores. Duck-typed against that Protocol, like
[infrastructure.evaluation.random_retriever.RandomRetriever].

Everything here is in-memory and constructor-injected: no database, no I/O. The
index is rebuilt from ``build/chunks.jsonl`` per run (a few seconds for ~4.5k
chunks -- unlike the 41-minute embedding run, it does not justify an on-disk
cache in this issue).

[M3-04] adds the ``metadata_filter`` kwarg and ``retrieve_scored`` -- this leg
of the hybrid retriever, filtered and score-exposing -- without changing the
unfiltered ``retrieve(question, k=k)`` path or its committed numbers.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from infrastructure.rag.bm25 import BM25Index, build_bm25_index, top_n
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.lexical_config import BM25_B, BM25_K1, LEXICAL_INDEX_TEXT_FIELD
from infrastructure.rag.retrieval_filter import RetrievalFilter


class _Analyzer(Protocol):
    """The text->tokens contract; [infrastructure.rag.lexical_analyzer.TextAnalyzer]."""

    def analyze(self, text: str) -> list[str]:
        """Turn raw text into the BM25 term list."""
        ...


def _chunk_text(chunk: ChunkRecord, text_field: str) -> str:
    """The chunk string BM25 indexes -- ``text`` (default) or ``display_text``."""
    if text_field == "display_text":
        return chunk.display_text
    return chunk.text


class LexicalRetriever:
    """Ranks clause ids for a question by BM25 over chunk text."""

    def __init__(
        self,
        index: BM25Index,
        analyzer: _Analyzer,
        chunk_to_clauses: Mapping[str, Sequence[str]],
        chunks_by_id: Mapping[str, ChunkRecord],
    ) -> None:
        """Build over a prepared index, its analyzer, and the chunk lookups.

        ``chunks_by_id`` carries the metadata [M3-04]'s pre-filter matches
        against; it is untouched when ``retrieve`` is called without a filter.
        """
        self._index = index
        self._analyzer = analyzer
        self._chunk_to_clauses = chunk_to_clauses
        self._chunks_by_id = chunks_by_id

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[ChunkRecord],
        analyzer: _Analyzer,
        *,
        text_field: str = LEXICAL_INDEX_TEXT_FIELD,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "LexicalRetriever":
        """Analyse the chosen text field of every chunk and build the BM25 index."""
        docs = [
            (chunk.chunk_id, analyzer.analyze(_chunk_text(chunk, text_field)))
            for chunk in chunks
        ]
        index = build_bm25_index(docs, k1=k1, b=b)
        chunk_to_clauses = {
            chunk.chunk_id: tuple(chunk.source_clause_ids) for chunk in chunks
        }
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return cls(index, analyzer, chunk_to_clauses, chunks_by_id)

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
        """Up to ``k`` ``(clause_id, BM25 score)`` pairs, best match first.

        Scores every matching chunk, not just the top k: split chunks
        (``clause_id#0``, ``#1``, ...) and merged chunks collapse on the roll-up
        to clause ids, so a fixed oversample could still under-fill. A clause's
        score is its best chunk's -- and since chunks arrive in descending score
        order, that is the first chunk it appears in. The scores are what
        [M3-04]'s weighted-score fusion needs and :meth:`retrieve` discards;
        ``metadata_filter`` drops non-matching chunks before the roll-up
        ([M3-04]'s pre-filter, applied to this leg).
        """
        if k <= 0:
            return []
        query_tokens = self._analyzer.analyze(question)
        ranked_chunks = top_n(self._index, query_tokens, len(self._index.doc_ids))
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()
        for chunk_id, score in ranked_chunks:
            if metadata_filter is not None and not metadata_filter.matches(
                self._chunks_by_id[chunk_id]
            ):
                continue
            for clause_id in self._chunk_to_clauses[chunk_id]:
                if clause_id not in seen:
                    seen.add(clause_id)
                    scored.append((clause_id, score))
                    if len(scored) == k:
                        return scored
        return scored
