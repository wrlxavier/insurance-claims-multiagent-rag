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
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from infrastructure.rag.bm25 import BM25Index, build_bm25_index, top_n
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.lexical_config import BM25_B, BM25_K1, LEXICAL_INDEX_TEXT_FIELD


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
    ) -> None:
        """Build over a prepared index, its analyzer, and the chunk->clauses map."""
        self._index = index
        self._analyzer = analyzer
        self._chunk_to_clauses = chunk_to_clauses

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
        return cls(index, analyzer, chunk_to_clauses)

    def retrieve(self, question: str, *, k: int) -> list[str]:
        """Up to ``k`` clause ids, best match first. May return fewer; never pads.

        Scores every matching chunk, not just the top k: split chunks
        (``clause_id#0``, ``#1``, ...) and merged chunks collapse on the roll-up
        to clause ids, so a fixed oversample could still under-fill.
        """
        if k <= 0:
            return []
        query_tokens = self._analyzer.analyze(question)
        ranked_chunks = top_n(self._index, query_tokens, len(self._index.doc_ids))
        clause_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id, _score in ranked_chunks:
            for clause_id in self._chunk_to_clauses[chunk_id]:
                if clause_id not in seen:
                    seen.add(clause_id)
                    clause_ids.append(clause_id)
                    if len(clause_ids) == k:
                        return clause_ids
        return clause_ids
