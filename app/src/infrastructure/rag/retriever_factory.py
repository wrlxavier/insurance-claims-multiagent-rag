"""Build the production retrieval stack behind the graph's ``RetrievalPort`` -- [M5-04].

Until [M5-04] this pipeline -- hybrid RRF (BM25 + dense) + cross-encoder rerank +
[M3-06] exclusion co-retrieval, the [M3-08] best config -- was assembled ad hoc
in every eval script (``scripts/eval_retrieval_node.py::_build_adapter`` and
friends). The API composition root needs the same thing, so it lives here once.

Two-phase, because the pieces have very different lifecycles:

- ``load_retriever_components()`` loads the heavy, process-wide parts -- the
  sentence-transformers query embedder and the cross-encoder reranker (the
  optional ``embed`` uv group; torch), the BM25 index over the chunk corpus, the
  clause index and the exclusion clause graph. Call it once, at startup.
- ``build_graph_retriever(session, components)`` is cheap and per-run: it binds
  the dense leg to a live Postgres session and composes the adapter.

``load_retriever_components`` raises a ``RuntimeError`` naming the fix
(``uv sync --group embed`` / ``make build-index``) when the group or the embedded
chunk corpus is missing, so a misconfigured deployment fails at boot with an
actionable message rather than on the first request.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import (
    JSONL_PATH,
    read_parsed_clauses_jsonl,
)
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.dense_retriever import DenseRetriever
from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.exclusion_co_retrieval import ClauseGraph
from infrastructure.rag.graph_retrieval_adapter import (
    GraphRetrievalAdapter,
    IndexedClause,
    SpanRecorder,
    build_clause_index,
)
from infrastructure.rag.hybrid_retriever import HybridRetriever
from infrastructure.rag.lexical_analyzer import build_analyzer
from infrastructure.rag.lexical_retriever import LexicalRetriever
from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_cache import CachingReranker


@dataclass(frozen=True)
class RetrieverComponents:
    """The load-once parts of the retrieval stack -- everything but the DB session."""

    embedder: CachingEmbedder
    reranker: CachingReranker
    lexical: LexicalRetriever
    clause_index: dict[str, IndexedClause]
    clause_graph: ClauseGraph


def load_retriever_components() -> RetrieverComponents:
    """Load the heavy, process-wide retrieval components (needs the ``embed`` group)."""
    if not CHUNKS_JSONL_PATH.exists():
        raise RuntimeError(
            f"{CHUNKS_JSONL_PATH} not found -- the API needs the built chunk "
            "corpus. Run `make build-index` (raw PDFs -> Postgres + embeddings)."
        )
    if not JSONL_PATH.exists():
        raise RuntimeError(
            f"{JSONL_PATH} not found -- run `make fetch-corpus-artifacts` or "
            "`make parse` first."
        )

    return retriever_components_from_corpora(
        read_chunks_jsonl(CHUNKS_JSONL_PATH),
        read_parsed_clauses_jsonl(JSONL_PATH),
    )


def retriever_components_from_corpora(
    chunks: Sequence[ChunkRecord],
    corpus: Sequence[ParsedClauseRecord],
) -> RetrieverComponents:
    """Build the load-once components from already-read corpora (for eval scripts)."""
    return RetrieverComponents(
        embedder=CachingEmbedder(_load_query_embedder()),
        reranker=CachingReranker(_load_reranker()),
        lexical=LexicalRetriever.from_chunks(list(chunks), build_analyzer()),
        clause_index=build_clause_index(chunks),
        clause_graph=ClauseGraph(corpus),
    )


def build_graph_retriever(
    session: Session,
    components: RetrieverComponents,
    *,
    tracer: SpanRecorder | None = None,
) -> GraphRetrievalAdapter:
    """Compose the per-run retrieval adapter over a live Postgres session.

    ``tracer`` ([M5-07]) is the graph's ``TracePort``, passed through so the
    adapter can record its ``retrieval.rerank`` span. ``SpanRecorder`` is the
    same shape re-declared in ``infrastructure.rag`` -- this layer does not
    import ``infrastructure.graph``. ``None`` (every caller but the composition
    root) leaves the adapter untraced.
    """
    dense = DenseRetriever(session, components.embedder)
    hybrid = HybridRetriever(components.lexical, dense)
    return GraphRetrievalAdapter(
        hybrid,
        components.reranker,
        components.clause_index,
        co_retrieval=components.clause_graph,
        tracer=tracer,
    )


def _load_query_embedder() -> Embedder:
    """The real sentence-transformers query embedder ([M3-02]).

    Deferred import so the heavy ``embed`` group is only required where the real
    retrieval stack is built (mirrors ``scripts/eval_retrieval._load_query_embedder``).
    """
    from infrastructure.rag.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    return SentenceTransformerEmbedder()


def _load_reranker() -> Reranker:
    """The real local cross-encoder reranker ([M3-05]); deferred import as above."""
    from infrastructure.rag.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker()
