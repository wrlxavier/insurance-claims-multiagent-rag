"""The reproducibility stamp for one retrieval-evaluation run [M2-06].

Mirrors [infrastructure.parsing.corpus_artifact.BuildManifest]'s role for
``make parse``: pins exactly what produced one run's numbers -- retriever,
k values, golden-set and corpus identity, seed, timestamp -- so a report
read in isolation still says what generated it. Embedded as one key of the
larger report dict both the JSON and Markdown outputs render from (see
``scripts/eval_retrieval.py``), which is otherwise a plain dict rather than
a frozen schema, since its shape is expected to grow with new breakdowns.

``v2`` [M3-03]: the lexical retriever adds the chunk-corpus identity and its
BM25/analyzer contract. ``v3`` [M3-04]: the dense and hybrid retrievers add the
metadata-filter mode, the fusion strategy and its constants, and the embedding /
hybrid config fingerprints. ``v4`` [M3-05]: the ``--rerank`` path adds the
cross-encoder model id/revision, the candidate depth and the reranker config
fingerprint. Every added field is optional -- a run without that leg leaves the
new ones ``None``.
"""

from datetime import datetime

from pydantic import BaseModel

SCHEMA_VERSION = "v4"


class RetrievalRunConfig(BaseModel):
    """Config that produced one retrieval-evaluation run's report."""

    schema_version: str
    retriever_name: str
    k_values: list[int]
    ndcg_k: int
    golden_set_dir: str
    golden_set_question_count: int
    corpus_path: str
    corpus_clause_count: int
    seed: int | None
    run_at_utc: datetime

    # [M3-03] lexical retriever (and the lexical leg of `hybrid`); None for
    # `random` and `dense`.
    chunk_corpus_path: str | None = None
    chunk_corpus_chunk_count: int | None = None
    lexical_analyzer_version: str | None = None
    bm25_k1: float | None = None
    bm25_b: float | None = None
    lexical_idf_variant: str | None = None
    lexical_index_text_field: str | None = None
    stemming_exception_count: int | None = None
    lexical_config_fingerprint: str | None = None

    # [M3-04] metadata pre-filter: `none` or `default` (per-question SUSEP
    # process + CNPJ from the manifest join). None for pre-M3-04 runs.
    filter_mode: str | None = None

    # [M3-04] dense retriever (and the dense leg of `hybrid`); None otherwise.
    dense_model_id: str | None = None
    dense_model_revision: str | None = None
    embedding_config_fingerprint: str | None = None

    # [M3-04] `hybrid` only; None otherwise.
    fusion_strategy: str | None = None
    rrf_k: int | None = None
    fusion_weights: list[float] | None = None
    candidate_depth: int | None = None
    hybrid_config_fingerprint: str | None = None

    # [M3-05] `--rerank` only; None otherwise. `rerank_candidate_depth` is how
    # many of the base retriever's candidates the cross-encoder re-scored before
    # the top-k cut.
    reranker_model_id: str | None = None
    reranker_model_revision: str | None = None
    rerank_candidate_depth: int | None = None
    reranker_config_fingerprint: str | None = None
