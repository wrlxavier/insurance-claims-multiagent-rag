"""The reproducibility stamp for one retrieval-evaluation run [M2-06].

Mirrors [infrastructure.parsing.corpus_artifact.BuildManifest]'s role for
``make parse``: pins exactly what produced one run's numbers -- retriever,
k values, golden-set and corpus identity, seed, timestamp -- so a report
read in isolation still says what generated it. Embedded as one key of the
larger report dict both the JSON and Markdown outputs render from (see
``scripts/eval_retrieval.py``), which is otherwise a plain dict rather than
a frozen schema, since its shape is expected to grow with new breakdowns.

``v2`` [M3-03]: the lexical retriever adds the chunk-corpus identity and its
BM25/analyzer contract. Every added field is optional -- the ``random``
retriever leaves them ``None`` and its report is byte-identical to ``v1``.
"""

from datetime import datetime

from pydantic import BaseModel

SCHEMA_VERSION = "v2"


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

    # [M3-03] lexical retriever only; None for `random`.
    chunk_corpus_path: str | None = None
    chunk_corpus_chunk_count: int | None = None
    lexical_analyzer_version: str | None = None
    bm25_k1: float | None = None
    bm25_b: float | None = None
    lexical_idf_variant: str | None = None
    lexical_index_text_field: str | None = None
    stemming_exception_count: int | None = None
    lexical_config_fingerprint: str | None = None
