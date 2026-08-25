"""The reproducibility stamp for one retrieval-evaluation run [M2-06].

Mirrors [infrastructure.parsing.corpus_artifact.BuildManifest]'s role for
``make parse``: pins exactly what produced one run's numbers -- retriever,
k values, golden-set and corpus identity, seed, timestamp -- so a report
read in isolation still says what generated it. Embedded as one key of the
larger report dict both the JSON and Markdown outputs render from (see
``scripts/eval_retrieval.py``), which is otherwise a plain dict rather than
a frozen schema, since its shape is expected to grow with new breakdowns.
"""

from datetime import datetime

from pydantic import BaseModel

SCHEMA_VERSION = "v1"


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
