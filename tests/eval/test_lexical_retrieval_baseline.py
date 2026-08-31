"""Sanity ceiling for the [M3-03] BM25 lexical retriever.

The mirror of ``test_retrieval_eval_collapse.py``: that one proves a garbage
retriever collapses toward zero; this one proves the real lexical retriever
clears a floor far above garbage on the real golden set. It is a smoke check,
not a quality gate -- the published numbers and the verdict live in
``docs/LEXICAL_RETRIEVAL.md``. Run via ``make test-eval``; skipped if
``build/chunks.jsonl`` has not been built.
"""

import pytest
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    K_VALUES,
    MANIFEST_PATH,
    NDCG_K,
    aggregate,
    build_lexical_retriever,
    evaluate_questions,
    load_chunk_corpus,
    load_document_metadata,
    load_golden_questions,
)

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

# Random retrieval collapses below 0.05 (see test_retrieval_eval_collapse.py);
# M3's exit target is Recall@10 >= 0.85. The unfiltered lexical baseline sits
# well between the two -- cross-document leakage (no metadata filter until
# [M3-04]) caps it below target. 0.40 / 0.20 leave generous headroom under the
# measured ~0.59 / ~0.39 so this stays a floor, not a brittle equality.
RECALL_FLOOR = 0.40
MRR_FLOOR = 0.20


@pytest.mark.eval
def test_lexical_retriever_clears_a_floor_far_above_random() -> None:
    if not CHUNKS_JSONL_PATH.exists():
        pytest.skip("build/chunks.jsonl not built; run `make build-chunks`")

    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)
    retriever = build_lexical_retriever(load_chunk_corpus())

    rows, _ = evaluate_questions(questions, retriever, document_meta)
    overall = aggregate(rows, K_VALUES, NDCG_K)

    assert overall["recall@10"] > RECALL_FLOOR
    assert overall["mrr"] > MRR_FLOOR
