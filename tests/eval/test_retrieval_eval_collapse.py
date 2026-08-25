"""Validate the harness against a deliberately broken retriever [M2-06].

Runs against the real golden set and the real built corpus (no mocking):
if `build/parsed_clauses.jsonl` is missing, run `make fetch-corpus-
artifacts` or `make parse` first, exactly as `scripts/validate_golden_set.py`
requires. Run via `make test-eval`.
"""

import pytest
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    K_VALUES,
    MANIFEST_PATH,
    NDCG_K,
    aggregate,
    evaluate_questions,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)

from infrastructure.evaluation.random_retriever import RandomRetriever
from infrastructure.parsing.corpus_artifact import JSONL_PATH

# A retriever that ignores the question and samples 10 clauses uniformly at
# random from a corpus of thousands should score far below any plausible
# real retriever. With ~4,925 clauses in the current corpus, the expected
# per-reference hit probability of a 10-draw is ~0.2%; even a corpus an
# order of magnitude smaller still expects ~1%. 0.05 leaves a 5-25x safety
# margin over that expectation while sitting an order of magnitude below
# M3's own target thresholds (Recall@10 >= 0.85, MRR >= 0.60 per
# MILESTONES.md) -- a fixed seed makes this assertion reproducible, not
# just usually true.
COLLAPSE_THRESHOLD = 0.05


@pytest.mark.eval
def test_random_retriever_recall_mrr_ndcg_collapse_on_golden_set() -> None:
    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)

    retriever = RandomRetriever([record.clause_id for record in corpus], seed=42)
    rows, _ = evaluate_questions(questions, retriever, document_meta)
    overall = aggregate(rows, K_VALUES, NDCG_K)

    assert overall["recall@10"] < COLLAPSE_THRESHOLD
    assert overall["mrr"] < COLLAPSE_THRESHOLD
    assert overall[f"ndcg@{NDCG_K}"] < COLLAPSE_THRESHOLD
