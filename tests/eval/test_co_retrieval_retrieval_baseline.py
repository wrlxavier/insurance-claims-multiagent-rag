"""Sanity floor for the [M3-06] exclusion co-retrieval step.

The mirror of ``test_rerank_retrieval_baseline.py`` for the co-retrieval stage:
proves that hybrid RRF + rerank + exclusion co-retrieval, filtered to each
question's SUSEP process + CNPJ, still clears a recall floor **and** does not
lose ground against the no-co-retrieval baseline on the two numbers M3-06 exists
to move -- pooled exclusion-clause recall and the ``coverage_with_exclusion``
Recall@10. A smoke check, not a quality gate -- the committed numbers, the slot
sweep and the verdict live in ``docs/EXCLUSION_CO_RETRIEVAL.md``.

Needs ``build/chunks.jsonl``, a reachable Postgres with embedded chunks, and the
optional ``embed`` uv group (which also ships the cross-encoder). Skips cleanly
when any is absent, so ``make test-eval`` stays green on a machine without the
stack.
"""

import argparse

import pytest
from sqlalchemy import func, select

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

pytestmark = pytest.mark.eval

RECALL_FLOOR = 0.85
MRR_FLOOR = 0.65


def _skip_unless_ready() -> None:
    if not CHUNKS_JSONL_PATH.exists():
        pytest.skip("build/chunks.jsonl not built; run `make build-chunks`")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("optional `embed` group not installed; `uv sync --group embed`")
    from infrastructure.database import (
        create_engine_from_settings,
        create_session_factory,
    )
    from infrastructure.database.models import ChunkRow

    try:
        engine = create_engine_from_settings()
        with create_session_factory(engine=engine)() as session:
            embedded = session.execute(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.embedding.is_not(None))
            ).scalar_one()
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 - any DB failure is a skip, not a fail
        pytest.skip(f"database not ready for the co-retrieval eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_co_retrieval_clears_a_floor_and_does_not_lose_exclusion_ground() -> None:
    _skip_unless_ready()

    from scripts.eval_retrieval import (
        GOLDEN_SET_DIR,
        MANIFEST_PATH,
        _build_filter_for,
        _open_retriever,
        aggregate,
        compute_exclusion_clause_recall,
        evaluate_questions,
        load_corpus,
        load_document_metadata,
        load_golden_questions,
    )

    from infrastructure.parsing.corpus_artifact import JSONL_PATH

    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)
    filter_for = _build_filter_for("default", document_meta)

    def score(co_retrieval: bool) -> tuple[dict[str, float], float | None, float]:
        args = argparse.Namespace(
            retriever="hybrid",
            fusion="rrf",
            filter_mode="default",
            seed=42,
            rerank=True,
            co_retrieval=co_retrieval,
        )
        with _open_retriever(args, corpus) as (retriever, _config):
            rows, _ = evaluate_questions(
                questions, retriever, document_meta, filter_for=filter_for
            )
        exclusion = compute_exclusion_clause_recall(rows, clause_by_id, k=10)
        subset = aggregate(
            [row for row in rows if row.question_type == "coverage_with_exclusion"],
            (1, 5, 10),
            10,
        )
        return aggregate(rows, (1, 5, 10), 10), exclusion["recall"], subset["recall@10"]

    base_overall, base_exclusion, base_subset = score(co_retrieval=False)
    co_overall, co_exclusion, co_subset = score(co_retrieval=True)

    assert co_overall["recall@10"] > RECALL_FLOOR
    assert co_overall["mrr"] > MRR_FLOOR
    # M3-06's purpose: never regress the exclusion-side numbers.
    assert base_exclusion is not None and co_exclusion is not None
    assert co_exclusion >= base_exclusion - 1e-9
    assert co_subset >= base_subset - 1e-9
