"""Sanity floor for the [M3-05] cross-encoder rerank of the filtered hybrid.

The mirror of ``test_hybrid_retrieval_baseline.py`` for the rerank stage: proves
that hybrid RRF + cross-encoder rerank, filtered to each question's SUSEP
process + CNPJ, still clears a recall floor **and** does not push exclusion
clauses out of the kept context relative to the no-rerank baseline (the M3-05
DoD's named regression). A smoke check, not a quality gate -- the committed
numbers, the curve and the verdict live in ``docs/RERANKING.md``.

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
        pytest.skip(f"database not ready for the rerank eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_filtered_hybrid_rerank_clears_a_floor_and_keeps_exclusion_clauses() -> None:
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

    def score(rerank: bool) -> tuple[dict[str, float], float | None]:
        args = argparse.Namespace(
            retriever="hybrid",
            fusion="rrf",
            filter_mode="default",
            seed=42,
            rerank=rerank,
            co_retrieval=False,
        )
        with _open_retriever(args, corpus) as (retriever, _config):
            rows, _ = evaluate_questions(
                questions, retriever, document_meta, filter_for=filter_for
            )
        exclusion = compute_exclusion_clause_recall(rows, clause_by_id, k=10)
        return aggregate(rows, (1, 5, 10), 10), exclusion["recall"]

    baseline_overall, baseline_exclusion = score(rerank=False)
    rerank_overall, rerank_exclusion = score(rerank=True)

    assert rerank_overall["recall@10"] > RECALL_FLOOR
    assert rerank_overall["mrr"] > MRR_FLOOR
    # DoD item 4: reranking must not push exclusion clauses out of the context.
    assert baseline_exclusion is not None and rerank_exclusion is not None
    assert rerank_exclusion >= baseline_exclusion - 1e-9
