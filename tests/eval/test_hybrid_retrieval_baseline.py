"""Sanity floor for the [M3-04] hybrid retriever + metadata pre-filter.

The mirror of ``test_lexical_retrieval_baseline.py`` for the full pipeline:
proves the RRF hybrid, filtered to each question's SUSEP process + CNPJ,
clears a floor far above the unfiltered legs on the real golden set, and that
the pre-filter drives the foreign-document rate to exactly zero ([M3-04] item
4). A smoke check, not a quality gate -- the committed numbers and the verdict
live in ``docs/HYBRID_RETRIEVAL.md``.

Needs more than the lexical baseline: ``build/chunks.jsonl``, a reachable
Postgres with embedded chunks, and the optional ``embed`` uv group. Skips
cleanly when any is absent, so ``make test-eval`` stays green on a machine
without the stack.
"""

import argparse

import pytest
from sqlalchemy import func, select

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

pytestmark = pytest.mark.eval

RECALL_FLOOR = 0.80
MRR_FLOOR = 0.55


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
        pytest.skip(f"database not ready for the hybrid eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_filtered_hybrid_clears_a_floor_and_stays_in_document() -> None:
    _skip_unless_ready()

    from scripts.eval_retrieval import (
        GOLDEN_SET_DIR,
        MANIFEST_PATH,
        _build_filter_for,
        _open_retriever,
        aggregate,
        compute_foreign_document_rate,
        evaluate_questions,
        load_corpus,
        load_document_metadata,
        load_golden_questions,
    )

    from infrastructure.parsing.corpus_artifact import JSONL_PATH

    args = argparse.Namespace(
        retriever="hybrid",
        fusion="rrf",
        filter_mode="default",
        seed=42,
        rerank=False,
        co_retrieval=False,
    )
    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)
    filter_for = _build_filter_for("default", document_meta)

    with _open_retriever(args, load_corpus(JSONL_PATH)) as (retriever, _config):
        rows, _ = evaluate_questions(
            questions, retriever, document_meta, filter_for=filter_for
        )

    overall = aggregate(rows, (1, 5, 10), 10)
    assert overall["recall@10"] > RECALL_FLOOR
    assert overall["mrr"] > MRR_FLOOR
    # DoD item 4: the process + CNPJ filter never leaks into another document.
    assert compute_foreign_document_rate(rows, k=10)["rate"] == 0.0
