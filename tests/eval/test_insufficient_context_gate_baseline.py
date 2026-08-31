"""Live re-derivation of the [M3-07] insufficient-context gate signals.

The always-on enforcement of the "100% recall on the unanswerable subset"
guarantee is a pure unit test over the committed signal snapshot
(``tests/unit/infrastructure/rag/test_insufficient_context_gate.py``). This
eval-marked test is the **drift guard**: it re-runs hybrid RRF + rerank over the
23 ``unanswerable`` golden questions live, so a retrieval change that lifts one
of them over the abstain threshold fails here even though the frozen snapshot
would still pass.

Needs ``build/chunks.jsonl``, a reachable Postgres with embedded chunks, and the
optional ``embed`` uv group (which also ships the cross-encoder). Skips cleanly
when any is absent, so ``make test-eval`` stays green on a machine without the
stack -- the mirror of ``test_rerank_retrieval_baseline.py``.
"""

import pytest
from sqlalchemy import func, select

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

pytestmark = pytest.mark.eval

# Published precision in docs/INSUFFICIENT_CONTEXT_GATE.md is 100% (0 FP). This
# floor only guards against a regression that keeps recall but wrecks precision;
# the exact number is the doc's, re-measured by `make eval-insufficient-context-gate`.
GATE_PRECISION_FLOOR = 0.80


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
        pytest.skip(f"database not ready for the gate eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_the_gate_abstains_on_every_unanswerable_question_live() -> None:
    _skip_unless_ready()

    from scripts.eval_insufficient_context_gate import collect_signals, operating_point
    from scripts.eval_retrieval import (
        GOLDEN_SET_DIR,
        MANIFEST_PATH,
        load_corpus,
        load_document_metadata,
        load_golden_questions,
    )

    from infrastructure.parsing.corpus_artifact import JSONL_PATH

    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)

    rows, _ = collect_signals(questions, document_meta, clause_by_id)
    op = operating_point(rows)

    assert op["unanswerable_missed"] == []
    assert op["recall"] == 1.0
    assert op["precision"] >= GATE_PRECISION_FLOOR
