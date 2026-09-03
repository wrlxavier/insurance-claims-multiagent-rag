"""Sanity floor for the [M4-04] retrieval node against ``golden-set-v1``.

Runs the real ``infrastructure.graph.nodes.retrieval.retrieval`` node over every
golden question through the production
``infrastructure.rag.graph_retrieval_adapter.GraphRetrievalAdapter`` and checks
it clears a retrieval floor, keeps citations inside the target document, and
that the [M3-07] gate -- reached here via the node's ``context_sufficient`` --
still abstains on every ``unanswerable`` question. A smoke check, not a quality
gate; the committed numbers and the analysis live in ``docs/RETRIEVAL_NODE.md``.

Needs ``build/chunks.jsonl``, a reachable Postgres with embedded chunks, and the
optional ``embed`` uv group. Skips cleanly when any is absent, so
``make test-eval`` stays green on a machine without the stack.

Floors are deliberately loose -- a target is a bet, and the analysis in the
committed report is the deliverable.
"""

import pytest
from sqlalchemy import func, select

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

pytestmark = pytest.mark.eval

_RECALL_FLOOR = 0.80
_MRR_FLOOR = 0.55
_GATE_FALSE_ABSTENTION_CEILING = 0.20


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
        pytest.skip(f"database not ready for the retrieval-node eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_retrieval_node_clears_a_floor_and_keeps_the_gate_recall() -> None:
    _skip_unless_ready()

    from scripts.eval_retrieval_node import run_retrieval_node_eval

    result = run_retrieval_node_eval()

    assert result.error_question_ids == []
    assert result.overall.n >= 90
    assert result.overall.recall_at_10 >= _RECALL_FLOOR
    assert result.overall.mrr >= _MRR_FLOOR
    assert result.overall.foreign_document_rate == 0.0
    # [M3-07] guarantee: every unanswerable question trips the gate.
    assert result.gate.unanswerable_missed == ()
    assert result.gate.gate_recall == 1.0
    assert result.gate.gate_false_abstention_rate <= _GATE_FALSE_ABSTENTION_CEILING
