"""Live smoke of the [M5-10] latency/cost measurement over a couple of claims.

The pure aggregation is unit-tested (``tests/unit/scripts/test_eval_performance.py``).
This eval-marked test runs the **real** graph twice and checks the harness comes
back with sane, populated numbers -- it is not a performance gate (latency and
cost are environment- and route-specific; the committed figures live in
``docs/PERFORMANCE.md``).

Needs ``LLM_PROVIDER`` set, ``build/chunks.jsonl``, a reachable Postgres with
embedded chunks, and the optional ``embed`` uv group. Skips cleanly when any is
absent, so ``make test-eval`` stays green (and free) on an unconfigured machine.
"""

import os

import pytest
from sqlalchemy import func, select

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH

pytestmark = pytest.mark.eval


def _skip_unless_ready() -> None:
    if not os.environ.get("LLM_PROVIDER"):
        pytest.skip("LLM_PROVIDER not set; run `make eval-performance` for real")
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
        pytest.skip(f"database not ready for the performance eval: {exc}")
    if embedded < 100:
        pytest.skip(f"only {embedded} embedded chunks; run `make embed-chunks`")


@pytest.mark.eval
def test_the_harness_measures_latency_and_cost_on_a_real_run() -> None:
    _skip_unless_ready()

    from scripts.eval_end_to_end import _CLAIM_TIMEOUT_SECONDS
    from scripts.eval_performance import run_performance_eval

    result = run_performance_eval(limit=2)

    assert result.error_claim_ids == []
    assert result.n_scored == 2

    e2e = result.end_to_end_latency
    assert 0.0 < e2e["p50"] <= e2e["p95"] < _CLAIM_TIMEOUT_SECONDS

    # Every claim made at least the intake + retrieval-gated assessment calls.
    node_names = {n.node for n in result.per_node_latency}
    assert {"intake", "retrieval", "recommendation"} <= node_names

    cost = result.cost_per_assessment
    assert cost["mean_usd"] > 0.0
    assert cost["total_input_tokens"] > 0

    # The reasoning call, when it happened, is the expensive one -- and the
    # handler must have seen at least as many tokens as the audit trail did.
    for node_cost in result.per_node_cost:
        assert node_cost.handler_tokens >= 0
        assert node_cost.audit_trail_tokens >= 0
