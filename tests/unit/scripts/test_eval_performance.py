"""The [M5-10] performance eval's own logic, offline.

The live run is `make eval-performance`; what is pinned here is the capture and
aggregation *around* the graph -- the node-log handler, the token callback's
run-id attribution, and the latency / cost / parallel-vs-sequential arithmetic.
All pure functions over constructed rows: no graph, no models, no database.
"""

import logging
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult
from scripts.eval_performance import (
    _ClaimPerf,
    _LatencySegments,
    _LlmCall,
    _NodeCostCallback,
    _NodeLatencyCollector,
    _summarise,
    render_markdown,
    snapshot_payload,
)

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings

pytestmark = pytest.mark.unit

_META: dict[str, Any] = {
    "generated_at_utc": "2026-09-07T00:00:00+00:00",
    "platform": "test",
    "reasoning_model": "reasoning-m",
    "fast_model": "fast-m",
    "reasoning_provider_order": ["alibaba"],
    "fast_provider_order": ["baidu/fp8"],
    "reasoning_price_per_1m_usd": [1.0, 4.0],
    "fast_price_per_1m_usd": [0.1, 0.2],
    "claim_count": 2,
    "cohort_counts": {"compatible": 2},
    "repeats": 1,
    "limit": None,
    "claim_timeout_seconds": 420.0,
    "method_note": "note",
}


def _settings() -> LlmSettings:
    return LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY="k",
        LLM_MODEL_FAST="fast-m",
        LLM_MODEL_REASONING="reasoning-m",
        EMBEDDING_MODEL="e",
        RERANKER_MODEL="r",
        LLM_FAST_INPUT_COST_PER_1M_TOKENS_USD=0.1,
        LLM_FAST_OUTPUT_COST_PER_1M_TOKENS_USD=0.2,
        LLM_REASONING_INPUT_COST_PER_1M_TOKENS_USD=1.0,
        LLM_REASONING_OUTPUT_COST_PER_1M_TOKENS_USD=4.0,
        _env_file=None,
    )


def _log_record(message: str, **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="infrastructure.graph.node",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- the node-latency handler --------------------------------------------- #


def test_collector_keeps_only_node_completed_and_failed_lines() -> None:
    collector = _NodeLatencyCollector()
    collector.emit(_log_record("node.start", node="intake", correlation_id="c1"))
    collector.emit(
        _log_record(
            "node.completed", node="intake", duration_ms=12.5, correlation_id="c1"
        )
    )
    collector.emit(
        _log_record(
            "node.failed", node="compatibility", duration_ms=3.0, correlation_id="c1"
        )
    )
    collector.emit(_log_record("request", path="/x", correlation_id="c1"))

    records = collector.drain("c1")
    assert records == [("intake", 12.5, False), ("compatibility", 3.0, True)]
    # drain clears
    assert collector.drain("c1") == []


def test_collector_demultiplexes_by_correlation_id() -> None:
    collector = _NodeLatencyCollector()
    collector.emit(
        _log_record(
            "node.completed", node="intake", duration_ms=1.0, correlation_id="a"
        )
    )
    collector.emit(
        _log_record(
            "node.completed", node="intake", duration_ms=2.0, correlation_id="b"
        )
    )
    assert collector.drain("a") == [("intake", 1.0, False)]
    assert collector.drain("b") == [("intake", 2.0, False)]


def test_collector_ignores_records_missing_the_fields() -> None:
    collector = _NodeLatencyCollector()
    collector.emit(
        _log_record("node.completed", correlation_id="c1")
    )  # no node/duration
    assert collector.drain("c1") == []


# --- the token callback --------------------------------------------------- #


class _FakeMessage:
    def __init__(self, usage_metadata: dict[str, Any] | None) -> None:
        self.usage_metadata = usage_metadata


class _FakeGeneration:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeLLMResult:
    def __init__(self, usage_metadata: dict[str, Any] | None) -> None:
        self.generations = [[_FakeGeneration(_FakeMessage(usage_metadata))]]


def _result(usage_metadata: dict[str, Any] | None) -> LLMResult:
    """A stand-in ``LLMResult`` -- ``_usage_of`` only reaches ``.generations``."""
    return cast(LLMResult, _FakeLLMResult(usage_metadata))


def test_callback_attributes_a_call_to_its_node_and_keeps_reasoning_tokens() -> None:
    cb = _NodeCostCallback()
    run_id = uuid4()
    cb.on_chat_model_start(
        {"id": ["x"]},
        [],
        run_id=run_id,
        metadata={"langgraph_node": "compatibility", "ls_model_name": "reasoning-m"},
    )
    cb.on_llm_end(
        _result(
            {
                "input_tokens": 3000,
                "output_tokens": 4000,
                "total_tokens": 7000,
                "output_token_details": {"reasoning": 3725},
                "input_token_details": {"cache_read": 100},
            }
        ),
        run_id=run_id,
    )
    assert len(cb.calls) == 1
    call = cb.calls[0]
    assert call.node == "compatibility"
    assert call.model == "reasoning-m"
    assert call.input_tokens == 3000
    assert call.output_tokens == 4000
    assert call.reasoning_tokens == 3725
    assert call.cache_read_tokens == 100
    assert call.tier == "reasoning"


def test_callback_unknown_node_when_metadata_absent() -> None:
    cb = _NodeCostCallback()
    run_id = uuid4()
    cb.on_chat_model_start({"id": ["x"]}, [], run_id=run_id, metadata=None)
    cb.on_llm_end(_result({"input_tokens": 1, "output_tokens": 2}), run_id=run_id)
    assert cb.calls[0].node == "unknown"
    assert cb.calls[0].tier == "fast"


def test_callback_records_a_failed_call_with_zero_tokens() -> None:
    cb = _NodeCostCallback()
    run_id = uuid4()
    cb.on_chat_model_start(
        {"id": ["x"]}, [], run_id=run_id, metadata={"langgraph_node": "intake"}
    )
    cb.on_llm_error(RuntimeError("boom"), run_id=run_id)
    assert cb.calls == [_LlmCall("intake", None, 0, 0, 0, 0, failed=True)]


def test_callback_run_ids_do_not_cross_in_a_concurrent_superstep() -> None:
    cb = _NodeCostCallback()
    compat_id, consist_id = uuid4(), uuid4()
    cb.on_chat_model_start(
        {}, [], run_id=compat_id, metadata={"langgraph_node": "compatibility"}
    )
    cb.on_chat_model_start(
        {}, [], run_id=consist_id, metadata={"langgraph_node": "consistency"}
    )
    cb.on_llm_end(_result({"input_tokens": 10, "output_tokens": 1}), run_id=consist_id)
    cb.on_llm_end(_result({"input_tokens": 20, "output_tokens": 2}), run_id=compat_id)
    by_node = {c.node: c.input_tokens for c in cb.calls}
    assert by_node == {"consistency": 10, "compatibility": 20}


# --- aggregation -------------------------------------------------------- #


def _row(**overrides: Any) -> _ClaimPerf:
    base: dict[str, Any] = {
        "claim_id": "compatible-001",
        "cohort": "compatible",
        "repeat": 1,
        "reached_checkpoint": True,
        "completed": True,
        "context_sufficient": True,
        "clarification_rounds": 0,
        "predicted_verdict": "compatible",
        "latency": _LatencySegments(to_interrupt_s=90.0, resume_s=10.0),
        "node_latencies_ms": {
            "intake": [8000.0],
            "retrieval": [5000.0],
            "compatibility": [60000.0],
            "consistency": [20000.0],
            "injection_scan": [5.0],
            "recommendation": [8000.0],
            "human_review": [2.0],
        },
        "node_failed": (),
        "calls": (
            _LlmCall("intake", "fast-m", 1000, 100, 0, 0),
            _LlmCall("compatibility", "reasoning-m", 3000, 4000, 3725, 0),
            _LlmCall("consistency", "fast-m", 800, 200, 0, 0),
            _LlmCall("recommendation", "fast-m", 1200, 250, 0, 0),
        ),
        "audit_tokens_by_node": {
            "intake": 1100,
            "compatibility": 7275,  # audit trail folds reasoning into total
            "consistency": 1000,
            "recommendation": 1450,
        },
        "error": None,
    }
    base.update(overrides)
    return _ClaimPerf(**base)


def test_summarise_end_to_end_latency_and_split() -> None:
    result = _summarise(
        [_row(), _row(claim_id="c2")], [], settings=_settings(), meta=_META
    )
    assert result.n_scored == 2
    assert result.end_to_end_latency["p50"] == 100.0
    assert result.latency_split["mean_to_interrupt_s"] == 90.0
    assert result.latency_split["mean_resume_s"] == 10.0


def test_summarise_per_node_latency_has_a_row_per_node_seen() -> None:
    result = _summarise([_row()], [], settings=_settings(), meta=_META)
    nodes = {n.node for n in result.per_node_latency}
    assert {"intake", "compatibility", "consistency", "recommendation"} <= nodes
    compat = next(n for n in result.per_node_latency if n.node == "compatibility")
    assert compat.mean_ms == 60000.0
    assert compat.calls_per_assessment == 1.0


def test_summarise_costs_the_reasoning_call_at_the_reasoning_rate() -> None:
    result = _summarise([_row()], [], settings=_settings(), meta=_META)
    # compatibility: 3000 in * 1.0/1e6 + 4000 out * 4.0/1e6 = 0.003 + 0.016 = 0.019
    compat = next(c for c in result.per_node_cost if c.node == "compatibility")
    assert compat.mean_usd == pytest.approx(0.019, abs=1e-9)
    assert compat.mean_reasoning_tokens == 3725.0
    assert result.cost_per_assessment["total_reasoning_tokens"] == 3725


def test_summarise_reports_the_audit_trail_token_gap() -> None:
    result = _summarise([_row()], [], settings=_settings(), meta=_META)
    compat = next(c for c in result.per_node_cost if c.node == "compatibility")
    assert compat.handler_tokens == 7000  # 3000 + 4000
    assert compat.audit_trail_tokens == 7275


def test_summarise_parallel_saving_is_seq_minus_max_and_cost_is_identical() -> None:
    result = _summarise([_row()], [], settings=_settings(), meta=_META)
    assert result.parallel is not None
    p = result.parallel
    # t_seq = 60000 + 20000 + 5 = 80005 ; t_par = max(...) = 60000
    assert p.saving_mean_ms == pytest.approx(20005.0)
    assert p.n_claims == 1
    # assessment-branch cost = compatibility 0.019 + consistency 0.00012, to 4dp
    assert p.token_cost_identical_usd == pytest.approx(0.0191)


def test_summarise_excludes_errored_claims_from_scoring() -> None:
    errored = _row(
        claim_id="e1", error="boom", completed=False, calls=(), node_latencies_ms={}
    )
    result = _summarise([_row(), errored], ["e1"], settings=_settings(), meta=_META)
    assert result.n_scored == 1
    assert result.n_claims == 2
    assert result.error_claim_ids == ["e1"]


def test_summarise_handles_a_run_with_no_parallel_branch() -> None:
    starved = _row(
        context_sufficient=False,
        node_latencies_ms={
            "intake": [8000.0],
            "retrieval": [5000.0],
            "recommendation": [7000.0],
        },
        calls=(_LlmCall("intake", "fast-m", 1000, 100, 0, 0),),
        audit_tokens_by_node={"intake": 1100},
    )
    result = _summarise([starved], [], settings=_settings(), meta=_META)
    assert result.parallel is None


# --- rendering --------------------------------------------------------- #


def test_render_markdown_carries_every_dod_section() -> None:
    markdown = render_markdown(
        _summarise([_row()], [], settings=_settings(), meta=_META)
    )
    for heading in (
        "## End-to-end latency",
        "## Per-node latency",
        "## Token cost per assessment",
        "## Parallel branch vs sequential",
        "## One-off corpus indexing cost",
        "## Findings",
    ):
        assert heading in markdown


def test_snapshot_payload_has_provenance_and_no_per_claim_rows() -> None:
    payload = snapshot_payload(
        _summarise([_row()], [], settings=_settings(), meta=_META)
    )
    assert payload["provenance"]["generated_by"] == "scripts/eval_performance.py"
    assert "rows" not in payload
    assert "per_node_latency_ms" in payload
