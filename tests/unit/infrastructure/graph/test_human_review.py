"""Unit tests for the human checkpoint node ([M4-09]).

A one-node ``StateGraph`` over an ``InMemorySaver``, with the recommendation
seeded into the input state -- the node makes no LLM call, so there is no fake
model here. The checkpointer is not optional: ``interrupt()`` needs somewhere to
park, and without one the graph would return at the pause with no error.

The load-bearing checks are the two the DoD names: the node does nothing
observable before the pause even though everything above it runs twice, and the
analyst's decision is recorded *beside* the system recommendation rather than
over it. Persistence across a real process restart is
``tests/integration/test_human_checkpoint.py``.
"""

from collections.abc import Sequence
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from domain.clause_classification import ClauseType
from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.graph.context import AuditTrailSink, GraphContext, RetrievalPort
from infrastructure.graph.nodes.human_review import DECISION_OPTIONS, human_review
from infrastructure.graph.state import (
    AuditEvent,
    AuditRecord,
    Citation,
    ClaimState,
    ConsistencySignal,
    HumanDecision,
    Recommendation,
)
from infrastructure.rag.retrieved_clause import RetrievedClause

THREAD_ID = "thread-1"


class _SpySink:
    """Counts ``record`` calls and keeps what it was handed."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, list[AuditRecord]]] = []
        self._fail = fail

    def record(
        self, *, claim_id: str, thread_id: str, records: Sequence[AuditRecord]
    ) -> int:
        self.calls.append((claim_id, thread_id, list(records)))
        if self._fail:
            raise RuntimeError("audit store unreachable")
        return len(records)


class _StubRetriever:
    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
        return []


def _citation(clause_id: str = "doc-1:1.1") -> Citation:
    return Citation(
        clause_id=clause_id,
        document_id="doc-1",
        susep_process="15414.900000/2013-00",
        clause_type=ClauseType.COVERAGE,
        relevance_score=0.9,
        excerpt="A seguradora cobre colisao.",
    )


def _recommendation(justification: str = "Resumo do sistema.") -> Recommendation:
    return Recommendation(
        recommended_action="Encaminhar para revisão humana.",
        justification=justification,
        citations=[_citation()],
        consistency_flags=[
            ConsistencySignal(
                check="date_in_future",
                severity="attention",
                detail="data do evento no futuro",
                source="deterministic",
            )
        ],
        confidence=0.6,
    )


def _context(sink: AuditTrailSink | None = None) -> GraphContext:
    model = cast(BaseChatModel, object())
    return GraphContext(
        fast_model=model,
        reasoning_model=model,
        retriever=cast(RetrievalPort, _StubRetriever()),
        llm_settings=LlmSettings(
            LLM_PROVIDER=LlmProvider.OPENAI,
            LLM_API_KEY="test-key",
            LLM_MODEL_FAST="fake-fast-model",
            LLM_MODEL_REASONING="fake-reasoning-model",
            EMBEDDING_MODEL="embed-model",
            RERANKER_MODEL="rerank-model",
            _env_file=None,
        ),
        audit_sink=sink,
    )


def _compiled() -> Any:
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("human_review", human_review)
    builder.add_edge(START, "human_review")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=InMemorySaver())


def _state(**overrides: Any) -> Any:
    state: dict[str, Any] = {
        "claim_id": "c1",
        "raw_claim_text": "bati o carro",
        "recommendation": _recommendation(),
        "audit_trail": [AuditEvent(node="recommendation", action="consolidate")],
    }
    state.update(overrides)
    return state


def _pause(compiled: Any, context: GraphContext, **overrides: Any) -> tuple[Any, Any]:
    """Run to the checkpoint. Return ``(config, the interrupt payload)``."""
    config: Any = {"configurable": {"thread_id": THREAD_ID}}
    out = compiled.invoke(_state(**overrides), config=config, context=context)
    return config, out["__interrupt__"][0].value


# --- the pause -------------------------------------------------------------


@pytest.mark.unit
def test_the_graph_stops_and_surfaces_the_whole_recommendation() -> None:
    sink = _SpySink()
    _, payload = _pause(_compiled(), _context(sink))

    assert payload["claim_id"] == "c1"
    assert payload["decision_options"] == list(DECISION_OPTIONS)
    surfaced = payload["recommendation"]
    assert surfaced["justification"] == "Resumo do sistema."
    assert [c["clause_id"] for c in surfaced["citations"]] == ["doc-1:1.1"]
    assert len(surfaced["consistency_flags"]) == 1
    assert surfaced["confidence"] == 0.6


@pytest.mark.unit
def test_the_review_payload_is_plain_json() -> None:
    # It crosses a checkpoint and, in [M5-04], an HTTP boundary: the reader must
    # not need this project's Pydantic models to make sense of it.
    import json

    _, payload = _pause(_compiled(), _context())
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.unit
def test_missing_recommendation_is_an_assembly_error() -> None:
    compiled = _compiled()
    config: Any = {"configurable": {"thread_id": THREAD_ID}}
    with pytest.raises(ValueError, match="no recommendation in state"):
        compiled.invoke(_state(recommendation=None), config=config, context=_context())


# --- re-execution semantics (DoD: no side effect before the pause) ---------


@pytest.mark.unit
def test_the_durable_write_happens_once_and_only_after_the_pause() -> None:
    # LangGraph re-runs an interrupted node from the top, so everything above
    # the interrupt executes twice. The audit write sits below it and must
    # therefore fire exactly once across the whole approve cycle.
    sink = _SpySink()
    context = _context(sink)
    compiled = _compiled()

    config, _ = _pause(compiled, context)
    assert sink.calls == [], "nothing may be written before the human decides"

    compiled.invoke(
        Command(resume={"decision": "approve"}), config=config, context=context
    )
    assert len(sink.calls) == 1


@pytest.mark.unit
def test_the_durable_write_carries_the_whole_trail_and_the_decision_payload() -> None:
    sink = _SpySink()
    context = _context(sink)
    compiled = _compiled()
    config, _ = _pause(compiled, context)

    compiled.invoke(
        Command(resume={"decision": "reject", "notes": "fora de vigência"}),
        config=config,
        context=context,
    )

    claim_id, thread_id, records = sink.calls[0]
    assert (claim_id, thread_id) == ("c1", THREAD_ID)
    # the upstream event, then the checkpoint's own -- in order, so the index of
    # a record is its `sequence` in the table
    assert [r.event.node for r in records] == ["recommendation", "human_review"]
    assert records[0].payload is None
    decision_payload = records[-1].payload
    assert decision_payload is not None
    assert decision_payload["decision"] == "reject"
    assert decision_payload["notes"] == "fora de vigência"
    assert "decided_at" in decision_payload


@pytest.mark.unit
def test_a_failing_sink_does_not_cost_the_decision() -> None:
    # Raising here would strand the resume value in the checkpoint and make the
    # thread unfinishable -- the decision would be lost for good.
    context = _context(_SpySink(fail=True))
    compiled = _compiled()
    config, _ = _pause(compiled, context)

    out = compiled.invoke(
        Command(resume={"decision": "approve"}), config=config, context=context
    )

    assert cast(HumanDecision, out["human_decision"]).decision == "approve"
    actions = [event.action for event in out["audit_trail"]]
    assert "persist_audit_trail_failed" in actions


@pytest.mark.unit
def test_no_sink_means_no_durable_record_and_no_complaint() -> None:
    compiled = _compiled()
    context = _context(None)
    config, _ = _pause(compiled, context)

    out = compiled.invoke(
        Command(resume={"decision": "approve"}), config=config, context=context
    )

    assert cast(HumanDecision, out["human_decision"]).decision == "approve"
    assert [e.node for e in out["audit_trail"]] == ["recommendation", "human_review"]


# --- the decision ---------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("verdict", ["approve", "reject"])
def test_a_decision_is_recorded_beside_the_system_recommendation(verdict: str) -> None:
    compiled = _compiled()
    context = _context()
    config, _ = _pause(compiled, context)

    out = compiled.invoke(
        Command(resume={"decision": verdict, "notes": "conferido"}),
        config=config,
        context=context,
    )

    decision = cast(HumanDecision, out["human_decision"])
    assert decision.decision == verdict
    assert decision.notes == "conferido"
    assert decision.edited_recommendation is None
    assert out["recommendation"] == _recommendation()
    assert out["audit_trail"][-1].action == f"human_decision:{verdict}"


@pytest.mark.unit
def test_an_edit_never_overwrites_the_original_recommendation() -> None:
    compiled = _compiled()
    context = _context()
    config, _ = _pause(compiled, context)
    edited = _recommendation("O analista reescreveu a justificativa.")

    out = compiled.invoke(
        Command(
            resume={
                "decision": "edit",
                "notes": "reescrito",
                "edited_recommendation": edited.model_dump(mode="json"),
            }
        ),
        config=config,
        context=context,
    )

    decision = cast(HumanDecision, out["human_decision"])
    assert decision.edited_recommendation == edited
    # the system's original opinion survives untouched, side by side with it
    assert out["recommendation"] == _recommendation()
    assert out["recommendation"] != decision.edited_recommendation


@pytest.mark.unit
def test_a_human_decision_instance_resumes_as_well_as_a_mapping() -> None:
    compiled = _compiled()
    context = _context()
    config, _ = _pause(compiled, context)

    out = compiled.invoke(
        Command(resume=HumanDecision(decision="approve")),
        config=config,
        context=context,
    )

    assert cast(HumanDecision, out["human_decision"]).decision == "approve"


# --- a malformed decision re-asks rather than bricking the thread ----------


@pytest.mark.unit
def test_an_invalid_decision_reopens_the_checkpoint_with_the_error() -> None:
    # Raising instead would be unrecoverable: the bad resume value stays in the
    # checkpoint's pending writes and replays on every later resume.
    sink = _SpySink()
    context = _context(sink)
    compiled = _compiled()
    config, first = _pause(compiled, context)
    assert "error" not in first

    # "edit" with no revision -- rejected by HumanDecision's own validator
    again = compiled.invoke(
        Command(resume={"decision": "edit"}), config=config, context=context
    )
    payload = again["__interrupt__"][0].value
    assert "edited_recommendation" in payload["error"]
    assert payload["recommendation"] == first["recommendation"]
    assert sink.calls == [], "an invalid decision writes nothing"

    out = compiled.invoke(
        Command(resume={"decision": "approve"}), config=config, context=context
    )
    assert cast(HumanDecision, out["human_decision"]).decision == "approve"
    assert len(sink.calls) == 1


@pytest.mark.unit
def test_a_decision_outside_the_vocabulary_also_reasks() -> None:
    compiled = _compiled()
    context = _context()
    config, _ = _pause(compiled, context)

    again = compiled.invoke(
        Command(resume={"decision": "maybe"}), config=config, context=context
    )

    assert "__interrupt__" in again
    assert "human_decision" not in again
