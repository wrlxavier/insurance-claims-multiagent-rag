"""State merges for the agent graph, including the parallel fan-in [M4-01].

The DoD asks for "reducers for fields written by parallel nodes so the fan-in
has defined semantics rather than a race", tested "including the parallel
case". The faithful test drives a real ``StateGraph`` with two branches
writing the same channel in one superstep.
"""

from typing import Annotated, TypedDict

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

from domain.verdict import Verdict
from infrastructure.graph.state import (
    AuditEvent,
    ClaimState,
    CompatibilityAssessment,
    ConsistencyReport,
    append_audit_events,
)


def _event(node: str) -> AuditEvent:
    return AuditEvent(node=node, action="run")


# --- the reducer as a pure function ----------------------------------------


@pytest.mark.unit
def test_append_audit_events_concatenates_accumulated_first() -> None:
    left = [_event("a")]
    right = [_event("b")]

    merged = append_audit_events(left, right)

    assert [e.node for e in merged] == ["a", "b"]


@pytest.mark.unit
def test_append_audit_events_handles_empty_sides() -> None:
    only = [_event("a")]

    assert append_audit_events([], only) == only
    assert append_audit_events(only, []) == only
    assert append_audit_events([], []) == []


@pytest.mark.unit
def test_append_audit_events_does_not_mutate_its_inputs() -> None:
    left = [_event("a")]
    right = [_event("b")]

    append_audit_events(left, right)

    assert len(left) == 1
    assert len(right) == 1


# --- the parallel fan-in through a real StateGraph -----------------------


def _build_parallel_graph() -> object:
    def assess(state: ClaimState) -> dict[str, object]:
        return {
            "compatibility": CompatibilityAssessment(
                verdict=Verdict.COMPATIBLE,
                reasoning="ok",
                citations=[],
                confidence=0.6,
            ),
            "audit_trail": [_event("assess")],
        }

    def consistency(state: ClaimState) -> dict[str, object]:
        return {
            "consistency": ConsistencyReport(signals=[]),
            "audit_trail": [_event("consistency")],
        }

    graph = StateGraph(ClaimState)
    graph.add_node("assess", assess)
    graph.add_node("consistency", consistency)
    graph.add_edge(START, "assess")
    graph.add_edge(START, "consistency")
    graph.add_edge("assess", END)
    graph.add_edge("consistency", END)
    return graph.compile()


@pytest.mark.unit
def test_parallel_branches_both_land_in_the_audit_trail() -> None:
    out = _build_parallel_graph().invoke(  # type: ignore[attr-defined]
        {"claim_id": "c1", "raw_claim_text": "bati o carro no portao"}
    )

    assert len(out["audit_trail"]) == 2
    assert {e.node for e in out["audit_trail"]} == {"assess", "consistency"}


@pytest.mark.unit
def test_parallel_branches_write_their_disjoint_fields_without_clobbering() -> None:
    out = _build_parallel_graph().invoke(  # type: ignore[attr-defined]
        {"claim_id": "c1", "raw_claim_text": "bati o carro no portao"}
    )

    assert out["compatibility"] is not None
    assert out["consistency"] is not None
    assert out["claim_id"] == "c1"


@pytest.mark.unit
def test_a_shared_channel_without_a_reducer_races() -> None:
    # Why audit_trail carries a reducer: the same concurrent write to a plain
    # channel is a hard error, not a silent last-writer-wins.
    class _Racy(TypedDict, total=False):
        seen: list[str]
        # audit_trail keeps its reducer so only `seen` is the racy channel
        audit_trail: Annotated[list[AuditEvent], append_audit_events]

    def one(state: _Racy) -> dict[str, object]:
        return {"seen": ["one"]}

    def two(state: _Racy) -> dict[str, object]:
        return {"seen": ["two"]}

    graph = StateGraph(_Racy)
    graph.add_node("one", one)
    graph.add_node("two", two)
    graph.add_edge(START, "one")
    graph.add_edge(START, "two")
    graph.add_edge("one", END)
    graph.add_edge("two", END)
    compiled = graph.compile()

    with pytest.raises(InvalidUpdateError):
        compiled.invoke({})
