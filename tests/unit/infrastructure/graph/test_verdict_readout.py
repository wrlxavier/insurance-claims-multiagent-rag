"""Reading the effective verdict from the recommendation node's audit event [M4-08]."""

import pytest

from domain.verdict import Verdict
from infrastructure.graph.state import AuditEvent
from infrastructure.graph.verdict_readout import effective_verdict, posture_of


def _rec(node_input: str) -> AuditEvent:
    return AuditEvent(
        node="recommendation", action="consolidate", node_input=node_input
    )


@pytest.mark.unit
def test_reads_the_explicit_verdict_token() -> None:
    trail = [_rec("posture=incompatible verdict=incompatible n_clauses=2")]
    assert effective_verdict(trail) is Verdict.INCOMPATIBLE


@pytest.mark.unit
def test_falls_back_to_the_posture_map() -> None:
    trail = [_rec("posture=retrieval_miss n_clauses=0")]
    assert effective_verdict(trail) is Verdict.INSUFFICIENT_INFORMATION


@pytest.mark.unit
def test_uses_the_last_recommendation_event() -> None:
    trail = [
        _rec("posture=compatible verdict=compatible"),
        AuditEvent(node="human_review", action="human_decision:approve"),
        _rec("posture=incompatible verdict=incompatible"),
    ]
    assert effective_verdict(trail) is Verdict.INCOMPATIBLE


@pytest.mark.unit
def test_none_when_no_recommendation_event() -> None:
    trail = [AuditEvent(node="retrieval", action="retrieve_clauses", node_input="x")]
    assert effective_verdict(trail) is None
    assert posture_of(trail) is None


@pytest.mark.unit
def test_posture_of_extracts_the_posture() -> None:
    trail = [_rec("posture=claimant_gaps verdict=insufficient_information")]
    assert posture_of(trail) == "claimant_gaps"
