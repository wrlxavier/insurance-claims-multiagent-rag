"""The graph state <-> application boundary mapper [M5-04]."""

from datetime import UTC, datetime

import pytest

from domain.assessment import Assessment
from domain.citation import Citation as DomainCitation
from domain.clause_classification import ClauseType
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict
from infrastructure.graph import state
from infrastructure.graph.state_mapper import (
    audit_entries_from_records,
    result_from_final_state,
    resume_payload,
)

_SUSEP = "15414.610650/2024-59"


def _state_citation(clause_id: str = "doc:1.1") -> state.Citation:
    return state.Citation(
        clause_id=clause_id,
        document_id="doc",
        susep_process=_SUSEP,
        clause_type=ClauseType.COVERAGE,
        relevance_score=0.9,
        excerpt="A cobertura compreende colisao.",
    )


def _recommendation(**overrides: object) -> state.Recommendation:
    fields: dict[str, object] = {
        "recommended_action": "Encaminhar para analise.",
        "justification": "A colisao esta coberta pela clausula 1.1.",
        "citations": [_state_citation()],
        "consistency_flags": [],
        "confidence": 0.7,
    }
    fields.update(overrides)
    return state.Recommendation(**fields)


def _rec_event(
    verdict: str = "compatible", posture: str = "compatible"
) -> state.AuditEvent:
    return state.AuditEvent(
        node="recommendation",
        action="consolidate",
        node_input=f"posture={posture} verdict={verdict} n_clauses=1",
    )


@pytest.mark.unit
def test_result_from_finished_state() -> None:
    final_state = {
        "recommendation": _recommendation(),
        "audit_trail": [_rec_event()],
        "context_sufficient": True,
        "clarification_exhausted": False,
        "missing_information": [],
    }

    result = result_from_final_state(final_state, awaiting_review=False)

    assert result.verdict is Verdict.COMPATIBLE
    assert result.awaiting_review is False
    assert result.reasoning == "A colisao esta coberta pela clausula 1.1."
    assert result.recommended_action == "Encaminhar para analise."
    assert isinstance(result.citations[0].susep_process, SusepProcess)
    assert result.citations[0].susep_process.value == _SUSEP


@pytest.mark.unit
def test_missing_recommendation_is_a_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="no recommendation"):
        result_from_final_state({"audit_trail": []}, awaiting_review=True)


@pytest.mark.unit
def test_verdict_falls_back_to_insufficient_information() -> None:
    final_state = {"recommendation": _recommendation(), "audit_trail": []}

    result = result_from_final_state(final_state, awaiting_review=True)

    assert result.verdict is Verdict.INSUFFICIENT_INFORMATION


@pytest.mark.unit
def test_abstain_recommendation_maps_to_empty_citations() -> None:
    final_state = {
        "recommendation": _recommendation(citations=[]),
        "audit_trail": [_rec_event("insufficient_information", "retrieval_miss")],
        "context_sufficient": False,
    }

    result = result_from_final_state(final_state, awaiting_review=True)

    assert result.citations == ()
    assert result.context_sufficient is False


@pytest.mark.unit
def test_audit_entries_from_records_number_and_flatten() -> None:
    records = [
        state.AuditRecord(state.AuditEvent(node="retrieval", action="retrieve")),
        state.AuditRecord(
            state.AuditEvent(
                node="compatibility",
                action="assess",
                token_usage=state.TokenUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
            ),
            {"detail": 1},
        ),
    ]

    entries = audit_entries_from_records(records)

    assert [e.sequence for e in entries] == [0, 1]
    assert entries[1].input_tokens == 10
    assert entries[1].total_tokens == 15
    assert entries[1].payload == {"detail": 1}
    assert entries[0].input_tokens is None


@pytest.mark.unit
def test_resume_payload_for_approve() -> None:
    decision = HumanDecision(
        assessment_id="a1",
        decision=DecisionOutcome.APPROVE,
        decided_at=datetime(2026, 9, 3, tzinfo=UTC),
        notes="ok",
    )

    payload = resume_payload(decision)

    assert payload == {
        "decision": "approve",
        "notes": "ok",
        "decided_at": "2026-09-03T00:00:00+00:00",
    }


@pytest.mark.unit
def test_resume_payload_for_edit_carries_a_flat_recommendation() -> None:
    edited = Assessment(
        assessment_id="a1",
        claim_id="c1",
        verdict=Verdict.INCOMPATIBLE,
        reasoning="Exclusao aplicavel.",
        citations=(
            DomainCitation(
                clause_id="doc:3.1",
                document_id="doc",
                susep_process=SusepProcess(_SUSEP),
                clause_type=ClauseType.EXCLUSION,
                excerpt="trecho",
            ),
        ),
        confidence=0.6,
        recommended_action="Negar.",
    )
    decision = HumanDecision(
        assessment_id="a1",
        decision=DecisionOutcome.EDIT,
        decided_at=datetime(2026, 9, 3, tzinfo=UTC),
        edited_assessment=edited,
    )

    payload = resume_payload(decision)
    edited_rec = payload["edited_recommendation"]

    assert isinstance(edited_rec, dict)
    assert edited_rec["justification"] == "Exclusao aplicavel."
    assert edited_rec["consistency_flags"] == []
    # the mapping round-trips through the graph twin's validator
    graph_decision = state.HumanDecision.model_validate(payload)
    assert graph_decision.edited_recommendation is not None
