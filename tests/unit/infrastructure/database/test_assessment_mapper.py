"""The assessment aggregate <-> ORM-row mapper -- [M5-03].

Pure round-trip checks, no database: ``rows_to_record(record_to_rows(x)) == x``
across the shapes that matter -- a grounded record, a 0-citation abstain, and a
settled record for each decision outcome (an ``edit`` carries a nested
``Assessment``).
"""

from datetime import UTC, datetime

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from domain.assessment import Assessment
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from infrastructure.database.assessment_mapper import record_to_rows, rows_to_record
from tests.unit.application.fakes import (
    make_citation,
    make_consistency_flag,
    make_record,
)

_DECIDED_AT = datetime(2026, 9, 3, 15, 30, tzinfo=UTC)


def _round_trip(record: AssessmentRecord) -> AssessmentRecord:
    assessment_row, decision_row = record_to_rows(record)
    return rows_to_record(assessment_row, decision_row)


@pytest.mark.unit
def test_grounded_record_round_trips() -> None:
    record = make_record(
        citations=(make_citation(), make_citation(clause_id="doc:3.2")),
        consistency_flags=(make_consistency_flag(),),
        missing_information=("vehicle_info",),
        context_sufficient=True,
    )

    assert _round_trip(record) == record


@pytest.mark.unit
def test_abstain_record_round_trips_with_empty_json_arrays() -> None:
    record = make_record(
        verdict=Verdict.INSUFFICIENT_INFORMATION,
        citations=(),
        consistency_flags=(),
        confidence=0.2,
        context_sufficient=False,
        missing_information=(),
    )

    assessment_row, decision_row = record_to_rows(record)

    assert decision_row is None
    assert assessment_row.citations == []
    assert assessment_row.consistency_flags == []
    assert rows_to_record(assessment_row, decision_row) == record


@pytest.mark.unit
@pytest.mark.parametrize("outcome", [DecisionOutcome.APPROVE, DecisionOutcome.REJECT])
def test_settled_record_round_trips_without_an_edit(
    outcome: DecisionOutcome,
) -> None:
    decision = HumanDecision(
        assessment_id="assessment-1",
        decision=outcome,
        decided_at=_DECIDED_AT,
        notes="conferido",
    )
    record = make_record(status=AssessmentStatus.DECIDED, decision=decision)

    assessment_row, decision_row = record_to_rows(record)

    assert decision_row is not None
    assert decision_row.edited_assessment is None
    assert _round_trip(record) == record


@pytest.mark.unit
def test_settled_record_round_trips_with_a_nested_edited_assessment() -> None:
    edited = Assessment(
        assessment_id="assessment-1",
        claim_id="claim-1",
        verdict=Verdict.INCOMPATIBLE,
        reasoning="O evento e uma enchente, excluida pela clausula 5.1.",
        citations=(make_citation(clause_id="doc:5.1"),),
        confidence=0.55,
        recommended_action="Negar a cobertura.",
    )
    decision = HumanDecision(
        assessment_id="assessment-1",
        decision=DecisionOutcome.EDIT,
        decided_at=_DECIDED_AT,
        notes="reclassificado",
        edited_assessment=edited,
    )
    record = make_record(status=AssessmentStatus.DECIDED, decision=decision)

    _, decision_row = record_to_rows(record)
    assert decision_row is not None
    assert decision_row.edited_assessment is not None

    restored = _round_trip(record)
    assert restored == record
    assert restored.decision is not None
    assert restored.decision.edited_assessment == edited
