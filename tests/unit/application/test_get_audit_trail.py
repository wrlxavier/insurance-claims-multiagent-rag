"""The GetAuditTrail use case [M5-04]."""

import pytest

from application.audit_trail_entry import AuditTrailEntry
from application.errors import AssessmentNotFoundError
from application.use_cases.get_audit_trail import GetAuditTrail
from tests.unit.application.fakes import (
    InMemoryAssessmentRepository,
    InMemoryAuditTrailReader,
    make_audit_entry,
    make_record,
)


def _use_case(
    *, seeded: bool, trail: list[AuditTrailEntry] | None = None
) -> GetAuditTrail:
    store = {"a1": make_record(assessment_id="a1")} if seeded else {}
    return GetAuditTrail(
        assessments=InMemoryAssessmentRepository(store),
        audit=InMemoryAuditTrailReader({"a1": trail or []}),
    )


@pytest.mark.unit
def test_unknown_id_raises_not_found() -> None:
    with pytest.raises(AssessmentNotFoundError):
        _use_case(seeded=False)("a1")


@pytest.mark.unit
def test_awaiting_review_assessment_has_an_empty_trail() -> None:
    assert _use_case(seeded=True)("a1") == ()


@pytest.mark.unit
def test_returns_the_trail_in_sequence_order() -> None:
    trail = [make_audit_entry(sequence=1), make_audit_entry(sequence=0)]
    result = _use_case(seeded=True, trail=trail)("a1")

    assert [e.sequence for e in result] == [0, 1]
