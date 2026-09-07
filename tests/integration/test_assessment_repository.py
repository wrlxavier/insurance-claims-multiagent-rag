"""The SQLAlchemy ``AssessmentRepository`` against a real Postgres -- [M5-03].

Covers what only the database proves: the aggregate round-trips through the
``assessment`` / ``human_decision`` tables (grounded, abstain, and each decision
outcome), ``list`` orders and filters in SQL, timestamps come back tz-aware, and
a duplicate ``add`` is rejected by the primary key.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from application.assessment_record import AssessmentStatus
from domain.assessment import Assessment
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from infrastructure.database.assessment_repository import SqlAlchemyAssessmentRepository
from tests.unit.application.fakes import (
    make_citation,
    make_consistency_flag,
    make_record,
)

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _repo(session: Session) -> SqlAlchemyAssessmentRepository:
    return SqlAlchemyAssessmentRepository(session)


def _decision(
    assessment_id: str,
    outcome: DecisionOutcome = DecisionOutcome.APPROVE,
    *,
    edited: Assessment | None = None,
) -> HumanDecision:
    return HumanDecision(
        assessment_id=assessment_id,
        decision=outcome,
        decided_at=_NOW + timedelta(hours=1),
        notes="conferido",
        edited_assessment=edited,
    )


def test_grounded_record_round_trips(db_session: Session) -> None:
    repo = _repo(db_session)
    record = make_record(
        assessment_id="a-grounded",
        citations=(make_citation(), make_citation(clause_id="doc:3.2")),
        consistency_flags=(make_consistency_flag(),),
        missing_information=("vehicle_info",),
        context_sufficient=True,
    )

    repo.add(record)
    db_session.commit()

    assert repo.get("a-grounded") == record


def test_abstain_record_round_trips(db_session: Session) -> None:
    repo = _repo(db_session)
    record = make_record(
        assessment_id="a-abstain",
        verdict=Verdict.INSUFFICIENT_INFORMATION,
        citations=(),
        consistency_flags=(),
        confidence=0.2,
        context_sufficient=False,
        missing_information=(),
    )

    repo.add(record)
    db_session.commit()

    fetched = repo.get("a-abstain")
    assert fetched == record
    assert fetched is not None and fetched.citations == ()


def test_get_unknown_id_returns_none(db_session: Session) -> None:
    assert _repo(db_session).get("nope") is None


def test_duplicate_add_is_rejected(db_session: Session) -> None:
    repo = _repo(db_session)
    repo.add(make_record(assessment_id="a-dup"))
    db_session.commit()

    with pytest.raises(IntegrityError):
        repo.add(make_record(assessment_id="a-dup"))
        db_session.flush()


@pytest.mark.parametrize("outcome", [DecisionOutcome.APPROVE, DecisionOutcome.REJECT])
def test_update_to_decided_without_an_edit(
    db_session: Session, outcome: DecisionOutcome
) -> None:
    repo = _repo(db_session)
    repo.add(make_record(assessment_id="a-decide"))
    db_session.commit()

    decided = make_record(
        assessment_id="a-decide",
        status=AssessmentStatus.DECIDED,
        decision=_decision("a-decide", outcome),
    )
    repo.update(decided)
    db_session.commit()

    fetched = repo.get("a-decide")
    assert fetched == decided
    assert fetched is not None and fetched.decision is not None
    assert fetched.decision.decision is outcome
    assert fetched.decision.decided_at.tzinfo is not None


def test_update_to_decided_with_a_nested_edited_assessment(
    db_session: Session,
) -> None:
    repo = _repo(db_session)
    repo.add(make_record(assessment_id="a-edit"))
    db_session.commit()

    edited = Assessment(
        assessment_id="a-edit",
        claim_id="claim-1",
        verdict=Verdict.INCOMPATIBLE,
        reasoning="Reclassificado como enchente, excluida.",
        citations=(make_citation(clause_id="doc:5.1"),),
        confidence=0.6,
        recommended_action="Negar.",
    )
    decided = make_record(
        assessment_id="a-edit",
        status=AssessmentStatus.DECIDED,
        decision=_decision("a-edit", DecisionOutcome.EDIT, edited=edited),
    )
    repo.update(decided)
    db_session.commit()

    fetched = repo.get("a-edit")
    assert fetched == decided
    assert fetched is not None and fetched.decision is not None
    assert fetched.decision.edited_assessment == edited


def test_update_missing_record_raises(db_session: Session) -> None:
    with pytest.raises(KeyError):
        _repo(db_session).update(make_record(assessment_id="a-ghost"))


def test_list_orders_newest_first_and_filters(db_session: Session) -> None:
    repo = _repo(db_session)
    repo.add(make_record(assessment_id="a-old", claim_id="c-1", created_at=_NOW))
    repo.add(
        make_record(
            assessment_id="a-mid",
            claim_id="c-2",
            created_at=_NOW + timedelta(hours=1),
        )
    )
    newest = make_record(
        assessment_id="a-new",
        claim_id="c-1",
        created_at=_NOW + timedelta(hours=2),
    )
    repo.add(newest)
    db_session.commit()

    assert [r.assessment_id for r in repo.list()] == ["a-new", "a-mid", "a-old"]
    assert [r.assessment_id for r in repo.list(claim_id="c-1")] == [
        "a-new",
        "a-old",
    ]
    assert [
        r.assessment_id for r in repo.list(status=AssessmentStatus.AWAITING_REVIEW)
    ] == ["a-new", "a-mid", "a-old"]
    assert [r.assessment_id for r in repo.list(limit=1, offset=1)] == ["a-mid"]


def test_list_hydrates_the_decision_for_a_settled_row(db_session: Session) -> None:
    repo = _repo(db_session)
    repo.add(make_record(assessment_id="a-1", claim_id="c", created_at=_NOW))
    settled = make_record(
        assessment_id="a-2",
        claim_id="c",
        created_at=_NOW + timedelta(hours=1),
        status=AssessmentStatus.DECIDED,
        decision=_decision("a-2", DecisionOutcome.REJECT),
    )
    repo.add(settled)
    db_session.commit()

    listed = {r.assessment_id: r for r in repo.list(claim_id="c")}
    assert listed["a-1"].decision is None
    assert listed["a-2"].decision is not None
    assert listed["a-2"].decision.decision is DecisionOutcome.REJECT
