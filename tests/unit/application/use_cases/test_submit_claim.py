"""The SubmitClaim use case [M5-02]."""

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.errors import OrchestratorContractError
from application.use_cases.submit_claim import SubmitClaim
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    SUSEP,
    FakeClaimAssessmentOrchestrator,
    FixedClock,
    NaiveClock,
    SequentialIds,
    abstain_result,
    make_orchestrator_result,
    make_uow_factory,
)

_NARRATIVE = "Bati o carro em uma colisao na avenida no dia 05/01/2026."


def _build(
    *,
    orchestrator: FakeClaimAssessmentOrchestrator | None = None,
    store: dict[str, AssessmentRecord] | None = None,
    clock: object | None = None,
    ids: SequentialIds | None = None,
) -> tuple[SubmitClaim, dict[str, AssessmentRecord], FakeClaimAssessmentOrchestrator]:
    resolved_store = store if store is not None else {}
    resolved_orch = orchestrator or FakeClaimAssessmentOrchestrator()
    use_case = SubmitClaim(
        clock=clock or FixedClock(),  # type: ignore[arg-type]
        orchestrator=resolved_orch,
        uow_factory=make_uow_factory(resolved_store),
        new_id=ids or SequentialIds("id"),
    )
    return use_case, resolved_store, resolved_orch


@pytest.mark.unit
def test_happy_path_persists_an_awaiting_review_record() -> None:
    submit_claim, store, _ = _build()

    record = submit_claim(raw_text=_NARRATIVE)

    assert isinstance(record, AssessmentRecord)
    assert record.status is AssessmentStatus.AWAITING_REVIEW
    assert record.verdict is Verdict.COMPATIBLE
    assert store[record.assessment_id] == record
    assert record.created_at == FIXED_NOW


@pytest.mark.unit
def test_mints_distinct_claim_and_assessment_ids_from_new_id() -> None:
    ids = SequentialIds("id")
    submit_claim, _, orchestrator = _build(ids=ids)

    record = submit_claim(raw_text=_NARRATIVE)

    assert ids.issued == ["id-1", "id-2"]
    assert record.claim_id == "id-1"
    assert record.assessment_id == "id-2"
    assert orchestrator.started[0][0] == "id-2"
    assert orchestrator.started[0][1].claim_id == "id-1"


@pytest.mark.unit
def test_honours_an_explicit_claim_id() -> None:
    ids = SequentialIds("id")
    submit_claim, _, _ = _build(ids=ids)

    record = submit_claim(raw_text=_NARRATIVE, claim_id="claim-external-7")

    assert record.claim_id == "claim-external-7"
    assert record.assessment_id == "id-1"
    assert ids.issued == ["id-1"]


@pytest.mark.unit
def test_policy_ref_reaches_the_orchestrator_claim() -> None:
    submit_claim, _, orchestrator = _build()

    submit_claim(raw_text=_NARRATIVE, policy_ref=SUSEP)

    assert orchestrator.started[0][1].policy_ref == SUSEP


@pytest.mark.unit
def test_abstain_outcome_persists_with_no_citations() -> None:
    orchestrator = FakeClaimAssessmentOrchestrator(
        start_result=abstain_result(awaiting_review=True)
    )
    submit_claim, store, _ = _build(orchestrator=orchestrator)

    record = submit_claim(raw_text=_NARRATIVE)

    assert record.verdict is Verdict.INSUFFICIENT_INFORMATION
    assert record.citations == ()
    assert record.status is AssessmentStatus.AWAITING_REVIEW
    assert store[record.assessment_id].citations == ()


@pytest.mark.unit
def test_empty_narrative_is_rejected_and_nothing_is_persisted() -> None:
    submit_claim, store, orchestrator = _build()

    with pytest.raises(ValueError, match="raw_text must not be empty"):
        submit_claim(raw_text="")

    assert store == {}
    assert orchestrator.started == []


@pytest.mark.unit
def test_a_naive_clock_is_rejected_before_persisting() -> None:
    submit_claim, store, orchestrator = _build(clock=NaiveClock())

    with pytest.raises(ValueError, match="timezone-aware"):
        submit_claim(raw_text=_NARRATIVE)

    assert store == {}
    assert orchestrator.started == []


@pytest.mark.unit
def test_orchestrator_failure_leaves_nothing_persisted() -> None:
    orchestrator = FakeClaimAssessmentOrchestrator(
        raise_on_start=RuntimeError("graph exploded")
    )
    submit_claim, store, _ = _build(orchestrator=orchestrator)

    with pytest.raises(RuntimeError, match="graph exploded"):
        submit_claim(raw_text=_NARRATIVE)

    assert store == {}


@pytest.mark.unit
def test_a_run_that_does_not_pause_is_a_contract_error() -> None:
    orchestrator = FakeClaimAssessmentOrchestrator(
        start_result=make_orchestrator_result(awaiting_review=False)
    )
    submit_claim, store, _ = _build(orchestrator=orchestrator)

    with pytest.raises(OrchestratorContractError, match="did not pause"):
        submit_claim(raw_text=_NARRATIVE)

    assert store == {}
