"""The SubmitHumanDecision use case [M5-02]."""

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.edited_assessment_input import EditedAssessmentInput
from application.errors import (
    AssessmentAlreadyDecidedError,
    AssessmentNotFoundError,
    OrchestratorContractError,
    UnknownClauseError,
)
from application.use_cases.submit_human_decision import SubmitHumanDecision
from domain.errors import CitationRequiredError
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    AuditStore,
    FakeClaimAssessmentOrchestrator,
    FixedClock,
    InMemoryAssessmentRepository,
    InMemoryClauseRepository,
    InMemoryUnitOfWork,
    make_audit_entry,
    make_citation,
    make_orchestrator_result,
    make_policy_clause,
    make_record,
    make_uow_factory,
)

_KNOWN_CLAUSE_ID = "15414610650202459:3.4"


def _build(
    *,
    record: AssessmentRecord | None = None,
    orchestrator: FakeClaimAssessmentOrchestrator | None = None,
    clauses: InMemoryClauseRepository | None = None,
    audit_store: AuditStore | None = None,
) -> tuple[
    SubmitHumanDecision, dict[str, AssessmentRecord], FakeClaimAssessmentOrchestrator
]:
    seed = record if record is not None else make_record(assessment_id="assessment-1")
    store: dict[str, AssessmentRecord] = {seed.assessment_id: seed}
    resolved_orch = orchestrator or FakeClaimAssessmentOrchestrator(
        resume_result=make_orchestrator_result(awaiting_review=False)
    )
    use_case = SubmitHumanDecision(
        clock=FixedClock(),
        orchestrator=resolved_orch,
        assessments=InMemoryAssessmentRepository(store),
        clauses=clauses
        or InMemoryClauseRepository([make_policy_clause(clause_id=_KNOWN_CLAUSE_ID)]),
        uow_factory=make_uow_factory(store, audit_store),
    )
    return use_case, store, resolved_orch


def _edit(**overrides: object) -> EditedAssessmentInput:
    fields: dict[str, object] = {
        "verdict": Verdict.INCOMPATIBLE,
        "reasoning": "A exclusao 3.4 se aplica ao evento descrito.",
        "recommended_action": "Registrar recusa com fundamento na clausula 3.4.",
        "citations": (make_citation(clause_id=_KNOWN_CLAUSE_ID),),
        "confidence": 0.66,
    }
    fields.update(overrides)
    return EditedAssessmentInput(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_approve_settles_and_resumes_with_a_domain_decision() -> None:
    submit_decision, store, orchestrator = _build()

    updated = submit_decision(
        assessment_id="assessment-1", decision=DecisionOutcome.APPROVE
    )

    assert updated.status is AssessmentStatus.DECIDED
    assert updated.decision is not None
    assert updated.decision.decision is DecisionOutcome.APPROVE
    assert updated.decision.edited_assessment is None
    assert updated.decision.decided_at == FIXED_NOW
    assert store["assessment-1"].status is AssessmentStatus.DECIDED

    resumed_id, resumed_decision = orchestrator.resumed[0]
    assert resumed_id == "assessment-1"
    assert isinstance(resumed_decision, HumanDecision)
    assert resumed_decision.assessment_id == "assessment-1"


@pytest.mark.unit
def test_reject_stores_the_notes() -> None:
    submit_decision, store, _ = _build()

    updated = submit_decision(
        assessment_id="assessment-1",
        decision=DecisionOutcome.REJECT,
        notes="Evento fora do periodo de vigencia.",
    )

    assert updated.decision is not None
    assert updated.decision.notes == "Evento fora do periodo de vigencia."


@pytest.mark.unit
def test_edit_builds_a_grounded_assessment_and_keeps_the_system_verdict() -> None:
    submit_decision, store, _ = _build(
        record=make_record(assessment_id="assessment-1", claim_id="claim-77")
    )

    updated = submit_decision(
        assessment_id="assessment-1",
        decision=DecisionOutcome.EDIT,
        edited=_edit(),
    )

    assert updated.verdict is Verdict.COMPATIBLE  # system opinion, untouched
    assert updated.decision is not None
    edited = updated.decision.edited_assessment
    assert edited is not None
    assert edited.assessment_id == "assessment-1"
    assert edited.claim_id == "claim-77"
    assert edited.verdict is Verdict.INCOMPATIBLE
    assert len(edited.citations) == 1


@pytest.mark.unit
def test_edit_without_a_payload_is_rejected() -> None:
    submit_decision, store, orchestrator = _build()

    with pytest.raises(ValueError, match="requires an edited assessment"):
        submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.EDIT)

    assert orchestrator.resumed == []
    assert store["assessment-1"].status is AssessmentStatus.AWAITING_REVIEW


@pytest.mark.unit
def test_non_edit_carrying_a_payload_is_rejected() -> None:
    submit_decision, _, orchestrator = _build()

    with pytest.raises(ValueError, match="must not carry an edited assessment"):
        submit_decision(
            assessment_id="assessment-1",
            decision=DecisionOutcome.APPROVE,
            edited=_edit(),
        )

    assert orchestrator.resumed == []


@pytest.mark.unit
def test_edit_citing_nothing_hits_the_citation_rule() -> None:
    submit_decision, _, orchestrator = _build()

    with pytest.raises(CitationRequiredError):
        submit_decision(
            assessment_id="assessment-1",
            decision=DecisionOutcome.EDIT,
            edited=_edit(citations=()),
        )

    assert orchestrator.resumed == []


@pytest.mark.unit
def test_edit_citing_an_unknown_clause_is_rejected_before_resume() -> None:
    submit_decision, store, orchestrator = _build()

    with pytest.raises(UnknownClauseError) as excinfo:
        submit_decision(
            assessment_id="assessment-1",
            decision=DecisionOutcome.EDIT,
            edited=_edit(citations=(make_citation(clause_id="15414610650202459:9.9"),)),
        )

    assert excinfo.value.clause_ids == ("15414610650202459:9.9",)
    assert orchestrator.resumed == []
    assert store["assessment-1"].status is AssessmentStatus.AWAITING_REVIEW


@pytest.mark.unit
def test_unknown_assessment_id_raises_not_found() -> None:
    submit_decision, _, orchestrator = _build()

    with pytest.raises(AssessmentNotFoundError):
        submit_decision(assessment_id="missing", decision=DecisionOutcome.APPROVE)

    assert orchestrator.resumed == []


@pytest.mark.unit
def test_already_decided_assessment_is_rejected() -> None:
    decided = make_record(
        assessment_id="assessment-1",
        status=AssessmentStatus.DECIDED,
        decision=HumanDecision(
            assessment_id="assessment-1",
            decision=DecisionOutcome.APPROVE,
            decided_at=FIXED_NOW,
        ),
    )
    submit_decision, store, orchestrator = _build(record=decided)

    with pytest.raises(AssessmentAlreadyDecidedError):
        submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.REJECT)

    assert orchestrator.resumed == []
    assert store["assessment-1"] == decided


@pytest.mark.unit
def test_resume_failure_leaves_the_record_awaiting_review() -> None:
    orchestrator = FakeClaimAssessmentOrchestrator(
        raise_on_resume=RuntimeError("checkpoint gone")
    )
    submit_decision, store, _ = _build(orchestrator=orchestrator)

    with pytest.raises(RuntimeError, match="checkpoint gone"):
        submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.APPROVE)

    assert store["assessment-1"].status is AssessmentStatus.AWAITING_REVIEW


@pytest.mark.unit
def test_resume_that_re_pauses_is_a_contract_error() -> None:
    orchestrator = FakeClaimAssessmentOrchestrator(
        resume_result=make_orchestrator_result(awaiting_review=True)
    )
    submit_decision, store, _ = _build(orchestrator=orchestrator)

    with pytest.raises(OrchestratorContractError, match="did not complete"):
        submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.APPROVE)

    assert store["assessment-1"].status is AssessmentStatus.AWAITING_REVIEW


@pytest.mark.unit
def test_the_captured_audit_trail_is_persisted_with_the_record() -> None:
    audit_store: AuditStore = {}
    orchestrator = FakeClaimAssessmentOrchestrator(
        resume_result=make_orchestrator_result(
            awaiting_review=False,
            audit_records=(
                make_audit_entry(sequence=0, node="retrieval"),
                make_audit_entry(
                    sequence=1, node="human_review", action="human_decision:approve"
                ),
            ),
        )
    )
    submit_decision, _, _ = _build(orchestrator=orchestrator, audit_store=audit_store)

    submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.APPROVE)

    assert [e.sequence for e in audit_store["assessment-1"]] == [0, 1]


@pytest.mark.unit
def test_the_fold_is_atomic_record_and_trail_roll_back_together() -> None:
    seed = make_record(assessment_id="assessment-1")
    store: dict[str, AssessmentRecord] = {seed.assessment_id: seed}
    audit_store: AuditStore = {}

    class _FailingUow(InMemoryUnitOfWork):
        def commit(self) -> None:
            raise RuntimeError("disk full")

    orchestrator = FakeClaimAssessmentOrchestrator(
        resume_result=make_orchestrator_result(
            awaiting_review=False,
            audit_records=(make_audit_entry(sequence=0),),
        )
    )
    submit_decision = SubmitHumanDecision(
        clock=FixedClock(),
        orchestrator=orchestrator,
        assessments=InMemoryAssessmentRepository(store),
        clauses=InMemoryClauseRepository([make_policy_clause()]),
        uow_factory=lambda: _FailingUow(store, audit_store),
    )

    with pytest.raises(RuntimeError, match="disk full"):
        submit_decision(assessment_id="assessment-1", decision=DecisionOutcome.APPROVE)

    # both the record update and the audit append ran, then commit failed:
    # __exit__ rolls both stores back.
    assert store["assessment-1"].status is AssessmentStatus.AWAITING_REVIEW
    assert audit_store == {}


@pytest.mark.unit
def test_a_non_decision_outcome_value_is_rejected() -> None:
    submit_decision, _, orchestrator = _build()

    with pytest.raises(ValueError):
        submit_decision(assessment_id="assessment-1", decision="approve")  # type: ignore[arg-type]

    assert orchestrator.resumed == []
