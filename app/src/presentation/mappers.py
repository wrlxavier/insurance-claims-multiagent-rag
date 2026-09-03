"""Pure conversions between the API schemas and the application/domain types -- [M5-04].

One direction builds a response model from an ``AssessmentRecord`` /
``AuditTrailEntry`` (enum values rendered as strings, value objects as their
canonical form). The other turns a request body into the domain inputs the use
cases take -- and a malformed value object (a bad SUSEP process, an unknown
clause type, a non-permitted verdict) raises the domain ``ValueError`` /
``InvalidValueObjectError`` here, which the error edge maps to 422.
"""

from __future__ import annotations

from collections.abc import Sequence

from application.assessment_record import AssessmentRecord
from application.audit_trail_entry import AuditTrailEntry
from application.consistency_flag import ConsistencyFlag
from application.edited_assessment_input import EditedAssessmentInput
from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict
from presentation.schemas import (
    AssessmentResponse,
    AuditEntrySchema,
    AuditTrailResponse,
    CitationInput,
    CitationSchema,
    ConsistencyFlagSchema,
    DecisionRequest,
    EditedAssessmentPayload,
    EditedAssessmentView,
    HumanDecisionSchema,
)

# --------------------------------------------------------------------------- #
# domain/application -> response
# --------------------------------------------------------------------------- #


def _citation_schema(citation: Citation) -> CitationSchema:
    return CitationSchema(
        clause_id=citation.clause_id,
        document_id=citation.document_id,
        susep_process=citation.susep_process.value,
        clause_type=citation.clause_type.value,
        excerpt=citation.excerpt,
        relevance_score=citation.relevance_score,
    )


def _consistency_flag_schema(flag: ConsistencyFlag) -> ConsistencyFlagSchema:
    return ConsistencyFlagSchema(
        check=flag.check,
        severity=flag.severity,
        detail=flag.detail,
        source=flag.source,
    )


def _decision_schema(decision: HumanDecision) -> HumanDecisionSchema:
    edited = decision.edited_assessment
    return HumanDecisionSchema(
        assessment_id=decision.assessment_id,
        decision=decision.decision.value,
        decided_at=decision.decided_at,
        notes=decision.notes,
        edited_assessment=(
            EditedAssessmentView(
                verdict=edited.verdict.value,
                reasoning=edited.reasoning,
                recommended_action=edited.recommended_action,
                citations=[_citation_schema(c) for c in edited.citations],
                confidence=edited.confidence,
            )
            if edited is not None
            else None
        ),
    )


def assessment_response(record: AssessmentRecord) -> AssessmentResponse:
    """Render an ``AssessmentRecord`` as the API response model."""
    return AssessmentResponse(
        assessment_id=record.assessment_id,
        claim_id=record.claim_id,
        status=record.status.value,
        verdict=record.verdict.value,
        reasoning=record.reasoning,
        recommended_action=record.recommended_action,
        confidence=record.confidence,
        citations=[_citation_schema(c) for c in record.citations],
        consistency_flags=[
            _consistency_flag_schema(f) for f in record.consistency_flags
        ],
        context_sufficient=record.context_sufficient,
        clarification_exhausted=record.clarification_exhausted,
        missing_information=list(record.missing_information),
        is_grounded=record.is_grounded,
        created_at=record.created_at,
        decision=(
            _decision_schema(record.decision) if record.decision is not None else None
        ),
    )


def _audit_entry_schema(entry: AuditTrailEntry) -> AuditEntrySchema:
    return AuditEntrySchema(
        sequence=entry.sequence,
        timestamp=entry.timestamp,
        node=entry.node,
        action=entry.action,
        model=entry.model,
        model_version=entry.model_version,
        input_tokens=entry.input_tokens,
        output_tokens=entry.output_tokens,
        total_tokens=entry.total_tokens,
        confidence=entry.confidence,
        node_input=entry.node_input,
        payload=dict(entry.payload) if entry.payload is not None else None,
    )


def audit_trail_response(
    assessment_id: str, entries: Sequence[AuditTrailEntry]
) -> AuditTrailResponse:
    """Render a run's audit trail as the API response model."""
    return AuditTrailResponse(
        assessment_id=assessment_id,
        entries=[_audit_entry_schema(entry) for entry in entries],
    )


# --------------------------------------------------------------------------- #
# request -> domain/application
# --------------------------------------------------------------------------- #


def to_policy_ref(raw: str | None) -> SusepProcess | None:
    """Parse the optional ``policy_ref`` string into a ``SusepProcess``.

    Raises ``domain.errors.InvalidSusepProcessError`` (a ``ValueError``) on a
    malformed value -- mapped to 422 at the error edge.
    """
    if raw is None or not raw.strip():
        return None
    return SusepProcess.parse(raw)


def to_decision_outcome(value: str) -> DecisionOutcome:
    """Map the request's decision literal onto the domain enum."""
    return DecisionOutcome(value)


def _citation_from_input(payload: CitationInput) -> Citation:
    return Citation(
        clause_id=payload.clause_id,
        document_id=payload.document_id,
        susep_process=SusepProcess.parse(payload.susep_process),
        clause_type=ClauseType(payload.clause_type),
        excerpt=payload.excerpt,
        relevance_score=payload.relevance_score,
    )


def to_edited_assessment_input(
    payload: EditedAssessmentPayload,
) -> EditedAssessmentInput:
    """Build the ``EditedAssessmentInput`` the use case validates and grounds.

    Raises ``domain.errors.InvalidValueObjectError`` / ``ValueError`` on a
    malformed verdict, clause type or SUSEP process -- 422 at the edge.
    """
    return EditedAssessmentInput(
        verdict=Verdict(payload.verdict),
        reasoning=payload.reasoning,
        recommended_action=payload.recommended_action,
        citations=tuple(_citation_from_input(c) for c in payload.citations),
        confidence=payload.confidence,
    )


def edited_from_request(request: DecisionRequest) -> EditedAssessmentInput | None:
    """The ``EditedAssessmentInput`` for the request, or ``None`` when not editing."""
    if request.edited is None:
        return None
    return to_edited_assessment_input(request.edited)
