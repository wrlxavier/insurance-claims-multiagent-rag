"""The assessment endpoints -- [M5-04].

Five routes over the four use cases:

- ``POST /v1/assessments`` -> ``SubmitClaim``. Returns **202** with the id in the
  body and in ``Location``. The graph runs synchronously in the handler for now
  (minutes, LLM calls); [M5-05]'s queue makes it truly non-blocking and adds
  ``PENDING/RUNNING/FAILED`` run states.
- ``GET /v1/assessments`` -> ``ListAssessments`` (newest first).
- ``GET /v1/assessments/{id}`` -> ``GetAssessment`` (state, recommendation,
  structured citations).
- ``POST /v1/assessments/{id}/decision`` -> ``SubmitHumanDecision``. Resumes the
  paused run and returns the settled record (200 -- the resume completes).
- ``GET /v1/assessments/{id}/audit`` -> ``GetAuditTrail``. Empty until a decision
  is submitted (the durable trail is written at the human checkpoint).

Every domain/application error raised below is turned into the uniform envelope
by ``presentation.errors``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response

from application.assessment_record import AssessmentStatus
from application.use_cases.get_assessment import GetAssessment
from application.use_cases.get_audit_trail import GetAuditTrail
from application.use_cases.list_assessments import ListAssessments
from application.use_cases.submit_claim import SubmitClaim
from application.use_cases.submit_human_decision import SubmitHumanDecision
from presentation.dependencies import (
    get_get_assessment,
    get_get_audit_trail,
    get_list_assessments,
    get_submit_claim,
    get_submit_human_decision,
)
from presentation.mappers import (
    assessment_response,
    audit_trail_response,
    edited_from_request,
    to_decision_outcome,
    to_policy_ref,
)
from presentation.schemas import (
    AssessmentResponse,
    AuditTrailResponse,
    DecisionRequest,
    SubmitClaimRequest,
    SubmitClaimResponse,
)

router = APIRouter(prefix="/v1/assessments", tags=["assessments"])

_StatusFilter = Literal["awaiting_review", "decided"]

_SubmitClaim = Annotated[SubmitClaim, Depends(get_submit_claim)]
_ListAssessments = Annotated[ListAssessments, Depends(get_list_assessments)]
_GetAssessment = Annotated[GetAssessment, Depends(get_get_assessment)]
_SubmitHumanDecision = Annotated[
    SubmitHumanDecision, Depends(get_submit_human_decision)
]
_GetAuditTrail = Annotated[GetAuditTrail, Depends(get_get_audit_trail)]


@router.post("", status_code=202, response_model=SubmitClaimResponse)
def submit_claim(
    body: SubmitClaimRequest,
    response: Response,
    use_case: _SubmitClaim,
) -> SubmitClaimResponse:
    """Accept a claim, run it to the human checkpoint, return 202 + the id."""
    record = use_case(
        raw_text=body.raw_text,
        policy_ref=to_policy_ref(body.policy_ref),
        claim_id=body.claim_id,
    )
    response.headers["Location"] = f"/v1/assessments/{record.assessment_id}"
    return SubmitClaimResponse(
        assessment_id=record.assessment_id, status=record.status.value
    )


@router.get("", response_model=list[AssessmentResponse])
def list_assessments(
    use_case: _ListAssessments,
    claim_id: Annotated[str | None, Query()] = None,
    status: Annotated[_StatusFilter | None, Query()] = None,
    limit: Annotated[int, Query(gt=0)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AssessmentResponse]:
    """List assessment records, newest first, optionally filtered."""
    records = use_case(
        claim_id=claim_id,
        status=AssessmentStatus(status) if status is not None else None,
        limit=limit,
        offset=offset,
    )
    return [assessment_response(record) for record in records]


@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(
    assessment_id: str,
    use_case: _GetAssessment,
) -> AssessmentResponse:
    """Return one assessment's state, recommendation and citations."""
    return assessment_response(use_case(assessment_id))


@router.post("/{assessment_id}/decision", response_model=AssessmentResponse)
def submit_decision(
    assessment_id: str,
    body: DecisionRequest,
    use_case: _SubmitHumanDecision,
) -> AssessmentResponse:
    """Submit the analyst's decision and resume the paused run to completion."""
    record = use_case(
        assessment_id=assessment_id,
        decision=to_decision_outcome(body.decision),
        notes=body.notes,
        edited=edited_from_request(body),
    )
    return assessment_response(record)


@router.get("/{assessment_id}/audit", response_model=AuditTrailResponse)
def get_audit_trail(
    assessment_id: str,
    use_case: _GetAuditTrail,
) -> AuditTrailResponse:
    """Return the durable audit trail for one assessment run."""
    return audit_trail_response(assessment_id, use_case(assessment_id))
