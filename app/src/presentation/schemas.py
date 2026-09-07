"""Request and response models for the assessment API -- [M5-04].

Pydantic v2 models: the presentation layer is the one place in this codebase
that speaks JSON, so it is the one place Pydantic is allowed (``domain`` and
``application`` stay stdlib-only, enforced by
``tests/architecture/test_layer_boundaries.py``).

Citations are structured objects here, never prose (a DoD item): ``clause_id``,
``document_id``, ``susep_process``, ``clause_type``, ``excerpt``,
``relevance_score``. Enums are rendered as their string values.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# shared value shapes
# --------------------------------------------------------------------------- #

_VERDICTS = Literal["compatible", "incompatible", "insufficient_information"]
_DECISIONS = Literal["approve", "edit", "reject"]
# The lifecycle a caller polls ([M5-05]): `pending` / `running` while the queued
# run is in flight, `failed` if it dead-lettered, then `awaiting_review` /
# `decided` once the graph has produced a recommendation.
_READ_STATUS = Literal["pending", "running", "awaiting_review", "decided", "failed"]


class CitationSchema(BaseModel):
    """One clause an assessment's reasoning is grounded in -- a structured object."""

    clause_id: str
    document_id: str
    susep_process: str
    clause_type: str
    excerpt: str
    relevance_score: float


class ConsistencyFlagSchema(BaseModel):
    """One attention point raised while assessing -- kept beside the verdict."""

    check: str
    severity: Literal["info", "attention"]
    detail: str
    source: Literal["deterministic", "llm"]


class EditedAssessmentView(BaseModel):
    """The analyst's revised assessment, as recorded on an ``edit`` decision."""

    verdict: _VERDICTS
    reasoning: str
    recommended_action: str
    citations: list[CitationSchema]
    confidence: float


class HumanDecisionSchema(BaseModel):
    """The analyst's decision, recorded beside the system's opinion, never over it."""

    assessment_id: str
    decision: _DECISIONS
    decided_at: datetime
    notes: str
    edited_assessment: EditedAssessmentView | None = None


class AssessmentResponse(BaseModel):
    """One claim's assessment across its whole lifecycle -- the servable view.

    While the run is still ``pending`` / ``running`` (or has ``failed``), no
    recommendation exists yet: the verdict/prose/citation fields are ``null`` /
    empty and ``error`` carries the failure cause. They are populated once
    ``status`` reaches ``awaiting_review``.
    """

    assessment_id: str
    claim_id: str
    status: _READ_STATUS
    error: str | None = None
    verdict: _VERDICTS | None = None
    reasoning: str | None = None
    recommended_action: str | None = None
    confidence: float | None = None
    citations: list[CitationSchema] = Field(default_factory=list)
    consistency_flags: list[ConsistencyFlagSchema] = Field(default_factory=list)
    context_sufficient: bool | None = None
    clarification_exhausted: bool = False
    missing_information: list[str] = Field(default_factory=list)
    is_grounded: bool = False
    created_at: datetime
    decision: HumanDecisionSchema | None = None


class AuditEntrySchema(BaseModel):
    """One entry of a run's durable audit trail."""

    sequence: int
    timestamp: datetime
    node: str
    action: str
    model: str | None = None
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    confidence: float | None = None
    node_input: str | None = None
    payload: dict[str, object] | None = None


class AuditTrailResponse(BaseModel):
    """The audit trail for one assessment run, in ``sequence`` order."""

    assessment_id: str
    entries: list[AuditEntrySchema]


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #


class SubmitClaimRequest(BaseModel):
    """The body of ``POST /v1/assessments``."""

    raw_text: str = Field(min_length=1)
    policy_ref: str | None = Field(
        default=None,
        description=(
            "The registered product the claim is filed against -- a SUSEP "
            "process number (canonical NNNNN.NNNNNN/NNNN-NN or 17 bare digits)."
        ),
    )
    claim_id: str | None = None


class SubmitClaimResponse(BaseModel):
    """The 202 body of ``POST /v1/assessments``; the id also rides in ``Location``.

    ``status`` is ``pending`` -- the run is queued. Poll ``GET /v1/assessments/{id}``
    for ``running`` -> ``awaiting_review`` (or ``failed``).
    """

    assessment_id: str
    status: _READ_STATUS


class CitationInput(BaseModel):
    """One clause the analyst grounds an edited assessment in."""

    clause_id: str
    document_id: str
    susep_process: str
    clause_type: str
    excerpt: str
    relevance_score: float = 0.0


class EditedAssessmentPayload(BaseModel):
    """The analyst's replacement assessment on an ``edit`` decision."""

    verdict: _VERDICTS
    reasoning: str
    recommended_action: str
    citations: list[CitationInput]
    confidence: float


class DecisionRequest(BaseModel):
    """The body of ``POST /v1/assessments/{id}/decision``."""

    decision: _DECISIONS
    notes: str = ""
    edited: EditedAssessmentPayload | None = None


class ErrorBody(BaseModel):
    """The inner object of every error response."""

    code: str
    message: str
    details: dict[str, object] | None = None


class ErrorResponse(BaseModel):
    """The uniform error envelope: ``{"error": {"code", "message", "details"}}``."""

    error: ErrorBody
