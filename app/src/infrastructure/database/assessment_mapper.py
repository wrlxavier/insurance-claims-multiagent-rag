"""Map the assessment aggregate to and from its database rows -- [M5-03].

``docs/DOMAIN.md`` names "domain <-> ORM row" mappers as [M5-03]'s deliverable.
This module is that mapper for the assessment surface: pure functions, no
session, between [application.assessment_record.AssessmentRecord] (plus the
[domain.human_decision.HumanDecision] recorded beside it) and the
[infrastructure.database.models.AssessmentRow] /
[infrastructure.database.models.HumanDecisionRow] pair.

``citations`` / ``consistency_flags`` / ``edited_assessment`` are stored as
``JSONB``; the ``_*_to_json`` / ``_*_from_json`` helpers are the one place their
wire shape is defined. Value objects that carry a stricter type than JSON can
(``SusepProcess``, the ``ClauseType`` / ``Verdict`` / ``DecisionOutcome`` /
``AssessmentStatus`` enums) are written as their canonical string and rebuilt on
read -- the domain constructors re-validate, so a corrupted row fails loudly
rather than flowing through.

The graph's Pydantic ``state.py`` twin is deliberately *not* handled here: the
only consumer of a ``state`` <-> domain mapper is the LangGraph orchestrator
adapter, which is [M5-04]'s. ``AssessmentRecord.from_orchestrator_result``
already bridges the graph-free DTO.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast

from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.consistency_flag import ConsistencyFlag
from domain.assessment import Assessment
from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict
from infrastructure.database.models import AssessmentRow, HumanDecisionRow


def _require_str(data: Mapping[str, object], key: str) -> str:
    """Read a required string field from a JSON object, or raise."""
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"expected a string for {key!r}, got {value!r}")
    return value


def _require_number(data: Mapping[str, object], key: str) -> float:
    """Read a required numeric field from a JSON object, or raise."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number for {key!r}, got {value!r}")
    return float(value)


# --------------------------------------------------------------------------- #
# citations
# --------------------------------------------------------------------------- #
def _citation_to_json(citation: Citation) -> dict[str, object]:
    return {
        "clause_id": citation.clause_id,
        "document_id": citation.document_id,
        "susep_process": citation.susep_process.value,
        "clause_type": citation.clause_type.value,
        "excerpt": citation.excerpt,
        "relevance_score": citation.relevance_score,
    }


def _citation_from_json(data: Mapping[str, object]) -> Citation:
    return Citation(
        clause_id=_require_str(data, "clause_id"),
        document_id=_require_str(data, "document_id"),
        susep_process=SusepProcess(_require_str(data, "susep_process")),
        clause_type=ClauseType(_require_str(data, "clause_type")),
        excerpt=_require_str(data, "excerpt"),
        relevance_score=_require_number(data, "relevance_score"),
    )


# --------------------------------------------------------------------------- #
# consistency flags
# --------------------------------------------------------------------------- #
def _flag_to_json(flag: ConsistencyFlag) -> dict[str, object]:
    return {
        "check": flag.check,
        "severity": flag.severity,
        "detail": flag.detail,
        "source": flag.source,
    }


def _flag_from_json(data: Mapping[str, object]) -> ConsistencyFlag:
    # `ConsistencyFlag.__post_init__` re-validates the closed vocabularies; the
    # casts only satisfy the Literal annotations.
    return ConsistencyFlag(
        check=_require_str(data, "check"),
        severity=cast(Literal["info", "attention"], _require_str(data, "severity")),
        detail=_require_str(data, "detail"),
        source=cast(Literal["deterministic", "llm"], _require_str(data, "source")),
    )


# --------------------------------------------------------------------------- #
# the edited assessment nested in an `edit` decision
# --------------------------------------------------------------------------- #
def _assessment_to_json(assessment: Assessment) -> dict[str, object]:
    return {
        "assessment_id": assessment.assessment_id,
        "claim_id": assessment.claim_id,
        "verdict": assessment.verdict.value,
        "reasoning": assessment.reasoning,
        "citations": [_citation_to_json(c) for c in assessment.citations],
        "confidence": assessment.confidence,
        "recommended_action": assessment.recommended_action,
    }


def _assessment_from_json(data: Mapping[str, object]) -> Assessment:
    raw_citations = data.get("citations")
    if not isinstance(raw_citations, list):
        raise TypeError(f"expected a list for 'citations', got {raw_citations!r}")
    return Assessment(
        assessment_id=_require_str(data, "assessment_id"),
        claim_id=_require_str(data, "claim_id"),
        verdict=Verdict(_require_str(data, "verdict")),
        reasoning=_require_str(data, "reasoning"),
        citations=tuple(_citation_from_json(item) for item in raw_citations),
        confidence=_require_number(data, "confidence"),
        recommended_action=_require_str(data, "recommended_action"),
    )


# --------------------------------------------------------------------------- #
# the human decision
# --------------------------------------------------------------------------- #
def _decision_to_row(decision: HumanDecision) -> HumanDecisionRow:
    edited = decision.edited_assessment
    return HumanDecisionRow(
        assessment_id=decision.assessment_id,
        decision=decision.decision.value,
        decided_at=decision.decided_at,
        notes=decision.notes,
        edited_assessment=(_assessment_to_json(edited) if edited is not None else None),
    )


def _decision_from_row(row: HumanDecisionRow) -> HumanDecision:
    edited = row.edited_assessment
    return HumanDecision(
        assessment_id=row.assessment_id,
        decision=DecisionOutcome(row.decision),
        decided_at=_as_aware(row.decided_at),
        notes=row.notes,
        edited_assessment=(
            _assessment_from_json(edited) if edited is not None else None
        ),
    )


def _as_aware(value: datetime) -> datetime:
    """Guard: a ``timestamptz`` column must never hand back a naive datetime.

    psycopg returns tz-aware datetimes for ``timestamptz``; this makes a
    misconfigured column (or a driver change) fail here rather than several
    layers up, where the domain would reject it with less context.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"expected a timezone-aware datetime, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# the aggregate
# --------------------------------------------------------------------------- #
def record_to_rows(
    record: AssessmentRecord,
) -> tuple[AssessmentRow, HumanDecisionRow | None]:
    """Split an ``AssessmentRecord`` into its ``assessment`` (+ decision) rows."""
    assessment = AssessmentRow(
        assessment_id=record.assessment_id,
        claim_id=record.claim_id,
        verdict=record.verdict.value,
        reasoning=record.reasoning,
        recommended_action=record.recommended_action,
        confidence=record.confidence,
        context_sufficient=record.context_sufficient,
        clarification_exhausted=record.clarification_exhausted,
        missing_information=list(record.missing_information),
        citations=[_citation_to_json(c) for c in record.citations],
        consistency_flags=[_flag_to_json(f) for f in record.consistency_flags],
        status=record.status.value,
        created_at=record.created_at,
    )
    decision_row = (
        _decision_to_row(record.decision) if record.decision is not None else None
    )
    return assessment, decision_row


def rows_to_record(
    assessment: AssessmentRow, decision: HumanDecisionRow | None
) -> AssessmentRecord:
    """Rebuild an ``AssessmentRecord`` from its rows (the aggregate re-validates)."""
    return AssessmentRecord(
        assessment_id=assessment.assessment_id,
        claim_id=assessment.claim_id,
        verdict=Verdict(assessment.verdict),
        reasoning=assessment.reasoning,
        recommended_action=assessment.recommended_action,
        citations=tuple(_citation_from_json(item) for item in assessment.citations),
        confidence=assessment.confidence,
        consistency_flags=tuple(
            _flag_from_json(item) for item in assessment.consistency_flags
        ),
        context_sufficient=assessment.context_sufficient,
        clarification_exhausted=assessment.clarification_exhausted,
        missing_information=tuple(assessment.missing_information),
        status=AssessmentStatus(assessment.status),
        created_at=_as_aware(assessment.created_at),
        decision=_decision_from_row(decision) if decision is not None else None,
    )
