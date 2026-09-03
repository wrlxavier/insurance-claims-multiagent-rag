"""The graph ``state.py`` <-> application boundary -- [M5-04].

The one place a LangGraph ``ClaimState`` becomes an
``application.orchestrator_result.OrchestratorResult`` and a domain
``HumanDecision`` becomes the resume payload the ``human_review`` node validates.
Its only consumer is ``LangGraphClaimAssessmentOrchestrator`` -- which is why
this mapper is [M5-04]'s and not [M5-03]'s (``docs/DOMAIN.md`` "Deferred").

Three things it reconciles between the two representations:

- ``state.Recommendation`` has no ``verdict`` -- it is read from the
  recommendation node's audit event via
  ``infrastructure.graph.verdict_readout``;
- ``state.Citation.susep_process`` is a bare string, ``domain.citation.Citation``
  wants a ``SusepProcess`` value object;
- the graph twin ``state.HumanDecision`` (``decision`` a ``Literal``,
  ``edited_recommendation``) and the domain ``HumanDecision`` (``DecisionOutcome``,
  ``assessment_id``, ``edited_assessment``) differ in shape -- ``resume_payload``
  converts the domain one *to* the plain mapping the node's
  ``HumanDecision.model_validate`` accepts.

A reviewer ``edit`` drops the recommendation's consistency flags: an
``EditedAssessmentInput`` -> domain ``Assessment`` carries none, and flags are
attention points kept beside the verdict, not part of the decision
(``docs/ARCHITECTURE.md``, [M4-08]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from application.audit_trail_entry import AuditTrailEntry
from application.consistency_flag import ConsistencyFlag
from application.orchestrator_result import OrchestratorResult
from domain.citation import Citation
from domain.human_decision import HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict
from infrastructure.graph import state, verdict_readout

# Fallbacks for the two non-empty prose fields ``OrchestratorResult`` requires.
# The recommendation node always writes both (LLM or its deterministic template),
# so these guard an impossible-in-practice empty rather than a real path.
_MISSING_JUSTIFICATION = "(sem justificativa registrada)"
_MISSING_ACTION = "(nenhuma ação recomendada registrada)"
_MISSING_EXCERPT = "—"


def result_from_final_state(
    final_state: Mapping[str, object],
    *,
    awaiting_review: bool,
    audit_records: tuple[AuditTrailEntry, ...] = (),
) -> OrchestratorResult:
    """Project a compiled graph's final/paused state onto an ``OrchestratorResult``."""
    recommendation = final_state.get("recommendation")
    if not isinstance(recommendation, state.Recommendation):
        raise RuntimeError(
            "graph run produced no recommendation -- the recommendation node must "
            "run before the result is read (got "
            f"{type(recommendation).__name__})"
        )

    audit_trail = cast(
        "Sequence[state.AuditEvent]", final_state.get("audit_trail") or ()
    )
    verdict = verdict_readout.effective_verdict(audit_trail) or (
        Verdict.INSUFFICIENT_INFORMATION
    )

    missing_information = tuple(
        cast("Sequence[str]", final_state.get("missing_information") or ())
    )

    return OrchestratorResult(
        verdict=verdict,
        reasoning=recommendation.justification.strip() or _MISSING_JUSTIFICATION,
        recommended_action=(
            recommendation.recommended_action.strip() or _MISSING_ACTION
        ),
        citations=tuple(_domain_citation(c) for c in recommendation.citations),
        confidence=recommendation.confidence,
        consistency_flags=tuple(
            _consistency_flag(signal) for signal in recommendation.consistency_flags
        ),
        context_sufficient=cast("bool | None", final_state.get("context_sufficient")),
        clarification_exhausted=bool(final_state.get("clarification_exhausted")),
        missing_information=missing_information,
        awaiting_review=awaiting_review,
        audit_records=audit_records,
    )


def audit_entries_from_records(
    records: Sequence[state.AuditRecord],
) -> tuple[AuditTrailEntry, ...]:
    """Flatten the trail the ``human_review`` node built into application DTOs.

    ``sequence`` is the record's position in the run's whole trail -- the same
    numbering ``infrastructure.database.audit_repository.append_audit_events``
    uses, so a reader sees exactly what the graph's own sink would have written.
    """
    entries: list[AuditTrailEntry] = []
    for index, record in enumerate(records):
        event = record.event
        usage = event.token_usage
        entries.append(
            AuditTrailEntry(
                sequence=index,
                timestamp=event.timestamp,
                node=event.node,
                action=event.action,
                model=event.model,
                model_version=event.model_version,
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                total_tokens=usage.total_tokens if usage is not None else None,
                confidence=event.confidence,
                node_input=event.node_input,
                payload=record.payload,
            )
        )
    return tuple(entries)


def resume_payload(decision: HumanDecision) -> dict[str, object]:
    """The plain mapping ``human_review._await_decision`` validates as a decision.

    ``state.HumanDecision.model_validate`` accepts a mapping (the HTTP-boundary
    case its docstring anticipates) and enforces the same "``edit`` carries a
    revision" rule the domain ``HumanDecision`` already guaranteed.
    """
    payload: dict[str, object] = {
        "decision": decision.decision.value,
        "notes": decision.notes,
        "decided_at": decision.decided_at.isoformat(),
    }
    edited = decision.edited_assessment
    if edited is not None:
        payload["edited_recommendation"] = {
            "recommended_action": edited.recommended_action,
            "justification": edited.reasoning,
            "citations": [_state_citation_dict(c) for c in edited.citations],
            "consistency_flags": [],
            "confidence": edited.confidence,
        }
    return payload


def _domain_citation(citation: state.Citation) -> Citation:
    return Citation(
        clause_id=citation.clause_id,
        document_id=citation.document_id,
        susep_process=SusepProcess.parse(citation.susep_process),
        clause_type=citation.clause_type,
        excerpt=citation.excerpt or _MISSING_EXCERPT,
        relevance_score=max(citation.relevance_score, 0.0),
    )


def _consistency_flag(signal: state.ConsistencySignal) -> ConsistencyFlag:
    return ConsistencyFlag(
        check=signal.check,
        severity=signal.severity,
        detail=signal.detail,
        source=signal.source,
    )


def _state_citation_dict(citation: Citation) -> dict[str, object]:
    return {
        "clause_id": citation.clause_id,
        "document_id": citation.document_id,
        "susep_process": citation.susep_process.value,
        "clause_type": citation.clause_type.value,
        "relevance_score": citation.relevance_score,
        "excerpt": citation.excerpt,
    }
