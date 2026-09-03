"""The agent graph's state sub-models [M4-01]."""

import pytest
from pydantic import ValidationError

from domain.clause_classification import ClauseType
from domain.verdict import Verdict
from infrastructure.graph.state import (
    AuditEvent,
    Citation,
    CompatibilityAssessment,
    ConsistencyReport,
    ConsistencySignal,
    HumanDecision,
    Recommendation,
    TokenUsage,
)


def _citation(**overrides: object) -> Citation:
    fields: dict[str, object] = {
        "clause_id": "15414900666201489:2.1",
        "document_id": "1",
        "susep_process": "15414.900666/2014-89",
        "clause_type": ClauseType.COVERAGE,
        "relevance_score": 0.83,
        "excerpt": "A cobertura compreende colisao, incendio e roubo.",
    }
    fields.update(overrides)
    return Citation(**fields)


def _recommendation() -> Recommendation:
    return Recommendation(
        recommended_action="Encaminhar para analise humana.",
        justification="O evento e compativel com a cobertura basica.",
        citations=[_citation()],
        consistency_flags=[],
        confidence=0.7,
    )


# --- Citation -----------------------------------------------------------------


@pytest.mark.unit
def test_citation_accepts_a_full_row() -> None:
    citation = _citation()

    assert citation.clause_type is ClauseType.COVERAGE
    assert citation.relevance_score == 0.83


@pytest.mark.unit
@pytest.mark.parametrize("field", ["clause_id", "document_id", "susep_process"])
def test_citation_rejects_an_empty_identifier(field: str) -> None:
    with pytest.raises(ValidationError):
        _citation(**{field: ""})


@pytest.mark.unit
def test_citation_rejects_a_negative_relevance_score() -> None:
    with pytest.raises(ValidationError):
        _citation(relevance_score=-0.1)


@pytest.mark.unit
def test_citation_is_frozen() -> None:
    with pytest.raises(ValidationError):
        _citation().relevance_score = 0.9  # type: ignore[misc]


# --- TokenUsage --------------------------------------------------------------


@pytest.mark.unit
def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1, output_tokens=0, total_tokens=0)


# --- AuditEvent ------------------------------------------------------------


@pytest.mark.unit
def test_audit_event_stamps_an_aware_utc_timestamp_by_default() -> None:
    event = AuditEvent(node="intake", action="extract_entities")

    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() is not None
    assert event.timestamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


@pytest.mark.unit
def test_audit_event_leaves_model_fields_none_for_a_deterministic_node() -> None:
    event = AuditEvent(node="consistency", action="check_date_coherence")

    assert event.model is None
    assert event.model_version is None
    assert event.token_usage is None
    assert event.confidence is None


@pytest.mark.unit
def test_audit_event_carries_llm_call_metadata_when_given() -> None:
    event = AuditEvent(
        node="compatibility",
        action="assess",
        model="deepseek/deepseek-v4-pro-0813",
        model_version="0813",
        token_usage=TokenUsage(input_tokens=1200, output_tokens=310, total_tokens=1510),
        confidence=0.66,
        node_input="query=colisao franquia; citations=4",
    )

    assert event.token_usage is not None
    assert event.token_usage.total_tokens == 1510


@pytest.mark.unit
def test_audit_event_rejects_confidence_outside_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(node="x", action="y", confidence=1.5)


@pytest.mark.unit
def test_audit_event_rejects_an_empty_node() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(node="", action="y")


# --- CompatibilityAssessment / ConsistencyReport --------------------------


@pytest.mark.unit
def test_compatibility_assessment_holds_a_verdict_and_typed_citations() -> None:
    assessment = CompatibilityAssessment(
        verdict=Verdict.COMPATIBLE,
        reasoning="Compativel conforme clausula 2.1.",
        citations=[_citation()],
        confidence=0.8,
    )

    assert assessment.verdict is Verdict.COMPATIBLE
    assert isinstance(assessment.citations[0], Citation)


@pytest.mark.unit
def test_compatibility_assessment_rejects_a_bare_string_verdict() -> None:
    with pytest.raises(ValidationError):
        CompatibilityAssessment(
            verdict="compativel",
            reasoning="",
            citations=[],
            confidence=0.5,
        )


@pytest.mark.unit
def test_consistency_signal_rejects_an_unknown_source() -> None:
    with pytest.raises(ValidationError):
        ConsistencySignal(
            check="amount_plausibility",
            severity="attention",
            detail="Valor muito acima do padrao.",
            source="guessed",
        )


@pytest.mark.unit
def test_consistency_report_is_a_bag_of_signals_with_no_verdict() -> None:
    report = ConsistencyReport(
        signals=[
            ConsistencySignal(
                check="date_coherence",
                severity="info",
                detail="Data do evento dentro da faixa esperada.",
                source="deterministic",
            )
        ]
    )

    assert not hasattr(report, "verdict")
    assert report.signals[0].source == "deterministic"


# --- HumanDecision -------------------------------------------------------


@pytest.mark.unit
def test_human_decision_approve_needs_no_revision() -> None:
    decision = HumanDecision(decision="approve", notes="De acordo.")

    assert decision.edited_recommendation is None
    assert decision.decided_at.tzinfo is not None


@pytest.mark.unit
def test_human_decision_edit_requires_an_edited_recommendation() -> None:
    with pytest.raises(ValidationError):
        HumanDecision(decision="edit", notes="Trocar a acao recomendada.")


@pytest.mark.unit
def test_human_decision_non_edit_must_not_carry_an_edited_recommendation() -> None:
    with pytest.raises(ValidationError):
        HumanDecision(decision="reject", edited_recommendation=_recommendation())


@pytest.mark.unit
def test_human_decision_edit_round_trips_the_revision() -> None:
    decision = HumanDecision(decision="edit", edited_recommendation=_recommendation())

    assert decision.edited_recommendation is not None
    assert decision.edited_recommendation.confidence == 0.7


# --- JSON round-trips ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        _citation(),
        AuditEvent(
            node="compatibility",
            action="assess",
            model="deepseek/deepseek-v4-pro-0813",
            token_usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            confidence=0.5,
        ),
        CompatibilityAssessment(
            verdict=Verdict.INSUFFICIENT_INFORMATION,
            reasoning="Contexto insuficiente.",
            citations=[_citation()],
            confidence=0.4,
        ),
        _recommendation(),
    ],
)
def test_model_survives_a_json_round_trip(model: object) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())  # type: ignore[attr-defined]

    assert restored == model
