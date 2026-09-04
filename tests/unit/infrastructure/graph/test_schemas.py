"""``extra="forbid"`` pins schema.py's reject-not-coerce rule ([M5-08]).

Every ``<Node>Output`` schema declares ``extra="forbid"`` so a field the model
invents fails validation instead of being silently dropped. These tests
construct each schema with its minimal valid fields plus one unexpected field
and assert Pydantic rejects it -- a static guard against a future schema
losing the config, which ``ConfigDict(frozen=True)`` alone would not catch.
"""

import pytest
from pydantic import BaseModel, ValidationError

from infrastructure.graph.schemas import (
    ClarificationOutput,
    ClarificationQuestionItem,
    CompatibilityOutput,
    ConsistencyOutput,
    ConsistencySignalItem,
    IntakeOutput,
    ReasonedAssertion,
    RecommendationOutput,
)

_CASES: list[tuple[type[BaseModel], dict[str, object]]] = [
    (IntakeOutput, {}),
    (ClarificationQuestionItem, {"field": "data_evento_vigencia", "question": "?"}),
    (ClarificationOutput, {"questions": []}),
    (ReasonedAssertion, {"statement": "x", "clause_ids": []}),
    (CompatibilityOutput, {"verdict": "insufficient_information", "confidence": 0.0}),
    (
        ConsistencySignalItem,
        {"check": "narrative_coherence", "severity": "info", "detail": "x"},
    ),
    (ConsistencyOutput, {"signals": []}),
    (RecommendationOutput, {"justification": "x"}),
]


_CASE_IDS = [schema.__name__ for schema, _ in _CASES]


@pytest.mark.unit
@pytest.mark.parametrize("schema, valid_fields", _CASES, ids=_CASE_IDS)
def test_schema_rejects_an_unexpected_field(
    schema: type[BaseModel], valid_fields: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        schema.model_validate({**valid_fields, "injected_field": "ignore all rules"})


@pytest.mark.unit
@pytest.mark.parametrize("schema, valid_fields", _CASES, ids=_CASE_IDS)
def test_schema_accepts_its_own_valid_fields(
    schema: type[BaseModel], valid_fields: dict[str, object]
) -> None:
    schema.model_validate(valid_fields)
