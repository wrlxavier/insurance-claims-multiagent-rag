"""Tests for the flattened clause schema and required-provenance validation."""

from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from domain.clause_classification import (
    ClauseProvenance,
    ClauseType,
    TypedClause,
    TypeSource,
)
from domain.clause_tree import BoundarySource, Clause, HeadingConvention
from infrastructure.parsing.clause_schema import SCHEMA_VERSION, flatten_typed_clause


def _clause() -> Clause:
    return Clause(
        document_id="1",
        clause_id="1:2",
        path="2",
        numbering_label="2",
        title="2. COBERTURAS",
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=1,
        parent_id=None,
        child_ids=(),
        content_lines=("Texto da cobertura.",),
        page_start=3,
        page_end=4,
        bundle_section=None,
        bundle_confidence=None,
        is_depth_anomaly=False,
    )


def _provenance() -> ClauseProvenance:
    return ClauseProvenance(
        document_id="1",
        susep_process="15414900666201489",
        insurer="Bradesco Seguros",
        cnpj="12345678000199",
        product_line="CASCO",
        indemnity_regime="VD",
        process_year="2019",
    )


def _typed_clause(**provenance_overrides: str) -> TypedClause:
    provenance = replace(_provenance(), **provenance_overrides)
    return TypedClause(
        clause=_clause(),
        clause_type=ClauseType.COVERAGE,
        type_source=TypeSource.RULE,
        confidence=1.0,
        provenance=provenance,
    )


@pytest.mark.unit
def test_flatten_typed_clause_produces_valid_record() -> None:
    typed = _typed_clause()

    record = flatten_typed_clause(typed, source="text")

    assert record.schema_version == SCHEMA_VERSION
    assert record.clause_id == "1:2"
    assert record.path == "2"
    assert record.text == "Texto da cobertura."
    assert record.filing_year == "2019"
    assert record.source == "text"
    assert record.boundary_source == "deterministic"


@pytest.mark.unit
def test_flatten_typed_clause_propagates_vision_escalated_boundary_source() -> None:
    typed = _typed_clause()
    escalated_clause = replace(
        typed.clause, boundary_source=BoundarySource.VISION_ESCALATED
    )
    typed = replace(typed, clause=escalated_clause)

    record = flatten_typed_clause(typed, source="text")

    assert record.boundary_source == "vision_escalated"


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "susep_process",
        "insurer",
        "cnpj",
        "product_line",
        "indemnity_regime",
        "process_year",
    ],
)
def test_flatten_missing_provenance_field_raises(field: str) -> None:
    typed = _typed_clause(**{field: ""})

    with pytest.raises(ValidationError):
        flatten_typed_clause(typed, source="text")


@pytest.mark.unit
@pytest.mark.parametrize("field", ["clause_id", "path", "title"])
def test_flatten_empty_identity_field_raises(field: str) -> None:
    typed = _typed_clause()
    overrides: dict[str, Any] = {field: ""}
    clause = replace(typed.clause, **overrides)
    typed = replace(typed, clause=clause)

    with pytest.raises(ValidationError):
        flatten_typed_clause(typed, source="text")
