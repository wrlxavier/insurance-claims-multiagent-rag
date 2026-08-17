"""Tests for clause classification use case."""

import re

import pytest

from application.use_cases.clause_classification import (
    build_provenance,
    classify_and_enrich_clauses,
    normalize_heading,
)
from domain.clause_classification import (
    ClauseType,
    MissingProvenanceError,
    TypeSource,
)
from domain.clause_tree import (
    Clause,
    ClauseTree,
    ClauseTreeReport,
    HeadingConvention,
)


class DummyClassifier:
    """Mock for ClauseClassifierPort."""

    def __init__(
        self,
        expected_type: ClauseType,
        expected_confidence: float,
        should_fail: bool = False,
    ):
        self.expected_type = expected_type
        self.expected_confidence = expected_confidence
        self.should_fail = should_fail
        self.called = False

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        self.called = True
        if self.should_fail:
            raise RuntimeError("LLM Failure")
        return self.expected_type, self.expected_confidence


def build_dummy_tree(title: str, text: str = "body") -> ClauseTree:
    clause = Clause(
        document_id="1",
        clause_id="c1",
        path="1",
        numbering_label="1.",
        title=title,
        # Assuming enum is not strictly checked here if mock
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=1,
        parent_id=None,
        child_ids=(),
        content_lines=(text,),
        page_start=1,
        page_end=1,
    )
    return ClauseTree(
        document_id="1",
        filename="test.pdf",
        roots=(clause,),
        all_clauses=(clause,),
        report=ClauseTreeReport("1", "test.pdf", 1, 1, 0, 10, 0.0, "text", ()),
    )


def test_normalize_heading() -> None:
    assert normalize_heading("Riscos Excluídos") == "riscos excluidos"
    assert normalize_heading("  PREJUÍZOS NÃO INDENIZÁVEIS  ") == (
        "prejuizos nao indenizaveis"
    )
    assert normalize_heading("") == ""


def test_build_provenance_success() -> None:
    records = [
        {"id": "1", "susep_process": "123", "cnpj": "456", "insurer": "HDI Seguros"}
    ]
    prov = build_provenance("1", records)
    assert prov.document_id == "1"
    assert prov.susep_process == "123"
    assert prov.cnpj == "456"
    assert prov.insurer == "HDI Seguros"


def test_build_provenance_missing_error() -> None:
    records = [{"id": "2", "susep_process": "123"}]
    with pytest.raises(MissingProvenanceError):
        build_provenance("1", records)


def test_classify_and_enrich_rule_match() -> None:
    tree = build_dummy_tree("Riscos Excluídos")
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules = [(re.compile(r"^riscos excluidos$"), ClauseType.EXCLUSION)]
    classifier = DummyClassifier(ClauseType.COVERAGE, 0.9)

    results = classify_and_enrich_clauses(tree, records, rules, classifier)
    assert len(results) == 1
    assert results[0].clause_type == ClauseType.EXCLUSION
    assert results[0].type_source == TypeSource.RULE
    assert results[0].confidence == 1.0
    assert not classifier.called


def test_classify_and_enrich_llm_fallback() -> None:
    tree = build_dummy_tree("Condições Especiais")
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules = [(re.compile(r"^riscos excluidos$"), ClauseType.EXCLUSION)]
    classifier = DummyClassifier(ClauseType.CONDITION, 0.95)

    results = classify_and_enrich_clauses(tree, records, rules, classifier)
    assert len(results) == 1
    assert results[0].clause_type == ClauseType.CONDITION
    assert results[0].type_source == TypeSource.LLM
    assert results[0].confidence == 0.95
    assert classifier.called


def test_classify_and_enrich_llm_failure_fallback() -> None:
    tree = build_dummy_tree("Condições Especiais")
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules: list[tuple[re.Pattern[str], ClauseType]] = []
    classifier = DummyClassifier(ClauseType.CONDITION, 0.95, should_fail=True)

    results = classify_and_enrich_clauses(tree, records, rules, classifier)
    assert len(results) == 1
    assert results[0].clause_type == ClauseType.OTHER
    assert results[0].type_source == TypeSource.LLM
    assert results[0].confidence == 0.0
    assert classifier.called
