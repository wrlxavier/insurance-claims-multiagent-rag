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

    results = classify_and_enrich_clauses(
        tree, records, rules, classifier, sleep=lambda seconds: None
    )
    assert len(results) == 1
    assert results[0].clause_type == ClauseType.OTHER
    assert results[0].type_source == TypeSource.LLM
    assert results[0].confidence == 0.0
    assert classifier.called


class CountingClassifier:
    """Mock recording how many times it was called -- for retry tests."""

    def __init__(
        self, calls_to_fail: int, expected_type: ClauseType, confidence: float
    ):
        self.calls_to_fail = calls_to_fail
        self.expected_type = expected_type
        self.confidence = confidence
        self.call_count = 0

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        self.call_count += 1
        if self.call_count <= self.calls_to_fail:
            raise RuntimeError("transient failure")
        return self.expected_type, self.confidence


def test_classify_and_enrich_retries_then_succeeds() -> None:
    tree = build_dummy_tree("Condições Especiais")
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules: list[tuple[re.Pattern[str], ClauseType]] = []
    classifier = CountingClassifier(
        calls_to_fail=1, expected_type=ClauseType.COVERAGE, confidence=0.8
    )
    sleeps: list[float] = []

    results = classify_and_enrich_clauses(
        tree, records, rules, classifier, sleep=sleeps.append
    )

    assert len(results) == 1
    assert results[0].clause_type == ClauseType.COVERAGE
    assert results[0].confidence == 0.8
    assert classifier.call_count == 2
    assert sleeps == [5.0]


def test_classify_and_enrich_exhausts_retries_then_falls_back_to_other() -> None:
    tree = build_dummy_tree("Condições Especiais")
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules: list[tuple[re.Pattern[str], ClauseType]] = []
    classifier = CountingClassifier(
        calls_to_fail=99, expected_type=ClauseType.COVERAGE, confidence=0.8
    )
    sleeps: list[float] = []

    results = classify_and_enrich_clauses(
        tree, records, rules, classifier, sleep=sleeps.append
    )

    assert len(results) == 1
    assert results[0].clause_type == ClauseType.OTHER
    assert results[0].confidence == 0.0
    assert classifier.call_count == 3
    assert sleeps == [5.0, 5.0]


class MappingClassifier:
    """Mock returning a distinct type per title -- catches mixups under concurrency."""

    def __init__(
        self, mapping: dict[str, ClauseType], fail_titles: frozenset[str] = frozenset()
    ):
        self.mapping = mapping
        self.fail_titles = fail_titles

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        if clause_title in self.fail_titles:
            raise RuntimeError(f"boom on {clause_title}")
        return self.mapping[clause_title], 0.77


def build_multi_clause_tree(titles: list[str]) -> ClauseTree:
    clauses = tuple(
        Clause(
            document_id="1",
            clause_id=f"c{i}",
            path=str(i),
            numbering_label=f"{i}.",
            title=title,
            convention=HeadingConvention.NUMBERED_DECIMAL,
            depth=1,
            parent_id=None,
            child_ids=(),
            content_lines=("body",),
            page_start=1,
            page_end=1,
        )
        for i, title in enumerate(titles)
    )
    return ClauseTree(
        document_id="1",
        filename="test.pdf",
        roots=clauses,
        all_clauses=clauses,
        report=ClauseTreeReport(
            "1", "test.pdf", len(clauses), 1, 0, 10, 0.0, "text", ()
        ),
    )


def test_classify_and_enrich_parallel_matches_sequential_results() -> None:
    titles = [f"Clause {i}" for i in range(20)]
    types = [ClauseType.COVERAGE, ClauseType.EXCLUSION, ClauseType.CONDITION]
    mapping = {title: types[i % len(types)] for i, title in enumerate(titles)}
    tree = build_multi_clause_tree(titles)
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules: list[tuple[re.Pattern[str], ClauseType]] = []

    sequential = classify_and_enrich_clauses(
        tree, records, rules, MappingClassifier(mapping), max_workers=1
    )
    parallel = classify_and_enrich_clauses(
        tree, records, rules, MappingClassifier(mapping), max_workers=5
    )

    assert [(t.clause.clause_id, t.clause_type) for t in sequential] == [
        (t.clause.clause_id, t.clause_type) for t in parallel
    ]
    for typed in parallel:
        assert typed.clause_type == mapping[typed.clause.title]
        assert typed.type_source == TypeSource.LLM
        assert typed.confidence == 0.77


def test_classify_and_enrich_parallel_keeps_exception_fallback() -> None:
    titles = [f"Clause {i}" for i in range(10)]
    mapping = dict.fromkeys(titles, ClauseType.COVERAGE)
    fail_titles = frozenset({"Clause 2", "Clause 7"})
    tree = build_multi_clause_tree(titles)
    records = [{"id": "1", "susep_process": "123", "cnpj": "456"}]
    rules: list[tuple[re.Pattern[str], ClauseType]] = []

    results = classify_and_enrich_clauses(
        tree,
        records,
        rules,
        MappingClassifier(mapping, fail_titles),
        max_workers=4,
        sleep=lambda seconds: None,
    )

    for typed in results:
        if typed.clause.title in fail_titles:
            assert typed.clause_type == ClauseType.OTHER
            assert typed.confidence == 0.0
        else:
            assert typed.clause_type == ClauseType.COVERAGE
            assert typed.confidence == 0.77
        assert typed.type_source == TypeSource.LLM
