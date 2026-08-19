from pathlib import Path

import pytest

from domain.clause_classification import ClauseType
from infrastructure.parsing.llm_classification_cache import CachingClauseClassifier


class CountingClassifier:
    """Mock that records how many times it was actually invoked."""

    def __init__(self, result: tuple[ClauseType, float]) -> None:
        self.result = result
        self.call_count = 0

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        self.call_count += 1
        return self.result


@pytest.mark.unit
def test_cache_miss_calls_inner_and_persists(tmp_path: Path) -> None:
    inner = CountingClassifier((ClauseType.EXCLUSION, 0.9))
    cache_path = tmp_path / "cache.jsonl"
    classifier = CachingClauseClassifier(inner, model="m", cache_path=cache_path)

    result = classifier.classify("Riscos Excluídos", "texto")

    assert result == (ClauseType.EXCLUSION, 0.9)
    assert inner.call_count == 1
    assert cache_path.exists()
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 1


@pytest.mark.unit
def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    inner = CountingClassifier((ClauseType.COVERAGE, 0.8))
    cache_path = tmp_path / "cache.jsonl"
    classifier = CachingClauseClassifier(inner, model="m", cache_path=cache_path)

    first = classifier.classify("Coberturas", "texto")
    second = classifier.classify("Coberturas", "texto")

    assert first == second == (ClauseType.COVERAGE, 0.8)
    assert inner.call_count == 1


@pytest.mark.unit
def test_cache_persists_across_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    first_inner = CountingClassifier((ClauseType.DEFINITION, 0.7))
    CachingClauseClassifier(first_inner, model="m", cache_path=cache_path).classify(
        "Definições", "texto"
    )

    second_inner = CountingClassifier((ClauseType.OTHER, 0.1))
    second = CachingClauseClassifier(second_inner, model="m", cache_path=cache_path)
    result = second.classify("Definições", "texto")

    assert result == (ClauseType.DEFINITION, 0.7)
    assert second_inner.call_count == 0


@pytest.mark.unit
def test_different_model_is_a_separate_cache_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_a = CountingClassifier((ClauseType.PROCEDURE, 0.6))
    CachingClauseClassifier(inner_a, model="model-a", cache_path=cache_path).classify(
        "Procedimento", "texto"
    )

    inner_b = CountingClassifier((ClauseType.CONDITION, 0.5))
    result = CachingClauseClassifier(
        inner_b, model="model-b", cache_path=cache_path
    ).classify("Procedimento", "texto")

    assert result == (ClauseType.CONDITION, 0.5)
    assert inner_b.call_count == 1
