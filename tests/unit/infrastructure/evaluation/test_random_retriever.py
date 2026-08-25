"""Tests for the deliberately broken random retriever [M2-06]."""

import pytest

from infrastructure.evaluation.random_retriever import RandomRetriever

POOL = [f"clause-{i}" for i in range(20)]


@pytest.mark.unit
def test_retrieve_returns_k_unique_ids_from_the_pool() -> None:
    retriever = RandomRetriever(POOL, seed=1)
    result = retriever.retrieve("any question", k=5)
    assert len(result) == 5
    assert len(set(result)) == 5
    assert set(result) <= set(POOL)


@pytest.mark.unit
def test_retrieve_ignores_question_but_is_seed_deterministic() -> None:
    first = RandomRetriever(POOL, seed=1).retrieve("question A", k=5)
    second = RandomRetriever(POOL, seed=1).retrieve(
        "a completely different question", k=5
    )
    assert first == second


@pytest.mark.unit
def test_different_seeds_produce_different_output() -> None:
    first = RandomRetriever(POOL, seed=1).retrieve("q", k=5)
    second = RandomRetriever(POOL, seed=2).retrieve("q", k=5)
    assert first != second


@pytest.mark.unit
def test_k_larger_than_pool_returns_the_whole_pool() -> None:
    retriever = RandomRetriever(POOL, seed=1)
    result = retriever.retrieve("q", k=len(POOL) + 10)
    assert set(result) == set(POOL)
    assert len(result) == len(POOL)
