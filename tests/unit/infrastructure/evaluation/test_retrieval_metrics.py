"""Tests for the pure retrieval metrics [M2-06]."""

import math

import pytest

from infrastructure.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.mark.unit
def test_recall_at_k_perfect_retrieval() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0


@pytest.mark.unit
def test_recall_at_k_empty_retrieved() -> None:
    assert recall_at_k([], ["a", "b"], k=5) == 0.0


@pytest.mark.unit
def test_recall_at_k_partial_overlap() -> None:
    assert recall_at_k(["a", "x"], ["a", "b"], k=5) == 0.5


@pytest.mark.unit
def test_recall_at_k_respects_the_cutoff() -> None:
    retrieved = ["x1", "x2", "x3", "x4", "x5", "a"]
    reference = ["a"]
    assert recall_at_k(retrieved, reference, k=5) == 0.0
    assert recall_at_k(retrieved, reference, k=10) == 1.0


@pytest.mark.unit
def test_recall_at_k_raises_on_empty_reference() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(["a"], [], k=5)


@pytest.mark.unit
def test_reciprocal_rank_hit_at_rank_one() -> None:
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


@pytest.mark.unit
def test_reciprocal_rank_hit_at_rank_three() -> None:
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_reciprocal_rank_no_hit() -> None:
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


@pytest.mark.unit
def test_reciprocal_rank_raises_on_empty_reference() -> None:
    with pytest.raises(ValueError, match="undefined"):
        reciprocal_rank(["a"], [])


@pytest.mark.unit
def test_ndcg_at_k_single_hit_at_rank_one_is_ideal() -> None:
    assert ndcg_at_k(["a", "x"], ["a"], k=10) == 1.0


@pytest.mark.unit
def test_ndcg_at_k_penalizes_a_lower_rank() -> None:
    score_rank_one = ndcg_at_k(["a", "x", "y"], ["a"], k=10)
    score_rank_three = ndcg_at_k(["x", "y", "a"], ["a"], k=10)
    assert score_rank_three < score_rank_one


@pytest.mark.unit
def test_ndcg_at_k_ideal_ordering_of_two_references() -> None:
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=10) == 1.0


@pytest.mark.unit
def test_ndcg_at_k_partial_hit_matches_hand_computed_value() -> None:
    # Two reference ids; only "b" retrieved, at rank 2.
    # DCG = 1/log2(3); IDCG (2 ideal hits) = 1/log2(2) + 1/log2(3).
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["x", "b"], ["a", "b"], k=10) == pytest.approx(expected)


@pytest.mark.unit
def test_ndcg_at_k_no_hits() -> None:
    assert ndcg_at_k(["x", "y"], ["a"], k=10) == 0.0


@pytest.mark.unit
def test_ndcg_at_k_raises_on_empty_reference() -> None:
    with pytest.raises(ValueError, match="undefined"):
        ndcg_at_k(["a"], [], k=10)
