"""Rank / score fusion of the retrieval legs -- [M3-04]."""

import pytest

from infrastructure.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion


@pytest.mark.unit
def test_rrf_rewards_agreement_across_lists() -> None:
    lexical = ["a", "b", "c"]
    dense = ["c", "a", "d"]

    # `a` is rank 1+2, `c` is rank 3+1 -> both beat the singletons; `a` edges
    # `c` (1/61 + 1/62 > 1/63 + 1/61).
    assert reciprocal_rank_fusion([lexical, dense], k=60) == ["a", "c", "b", "d"]


@pytest.mark.unit
def test_rrf_tie_breaks_on_clause_id_ascending() -> None:
    # Symmetric single-list input: every id has the same score.
    assert reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60) == ["a", "b"]


@pytest.mark.unit
def test_rrf_handles_empty_and_single_lists() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
    assert reciprocal_rank_fusion([["a", "b"]], k=60) == ["a", "b"]


@pytest.mark.unit
def test_rrf_counts_a_repeated_id_once_at_its_best_rank() -> None:
    # A leg that (defensively) emits a dup must not double-score it.
    once = reciprocal_rank_fusion([["a", "b"]], k=60)
    twice = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
    assert once == twice == ["a", "b"]


@pytest.mark.unit
def test_weighted_fusion_normalises_each_list_independently() -> None:
    lexical = [("a", 10.0), ("b", 0.0)]  # -> a:1.0, b:0.0
    dense = [("b", 0.9), ("a", 0.1)]  # -> b:1.0, a:0.0
    # equal weights -> a and b both sum to 1.0 -> tie -> id order.
    assert weighted_score_fusion([lexical, dense], weights=(0.5, 0.5)) == ["a", "b"]


@pytest.mark.unit
def test_weighted_fusion_weights_shift_the_ranking() -> None:
    lexical = [("a", 10.0), ("b", 0.0)]
    dense = [("b", 0.9), ("a", 0.1)]
    assert weighted_score_fusion([lexical, dense], weights=(0.9, 0.1)) == ["a", "b"]
    assert weighted_score_fusion([lexical, dense], weights=(0.1, 0.9)) == ["b", "a"]


@pytest.mark.unit
def test_weighted_fusion_all_equal_scores_normalise_to_one() -> None:
    # A list whose scores are all equal (incl. a singleton) contributes its
    # full weight to every member rather than dividing by zero.
    result = weighted_score_fusion(
        [[("a", 5.0), ("b", 5.0)], [("a", 1.0)]], weights=(1.0, 1.0)
    )
    assert result == ["a", "b"]  # a: 1.0 + 1.0, b: 1.0 + 0


@pytest.mark.unit
def test_weighted_fusion_rejects_a_weight_count_mismatch() -> None:
    with pytest.raises(ValueError, match="2 lists but 1 weights"):
        weighted_score_fusion([[("a", 1.0)], [("b", 1.0)]], weights=(1.0,))
