"""Hand-rolled Okapi BM25 -- [M3-03]."""

import math

import pytest

from infrastructure.rag.bm25 import BM25Index, build_bm25_index, score, top_n

_DOCS = [
    ("d1", ["a", "b", "c"]),
    ("d2", ["a", "a", "d"]),
    ("d3", ["a", "b", "b", "e", "f"]),
]


def _index() -> BM25Index:
    return build_bm25_index(_DOCS, k1=1.5, b=0.75)


@pytest.mark.unit
def test_idf_is_the_lucene_plus_one_form_and_never_negative() -> None:
    index = _index()
    # 'a' is in every doc; classic Okapi IDF would be negative here.
    assert index.idf["a"] == pytest.approx(math.log(1 + 0.5 / 3.5))
    assert index.idf["a"] > 0.0
    assert all(value > 0.0 for value in index.idf.values())


@pytest.mark.unit
def test_score_matches_the_hand_computed_bm25() -> None:
    scores = score(_index(), ["b"])
    assert scores == pytest.approx({"d1": 0.51189, "d3": 0.60117}, abs=1e-4)


@pytest.mark.unit
def test_a_doc_sharing_no_term_is_absent_not_zero() -> None:
    scores = score(_index(), ["b"])
    assert "d2" not in scores  # d2 has no 'b'


@pytest.mark.unit
def test_unseen_term_empty_query_and_empty_corpus_return_nothing() -> None:
    index = _index()
    assert score(index, ["never-indexed"]) == {}
    assert score(index, []) == {}
    assert score(build_bm25_index([], k1=1.5, b=0.75), ["a"]) == {}


@pytest.mark.unit
def test_top_n_truncates_and_breaks_ties_by_doc_id() -> None:
    assert [doc_id for doc_id, _ in top_n(_index(), ["a", "b"], 2)] == ["d3", "d1"]
    tie = build_bm25_index([("z2", ["x"]), ("z1", ["x"])], k1=1.5, b=0.75)
    assert [doc_id for doc_id, _ in top_n(tie, ["x"], 2)] == ["z1", "z2"]
