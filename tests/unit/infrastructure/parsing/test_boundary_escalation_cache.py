from pathlib import Path

import pytest

from domain.boundary_escalation import BoundaryReview
from infrastructure.parsing.boundary_escalation_cache import (
    CachingBoundaryVisionReviewer,
)


class CountingReviewer:
    """Mock that records how many times it was actually invoked."""

    def __init__(self, result: BoundaryReview) -> None:
        self.result = result
        self.call_count = 0

    def review(
        self,
        *,
        clause_title: str,
        claimed_page_start: int,
        claimed_page_end: int,
        page_images: tuple[bytes, ...],
    ) -> BoundaryReview:
        self.call_count += 1
        return self.result


_REVIEW = BoundaryReview(
    confirmed=True,
    corrected_page_start=None,
    corrected_page_end=None,
    split_suggested=False,
    split_notes="",
    reasoning="ok",
)


@pytest.mark.unit
def test_cache_miss_calls_inner_and_persists(tmp_path: Path) -> None:
    inner = CountingReviewer(_REVIEW)
    cache_path = tmp_path / "cache.jsonl"
    reviewer = CachingBoundaryVisionReviewer(inner, model="m", cache_path=cache_path)

    result = reviewer.review(
        clause_title="Riscos Excluídos",
        claimed_page_start=1,
        claimed_page_end=2,
        page_images=(b"page-1",),
    )

    assert result == _REVIEW
    assert inner.call_count == 1
    assert cache_path.exists()
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 1


@pytest.mark.unit
def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    inner = CountingReviewer(_REVIEW)
    cache_path = tmp_path / "cache.jsonl"
    reviewer = CachingBoundaryVisionReviewer(inner, model="m", cache_path=cache_path)

    first = reviewer.review(
        clause_title="Coberturas",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )
    second = reviewer.review(
        clause_title="Coberturas",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    assert first == second == _REVIEW
    assert inner.call_count == 1


@pytest.mark.unit
def test_cache_persists_across_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    first_inner = CountingReviewer(_REVIEW)
    CachingBoundaryVisionReviewer(first_inner, model="m", cache_path=cache_path).review(
        clause_title="Definições",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    other_review = BoundaryReview(
        confirmed=False,
        corrected_page_start=2,
        corrected_page_end=3,
        split_suggested=False,
        split_notes="",
        reasoning="different",
    )
    second_inner = CountingReviewer(other_review)
    second = CachingBoundaryVisionReviewer(
        second_inner, model="m", cache_path=cache_path
    )
    result = second.review(
        clause_title="Definições",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    assert result == _REVIEW
    assert second_inner.call_count == 0


@pytest.mark.unit
def test_different_model_is_a_separate_cache_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_a = CountingReviewer(_REVIEW)
    CachingBoundaryVisionReviewer(
        inner_a, model="model-a", cache_path=cache_path
    ).review(
        clause_title="Procedimento",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    other_review = BoundaryReview(
        confirmed=False,
        corrected_page_start=2,
        corrected_page_end=2,
        split_suggested=False,
        split_notes="",
        reasoning="different",
    )
    inner_b = CountingReviewer(other_review)
    result = CachingBoundaryVisionReviewer(
        inner_b, model="model-b", cache_path=cache_path
    ).review(
        clause_title="Procedimento",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    assert result == other_review
    assert inner_b.call_count == 1


@pytest.mark.unit
def test_different_page_images_is_a_separate_cache_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_a = CountingReviewer(_REVIEW)
    CachingBoundaryVisionReviewer(inner_a, model="m", cache_path=cache_path).review(
        clause_title="Procedimento",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1",),
    )

    other_review = BoundaryReview(
        confirmed=False,
        corrected_page_start=2,
        corrected_page_end=2,
        split_suggested=False,
        split_notes="",
        reasoning="different image",
    )
    inner_b = CountingReviewer(other_review)
    result = CachingBoundaryVisionReviewer(
        inner_b, model="m", cache_path=cache_path
    ).review(
        clause_title="Procedimento",
        claimed_page_start=1,
        claimed_page_end=1,
        page_images=(b"page-1-different-bytes",),
    )

    assert result == other_review
    assert inner_b.call_count == 1
