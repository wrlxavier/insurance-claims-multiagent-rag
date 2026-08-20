"""Tests for [M1-04d]'s vision-LLM boundary-escalation use case."""

from pathlib import Path

import pytest

from application.use_cases.boundary_escalation import (
    escalate_boundaries,
    find_suspicious_clauses,
)
from domain.boundary_escalation import BoundaryReview
from domain.clause_tree import (
    BoundarySource,
    Clause,
    ClauseTree,
    ClauseTreeReport,
    HeadingConvention,
)


def _clause(
    clause_id: str,
    *,
    title: str = "Title",
    convention: HeadingConvention = HeadingConvention.NUMBERED_DECIMAL,
    parent_id: str | None = None,
    content_lines: tuple[str, ...] = (),
    content_line_pages: tuple[int, ...] = (),
    page_start: int,
    page_end: int,
    is_depth_anomaly: bool = False,
) -> Clause:
    return Clause(
        document_id="d1",
        clause_id=clause_id,
        path=clause_id,
        numbering_label="1",
        title=title,
        convention=convention,
        depth=1,
        parent_id=parent_id,
        child_ids=(),
        content_lines=content_lines,
        page_start=page_start,
        page_end=page_end,
        is_depth_anomaly=is_depth_anomaly,
        content_line_pages=content_line_pages,
    )


def _tree(clauses: tuple[Clause, ...], *, extraction_mode: str = "text") -> ClauseTree:
    roots = tuple(clause for clause in clauses if clause.parent_id is None)
    return ClauseTree(
        document_id="d1",
        filename="f.pdf",
        roots=roots,
        all_clauses=clauses,
        report=ClauseTreeReport(
            "d1", "f.pdf", len(clauses), 1, 0, 100, 0.0, extraction_mode, ()
        ),
    )


# ---------------------------------------------------------------------------
# find_suspicious_clauses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_suspicious_clauses_flags_oversized() -> None:
    small = _clause(
        "d1:a", content_lines=("x",), content_line_pages=(1,), page_start=1, page_end=1
    )
    big = _clause(
        "d1:b",
        content_lines=("y" * 50,),
        content_line_pages=(2,),
        page_start=2,
        page_end=2,
    )
    tree = _tree((small, big))

    flags = find_suspicious_clauses(tree, max_page_span=10, max_char_count=10)

    assert {flag.clause_id: flag.reasons for flag in flags} == {"d1:b": ("oversized",)}


@pytest.mark.unit
def test_find_suspicious_clauses_flags_every_clause_in_an_ocr_document() -> None:
    a = _clause(
        "d1:a", content_lines=("x",), content_line_pages=(1,), page_start=1, page_end=1
    )
    b = _clause(
        "d1:b", content_lines=("y",), content_line_pages=(2,), page_start=2, page_end=2
    )
    tree = _tree((a, b), extraction_mode="ocr_required")

    flags = find_suspicious_clauses(tree, max_page_span=100, max_char_count=100000)

    assert {flag.clause_id for flag in flags} == {"d1:a", "d1:b"}
    assert all(flag.reasons == ("ocr",) for flag in flags)


@pytest.mark.unit
def test_find_suspicious_clauses_flags_depth_anomaly() -> None:
    clause = _clause("d1:a", page_start=1, page_end=1, is_depth_anomaly=True)
    tree = _tree((clause,))

    flags = find_suspicious_clauses(tree, max_page_span=100, max_char_count=100000)

    assert len(flags) == 1
    assert flags[0].clause_id == "d1:a"
    assert "depth_anomaly" in flags[0].reasons


@pytest.mark.unit
def test_find_suspicious_clauses_flags_neighbor_of_unnumbered_part() -> None:
    part = _clause(
        "d1:part",
        convention=HeadingConvention.UNNUMBERED_PART,
        page_start=1,
        page_end=1,
    )
    after = _clause("d1:after", page_start=2, page_end=2)
    tree = _tree((part, after))

    flags = find_suspicious_clauses(tree, max_page_span=100, max_char_count=100000)

    assert len(flags) == 1
    assert flags[0].clause_id == "d1:after"
    assert "unnumbered_part_adjacent" in flags[0].reasons


@pytest.mark.unit
def test_find_suspicious_clauses_clean_tree_yields_no_flags() -> None:
    a = _clause(
        "d1:a", content_lines=("x",), content_line_pages=(1,), page_start=1, page_end=1
    )
    b = _clause(
        "d1:b", content_lines=("y",), content_line_pages=(2,), page_start=2, page_end=2
    )
    tree = _tree((a, b))

    flags = find_suspicious_clauses(tree, max_page_span=100, max_char_count=100000)

    assert flags == ()


# ---------------------------------------------------------------------------
# escalate_boundaries
# ---------------------------------------------------------------------------


class ScriptedReviewer:
    """Returns a fixed BoundaryReview for every call; records calls made."""

    def __init__(self, review: BoundaryReview) -> None:
        self.review_to_return = review
        self.calls: list[dict[str, object]] = []

    def review(
        self,
        *,
        clause_title: str,
        claimed_page_start: int,
        claimed_page_end: int,
        page_images: tuple[bytes, ...],
    ) -> BoundaryReview:
        self.calls.append(
            {
                "clause_title": clause_title,
                "claimed_page_start": claimed_page_start,
                "claimed_page_end": claimed_page_end,
                "page_images": page_images,
            }
        )
        return self.review_to_return


class CountingReviewer:
    """Fails the first ``calls_to_fail`` calls, then returns a fixed review."""

    def __init__(self, calls_to_fail: int, review: BoundaryReview) -> None:
        self.calls_to_fail = calls_to_fail
        self.review_to_return = review
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
        if self.call_count <= self.calls_to_fail:
            raise RuntimeError("transient failure")
        return self.review_to_return


class FakeRasterizer:
    """Returns deterministic placeholder page images; records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, int, int]] = []

    def rasterize(
        self, pdf_path: Path, document_id: str, first_page: int, last_page: int
    ) -> tuple[bytes, ...]:
        self.calls.append((pdf_path, document_id, first_page, last_page))
        return tuple(f"page-{p}".encode() for p in range(first_page, last_page + 1))


def _three_clause_tree() -> tuple[ClauseTree, Clause, Clause, Clause]:
    """A, B, C in document order. B alone trips the oversized-page-span flag."""
    a = _clause(
        "d1:a",
        content_lines=("a1", "a2"),
        content_line_pages=(1, 2),
        page_start=1,
        page_end=2,
    )
    b = _clause(
        "d1:b",
        content_lines=("b1", "b2", "b3"),
        content_line_pages=(3, 4, 5),
        page_start=3,
        page_end=5,
    )
    c = _clause(
        "d1:c",
        content_lines=("c1", "c2"),
        content_line_pages=(6, 7),
        page_start=6,
        page_end=7,
    )
    return _tree((a, b, c)), a, b, c


@pytest.mark.unit
def test_escalate_boundaries_confirmed_no_op() -> None:
    tree, a, b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=True,
        corrected_page_start=None,
        corrected_page_end=None,
        split_suggested=False,
        split_notes="",
        reasoning="looks right",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert len(outcomes) == 1
    assert outcomes[0].clause_id == "d1:b"
    assert outcomes[0].applied is False
    assert outcomes[0].note == "confirmed, no change"
    assert rasterizer.calls == [(Path("doc.pdf"), "d1", 2, 6)]

    revised = {clause.clause_id: clause for clause in revised_tree.all_clauses}
    assert revised["d1:b"].content_lines == b.content_lines
    assert revised["d1:b"].page_start == b.page_start
    assert revised["d1:b"].page_end == b.page_end
    assert revised["d1:b"].boundary_source == BoundarySource.VISION_ESCALATED
    assert revised["d1:a"].boundary_source == BoundarySource.DETERMINISTIC


@pytest.mark.unit
def test_escalate_boundaries_shrink_correction_moves_content_to_next_neighbor() -> None:
    tree, _a, b, c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=False,
        corrected_page_start=3,
        corrected_page_end=4,
        split_suggested=False,
        split_notes="",
        reasoning="ends a page earlier than claimed",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert outcomes[0].applied is True
    assert outcomes[0].note == "corrected page range applied"

    revised = {clause.clause_id: clause for clause in revised_tree.all_clauses}
    revised_b = revised["d1:b"]
    assert revised_b.page_start == 3
    assert revised_b.page_end == 4
    assert revised_b.content_lines == ("b1", "b2")
    assert revised_b.content_line_pages == (3, 4)

    revised_c = revised["d1:c"]
    assert revised_c.content_lines == ("b3", "c1", "c2")
    assert revised_c.content_line_pages == (5, 6, 7)
    assert revised_c.page_start == 5
    assert revised_c.page_end == c.page_end
    assert revised_c.boundary_source == BoundarySource.VISION_ESCALATED


@pytest.mark.unit
def test_escalate_boundaries_extend_correction_pulls_from_previous_neighbor() -> None:
    tree, a, _b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=False,
        corrected_page_start=2,
        corrected_page_end=5,
        split_suggested=False,
        split_notes="",
        reasoning="actually starts a page earlier",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert outcomes[0].applied is True

    revised = {clause.clause_id: clause for clause in revised_tree.all_clauses}
    revised_b = revised["d1:b"]
    assert revised_b.page_start == 2
    assert revised_b.content_lines == ("a2", "b1", "b2", "b3")
    assert revised_b.content_line_pages == (2, 3, 4, 5)

    revised_a = revised["d1:a"]
    assert revised_a.content_lines == ("a1",)
    assert revised_a.content_line_pages == (1,)
    assert revised_a.page_end == 1
    assert revised_a.page_start == a.page_start
    assert revised_a.boundary_source == BoundarySource.VISION_ESCALATED


@pytest.mark.unit
def test_escalate_boundaries_correction_outside_margin_not_applied() -> None:
    tree, _a, b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=False,
        corrected_page_start=1,
        corrected_page_end=5,
        split_suggested=False,
        split_notes="",
        reasoning="way outside what was shown",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert outcomes[0].applied is False
    assert "outside the rasterized margin" in outcomes[0].note

    revised_b = next(
        clause for clause in revised_tree.all_clauses if clause.clause_id == "d1:b"
    )
    assert revised_b.content_lines == b.content_lines
    assert revised_b.page_start == b.page_start
    assert revised_b.page_end == b.page_end


@pytest.mark.unit
def test_escalate_boundaries_neighbor_without_page_attribution_not_applied() -> None:
    a = _clause(
        "d1:a",
        content_lines=("a1", "a2"),
        content_line_pages=(1, 2),
        page_start=1,
        page_end=2,
    )
    b = _clause(
        "d1:b",
        content_lines=("b1", "b2", "b3"),
        content_line_pages=(3, 4, 5),
        page_start=3,
        page_end=5,
    )
    # Simulates a pre-[M1-04d] cached clause: content but no per-line pages.
    c = _clause(
        "d1:c",
        content_lines=("c1", "c2"),
        content_line_pages=(),
        page_start=6,
        page_end=7,
    )
    tree = _tree((a, b, c))
    review = BoundaryReview(
        confirmed=False,
        corrected_page_start=3,
        corrected_page_end=4,
        split_suggested=False,
        split_notes="",
        reasoning="ends a page earlier than claimed",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert outcomes[0].applied is False
    assert "per-line page attribution" in outcomes[0].note

    revised_b = next(
        clause for clause in revised_tree.all_clauses if clause.clause_id == "d1:b"
    )
    assert revised_b.content_lines == b.content_lines
    assert revised_b.boundary_source == BoundarySource.VISION_ESCALATED


@pytest.mark.unit
def test_escalate_boundaries_split_suggestion_is_recorded_not_applied() -> None:
    tree, _a, b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=True,
        corrected_page_start=None,
        corrected_page_end=None,
        split_suggested=True,
        split_notes="Looks like two sub-clauses on page 4.",
        reasoning="boundary ok but content should split",
    )
    reviewer = ScriptedReviewer(review)
    rasterizer = FakeRasterizer()

    revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
    )

    assert outcomes[0].applied is False
    assert "not auto-applied" in outcomes[0].note
    assert outcomes[0].review.split_suggested is True
    assert outcomes[0].review.split_notes == "Looks like two sub-clauses on page 4."

    revised = {clause.clause_id: clause for clause in revised_tree.all_clauses}
    assert revised["d1:b"].content_lines == b.content_lines
    assert revised["d1:b"].boundary_source == BoundarySource.VISION_ESCALATED
    assert revised["d1:a"].boundary_source == BoundarySource.DETERMINISTIC
    assert revised["d1:c"].boundary_source == BoundarySource.DETERMINISTIC


@pytest.mark.unit
def test_escalate_boundaries_retries_then_succeeds() -> None:
    tree, _a, _b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=True,
        corrected_page_start=None,
        corrected_page_end=None,
        split_suggested=False,
        split_notes="",
        reasoning="ok",
    )
    reviewer = CountingReviewer(calls_to_fail=1, review=review)
    rasterizer = FakeRasterizer()
    sleeps: list[float] = []

    _revised_tree, outcomes = escalate_boundaries(
        tree,
        Path("doc.pdf"),
        reviewer=reviewer,
        rasterizer=rasterizer,
        page_count=10,
        max_page_span=2,
        max_char_count=1000,
        sleep=sleeps.append,
    )

    assert reviewer.call_count == 2
    assert sleeps == [5.0]
    assert outcomes[0].applied is False
    assert outcomes[0].note == "confirmed, no change"


@pytest.mark.unit
def test_escalate_boundaries_exhausts_retries_and_reraises() -> None:
    tree, _a, _b, _c = _three_clause_tree()
    review = BoundaryReview(
        confirmed=True,
        corrected_page_start=None,
        corrected_page_end=None,
        split_suggested=False,
        split_notes="",
        reasoning="ok",
    )
    reviewer = CountingReviewer(calls_to_fail=99, review=review)
    rasterizer = FakeRasterizer()
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="transient failure"):
        escalate_boundaries(
            tree,
            Path("doc.pdf"),
            reviewer=reviewer,
            rasterizer=rasterizer,
            page_count=10,
            max_page_span=2,
            max_char_count=1000,
            sleep=sleeps.append,
        )

    assert reviewer.call_count == 3
    assert sleeps == [5.0, 5.0]
