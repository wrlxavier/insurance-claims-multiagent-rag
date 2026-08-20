"""Use case for [M1-04d]'s vision-LLM boundary-escalation pass.

A narrow, last-resort escalation on top of [application.use_cases.
clause_segmentation.segment_document]'s deterministic pass: only clauses
the deterministic pass already flags as suspicious (oversized, OCR-derived,
or adjacent to an UNNUMBERED_PART/depth-anomaly boundary -- see
[find_suspicious_clauses]) get a second, vision-based review.

A vision-proposed page-range correction is applied via a bounded, mechanical
line-reassignment between the flagged clause and its immediate
document-order neighbor, using the per-line page attribution
[application.use_cases.clause_segmentation] now records on every [Clause]
(``content_line_pages``). A vision-proposed sub-clause split is never
auto-applied -- by explicit project decision, confirmed with the project
owner: splitting would mint new clause_id/path values and restructure the
tree, a materially bigger, riskier change than this issue's scope, and
would risk exactly the clause_id churn [M1-07]'s determinism guarantee
exists to prevent. A split suggestion is recorded (see [BoundaryReview.
split_notes]) for a human or [M1-08c] to act on later.

A corrected page is only ever trusted if it falls inside the page range
that was actually rasterized and shown to the model -- the model never
gets to claim a boundary it wasn't shown.
"""

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from application.ports.boundary_vision_reviewer import BoundaryVisionReviewerPort
from application.ports.page_rasterizer import PageRasterizerPort
from application.use_cases.clause_segmentation import find_oversized_clauses
from domain.boundary_escalation import (
    BoundaryEscalationOutcome,
    BoundaryReview,
    SuspicionFlag,
)
from domain.clause_tree import BoundarySource, Clause, ClauseTree, HeadingConvention

BOUNDARY_ESCALATION_MAX_ATTEMPTS = 3
BOUNDARY_ESCALATION_RETRY_DELAY_SECONDS = 5.0
# The same +/-1-page margin already validated in [M1-08b]'s
# scripts/validate_parsing_quality_sample.py.
BOUNDARY_ESCALATION_PAGE_MARGIN = 1


def find_suspicious_clauses(
    tree: ClauseTree, *, max_page_span: int, max_char_count: int
) -> tuple[SuspicionFlag, ...]:
    """Flag clauses worth a vision review. Pure, never raises.

    A clause is flagged when any of the following holds:

    - ``"oversized"``: it trips [find_oversized_clauses]'s existing
      page-span/char-count safeguard.
    - ``"ocr"``: its document has no font/position heading signal at all
      (``tree.report.extraction_mode == "ocr_required"`` -- a document-level
      signal, since [Clause] carries no per-clause OCR flag).
    - ``"depth_anomaly"``: [Clause.is_depth_anomaly] is True.
    - ``"unnumbered_part_adjacent"`` / ``"depth_anomaly_adjacent"``: the
      clause immediately before or after it in ``tree.all_clauses`` (true
      document reading order) has convention UNNUMBERED_PART, or is itself
      a depth anomaly, respectively.
    """
    oversized_ids = {
        clause.clause_id
        for clause in find_oversized_clauses(
            tree.all_clauses,
            max_page_span=max_page_span,
            max_char_count=max_char_count,
        )
    }
    is_ocr_document = tree.report.extraction_mode == "ocr_required"
    clauses = tree.all_clauses

    flags: list[SuspicionFlag] = []
    for index, clause in enumerate(clauses):
        reasons: list[str] = []
        if clause.clause_id in oversized_ids:
            reasons.append("oversized")
        if is_ocr_document:
            reasons.append("ocr")
        if clause.is_depth_anomaly:
            reasons.append("depth_anomaly")

        neighbors = (
            clauses[index - 1] if index > 0 else None,
            clauses[index + 1] if index + 1 < len(clauses) else None,
        )
        for neighbor in neighbors:
            if neighbor is None:
                continue
            if (
                neighbor.convention == HeadingConvention.UNNUMBERED_PART
                and "unnumbered_part_adjacent" not in reasons
            ):
                reasons.append("unnumbered_part_adjacent")
            if neighbor.is_depth_anomaly and "depth_anomaly_adjacent" not in reasons:
                reasons.append("depth_anomaly_adjacent")

        if reasons:
            flags.append(
                SuspicionFlag(clause_id=clause.clause_id, reasons=tuple(reasons))
            )

    return tuple(flags)


def _resolve_rasterize_range(
    page_start: int, page_end: int, margin: int, page_count: int
) -> tuple[int, int]:
    """Same clamp as validate_parsing_quality_sample.resolve_page_range."""
    first = max(1, page_start - margin)
    last = min(page_count, page_end + margin)
    return first, last


def _review_with_retry(
    reviewer: BoundaryVisionReviewerPort,
    *,
    clause_title: str,
    claimed_page_start: int,
    claimed_page_end: int,
    page_images: tuple[bytes, ...],
    max_attempts: int = BOUNDARY_ESCALATION_MAX_ATTEMPTS,
    retry_delay_seconds: float = BOUNDARY_ESCALATION_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> BoundaryReview:
    """Call the vision port, retrying transient failures.

    Retries up to ``max_attempts`` times, sleeping ``retry_delay_seconds``
    between attempts, then RE-RAISES -- matching
    ``scripts/validate_parsing_quality_sample.py``'s ``call_llm_with_retry``
    precedent, not [application.use_cases.clause_classification]'s
    OTHER/0.0 silent fallback: a boundary review that silently defaulted to
    "confirmed" on failure would be indistinguishable from a real
    confirmation, which is worse than a loud failure.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return reviewer.review(
                clause_title=clause_title,
                claimed_page_start=claimed_page_start,
                claimed_page_end=claimed_page_end,
                page_images=page_images,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                sleep(retry_delay_seconds)
    assert last_exc is not None
    raise last_exc


def _partition_by_page(
    lines: tuple[str, ...],
    pages: tuple[int, ...],
    *,
    keep: Callable[[int], bool],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    """Split parallel (lines, pages) into (kept, moved), preserving order."""
    kept_lines: list[str] = []
    kept_pages: list[int] = []
    moved_lines: list[str] = []
    moved_pages: list[int] = []
    for line, page in zip(lines, pages, strict=True):
        if keep(page):
            kept_lines.append(line)
            kept_pages.append(page)
        else:
            moved_lines.append(line)
            moved_pages.append(page)
    return tuple(kept_lines), tuple(kept_pages), tuple(moved_lines), tuple(moved_pages)


def _has_page_attribution(clause: Clause) -> bool:
    return bool(clause.content_lines) and len(clause.content_line_pages) == len(
        clause.content_lines
    )


def _pages_shared_with_others(
    clauses_by_id: dict[str, Clause],
    order: list[str],
    pages: set[int],
    involved: set[str],
) -> set[int]:
    """Return which of ``pages`` also carry content from clauses outside ``involved``.

    A vision review can only reason at page granularity -- it is shown page
    images -- but a page routinely carries several clauses (a glossary page
    in doc 23 carries 13). Reassigning a whole page's lines between two
    neighbours therefore sweeps up any third clause sharing that page.
    [M1-08c] measured this as the dominant cause of the boundary
    regressions the first escalation run introduced: 95 of its 136 applied
    corrections moved lines on a page shared with a third clause. See
    ``docs/PARSING.md``'s third-measurement section for the evidence.
    """
    if not pages:
        return set()
    shared: set[int] = set()
    for clause_id in order:
        if clause_id in involved:
            continue
        for page in clauses_by_id[clause_id].content_line_pages:
            if page in pages:
                shared.add(page)
    return shared


def _apply_start_edge(
    clauses_by_id: dict[str, Clause], order: list[str], index: int, new_start: int
) -> tuple[dict[str, Clause], bool, str]:
    """Reassign lines between a clause and its preceding neighbor.

    Shrinking (``new_start`` > current ``page_start``): the clause's own
    leading lines on pages before ``new_start`` move to the preceding
    neighbor. Widening (``new_start`` < current ``page_start``): the
    clause pulls the preceding neighbor's trailing lines on pages
    ``>= new_start``.
    """
    clause = clauses_by_id[order[index]]
    if not _has_page_attribution(clause):
        return (
            clauses_by_id,
            False,
            "clause has no per-line page attribution; front-edge correction "
            "not applied",
        )
    if index == 0:
        return (
            clauses_by_id,
            False,
            "no preceding neighbor for the front-edge correction; not applied",
        )

    prev_id = order[index - 1]
    prev = clauses_by_id[prev_id]
    if not _has_page_attribution(prev) and prev.content_lines:
        return (
            clauses_by_id,
            False,
            "preceding neighbor has no per-line page attribution; "
            "front-edge correction not applied",
        )

    if new_start > clause.page_start:
        moving_pages = {p for p in clause.content_line_pages if p < new_start}
    else:
        moving_pages = {p for p in prev.content_line_pages if p >= new_start}
    shared = _pages_shared_with_others(
        clauses_by_id, order, moving_pages, {clause.clause_id, prev_id}
    )
    if shared:
        return (
            clauses_by_id,
            False,
            f"page(s) {sorted(shared)} also carry other clauses' content; a "
            "page-granular front-edge correction would sweep them up; not applied",
        )

    if new_start > clause.page_start:
        kept_lines, kept_pages, moved_lines, moved_pages = _partition_by_page(
            clause.content_lines,
            clause.content_line_pages,
            keep=lambda p: p >= new_start,
        )
        if not kept_lines:
            return (
                clauses_by_id,
                False,
                "front-edge correction would empty the clause; not applied",
            )
        if not moved_lines:
            return (
                clauses_by_id,
                False,
                "no content found on the trimmed front pages; not applied",
            )
        clauses_by_id[clause.clause_id] = replace(
            clause,
            content_lines=kept_lines,
            content_line_pages=kept_pages,
            page_start=kept_pages[0],
        )
        clauses_by_id[prev_id] = replace(
            prev,
            content_lines=prev.content_lines + moved_lines,
            content_line_pages=prev.content_line_pages + moved_pages,
            page_end=moved_pages[-1],
            boundary_source=BoundarySource.VISION_ESCALATED,
        )
    else:
        kept_prev_lines, kept_prev_pages, moved_lines, moved_pages = _partition_by_page(
            prev.content_lines, prev.content_line_pages, keep=lambda p: p < new_start
        )
        if not moved_lines:
            return (
                clauses_by_id,
                False,
                "preceding neighbor has no content on the claimed pages; not applied",
            )
        clauses_by_id[clause.clause_id] = replace(
            clause,
            content_lines=moved_lines + clause.content_lines,
            content_line_pages=moved_pages + clause.content_line_pages,
            page_start=new_start,
        )
        clauses_by_id[prev_id] = replace(
            prev,
            content_lines=kept_prev_lines,
            content_line_pages=kept_prev_pages,
            page_end=kept_prev_pages[-1] if kept_prev_pages else prev.page_start,
            boundary_source=BoundarySource.VISION_ESCALATED,
        )
    return clauses_by_id, True, ""


def _apply_end_edge(
    clauses_by_id: dict[str, Clause], order: list[str], index: int, new_end: int
) -> tuple[dict[str, Clause], bool, str]:
    """Reassign lines between a clause and its following neighbor.

    Mirror of [_apply_start_edge] for the trailing boundary.
    """
    clause = clauses_by_id[order[index]]
    if not _has_page_attribution(clause):
        return (
            clauses_by_id,
            False,
            "clause has no per-line page attribution; back-edge correction not applied",
        )
    if index + 1 >= len(order):
        return (
            clauses_by_id,
            False,
            "no following neighbor for the back-edge correction; not applied",
        )

    next_id = order[index + 1]
    next_clause = clauses_by_id[next_id]
    if not _has_page_attribution(next_clause) and next_clause.content_lines:
        return (
            clauses_by_id,
            False,
            "following neighbor has no per-line page attribution; "
            "back-edge correction not applied",
        )

    if new_end < clause.page_end:
        moving_pages = {p for p in clause.content_line_pages if p > new_end}
    else:
        moving_pages = {p for p in next_clause.content_line_pages if p <= new_end}
    shared = _pages_shared_with_others(
        clauses_by_id, order, moving_pages, {clause.clause_id, next_id}
    )
    if shared:
        return (
            clauses_by_id,
            False,
            f"page(s) {sorted(shared)} also carry other clauses' content; a "
            "page-granular back-edge correction would sweep them up; not applied",
        )

    if new_end < clause.page_end:
        kept_lines, kept_pages, moved_lines, moved_pages = _partition_by_page(
            clause.content_lines, clause.content_line_pages, keep=lambda p: p <= new_end
        )
        if not kept_lines:
            return (
                clauses_by_id,
                False,
                "back-edge correction would empty the clause; not applied",
            )
        if not moved_lines:
            return (
                clauses_by_id,
                False,
                "no content found on the trimmed back pages; not applied",
            )
        clauses_by_id[clause.clause_id] = replace(
            clause,
            content_lines=kept_lines,
            content_line_pages=kept_pages,
            page_end=kept_pages[-1],
        )
        clauses_by_id[next_id] = replace(
            next_clause,
            content_lines=moved_lines + next_clause.content_lines,
            content_line_pages=moved_pages + next_clause.content_line_pages,
            page_start=moved_pages[0],
            boundary_source=BoundarySource.VISION_ESCALATED,
        )
    else:
        kept_next_lines, kept_next_pages, moved_lines, moved_pages = _partition_by_page(
            next_clause.content_lines,
            next_clause.content_line_pages,
            keep=lambda p: p > new_end,
        )
        if not moved_lines:
            return (
                clauses_by_id,
                False,
                "following neighbor has no content on the claimed pages; not applied",
            )
        clauses_by_id[clause.clause_id] = replace(
            clause,
            content_lines=clause.content_lines + moved_lines,
            content_line_pages=clause.content_line_pages + moved_pages,
            page_end=new_end,
        )
        clauses_by_id[next_id] = replace(
            next_clause,
            content_lines=kept_next_lines,
            content_line_pages=kept_next_pages,
            page_start=kept_next_pages[0] if kept_next_pages else next_clause.page_end,
            boundary_source=BoundarySource.VISION_ESCALATED,
        )
    return clauses_by_id, True, ""


def _apply_boundary_review(
    clauses_by_id: dict[str, Clause],
    order: list[str],
    clause_id: str,
    review: BoundaryReview,
    *,
    rasterized_first_page: int,
    rasterized_last_page: int,
) -> tuple[dict[str, Clause], bool, str]:
    """Apply one clause's boundary review. Returns (updated map, applied, note)."""
    clauses_by_id = dict(clauses_by_id)
    clause = replace(
        clauses_by_id[clause_id], boundary_source=BoundarySource.VISION_ESCALATED
    )
    clauses_by_id[clause_id] = clause

    if review.split_suggested:
        return (
            clauses_by_id,
            False,
            "sub-clause split suggested; not auto-applied (see split_notes)",
        )
    if review.confirmed:
        return clauses_by_id, False, "confirmed, no change"

    new_start = review.corrected_page_start
    new_end = review.corrected_page_end
    if new_start is None or new_end is None:
        return clauses_by_id, False, "no corrected page range provided"
    if new_start > new_end:
        return (
            clauses_by_id,
            False,
            "corrected page range is invalid (start > end); not applied",
        )
    if not (rasterized_first_page <= new_start <= rasterized_last_page) or not (
        rasterized_first_page <= new_end <= rasterized_last_page
    ):
        return (
            clauses_by_id,
            False,
            "corrected page range falls outside the rasterized margin; not applied",
        )
    if new_start == clause.page_start and new_end == clause.page_end:
        return (
            clauses_by_id,
            False,
            "corrected range matches current boundary; no change",
        )

    index = order.index(clause_id)
    applied_any = False
    notes: list[str] = []
    # Snapshot taken after the boundary_source stamp but before any edge is
    # applied, so the identity-retention guard below can revert the content
    # changes without discarding the record that this clause was reviewed.
    stamped_state = dict(clauses_by_id)
    original_lines = set(clause.content_lines)

    if new_start != clause.page_start:
        clauses_by_id, ok, note = _apply_start_edge(
            clauses_by_id, order, index, new_start
        )
        applied_any = applied_any or ok
        if note:
            notes.append(note)

    clause = clauses_by_id[clause_id]
    if new_end != clause.page_end:
        clauses_by_id, ok, note = _apply_end_edge(clauses_by_id, order, index, new_end)
        applied_any = applied_any or ok
        if note:
            notes.append(note)

    # A boundary *refinement* must leave the clause substantially itself. If
    # applying both edges moved every one of its own lines elsewhere, the
    # model is not refining a boundary -- it is asserting the clause lives
    # somewhere else entirely, which line reassignment cannot express safely.
    # [M1-08c] measured one such case replacing a clause's 36 correct lines
    # with 32 lines belonging to its neighbour.
    if applied_any and original_lines:
        retained = set(clauses_by_id[clause_id].content_lines) & original_lines
        if not retained:
            return (
                stamped_state,
                False,
                "correction would move the clause's entire content elsewhere; "
                "not applied",
            )

    if not notes:
        return clauses_by_id, True, "corrected page range applied"
    return clauses_by_id, applied_any, "; ".join(notes)


def escalate_boundaries(
    tree: ClauseTree,
    pdf_path: Path,
    *,
    reviewer: BoundaryVisionReviewerPort,
    rasterizer: PageRasterizerPort,
    page_count: int,
    max_page_span: int,
    max_char_count: int,
    page_margin: int = BOUNDARY_ESCALATION_PAGE_MARGIN,
    max_attempts: int = BOUNDARY_ESCALATION_MAX_ATTEMPTS,
    retry_delay_seconds: float = BOUNDARY_ESCALATION_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ClauseTree, tuple[BoundaryEscalationOutcome, ...]]:
    """Run the vision-escalation pass over every suspicious clause in ``tree``.

    Returns the (possibly boundary-corrected) tree plus one outcome per
    flagged clause, so escalated and non-escalated clauses stay measurable
    separately downstream (mirroring [domain.clause_classification.
    TypeSource]'s rule/llm distinction).
    """
    flags = find_suspicious_clauses(
        tree, max_page_span=max_page_span, max_char_count=max_char_count
    )
    if not flags:
        return tree, ()

    clauses_by_id: dict[str, Clause] = {
        clause.clause_id: clause for clause in tree.all_clauses
    }
    order = [clause.clause_id for clause in tree.all_clauses]
    outcomes: list[BoundaryEscalationOutcome] = []

    for flag in flags:
        clause = clauses_by_id[flag.clause_id]
        first_page, last_page = _resolve_rasterize_range(
            clause.page_start, clause.page_end, page_margin, page_count
        )
        page_images = rasterizer.rasterize(
            pdf_path, tree.document_id, first_page, last_page
        )
        review = _review_with_retry(
            reviewer,
            clause_title=clause.title,
            claimed_page_start=clause.page_start,
            claimed_page_end=clause.page_end,
            page_images=page_images,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleep=sleep,
        )
        clauses_by_id, applied, note = _apply_boundary_review(
            clauses_by_id,
            order,
            flag.clause_id,
            review,
            rasterized_first_page=first_page,
            rasterized_last_page=last_page,
        )
        outcomes.append(
            BoundaryEscalationOutcome(
                clause_id=flag.clause_id,
                reasons=flag.reasons,
                review=review,
                applied=applied,
                note=note,
            )
        )

    revised_clauses = tuple(clauses_by_id[cid] for cid in order)
    revised_roots = tuple(
        clause for clause in revised_clauses if clause.parent_id is None
    )
    revised_tree = replace(tree, roots=revised_roots, all_clauses=revised_clauses)
    return revised_tree, tuple(outcomes)
