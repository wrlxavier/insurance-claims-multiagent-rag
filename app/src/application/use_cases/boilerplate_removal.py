"""Pure boilerplate-detection logic for the pre-segmentation cleanup pass.

Removes table-of-contents lines, repeated headers/footers and front-matter
marketing/cover pages from an [domain.extracted_text.ExtractedDocument]
before it reaches clause segmentation ([M1-04]) -- this text would otherwise
sit in the embedding index as high-similarity noise. The glossary must never
be touched: it is a legitimate retrieval target, not boilerplate.

Thresholds below were calibrated against real extracted spans from the three
longest documents in the corpus (207, 146 and 122 pages), not assumed
generically -- see ``.ai_context/github_labels_milestones_and_issues.md``
[M1-03] and ``docs/BOILERPLATE_REMOVAL_REPORT.md`` for the evidence.

Processing order is load-bearing, not just tidiness. Headers/footers are
stripped first: a bare footer page-number line (e.g. ``"12"``) would
otherwise inflate every page's TOC entry-ratio, and marketing-page detection
compares a page's character count against the document median -- a
constant-size header is a much larger fraction of a genuinely sparse cover
page's count than of an ordinary page's, so it must already be gone before
that comparison runs. Marketing pages are then removed *before* TOC pages,
not after: a cover page can legitimately contain a few short trailing
numbers of its own (a CNPJ, a process number, a version date), which can
push its TOC entry-ratio over threshold by coincidence -- confirmed against
the real corpus, where a Youse cover page's ratio (3 of 7 lines) landed at
0.43, just over the 0.4 cutoff. Running marketing detection first lets the
sparse-page-with-oversized-font signal claim that page correctly, before
the coincidental digit-ending lines can mislabel it as a TOC page instead.

``BOILERPLATE_REMOVAL_VERSION`` feeds the downstream cache key (see
[infrastructure.parsing.boilerplate_caching]): bump it whenever any
detection rule here changes, so stale cached output is invalidated.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace

from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan

BOILERPLATE_REMOVAL_VERSION = "v1"

HEADER_FOOTER_FREQUENCY_THRESHOLD = 0.9
HEADER_FOOTER_Y0_ROUNDING = 8.0
TOC_ENTRY_RATIO_THRESHOLD = 0.4
MARKETING_CHAR_COUNT_RATIO_THRESHOLD = 0.4
MARKETING_FONT_SIZE_RATIO_THRESHOLD = 1.5

_DIGIT_RUN = re.compile(r"\d+")
_DOT_LEADER_TOC_LINE = re.compile(r"\.{4,}\s*\d{1,4}\s*$")
_ENDS_IN_SHORT_NUMBER = re.compile(r"\d{1,4}\s*$")

_LogicalLine = tuple[int, str, float]


def normalize_for_grouping(text: str) -> str:
    """Collapse digit runs to a placeholder, so page numbers don't split a header."""
    return _DIGIT_RUN.sub("#", text.strip())


def is_dot_leader_toc_line(text: str) -> bool:
    """True for a classic dot-leader TOC entry, e.g. ``'GLOSSÁRIO ..... 3'``.

    Real TOC pages also contain entries with no dots at all (plain
    whitespace alignment); those are caught by [find_toc_pages] instead,
    since a per-line regex cannot distinguish them from ordinary prose.
    """
    return bool(_DOT_LEADER_TOC_LINE.search(text))


def front_matter_window(page_count: int) -> int:
    """How many leading pages count as "front matter" for TOC/marketing checks.

    ``max(3, min(10, round(0.05 * page_count)))``, clamped to ``page_count``
    itself so a document shorter than the floor never gets a window bigger
    than the document.
    """
    window = max(3, min(10, round(0.05 * page_count)))
    return min(window, page_count)


def _logical_lines(page: ExtractedPage) -> list[_LogicalLine]:
    """Group a page's spans into (line_id, joined text, top y0) triples."""
    lines: dict[int, list[ExtractedSpan]] = {}
    for span in page.spans:
        lines.setdefault(span.line_id, []).append(span)
    result: list[_LogicalLine] = []
    for line_id, spans in lines.items():
        spans_sorted = sorted(spans, key=lambda span: span.order)
        text = "".join(span.text for span in spans_sorted).strip()
        y0 = min(span.bbox[1] for span in spans_sorted)
        result.append((line_id, text, y0))
    return result


def _char_count(spans: tuple[ExtractedSpan, ...]) -> int:
    return sum(len(span.text) for span in spans)


def toc_entry_ratio(page: ExtractedPage) -> float:
    """Fraction of a page's non-blank lines that end in a short number.

    A high ratio means the page reads like a table of contents (each line
    is "title ... page number"), whether or not it uses dot leaders.
    """
    texts = [text for _, text, _y0 in _logical_lines(page) if text]
    if not texts:
        return 0.0
    hits = sum(1 for text in texts if _ENDS_IN_SHORT_NUMBER.search(text))
    return hits / len(texts)


def find_header_footer_keys(
    document: ExtractedDocument,
    *,
    frequency_threshold: float = HEADER_FOOTER_FREQUENCY_THRESHOLD,
    y0_rounding: float = HEADER_FOOTER_Y0_ROUNDING,
) -> frozenset[tuple[str, float]]:
    """Lines repeated at the same position across most of the document.

    Keyed by (digit-normalized text, y0 rounded to ``y0_rounding``) so a
    header carrying a page number is still recognized as one recurring
    line, and so an unrelated line that happens to share text at a
    different vertical position isn't conflated with it.
    """
    page_count = len(document.pages)
    if page_count == 0:
        return frozenset()

    counts: Counter[tuple[str, float]] = Counter()
    for page in document.pages:
        keys_on_page: set[tuple[str, float]] = set()
        for _line_id, text, y0 in _logical_lines(page):
            if not text:
                continue
            rounded_y0 = round(y0 / y0_rounding) * y0_rounding
            keys_on_page.add((normalize_for_grouping(text), rounded_y0))
        counts.update(keys_on_page)

    threshold_count = frequency_threshold * page_count
    return frozenset(key for key, count in counts.items() if count >= threshold_count)


def remove_headers_and_footers(
    document: ExtractedDocument,
    *,
    frequency_threshold: float = HEADER_FOOTER_FREQUENCY_THRESHOLD,
    y0_rounding: float = HEADER_FOOTER_Y0_ROUNDING,
) -> tuple[ExtractedDocument, int]:
    """Drop lines matching [find_header_footer_keys]. Returns lines removed."""
    keys = find_header_footer_keys(
        document, frequency_threshold=frequency_threshold, y0_rounding=y0_rounding
    )
    if not keys:
        return document, 0

    lines_removed = 0
    new_pages = []
    for page in document.pages:
        drop_line_ids: set[int] = set()
        for line_id, text, y0 in _logical_lines(page):
            if not text:
                continue
            rounded_y0 = round(y0 / y0_rounding) * y0_rounding
            if (normalize_for_grouping(text), rounded_y0) in keys:
                drop_line_ids.add(line_id)
        lines_removed += len(drop_line_ids)
        kept_spans = tuple(
            span for span in page.spans if span.line_id not in drop_line_ids
        )
        new_pages.append(
            ExtractedPage(
                page_number=page.page_number,
                spans=kept_spans,
                char_count=_char_count(kept_spans),
            )
        )
    return replace(document, pages=tuple(new_pages)), lines_removed


def find_toc_pages(
    document: ExtractedDocument,
    *,
    entry_ratio_threshold: float = TOC_ENTRY_RATIO_THRESHOLD,
) -> frozenset[int]:
    """Front-matter pages whose [toc_entry_ratio] clears the threshold."""
    window = front_matter_window(len(document.pages))
    result = set()
    for page in document.pages:
        if page.page_number > window:
            break
        if toc_entry_ratio(page) >= entry_ratio_threshold:
            result.add(page.page_number)
    return frozenset(result)


def remove_toc(
    document: ExtractedDocument,
    *,
    entry_ratio_threshold: float = TOC_ENTRY_RATIO_THRESHOLD,
) -> tuple[ExtractedDocument, int, frozenset[int], int]:
    """Drop dot-leader lines everywhere and empty out detected TOC pages.

    Returns (new_document, dot_leader_lines_removed, toc_pages,
    toc_page_lines_removed). A detected TOC page is dropped in full -- it
    has no other content worth keeping -- so its dot-leader lines are
    counted once, under ``toc_page_lines_removed``, not double-counted
    against ``dot_leader_lines_removed``.
    """
    toc_pages = find_toc_pages(document, entry_ratio_threshold=entry_ratio_threshold)

    dot_leader_lines_removed = 0
    toc_page_lines_removed = 0
    new_pages = []
    for page in document.pages:
        if page.page_number in toc_pages:
            toc_page_lines_removed += len({span.line_id for span in page.spans})
            new_pages.append(
                ExtractedPage(page_number=page.page_number, spans=(), char_count=0)
            )
            continue

        drop_line_ids = {
            line_id
            for line_id, text, _y0 in _logical_lines(page)
            if is_dot_leader_toc_line(text)
        }
        dot_leader_lines_removed += len(drop_line_ids)
        kept_spans = tuple(
            span for span in page.spans if span.line_id not in drop_line_ids
        )
        new_pages.append(
            ExtractedPage(
                page_number=page.page_number,
                spans=kept_spans,
                char_count=_char_count(kept_spans),
            )
        )

    new_document = replace(document, pages=tuple(new_pages))
    return new_document, dot_leader_lines_removed, toc_pages, toc_page_lines_removed


def _weighted_modal_font_size(document: ExtractedDocument) -> float:
    """Most common font size, weighted by character count.

    A page of many short table-cell spans at a secondary size would skew a
    span-count-weighted mode away from the true body-paragraph font, so
    weight by how much text is actually set in each size instead.
    """
    weights: Counter[float] = Counter()
    for page in document.pages:
        for span in page.spans:
            weights[span.font_size] += len(span.text)
    if not weights:
        return 0.0
    return max(weights, key=lambda font_size: weights[font_size])


def find_marketing_pages(
    document: ExtractedDocument,
    *,
    char_count_ratio_threshold: float = MARKETING_CHAR_COUNT_RATIO_THRESHOLD,
    font_size_ratio_threshold: float = MARKETING_FONT_SIZE_RATIO_THRESHOLD,
) -> frozenset[int]:
    """Front-matter pages that are sparse AND carry a headline-sized font.

    Both conditions are required: a high font ratio alone is not enough --
    a dense front-matter page (e.g. a first page that is itself the start
    of the table of contents) can have a large header font too, but it is
    not a marketing page. Pages already emptied by an earlier stage are
    skipped, both to avoid a division-by-zero on their (nonexistent) max
    font size and to avoid double-counting a page in two report
    categories.
    """
    page_count = len(document.pages)
    if page_count == 0:
        return frozenset()

    window = front_matter_window(page_count)
    median_chars = statistics.median(page.char_count for page in document.pages)
    modal_font = _weighted_modal_font_size(document)
    if median_chars == 0 or modal_font == 0:
        return frozenset()

    result = set()
    for page in document.pages:
        if page.page_number > window:
            break
        if not page.spans:
            continue
        if page.char_count >= char_count_ratio_threshold * median_chars:
            continue
        page_max_font = max(span.font_size for span in page.spans)
        if page_max_font >= font_size_ratio_threshold * modal_font:
            result.add(page.page_number)
    return frozenset(result)


def remove_marketing_pages(
    document: ExtractedDocument,
    *,
    char_count_ratio_threshold: float = MARKETING_CHAR_COUNT_RATIO_THRESHOLD,
    font_size_ratio_threshold: float = MARKETING_FONT_SIZE_RATIO_THRESHOLD,
) -> tuple[ExtractedDocument, frozenset[int], int]:
    """Drop pages detected by [find_marketing_pages]. Returns (doc, pages, lines)."""
    marketing_pages = find_marketing_pages(
        document,
        char_count_ratio_threshold=char_count_ratio_threshold,
        font_size_ratio_threshold=font_size_ratio_threshold,
    )
    if not marketing_pages:
        return document, marketing_pages, 0

    lines_removed = 0
    new_pages = []
    for page in document.pages:
        if page.page_number in marketing_pages:
            lines_removed += len({span.line_id for span in page.spans})
            new_pages.append(
                ExtractedPage(page_number=page.page_number, spans=(), char_count=0)
            )
        else:
            new_pages.append(page)
    return replace(document, pages=tuple(new_pages)), marketing_pages, lines_removed


@dataclass(frozen=True)
class BoilerplateRemovalCounts:
    """Per-document removal counts, broken out by category -- never silent."""

    header_footer_lines_removed: int
    toc_dot_leader_lines_removed: int
    toc_pages_removed: int
    toc_page_lines_removed: int
    marketing_pages_removed: int
    marketing_page_lines_removed: int
    removed_page_numbers: tuple[int, ...]


def remove_boilerplate(
    document: ExtractedDocument,
) -> tuple[ExtractedDocument, BoilerplateRemovalCounts]:
    """Run the full pipeline: headers/footers, then marketing pages, then TOC.

    See the module docstring for why this order is required, not
    arbitrary.
    """
    document, header_footer_lines_removed = remove_headers_and_footers(document)
    document, marketing_pages, marketing_page_lines_removed = remove_marketing_pages(
        document
    )
    document, toc_dot_leader_lines_removed, toc_pages, toc_page_lines_removed = (
        remove_toc(document)
    )

    counts = BoilerplateRemovalCounts(
        header_footer_lines_removed=header_footer_lines_removed,
        toc_dot_leader_lines_removed=toc_dot_leader_lines_removed,
        toc_pages_removed=len(toc_pages),
        toc_page_lines_removed=toc_page_lines_removed,
        marketing_pages_removed=len(marketing_pages),
        marketing_page_lines_removed=marketing_page_lines_removed,
        removed_page_numbers=tuple(sorted(toc_pages | marketing_pages)),
    )
    return document, counts
