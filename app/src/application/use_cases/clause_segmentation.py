"""Pure clause-tree recovery logic for the post-boilerplate segmentation pass.

Recovers a parent-referenced hierarchy of clauses/sub-clauses from an
already-cleaned [domain.extracted_text.ExtractedDocument] ([M1-03]'s
output), respecting the document's own heading structure instead of fixed
token windows that cut a clause in half.

Heading-detection signals were calibrated against real spans from the
corpus (docs 1, 9, 10, 12, 14, 15, 21, 28), not assumed generically -- see
``.ai_context/github_labels_milestones_and_issues.md`` [M1-04] for the DoD
and the plan file this module implements for the full evidence.

Two deliberate deviations from a literal "numbering + font size + position"
reading, both evidence-backed:

- **Bold-character fraction of a line, not absolute or relative font size,
  is the primary secondary gate for numbered headings.** Font size is flat
  between a heading and the very next numbered body paragraph in most of
  the sampled templates (e.g. doc 1 Porto Seguro: both 13.0pt; doc 15
  Mapfre: both 12.0pt; doc 12 HDI: the *body* line is even larger than its
  own heading, 12.0 vs 11.9). Only one sampled document (doc 10, Bradesco)
  has a real font-size ladder by depth. Bold fraction, by contrast, cleanly
  separates a heading (~100% bold) from a numbered body-paragraph opening
  line (only its numeric prefix is bold, diluting to under ~10%) across
  three unrelated font families (Helvetica, Arial/ArialNarrow,
  TimesNewRoman). It is also what defeats doc 12's ~18 false-positive
  "CLAUSULA 2" cross-references, which match the numbering regex but sit at
  body font size, unbolded, inside a table.
- **The numbering-pattern regex remains a hard, non-optional gate** for the
  decimal/CLÁUSULA conventions specifically because doc 10 has a block of
  body prose that is entirely bold by legacy template choice; without the
  numbering-pattern gate, bold fraction alone would misfire there. Its
  lettered items (``a)``, ``b)``...) don't match the numbering regex, so
  they're correctly excluded before bold fraction is even considered.

Font size is not discarded -- it is demoted from a required gate to
provenance carried on each detected line, available for future tuning, but
not load-bearing for the current heuristic: numbering-pattern depth is the
authority for tree structure, per the DoD's own framing ("respecting the
document's own structure").

``CLAUSE_SEGMENTATION_VERSION`` feeds the downstream cache key (see
[infrastructure.parsing.clause_tree_caching]): bump it whenever any
detection rule here changes, so stale cached output is invalidated.
"""

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace

from domain.clause_tree import (
    Clause,
    ClauseTree,
    ClauseTreeReport,
    ClauseTreeWarning,
    HeadingConvention,
)
from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan

CLAUSE_SEGMENTATION_VERSION = "v1"

NUMBERED_BOLD_THRESHOLD = 0.5
PART_BOLD_THRESHOLD = 0.9
MIN_PART_ALPHA_CHARS = 4
# Real short part titles (GLOSSÁRIO, DISPOSIÇÕES PRELIMINARES, the wrapped
# roman-numeral header in doc 12) run well under 10 words per line. Bold
# ALL-CAPS text longer than that is, in the corpus, reliably a body sentence
# highlighted per the legal convention of capitalizing limiting/exclusionary
# clauses (confirmed in doc 15/Mapfre and doc 10/Bradesco), not a heading --
# without this cap those sentences flood the tree with spurious part nodes
# that reset the stack mid-document.
MAX_PART_WORD_COUNT = 10

# Multi-column detection: thresholds calibrated against the corpus (see
# module docstring) to isolate only genuine sustained parallel-column prose
# (doc 28) and reject tables and justified-text artifacts (doc 1 p.28,
# doc 10 p.20, doc 21).
LEFT_BAND_MAX_RATIO = 0.35
RIGHT_BAND_MIN_RATIO = 0.42
RIGHT_BAND_MAX_RATIO = 0.60
MIN_COLUMN_LINES = 12
MIN_COLUMN_Y_OVERLAP_RATIO = 0.6

# Some documents embed subset CID fonts (e.g. "CIDFont+F2") that carry no
# readable weight in their name -- confirmed real case: doc 4 (Sura) uses a
# dedicated font purely for headings and another for body, with the exact
# same all-heading/prefix-only-in-body pattern as literal-bold documents,
# but zero characters anywhere named "bold". Working documents in the
# corpus sit at 10-39% overall bold-named-character share; this failure
# mode sits at 0.00-0.13%, so this threshold cleanly separates "the literal
# 'bold' signal is trustworthy for this document" from "it is not".
HAS_BOLD_FONT_CHAR_RATIO_THRESHOLD = 0.02

# A depth-1 token (bare integer) requires its trailing dot -- "24 Horas" and
# "7 dias" are real table-cell content in the corpus (a service-tier grid),
# not headings, and only the dot distinguishes them from "1. OBJETO...".
# A depth-2+ token already carries an internal dot, so the trailing one
# stays optional -- matches real headings like "2.1 Coberturas Básicas"
# (no trailing dot) and "10.10     Pagamento em atraso" (extra spacing).
_NUMBERED_DECIMAL_DEEP = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")
_NUMBERED_DECIMAL_TOP = re.compile(r"^(\d+)\.\s+\S")
_CLAUSULA_KEYWORD = re.compile(r"^CL[ÁA]USULA\s+(\d+)[ªa°]?", re.IGNORECASE)
_ROMAN_PART_PREFIX = re.compile(r"^[IVXLCDM]+\s*[-–.]\s*")
# Case-insensitive and up to two letters: doc 15's "riscos excluídos" list
# runs past "z)" into "aa)", "cc)", "gg)" (confirmed real corpus usage,
# rendered in the source PDF as uppercase -- "G)", "AA)", "GG)"). Without
# the uppercase/double-letter allowance, each of those fragments falls
# through to heading detection and, being bold and short, is wrongly
# treated as a new unnumbered part, fragmenting one exclusion list into
# dozens of spurious top-level nodes.
_LETTERED_ITEM = re.compile(r"^[A-Za-z]{1,2}\)\s+\S")
_BULLET_ITEM = re.compile(r"^[•\-–]\s+\S")


def is_list_item_line(text: str) -> bool:
    """True for a lettered (``a)``/``AA)``) or bulleted (``•``/``-``/``–``) item.

    These are content of the enclosing clause, per the DoD, never their own
    tree node -- checked before heading detection so they never reach it.
    """
    return bool(_LETTERED_ITEM.match(text) or _BULLET_ITEM.match(text))


@dataclass(frozen=True)
class _Line:
    """One logical line, position/format-enriched for heading detection."""

    page_number: int
    text: str
    x0: float
    y0: float
    font_size: float
    bold_fraction: float
    is_ocr: bool


@dataclass(frozen=True)
class _HeadingMatch:
    """A line that passed a heading convention's gates."""

    convention: HeadingConvention
    numbering_label: str
    depth: int
    title: str


def _bold_fraction(
    spans: list[ExtractedSpan], is_heavy: Callable[[str], bool]
) -> float:
    """Fraction of a line's characters set in a "heavy" (heading-weight) font."""
    total = sum(len(span.text) for span in spans)
    if total == 0:
        return 0.0
    heavy = sum(len(span.text) for span in spans if is_heavy(span.font_name))
    return heavy / total


def _document_bold_ratio(document: ExtractedDocument) -> float:
    """Character-weighted fraction of the whole document in a "bold"-named font."""
    bold_chars = 0
    total_chars = 0
    for page in document.pages:
        for span in page.spans:
            total_chars += len(span.text)
            if "bold" in span.font_name.lower():
                bold_chars += len(span.text)
    return bold_chars / total_chars if total_chars else 0.0


def _document_modal_font_name(document: ExtractedDocument) -> str:
    """Character-weighted modal font name -- the document's dominant body font."""
    weights: dict[str, int] = {}
    for page in document.pages:
        for span in page.spans:
            weights[span.font_name] = weights.get(span.font_name, 0) + len(span.text)
    if not weights:
        return ""
    return max(weights, key=lambda name: weights[name])


def _heavy_font_predicate(document: ExtractedDocument) -> Callable[[str], bool]:
    """Build the per-document "is this font_name heading/emphasis weight" test.

    Ordinarily a font whose name literally contains "bold". When the whole
    document's bold-named-character ratio is below
    [HAS_BOLD_FONT_CHAR_RATIO_THRESHOLD] (see that constant's docstring for
    the corpus evidence), that literal signal is untrustworthy for this
    document, so fall back to "is not the document's dominant font" as the
    heavy-weight signal instead.
    """
    if _document_bold_ratio(document) >= HAS_BOLD_FONT_CHAR_RATIO_THRESHOLD:
        return lambda font_name: "bold" in font_name.lower()
    modal_font_name = _document_modal_font_name(document)
    return lambda font_name: font_name != modal_font_name and font_name != ""


def _is_ocr_page(page: ExtractedPage) -> bool:
    """True for a page produced by [M1-02]'s OCR path (no font/position signal).

    The OCR extractor always collapses a page to exactly one span with
    ``font_size=0.0`` -- a self-describing artifact of that contract, so
    this needs no external manifest lookup to detect.
    """
    return len(page.spans) == 1 and page.spans[0].font_size == 0.0


def _ocr_pseudo_lines(page: ExtractedPage) -> list[_Line]:
    """Recover pseudo-lines from an OCR page's newline-delimited text.

    No font/position signal is available (the whole page is one span), so
    ``bold_fraction`` is forced to 0.0 and ``x0``/``font_size`` are inert
    placeholders -- heading detection for these lines runs in
    pattern-only mode (see [_detect_heading]).
    """
    text = page.spans[0].text
    lines: list[_Line] = []
    for index, raw_line in enumerate(text.split("\n")):
        stripped = raw_line.strip()
        if not stripped:
            continue
        lines.append(
            _Line(
                page_number=page.page_number,
                text=stripped,
                x0=0.0,
                y0=float(index),
                font_size=0.0,
                bold_fraction=0.0,
                is_ocr=True,
            )
        )
    return lines


def _text_page_lines(
    page: ExtractedPage, is_heavy: Callable[[str], bool]
) -> list[_Line]:
    """Group a text-extraction page's spans into position-aware logical lines."""
    grouped: dict[int, list[ExtractedSpan]] = {}
    for span in page.spans:
        grouped.setdefault(span.line_id, []).append(span)

    lines: list[_Line] = []
    for spans in grouped.values():
        spans_sorted = sorted(spans, key=lambda span: span.order)
        text = "".join(span.text for span in spans_sorted).strip()
        if not text:
            continue
        font_size = next(
            (span.font_size for span in spans_sorted if span.text.strip()),
            spans_sorted[0].font_size,
        )
        lines.append(
            _Line(
                page_number=page.page_number,
                text=text,
                x0=min(span.bbox[0] for span in spans_sorted),
                y0=min(span.bbox[1] for span in spans_sorted),
                font_size=font_size,
                bold_fraction=_bold_fraction(spans_sorted, is_heavy),
                is_ocr=False,
            )
        )
    return lines


def _detect_and_reflow_columns(
    lines: list[_Line], page_width: float
) -> tuple[list[_Line], bool]:
    """Detect a genuine sustained two-column layout and reflow reading order.

    Requires two x0 bands each holding >= [MIN_COLUMN_LINES] lines whose
    vertical extents overlap by more than [MIN_COLUMN_Y_OVERLAP_RATIO] of
    their combined span -- i.e. two columns of prose running concurrently
    down the page, not a table (short-lived) or a single justified column
    (whose line-final word can land far right, but shares its line's real
    left-aligned x0 once spans are grouped by line, not by span).
    """
    if page_width <= 0:
        return lines, False

    left = [line for line in lines if line.x0 < LEFT_BAND_MAX_RATIO * page_width]
    right = [
        line
        for line in lines
        if RIGHT_BAND_MIN_RATIO * page_width
        <= line.x0
        <= RIGHT_BAND_MAX_RATIO * page_width
    ]
    if len(left) < MIN_COLUMN_LINES or len(right) < MIN_COLUMN_LINES:
        return lines, False

    left_y0, left_y1 = min(line.y0 for line in left), max(line.y0 for line in left)
    right_y0, right_y1 = min(line.y0 for line in right), max(line.y0 for line in right)
    overlap = min(left_y1, right_y1) - max(left_y0, right_y0)
    span = max(left_y1, right_y1) - min(left_y0, right_y0)
    if span <= 0 or overlap / span <= MIN_COLUMN_Y_OVERLAP_RATIO:
        return lines, False

    left_median_x0 = statistics.median(line.x0 for line in left)
    right_median_x0 = statistics.median(line.x0 for line in right)
    banded_ids = {id(line) for line in left} | {id(line) for line in right}
    for line in lines:
        if id(line) in banded_ids:
            continue
        if abs(line.x0 - left_median_x0) <= abs(line.x0 - right_median_x0):
            left.append(line)
        else:
            right.append(line)

    reflowed = sorted(left, key=lambda line: line.y0) + sorted(
        right, key=lambda line: line.y0
    )
    return reflowed, True


def _ordered_page_lines(
    page: ExtractedPage, is_heavy: Callable[[str], bool]
) -> tuple[list[_Line], bool]:
    """Return one page's lines in corrected reading order, and whether reflowed."""
    if _is_ocr_page(page):
        return _ocr_pseudo_lines(page), False

    lines = _text_page_lines(page, is_heavy)
    if not lines:
        return lines, False
    page_width = max((span.bbox[2] for span in page.spans), default=0.0)
    return _detect_and_reflow_columns(lines, page_width)


def _detect_heading(
    text: str, bold_fraction: float, *, ocr: bool
) -> _HeadingMatch | None:
    """Test one line against the numbered, CLÁUSULA and unnumbered-part gates.

    ``ocr`` drops the bold-fraction gates entirely (no font signal exists
    on OCR pseudo-lines), falling back to pattern-only matching -- a
    deliberately lower-confidence path, not a workaround to reach parity.
    """
    numbered_threshold = 0.0 if ocr else NUMBERED_BOLD_THRESHOLD
    part_threshold = 0.0 if ocr else PART_BOLD_THRESHOLD

    clausula_match = _CLAUSULA_KEYWORD.match(text)
    if clausula_match:
        if bold_fraction < numbered_threshold:
            return None
        return _HeadingMatch(
            convention=HeadingConvention.CLAUSULA_KEYWORD,
            numbering_label=clausula_match.group(1),
            depth=1,
            title=text,
        )

    decimal_match = _NUMBERED_DECIMAL_DEEP.match(text) or _NUMBERED_DECIMAL_TOP.match(
        text
    )
    if decimal_match:
        if bold_fraction < numbered_threshold:
            return None
        token = decimal_match.group(1)
        return _HeadingMatch(
            convention=HeadingConvention.NUMBERED_DECIMAL,
            numbering_label=token,
            depth=token.count(".") + 1,
            title=text,
        )

    if bold_fraction < part_threshold:
        return None
    candidate = _ROMAN_PART_PREFIX.sub("", text, count=1)
    alpha_chars = [char for char in candidate if char.isalpha()]
    if len(alpha_chars) < MIN_PART_ALPHA_CHARS:
        return None
    if len(candidate.split()) > MAX_PART_WORD_COUNT:
        return None
    if not all(char.isupper() for char in alpha_chars):
        return None
    return _HeadingMatch(
        convention=HeadingConvention.UNNUMBERED_PART,
        numbering_label="",
        depth=0,
        title=text,
    )


class _ClauseBuilder:
    """Mutable accumulator for one clause while the document is being walked."""

    def __init__(
        self,
        *,
        document_id: str,
        clause_id: str,
        numbering_label: str,
        title: str,
        convention: HeadingConvention,
        depth: int,
        parent_id: str | None,
        is_depth_anomaly: bool,
    ) -> None:
        self.document_id = document_id
        self.clause_id = clause_id
        self.numbering_label = numbering_label
        self.title = title
        self.convention = convention
        self.depth = depth
        self.parent_id = parent_id
        self.is_depth_anomaly = is_depth_anomaly
        self.child_ids: list[str] = []
        self.content_lines: list[str] = []
        self.page_start: int | None = None
        self.page_end: int | None = None
        self.bundle_section: str | None = None
        self.bundle_confidence: str | None = None

    def touch_page(self, page_number: int) -> None:
        if self.page_start is None:
            self.page_start = page_number
        self.page_end = page_number

    def freeze(self) -> Clause:
        return Clause(
            document_id=self.document_id,
            clause_id=self.clause_id,
            numbering_label=self.numbering_label,
            title=self.title,
            convention=self.convention,
            depth=self.depth,
            parent_id=self.parent_id,
            child_ids=tuple(self.child_ids),
            content_lines=tuple(self.content_lines),
            page_start=self.page_start if self.page_start is not None else 0,
            page_end=self.page_end if self.page_end is not None else 0,
            bundle_section=self.bundle_section,
            bundle_confidence=self.bundle_confidence,
            is_depth_anomaly=self.is_depth_anomaly,
        )


def segment_document(document: ExtractedDocument) -> ClauseTree:
    """Recover the clause tree for one boilerplate-cleaned document.

    Never raises -- always returns a complete [ClauseTree]/[ClauseTreeReport]
    with orphan counts and warnings, so a caller decides what "too broken"
    means (see ``scripts/build_clause_tree.py``'s [domain.clause_tree.
    OrphanTextExceedsThresholdError] check), rather than this function hiding a
    bad document behind an exception.
    """
    is_heavy = _heavy_font_predicate(document)
    lines: list[_Line] = []
    warnings: list[ClauseTreeWarning] = []
    for page in document.pages:
        page_lines, reflowed = _ordered_page_lines(page, is_heavy)
        if reflowed:
            warnings.append(
                ClauseTreeWarning(
                    document_id=document.document_id,
                    page_number=page.page_number,
                    kind="multi_column_reflow",
                    detail=(
                        "Two parallel text columns detected; reordered left "
                        "column top-to-bottom, then right column."
                    ),
                )
            )
        lines.extend(page_lines)

    has_ocr = any(line.is_ocr for line in lines)
    if has_ocr:
        warnings.append(
            ClauseTreeWarning(
                document_id=document.document_id,
                page_number=0,
                kind="ocr_relaxed_mode",
                detail=(
                    "OCR-extracted pages carry no font/position signal; "
                    "heading detection used pattern-only matching."
                ),
            )
        )

    builders: dict[str, _ClauseBuilder] = {}
    order: list[str] = []
    stack: list[_ClauseBuilder] = []
    orphan_char_count = 0
    total_char_count = 0
    clause_counter = 0

    def next_id() -> str:
        nonlocal clause_counter
        clause_counter += 1
        return f"{document.document_id}:{clause_counter}"

    def open_clause(match: _HeadingMatch, page_number: int, depth: int) -> None:
        if match.convention == HeadingConvention.UNNUMBERED_PART:
            stack.clear()
        else:
            while stack and stack[-1].depth >= depth:
                stack.pop()
        parent_id = stack[-1].clause_id if stack else None
        clause_id = next_id()
        builder = _ClauseBuilder(
            document_id=document.document_id,
            clause_id=clause_id,
            numbering_label=match.numbering_label,
            title=match.title,
            convention=match.convention,
            depth=depth,
            parent_id=parent_id,
            is_depth_anomaly=False,
        )
        builder.touch_page(page_number)
        builders[clause_id] = builder
        order.append(clause_id)
        if parent_id is not None:
            builders[parent_id].child_ids.append(clause_id)
        stack.append(builder)

    def attach_content(text: str, page_number: int) -> None:
        nonlocal orphan_char_count
        if stack:
            stack[-1].content_lines.append(text)
            stack[-1].touch_page(page_number)
        else:
            orphan_char_count += len(text)

    def consume_title_wrap(
        heading: _HeadingMatch, start_index: int, page_number: int
    ) -> tuple[_HeadingMatch, int]:
        """Absorb trailing bold/all-caps continuation lines into a heading's title.

        A heading's title can itself wrap across multiple lines (any
        convention, not just [HeadingConvention.UNNUMBERED_PART]) -- e.g.
        doc 28's "7. RISCOS EXCLUÍDOS/ BENS E INTERESSES" continuing as
        "NÃO GARANTIDOS" on the next line. Without this, a continuation
        line would independently pass the unnumbered-part gate and open a
        spurious top-level node, resetting the stack mid-clause.
        """
        nonlocal total_char_count
        title_parts = [heading.title]
        lookahead = start_index
        while lookahead < line_count:
            candidate_line = lines[lookahead]
            if candidate_line.page_number != page_number:
                break
            candidate = _detect_heading(
                candidate_line.text,
                candidate_line.bold_fraction,
                ocr=candidate_line.is_ocr,
            )
            if (
                candidate is None
                or candidate.convention != HeadingConvention.UNNUMBERED_PART
            ):
                break
            title_parts.append(candidate.title)
            total_char_count += len(candidate_line.text)
            lookahead += 1
        if len(title_parts) == 1:
            return heading, lookahead
        return replace(heading, title=" ".join(title_parts)), lookahead

    index = 0
    line_count = len(lines)
    while index < line_count:
        line = lines[index]
        total_char_count += len(line.text)

        if is_list_item_line(line.text):
            attach_content(line.text, line.page_number)
            index += 1
            continue

        heading = _detect_heading(line.text, line.bold_fraction, ocr=line.is_ocr)
        if heading is None:
            attach_content(line.text, line.page_number)
            index += 1
            continue

        if heading.convention == HeadingConvention.UNNUMBERED_PART:
            merged, next_index = consume_title_wrap(
                heading, index + 1, line.page_number
            )
            open_clause(merged, line.page_number, depth=0)
            index = next_index
            continue

        depth = heading.depth
        expected_max_depth = stack[-1].depth + 1 if stack else 1
        is_anomaly = depth > expected_max_depth
        if is_anomaly:
            depth = expected_max_depth
        merged, next_index = consume_title_wrap(heading, index + 1, line.page_number)
        open_clause(merged, line.page_number, depth=depth)
        if is_anomaly:
            builders[order[-1]].is_depth_anomaly = True
            warnings.append(
                ClauseTreeWarning(
                    document_id=document.document_id,
                    page_number=line.page_number,
                    kind="depth_anomaly",
                    detail=(
                        f'"{heading.numbering_label}" attached at depth {depth} '
                        "(an intermediate level was expected but missing)."
                    ),
                )
            )
        index = next_index

    def is_numbering_start(label: str) -> bool:
        clean = re.sub(r"[^0-9]", "", label)
        return clean == "1"

    restarting_roots = []
    root_ids = [cid for cid in order if builders[cid].parent_id is None]

    for root_id in root_ids:
        root_builder = builders[root_id]
        if root_builder.depth == 0:
            has_start = any(
                builders[child_id].depth == 1
                and is_numbering_start(builders[child_id].numbering_label)
                for child_id in root_builder.child_ids
            )
            if has_start:
                restarting_roots.append(root_id)

    is_bundle = len(restarting_roots) > 1

    if is_bundle:
        for root_id in root_ids:
            root_builder = builders[root_id]
            if root_id in restarting_roots:
                section_name = root_builder.title if root_builder.depth == 0 else None
                confidence = "high"
            else:
                section_name = None
                confidence = "low"

            def propagate(
                cid: str, current_section: str | None, current_confidence: str | None
            ) -> None:
                builders[cid].bundle_section = current_section
                builders[cid].bundle_confidence = current_confidence
                for child_id in builders[cid].child_ids:
                    propagate(child_id, current_section, current_confidence)

            propagate(root_id, section_name, confidence)

    clauses = tuple(builders[clause_id].freeze() for clause_id in order)
    roots = tuple(clause for clause in clauses if clause.parent_id is None)
    orphan_ratio = orphan_char_count / total_char_count if total_char_count else 0.0

    report = ClauseTreeReport(
        document_id=document.document_id,
        filename=document.filename,
        clause_count=len(clauses),
        max_depth=max((clause.depth for clause in clauses), default=0),
        orphan_char_count=orphan_char_count,
        total_char_count=total_char_count,
        orphan_ratio=orphan_ratio,
        extraction_mode="ocr_required" if has_ocr else "text",
        warnings=tuple(warnings),
    )
    return ClauseTree(
        document_id=document.document_id,
        filename=document.filename,
        roots=roots,
        all_clauses=clauses,
        report=report,
    )
