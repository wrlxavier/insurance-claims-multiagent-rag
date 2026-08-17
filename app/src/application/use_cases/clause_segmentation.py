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

[M1-04b] adds a third signal on top of the two above, needed because bold
fraction, the numbering-pattern gate, and font size (see below) are all
identical between a genuine UNNUMBERED_PART product title and pure
typographical noise in some documents: doc 10 (Bradesco)'s real product
intro "RESUMO DE COBERTURAS DA ASSISTÊNCIA À MOTOCICLETAS" and its noise
"NÃO ESTÃO COBERTOS:" (repeated ~20 times as a benefits-tier template) sit
at the exact same font (TimesNewRomanPS-BoldMT, 9.0pt), so no font/position
signal available in this corpus discriminates them; TOC text is discarded
entirely by [M1-03]'s boilerplate removal, so "is this in the TOC" isn't
an available signal without new extraction plumbing, out of scope here.
**A genuine part title is introduced exactly once in a document; repeated
list-item/table labels are not** -- so an UNNUMBERED_PART candidate whose
title recurs (per [_recurrence_key], which reuses [_slugify]'s accent/
punctuation folding to catch corpus inconsistencies like doc 10's "NÃO
ESTÃO COBERTOS" vs "NÃO ESTÃO COBERTOS:", or "ENVIO DE TÁXI" vs "ENVIO DE
TAXI") is demoted to plain content instead of opening a new root
([_prescan_recurring_unnumbered_titles]). This must be all-or-nothing per
title, not per-occurrence: doc 16 (AKAD RCF-V) reuses "CONDIÇÕES ESPECIAIS"
3 times for 3 distinct coverage variants, each correctly followed by its
own "1. Riscos Cobertos" restart, and must be kept; doc 10's "ARGENTINA,
PARAGUAI, URUGUAI E CHILE." recurs 8 times and only 3 of those occurrences
happen to precede an unrelated "1." elsewhere, so a per-occurrence rule
would leave it partially, wrongly promoted. Accepted trade-off: doc 3
(AKAD)'s "ÂMBITO GEOGRÁFICO" recurs twice for two legitimately separate
endorsement clauses that have zero numbered children either way (their
body bold fraction never clears [NUMBERED_BOLD_THRESHOLD], a pre-existing,
unrelated quirk) -- this rule demotes it, misattributing ~300 chars of
prose to the preceding clause. No content is lost and no existing test
covers it; a content-length heuristic doesn't rescue it without also
un-fixing doc 10, where noise items have larger content spans than doc
10's own genuine parts.

**Retroactive note (found while regenerating the corpus via ``make parse``
after [M1-04b] landed, applied 2026-08-17).** Doc 5 (KOVR) surfaced a case
[M1-04b]'s original DoD had flagged as worth investigating but the
recurrence signal above doesn't address: it recovered *zero* clauses
(orphan ratio 1.000), failing ``scripts/build_clause_tree.py``'s threshold
check outright, even though [scripts.audit_text_layer]'s two independent
backends agree the extracted text is fully legible (2,914 chars/page, no
low-character pages). The cause is structural, not textual: doc 5 has no
"bold"-named font (ratio 0.13%, same tier as doc 4) *and*, unlike doc 4,
no dedicated second font either -- 99.4% of its characters share one font
name (``NewJuneRegular``), so tier 2's "not the dominant font" fallback has
nothing to key off (every span reports the same name). Every other corpus
document that reaches tier 2 has a real, substantially-shared second font
(doc 4 78.8%, doc 8 50.8%, doc 18 82.8% modal-font share), so
[MODAL_FONT_CHAR_SHARE_THRESHOLD] cleanly isolates doc 5 as the one case
needing a third tier. That tier is font size: doc 5's 34 top-level headings
(``"1) DISPOSIÇÕES INICIAIS"`` ... ``"34) FORO"``, confirmed by inspecting
every line >= 11.9pt in the document) are set at a clean, isolated 12.0pt
against a 10.657pt modal body size (a 12.6% delta), with zero bold signal
whatsoever -- [HEADING_FONT_SIZE_RATIO_THRESHOLD] catches this while
staying under the document's next-largest recurring size (11.61pt, the
cover title and inline URLs mixed together, an 8.9% delta and not a
reliable heading signal). Fixed in [_heavy_font_predicate] (a third
fallback tier) and [_bold_fraction] (now keyed on ``(font_name,
font_size)`` instead of ``font_name`` alone); ``CLAUSE_SEGMENTATION_VERSION``
bumped ``v3`` -> ``v4``. Effect: doc 5 recovers 0 -> 33 root clauses and
its orphan ratio drops 1.000 -> 0.009, with the whole rest of the 30-doc
corpus byte-identical (confirmed via ``scripts/build_clause_tree.py
--continue-on-orphan-failure``), since doc 5 is the only document in the
corpus whose modal-font share clears [MODAL_FONT_CHAR_SHARE_THRESHOLD].
Doc 5 is not fully recovered -- its ``"N.M."`` sub-items share the same
font *and* size as body prose, so they correctly stay ungated content
under their parent ``"N)"`` root (``max_depth`` stays 0) rather than
guessing at a threshold with no corpus evidence behind it; a real signal
for that finer level is left for future work if [M1-08]'s sample surfaces
it as a material accuracy loss.

Font size is demoted from a required gate to provenance carried on each
detected line for the numbering-pattern conventions -- not load-bearing
there, since numbering-pattern depth is the authority for tree structure,
per the DoD's own framing ("respecting the document's own structure") --
but it *is* load-bearing as the tier-3 UNNUMBERED_PART fallback above, for
the narrow case where no font-weight signal survives at all.

``CLAUSE_SEGMENTATION_VERSION`` feeds the downstream cache key (see
[infrastructure.parsing.clause_tree_caching]): bump it whenever any
detection rule here changes, so stale cached output is invalidated.
"""

import re
import statistics
import unicodedata
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

CLAUSE_SEGMENTATION_VERSION = "v4"

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

# [M1-04b] doc 5 (KOVR) has no bold-named font AND no dedicated second font
# at all -- 99.4% of its characters share one font name, vs. every other
# corpus document that reaches this fallback tier (doc 4/Sura 78.8%, doc 8
# 50.8%, doc 18 82.8%) having a real, substantially-shared second font. This
# threshold cleanly separates "no dedicated heading font exists" (only doc 5
# in the corpus) from "a dedicated heading font exists, just not bold-named".
MODAL_FONT_CHAR_SHARE_THRESHOLD = 0.95
# Doc 5's real headings ("N) TÍTULO", confirmed across all 34 of them) sit
# at 12.0pt against a 10.657pt modal/body size -- a 12.6% delta. The next
# most common larger size in the document (11.61pt, used for the cover
# title and inline URLs, not a reliable heading signal) is only an 8.9%
# delta, so this threshold cleanly separates the two.
HEADING_FONT_SIZE_RATIO_THRESHOLD = 1.10

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
    spans: list[ExtractedSpan], is_heavy: Callable[[str, float], bool]
) -> float:
    """Fraction of a line's characters set in a "heavy" (heading-weight) font."""
    total = sum(len(span.text) for span in spans)
    if total == 0:
        return 0.0
    heavy = sum(
        len(span.text) for span in spans if is_heavy(span.font_name, span.font_size)
    )
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


def _document_font_name_weights(document: ExtractedDocument) -> dict[str, int]:
    """Character-weighted count of each font name used in the document."""
    weights: dict[str, int] = {}
    for page in document.pages:
        for span in page.spans:
            weights[span.font_name] = weights.get(span.font_name, 0) + len(span.text)
    return weights


def _document_modal_font_name(document: ExtractedDocument) -> str:
    """Character-weighted modal font name -- the document's dominant body font."""
    weights = _document_font_name_weights(document)
    if not weights:
        return ""
    return max(weights, key=lambda name: weights[name])


def _document_modal_font_share(document: ExtractedDocument) -> float:
    """Character-weighted share of the document set in its single modal font.

    Close to 1.0 means the document uses essentially one font throughout --
    i.e. there is no dedicated second font a heading could be set in, so the
    "is not the dominant font" fallback below has nothing to key off.
    """
    weights = _document_font_name_weights(document)
    total = sum(weights.values())
    if total == 0:
        return 0.0
    return max(weights.values()) / total


def _document_modal_font_size(document: ExtractedDocument) -> float:
    """Character-weighted modal font size -- the document's dominant body size."""
    weights: dict[float, int] = {}
    for page in document.pages:
        for span in page.spans:
            weights[span.font_size] = weights.get(span.font_size, 0) + len(span.text)
    if not weights:
        return 0.0
    return max(weights, key=lambda size: weights[size])


def _heavy_font_predicate(
    document: ExtractedDocument,
) -> Callable[[str, float], bool]:
    """Build the per-document "is this span heading/emphasis weight" test.

    Three tiers, each a fallback for when the previous one has nothing to
    key off:

    1. Ordinarily a font whose name literally contains "bold".
    2. When the whole document's bold-named-character ratio is below
       [HAS_BOLD_FONT_CHAR_RATIO_THRESHOLD] (see that constant's docstring
       for the corpus evidence), that literal signal is untrustworthy, so
       fall back to "is not the document's dominant font" -- the doc 4
       (Sura) case, where headings use a wholly separate subset font.
    3. [M1-04b]: doc 5 (KOVR) has neither -- 99.4% of its characters share
       one font name ([MODAL_FONT_CHAR_SHARE_THRESHOLD]), with no dedicated
       heading font at all, so tier 2 also has nothing to key off (every
       span reports the same font_name). Its real headings are set 12.6%
       larger than body text (12.0pt vs a 10.66pt modal size) with no bold
       signal whatsoever -- see the module docstring for the full
       calibration evidence.
    """
    if _document_bold_ratio(document) >= HAS_BOLD_FONT_CHAR_RATIO_THRESHOLD:
        return lambda font_name, font_size: "bold" in font_name.lower()
    if _document_modal_font_share(document) < MODAL_FONT_CHAR_SHARE_THRESHOLD:
        modal_font_name = _document_modal_font_name(document)
        return lambda font_name, font_size: (
            font_name != modal_font_name and font_name != ""
        )
    size_gate = _document_modal_font_size(document) * HEADING_FONT_SIZE_RATIO_THRESHOLD
    return lambda font_name, font_size: font_size >= size_gate


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
    page: ExtractedPage, is_heavy: Callable[[str, float], bool]
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
    page: ExtractedPage, is_heavy: Callable[[str, float], bool]
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


def _slugify(text: str) -> str:
    """Turn a heading title into a stable, human-readable path segment.

    Used only for [HeadingConvention.UNNUMBERED_PART], which carries no
    ``numbering_label``. Anchoring the segment to the title's own text
    (not to scan position) keeps the segment -- and therefore the
    [Clause.clause_id] built from it -- stable when an unrelated part is
    inserted or removed elsewhere in the document.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug or "part"


def _is_numbering_start(label: str) -> bool:
    """True if a numbering label restarts a sequence at "1" (e.g. "1.", "1")."""
    clean = re.sub(r"[^0-9]", "", label)
    return clean == "1"


def _consume_unnumbered_part_run(
    lines: list[_Line], start_index: int, page_number: int
) -> tuple[list[str], int]:
    """Collect consecutive same-page UNNUMBERED_PART title lines from start_index.

    Pure/side-effect-free title-wrap merge, shared by [consume_title_wrap]
    (the real segmentation pass) and [_prescan_recurring_unnumbered_titles]
    (the [M1-04b] noise pre-scan), so both agree on what one candidate
    title's full text is.
    """
    title_parts: list[str] = []
    lookahead = start_index
    while lookahead < len(lines):
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
        lookahead += 1
    return title_parts, lookahead


def _recurrence_key(title: str) -> str:
    """Dedup key for cross-document UNNUMBERED_PART title recurrence.

    Reuses [_slugify]'s NFKD/accent-fold/case-fold/punctuation-collapse
    behavior rather than a second normalizer: real corpus pairs like "NÃO
    ESTÃO COBERTOS" / "NÃO ESTÃO COBERTOS:" and "ENVIO DE TÁXI" / "ENVIO DE
    TAXI" (doc 10) are the same noise item with inconsistent trailing
    punctuation/accents across occurrences, and [_slugify] already collapses
    both differences by construction.
    """
    return _slugify(title)


def _next_heading_starts_numbering(lines: list[_Line], start_index: int) -> bool:
    """True if the next real heading at/after start_index restarts at "1".

    Skips lettered/bulleted list-item lines (mirroring [is_list_item_line]'s
    role in the main segmentation loop) before looking for the next
    [_detect_heading] match.
    """
    index = start_index
    while index < len(lines):
        line = lines[index]
        if is_list_item_line(line.text):
            index += 1
            continue
        heading = _detect_heading(line.text, line.bold_fraction, ocr=line.is_ocr)
        if heading is not None:
            return heading.convention in (
                HeadingConvention.NUMBERED_DECIMAL,
                HeadingConvention.CLAUSULA_KEYWORD,
            ) and _is_numbering_start(heading.numbering_label)
        index += 1
    return False


def _prescan_recurring_unnumbered_titles(lines: list[_Line]) -> set[str]:
    """Identify UNNUMBERED_PART candidate titles to demote to plain content.

    [M1-04b]: a title is demoted -- all its occurrences, not just the
    offending ones -- when it recurs more than once in the document AND at
    least one occurrence is not immediately followed by its own numbering
    restart at "1". See the module docstring for the doc 10/16/3 corpus
    evidence this all-or-nothing rule is calibrated against.
    """
    occurrence_flags: dict[str, list[bool]] = {}
    index = 0
    line_count = len(lines)
    while index < line_count:
        line = lines[index]
        heading = _detect_heading(line.text, line.bold_fraction, ocr=line.is_ocr)
        if heading is None or heading.convention != HeadingConvention.UNNUMBERED_PART:
            index += 1
            continue
        extra_parts, next_index = _consume_unnumbered_part_run(
            lines, index + 1, line.page_number
        )
        merged_title = " ".join([heading.title, *extra_parts])
        key = _recurrence_key(merged_title)
        starts_restart = _next_heading_starts_numbering(lines, next_index)
        occurrence_flags.setdefault(key, []).append(starts_restart)
        index = next_index
    return {
        key
        for key, flags in occurrence_flags.items()
        if len(flags) > 1 and not all(flags)
    }


class _ClauseBuilder:
    """Mutable accumulator for one clause while the document is being walked."""

    def __init__(
        self,
        *,
        document_id: str,
        clause_id: str,
        path: str,
        numbering_label: str,
        title: str,
        convention: HeadingConvention,
        depth: int,
        parent_id: str | None,
        is_depth_anomaly: bool,
    ) -> None:
        self.document_id = document_id
        self.clause_id = clause_id
        self.path = path
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
            path=self.path,
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

    noise_titles = _prescan_recurring_unnumbered_titles(lines)

    builders: dict[str, _ClauseBuilder] = {}
    order: list[str] = []
    stack: list[_ClauseBuilder] = []
    orphan_char_count = 0
    total_char_count = 0
    # Segment collisions are counted per parent bucket (keyed by parent_id,
    # None for document roots), so a duplicate segment under one parent
    # never affects sibling counts under a different parent -- the id
    # scheme's whole point is that an edit in one branch of the tree must
    # not shift the clause_id of a node in an unrelated branch.
    sibling_segment_counts: dict[str | None, dict[str, int]] = {}

    def next_segment(parent_id: str | None, base: str) -> str:
        counts = sibling_segment_counts.setdefault(parent_id, {})
        counts[base] = counts.get(base, 0) + 1
        occurrence = counts[base]
        return base if occurrence == 1 else f"{base}-{occurrence}"

    def open_clause(match: _HeadingMatch, page_number: int, depth: int) -> None:
        if match.convention == HeadingConvention.UNNUMBERED_PART:
            stack.clear()
        else:
            while stack and stack[-1].depth >= depth:
                stack.pop()
        parent = stack[-1] if stack else None
        parent_id = parent.clause_id if parent else None
        parent_path = parent.path if parent else None

        # numbering_label is empty only for UNNUMBERED_PART, whose segment
        # is anchored to the title's own text instead -- see [_slugify].
        base_segment = match.numbering_label or _slugify(match.title)
        segment = next_segment(parent_id, base_segment)
        path = f"{parent_path}/{segment}" if parent_path else segment
        clause_id = f"{document.document_id}:{path}"

        builder = _ClauseBuilder(
            document_id=document.document_id,
            clause_id=clause_id,
            path=path,
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
        extra_parts, lookahead = _consume_unnumbered_part_run(
            lines, start_index, page_number
        )
        for consumed_index in range(start_index, lookahead):
            total_char_count += len(lines[consumed_index].text)
        if not extra_parts:
            return heading, lookahead
        merged_title = " ".join([heading.title, *extra_parts])
        return replace(heading, title=merged_title), lookahead

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
            if _recurrence_key(merged.title) in noise_titles:
                attach_content(merged.title, line.page_number)
                index = next_index
                continue
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

    # [M1-04b]: with recurring noise no longer fragmenting the tree above,
    # a genuine product root's own numbered content doesn't reliably
    # restart at label "1" (doc 10's "MOTOCICLETAS" root: its assistance-
    # list items 1-9 never surface as bold NUMBERED_DECIMAL headings, a
    # separate corpus quirk, so its first real child is "10."). Any real
    # numbered/CLÁUSULA child is now enough signal that this part carries
    # its own content-bearing clause sequence, rather than requiring that
    # sequence to specifically restart at "1" -- validated this still
    # correctly excludes zero-numbered-child parts (e.g. "GLOSSÁRIO").
    restarting_roots = []
    root_ids = [cid for cid in order if builders[cid].parent_id is None]

    for root_id in root_ids:
        root_builder = builders[root_id]
        if root_builder.depth == 0:
            has_start = any(
                builders[child_id].depth == 1
                and builders[child_id].convention
                in (
                    HeadingConvention.NUMBERED_DECIMAL,
                    HeadingConvention.CLAUSULA_KEYWORD,
                )
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
