"""Pure clause-tree chunking logic for [M3-01].

Chunks a document's already-classified [domain.clause_classification.
TypedClause] list into embeddable [domain.chunk.Chunk]s, instead of fixed
token windows that ignore the clause tree [M1-04] already recovered.

Three rules apply, in order, per clause:

1. **Merge** -- a clause whose own content is too short to be meaningful in
   isolation folds into its parent (recursively, until a large-enough unit
   or the document root is reached). Applies uniformly to every clause
   node, not just leaves: an internal clause's own thin "chapeau" text
   (the prose between its heading and its first child's heading) is just
   as eligible to fold upward as a genuine leaf -- whatever independent,
   long-enough children it has are unaffected either way, since a
   clause's fold decision depends only on its own accumulated size, never
   a sibling's.
2. **Split on item boundaries** -- a merge-resolved unit whose body still
   exceeds the configured maximum is grouped on internal list-item
   boundaries (reusing [application.use_cases.clause_segmentation.
   is_list_item_line], never invented fresh) and packed into
   near-target-sized pieces, never breaking a line.
3. **Sliding window (last resort)** -- only a body with no internal item
   boundaries at all (a monolithic prose block, e.g. a glossary's
   term/definition pairs) or a single oversized item falls through to an
   overlapping, sentence-boundary-snapped window split.

Every emitted chunk's `text` is prefixed with its ancestor titles (see
[_ancestor_titles]) so an isolated sub-clause -- or an isolated split
piece -- keeps its place in the document's structure even outside the
full tree; see docs/PARSING.md's "isolated sub-clause loses context"
critique, which this directly addresses. `source_clause_ids` keeps every
folded-in clause id traceable, so a chunk-level retrieval hit can still be
rolled up to clause-id granularity for [infrastructure.evaluation.
retriever.Retriever] and ``golden-set-v1``, both frozen at that
granularity per [M2-06]/[M2-07].

When a unit is both merged and then split, the emitted chunks report
whichever split rule fired (item-boundary or sliding-window), not
``MERGED`` -- the split rule is what actually determined the chunk's
final boundaries.

Pure, total: like [segment_document], this never raises. Operates on one
document's `TypedClause` list at a time, mirroring
[classify_and_enrich_clauses]'s one-`ClauseTree` contract.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from application.use_cases.clause_segmentation import is_list_item_line
from domain.chunk import Chunk, ChunkingReport, ChunkRule
from domain.clause_classification import TypedClause
from domain.clause_tree import Clause

CHUNKING_VERSION = "v1"

# A best-effort "end of sentence" signal for the sliding-window fallback --
# content_lines are extracted visual lines, not sentence-segmented prose
# (see domain.extracted_text), so this is a heuristic, not a guarantee.
_SENTENCE_END_CHARS = (".", "!", "?", ";", ":")

# How many lines the sliding window will look backward to find a
# sentence-final line before giving up and cutting at a hard line boundary.
_SENTENCE_LOOKBACK_LINES = 5


@dataclass
class _Unit:
    """Mutable accumulator for one merge-resolved chunkable unit.

    `parts` holds every clause folded into this unit -- the anchor is
    always present (folding only ever moves a descendant's content into
    an ancestor's unit, so the anchor's pre-order position is always the
    earliest among its parts). `char_count` sums [Clause.content_lines]
    length across every part, mirroring [application.use_cases.
    clause_segmentation.find_oversized_clauses]'s own-content measure --
    used only to decide whether this unit is still too short to stand
    alone, never as the final chunk size (see [_unit_body_lines], which
    also injects each absorbed part's title as a sub-heading).
    """

    anchor: Clause
    parts: list[Clause] = field(default_factory=list)
    char_count: int = 0


@dataclass(frozen=True)
class _SplitPiece:
    """One line-group produced by the split pass, tagged with the rule that made it."""

    lines: tuple[str, ...]
    rule: ChunkRule


def _clause_own_char_count(clause: Clause) -> int:
    """Character count of a clause's own content lines, no separators added."""
    return sum(len(line) for line in clause.content_lines)


def _lines_char_count(lines: Sequence[str]) -> int:
    """Character count across a sequence of lines, no separators added."""
    return sum(len(line) for line in lines)


def _ancestor_titles(
    clause: Clause, clause_by_id: dict[str, Clause]
) -> tuple[str, ...]:
    """Ancestor titles, root-first, not including `clause`'s own title.

    [Clause.title] already carries its full matched heading line, numeral
    included (every branch of [application.use_cases.clause_segmentation.
    _detect_heading] sets ``title=text`` or the equivalent joined form),
    so ancestor titles are joined as-is -- never re-prefixed with
    ``numbering_label``, which would double the numeral.
    """
    titles: list[str] = []
    parent_id = clause.parent_id
    while parent_id is not None:
        parent = clause_by_id[parent_id]
        titles.append(parent.title)
        parent_id = parent.parent_id
    return tuple(reversed(titles))


def _fold_short_units(
    typed_clauses: list[TypedClause], *, min_char_count: int
) -> list[_Unit]:
    """Bottom-up fold pass: absorb a too-short clause into its parent's unit.

    Relies on `typed_clauses` sharing [domain.clause_tree.ClauseTree.
    all_clauses]'s pre-order (guaranteed by [classify_and_enrich_clauses],
    which builds it by iterating `tree.all_clauses` unchanged): in a
    pre-order walk, a node's descendants are contiguous immediately after
    it, so a single `reversed()` pass visits every node strictly after all
    of its own descendants -- exactly the order a bottom-up fold needs,
    with no separate tree-building step, and one that also guarantees
    multiple short siblings folding into the same parent are both
    resolved before the parent's own turn.

    A unit whose accumulated size (own content, plus anything already
    absorbed from its children in this same pass) is still below
    `min_char_count` folds its whole part list into its parent's unit and
    is dropped from the returned list; a root has nowhere left to fold, so
    it is always kept regardless of size (an all-empty root, or an
    all-empty subtree in general, is filtered later by the zero-char-count
    check in [chunk_typed_clauses], not here).
    """
    clause_by_id = {typed.clause.clause_id: typed.clause for typed in typed_clauses}
    units: dict[str, _Unit] = {}
    folded: set[str] = set()

    for typed in reversed(typed_clauses):
        clause = typed.clause
        unit = units.setdefault(clause.clause_id, _Unit(anchor=clause, parts=[clause]))
        unit.char_count += _clause_own_char_count(clause)

        if unit.char_count < min_char_count and clause.parent_id is not None:
            parent = clause_by_id[clause.parent_id]
            parent_unit = units.setdefault(
                parent.clause_id, _Unit(anchor=parent, parts=[parent])
            )
            parent_unit.parts.extend(unit.parts)
            parent_unit.char_count += unit.char_count
            folded.add(clause.clause_id)

    return [unit for clause_id, unit in units.items() if clause_id not in folded]


def _unit_body_lines(unit: _Unit, order_index: dict[str, int]) -> list[str]:
    """Anchor's own content lines, then each absorbed part in document order.

    Every part beyond the anchor is prefixed with its own title so the
    absorbed structure stays legible in the rendered body, not just
    silently concatenated.
    """
    ordered_parts = sorted(unit.parts, key=lambda clause: order_index[clause.clause_id])
    lines: list[str] = list(ordered_parts[0].content_lines)
    for part in ordered_parts[1:]:
        lines.append(part.title)
        lines.extend(part.content_lines)
    return lines


def _group_by_item_boundary(lines: Sequence[str]) -> list[list[str]]:
    """Group lines so a new group starts at every list-item line.

    Reuses [is_list_item_line] rather than re-detecting lettered/bulleted
    items -- exactly the same signal [application.use_cases.
    clause_segmentation] uses to keep list items out of the heading tree
    in the first place.
    """
    groups: list[list[str]] = []
    for line in lines:
        if not groups or is_list_item_line(line):
            groups.append([line])
        else:
            groups[-1].append(line)
    return groups


def _pack_groups(groups: list[list[str]], *, target_char_count: int) -> list[list[str]]:
    """Greedily pack whole item-groups into near-target-sized pieces.

    Never splits a group -- a single group already over `target_char_count`
    (e.g. one long list item) is still emitted as its own, possibly
    oversized, pack; the caller runs the sliding-window fallback on
    exactly that pack.
    """
    packs: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for group in groups:
        group_len = _lines_char_count(group)
        if current and current_len + group_len > target_char_count:
            packs.append(current)
            current = []
            current_len = 0
        current.extend(group)
        current_len += group_len
    if current:
        packs.append(current)
    return packs


def _sentence_aware_sliding_window(
    lines: Sequence[str], *, target_char_count: int, overlap_chars: int
) -> list[tuple[str, ...]]:
    """Overlapping windows over `lines`, snapped to a sentence-final line if possible.

    Operates at line granularity -- a line is never split, so this can
    never cut mid-word. Grows a window to roughly `target_char_count`,
    then looks backward up to [_SENTENCE_LOOKBACK_LINES] lines for the
    most recent line ending in [_SENTENCE_END_CHARS] to snap the boundary
    to; a documented, accepted gap when no such line exists in that
    lookback (a hard line-boundary cut instead) or when a single line
    alone exceeds `target_char_count` (emitted whole rather than cut).
    The next window then steps back by roughly `overlap_chars` worth of
    trailing lines, snapped forward at least one line to guarantee
    progress.
    """
    if not lines:
        return []

    windows: list[tuple[str, ...]] = []
    total = len(lines)
    start = 0
    while start < total:
        end = start
        length = 0
        while end < total and (
            length == 0 or length + len(lines[end]) <= target_char_count
        ):
            length += len(lines[end])
            end += 1

        snap_end = end
        lookback_floor = max(start + 1, end - _SENTENCE_LOOKBACK_LINES)
        for candidate in range(end - 1, lookback_floor - 1, -1):
            if lines[candidate].rstrip().endswith(_SENTENCE_END_CHARS):
                snap_end = candidate + 1
                break

        windows.append(tuple(lines[start:snap_end]))
        if snap_end >= total:
            break

        overlap_start = snap_end
        consumed = 0
        while overlap_start > start and consumed < overlap_chars:
            overlap_start -= 1
            consumed += len(lines[overlap_start])
        start = max(overlap_start, start + 1)

    return windows


def _split_oversized_body(
    body_lines: Sequence[str],
    *,
    target_char_count: int,
    max_char_count: int,
    overlap_chars: int,
) -> list[_SplitPiece]:
    """Split an over-long body: item boundaries first, sliding window as last resort."""
    groups = _group_by_item_boundary(body_lines)
    if len(groups) <= 1:
        windows = _sentence_aware_sliding_window(
            body_lines, target_char_count=target_char_count, overlap_chars=overlap_chars
        )
        return [
            _SplitPiece(lines=window, rule=ChunkRule.SLIDING_WINDOW_SPLIT)
            for window in windows
        ]

    pieces: list[_SplitPiece] = []
    for pack in _pack_groups(groups, target_char_count=target_char_count):
        if _lines_char_count(pack) > max_char_count:
            windows = _sentence_aware_sliding_window(
                pack, target_char_count=target_char_count, overlap_chars=overlap_chars
            )
            pieces.extend(
                _SplitPiece(lines=window, rule=ChunkRule.SLIDING_WINDOW_SPLIT)
                for window in windows
            )
        else:
            pieces.append(
                _SplitPiece(lines=tuple(pack), rule=ChunkRule.ITEM_BOUNDARY_SPLIT)
            )
    return pieces


def _render_piece(
    parent_path: str, anchor_title: str, body_lines: Sequence[str]
) -> str:
    """Render one chunk's embedding text: parent-path breadcrumb, anchor title, body."""
    header = f"{parent_path}\n{anchor_title}" if parent_path else anchor_title
    body = "\n".join(body_lines)
    return f"{header}\n\n{body}" if body else header


def _build_chunk(
    anchor_typed: TypedClause,
    source_clause_ids: tuple[str, ...],
    parent_path: str,
    text: str,
    *,
    chunk_index: int,
    chunk_count: int,
    rule: ChunkRule,
) -> Chunk:
    clause = anchor_typed.clause
    chunk_id = (
        clause.clause_id if chunk_count == 1 else f"{clause.clause_id}#{chunk_index}"
    )
    return Chunk(
        document_id=clause.document_id,
        chunk_id=chunk_id,
        clause_id=clause.clause_id,
        source_clause_ids=source_clause_ids,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        parent_path=parent_path,
        text=text,
        char_count=len(text),
        rule=rule,
        clause_type=anchor_typed.clause_type,
        type_source=anchor_typed.type_source,
        confidence=anchor_typed.confidence,
        bundle_section=clause.bundle_section,
        provenance=anchor_typed.provenance,
    )


def _build_report(document_id: str, chunks: list[Chunk]) -> ChunkingReport:
    if not chunks:
        return ChunkingReport(
            document_id=document_id,
            chunk_count=0,
            single_count=0,
            merged_count=0,
            item_boundary_split_count=0,
            sliding_window_split_count=0,
            min_char_count=0,
            p50_char_count=0,
            p90_char_count=0,
            max_char_count=0,
            mean_char_count=0.0,
        )
    lengths = sorted(chunk.char_count for chunk in chunks)
    count = len(lengths)

    def percentile(fraction: float) -> int:
        return lengths[min(int(count * fraction), count - 1)]

    return ChunkingReport(
        document_id=document_id,
        chunk_count=count,
        single_count=sum(1 for chunk in chunks if chunk.rule == ChunkRule.SINGLE),
        merged_count=sum(1 for chunk in chunks if chunk.rule == ChunkRule.MERGED),
        item_boundary_split_count=sum(
            1 for chunk in chunks if chunk.rule == ChunkRule.ITEM_BOUNDARY_SPLIT
        ),
        sliding_window_split_count=sum(
            1 for chunk in chunks if chunk.rule == ChunkRule.SLIDING_WINDOW_SPLIT
        ),
        min_char_count=lengths[0],
        p50_char_count=percentile(0.5),
        p90_char_count=percentile(0.9),
        max_char_count=lengths[-1],
        mean_char_count=sum(lengths) / count,
    )


def chunk_typed_clauses(
    typed_clauses: list[TypedClause],
    *,
    min_char_count: int,
    target_char_count: int,
    max_char_count: int,
    sliding_window_overlap_chars: int,
) -> tuple[list[Chunk], ChunkingReport]:
    """Chunk one document's classified clause list. Pure, never raises.

    Args:
        typed_clauses: One document's clauses, in [domain.clause_tree.
            ClauseTree.all_clauses] pre-order (as returned by
            [classify_and_enrich_clauses]).
        min_char_count: Below this, a unit folds into its parent.
        target_char_count: The split pass's packing target -- pieces aim
            for roughly this size, never exceeding it except for a single
            oversized item/line that cannot be broken further.
        max_char_count: Above this, a merge-resolved unit's body is split.
        sliding_window_overlap_chars: Overlap between consecutive
            sliding-window pieces, in the last-resort split rule only.

    Returns:
        The document's chunks (document order, then split order within a
        unit) and a [ChunkingReport] summarizing rule counts and the
        chunk-length distribution.
    """
    if not typed_clauses:
        return [], _build_report("", [])

    document_id = typed_clauses[0].clause.document_id
    clause_by_id = {typed.clause.clause_id: typed.clause for typed in typed_clauses}
    typed_by_id = {typed.clause.clause_id: typed for typed in typed_clauses}
    order_index = {
        typed.clause.clause_id: index for index, typed in enumerate(typed_clauses)
    }

    chunks: list[Chunk] = []
    for unit in _fold_short_units(typed_clauses, min_char_count=min_char_count):
        if unit.char_count == 0:
            continue

        anchor_typed = typed_by_id[unit.anchor.clause_id]
        parent_path = " > ".join(_ancestor_titles(unit.anchor, clause_by_id))
        body_lines = _unit_body_lines(unit, order_index)
        source_clause_ids = tuple(
            sorted(
                (part.clause_id for part in unit.parts),
                key=lambda clause_id: order_index[clause_id],
            )
        )
        rule = ChunkRule.SINGLE if len(unit.parts) == 1 else ChunkRule.MERGED

        if _lines_char_count(body_lines) <= max_char_count:
            text = _render_piece(parent_path, unit.anchor.title, body_lines)
            chunks.append(
                _build_chunk(
                    anchor_typed,
                    source_clause_ids,
                    parent_path,
                    text,
                    chunk_index=0,
                    chunk_count=1,
                    rule=rule,
                )
            )
            continue

        pieces = _split_oversized_body(
            body_lines,
            target_char_count=target_char_count,
            max_char_count=max_char_count,
            overlap_chars=sliding_window_overlap_chars,
        )
        for index, piece in enumerate(pieces):
            text = _render_piece(parent_path, unit.anchor.title, piece.lines)
            chunks.append(
                _build_chunk(
                    anchor_typed,
                    source_clause_ids,
                    parent_path,
                    text,
                    chunk_index=index,
                    chunk_count=len(pieces),
                    rule=piece.rule,
                )
            )

    return chunks, _build_report(document_id, chunks)
