#!/usr/bin/env python3
"""Deterministic candidate-pair selection for the adversarial golden set [M2-03].

Layer 1 of [M2-01]'s three-layer authoring flow, kept strictly separate from
the LLM phrasing layer in ``scripts/draft_golden_questions_adversarial.py`` --
the same split ``scripts/casco_clause_selection.py`` has from
``scripts/draft_golden_questions_casco.py``. Nothing here calls a model;
every candidate pair is a pure function of ``build/parsed_clauses.jsonl``.

Four adversarial categories, each with a different notion of "distractor":

- ``coverage_with_exclusion``: a coverage clause paired with the exclusion
  that limits it, inside one of the 15 CASCO documents. Reuses
  ``casco_clause_selection.py`` in full (its own-damage scope classification
  is valid there), scoped away from clause pairs already finalized in
  ``data/golden_set/coverage_with_exclusion.jsonl``.
- ``cross_document``: a clause in one document and a near-duplicate in the
  sibling document of the same insurer (same CNPJ) -- the eight same-insurer
  pairs in the corpus. Tests engineering robustness (a metadata filter must
  not leak across near-duplicate documents), not a real analyst flow: in
  practice the insurer is already known before the search starts.
- ``hdi_brand_collision``: a clause in an HDI Seguros document (CNPJ
  29980158000157) paired with a plausible-looking match in the HDI Global
  document (CNPJ 18096627000153) -- two different legal entities sharing the
  "HDI" brand name. Distinct from ``cross_document``: a name-vs-CNPJ trap,
  not a same-company near-duplicate (12<->30, same CNPJ, is a
  ``cross_document`` pair, not this one).
- ``bundle_section``: two clauses inside the single Bradesco bundle document
  (id 10, 207 pages) that live in different ``bundle_section`` values but
  read almost identically -- two of its glass-coverage variants, or two of
  its assistance tiers differing only by a coverage limit. The realistic
  broad-search case: within one insurer's multi-product filing, never
  between insurers.

Only one side of every ``cross_document``/``hdi_brand_collision`` pair is a
CASCO document (the other is RCF-A, CARTA VERDE or ASSIST), so
[casco_clause_selection.classify_scope] -- which encodes a CASCO own-damage
scope assumption -- is not reused for those categories. What IS reused
everywhere: [casco_clause_selection.build_duplicate_text_index] (twin
detection) and [casco_clause_selection.find_exclusion_reasons] (degenerate-
clause filter), via [is_clause_selectable].

Near-duplicate search is purely structural (stdlib ``difflib``), matching
[M2-08]'s "no LLM judging content relevance" philosophy -- the same caveat
applies: an empty or weak automated result does not mean no related clause
exists. Measured against the real corpus, three pairs return nothing usable
from the automated search (AKAD 3<->16, KOVR 5<->20, HDI 30<->21 -- best
ratios 0.39-0.45, heading-only matches) and the ``bundle_section`` pairs need
hand-picked clause ids rather than a bare ratio search (the highest-ratio
matches there are byte-identical shared boilerplate common to every product
variant -- a real retrieval confusion cannot be built on those). Both cases
are covered by the hardcoded pairs below, found by reading the source
documents directly rather than left as a silent gap.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from infrastructure.parsing.clause_schema import ParsedClauseRecord

try:
    # Direct execution: the script's own directory is sys.path[0].
    import casco_clause_selection as selection
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import casco_clause_selection as selection

CORE_CLAUSE_TYPES = selection.CORE_CLAUSE_TYPES

# The eight same-insurer document pairs in the corpus (same CNPJ, different
# filing) -- see docs/DATA_SOURCES.md. `cross_document` questions target
# these: the correct clause lives in the first document, a plausible
# near-duplicate in the second.
SAME_INSURER_PAIRS: tuple[tuple[str, str], ...] = (
    ("1", "11"),  # Porto Seguro
    ("3", "16"),  # AKAD
    ("4", "18"),  # Seguros SURA
    ("5", "20"),  # KOVR
    ("9", "19"),  # Caixa Seguradora
    ("10", "17"),  # Bradesco Auto/RE
    ("12", "30"),  # HDI Seguros
    ("15", "24"),  # MAPFRE
)

# HDI brand-collision pairs: HDI Seguros (CNPJ 29980158000157, docs 12/30)
# vs. HDI Global (CNPJ 18096627000153, doc 21) -- two different legal
# entities sharing the "HDI" brand. Deliberately excludes 12<->30 (same
# CNPJ -- already a SAME_INSURER_PAIRS entry, a cross_document case, not a
# brand-collision one).
HDI_BRAND_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (("12", "21"), ("30", "21"))

BUNDLE_DOCUMENT_ID = "10"

MIN_TEXT_CHARS_FOR_NEAR_DUP = 150
MAX_LENGTH_RATIO_BAND = 0.4
DEFAULT_MIN_RATIO = 0.75
DEFAULT_MAX_PAIRS = 20

# The automated near-dup search (both body-text ratio and title matching)
# returns nothing usable for these three pairs -- measured best ratios
# 0.39-0.45, all weak heading-only matches, against the real
# build/parsed_clauses.jsonl. Found by reading the source documents
# directly: same topic, same document position, different phrasing/OCR
# quality. An empty automated result does NOT mean no related clause
# exists -- see the module docstring.
MANUAL_NEAR_DUPLICATE_PAIRS: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("3", "16"): [
        (
            "3:franquia",
            "16:condicoes-gerais-de-responsabilidade-civil-facultativa-de-veiculos-rcf-v/16",
        ),
        (
            "3:salvados",
            "16:condicoes-gerais-de-responsabilidade-civil-facultativa-de-veiculos-rcf-v/26",
        ),
    ],
    ("5", "20"): [
        ("5:22-franquia", "20:22-franquia-do-seguro/22.1"),
        ("5:28-sub-rogacao-de-direitos", "20:24-sub-rogacao-de-direitos/242/242.1"),
    ],
    ("30", "21"): [
        ("30:3", "21:16-documentos-necessarios-para-a-liquidacao-do-sinistro/23"),
        ("30:13", "21:16-documentos-necessarios-para-a-liquidacao-do-sinistro/19"),
    ],
    ("9", "19"): [
        ("9:17", "19:membros-inferiores/17"),
        ("9:31", "19:membros-inferiores/16"),
    ],
}

# Hand-picked bundle_section pairs within the Bradesco document (id 10):
# three glass-coverage variants whose only substantive difference is the
# "Objeto e âmbito geográfico" clause (their other clauses -- Obrigações do
# Segurado, Condições de Atendimento -- are byte-for-byte shared
# boilerplate, ratio ~1.0, which is why a bare ratio search surfaces the
# wrong clause first, and why this list is hand-picked rather than
# search-generated), and one assistance-tier pair whose root clause is
# near-identical except for the extra Premium services it adds (measured
# ratio 0.945, confirming the differentiating fact sits inside otherwise
# near-duplicate text rather than a separate clause). A fourth candidate --
# the "100km"/"200km" motorcycle-assistance tiers -- was tried and dropped:
# the "200km" and "400km" tiers turned out to be byte-identical twins of
# each other (a genuine ground-truth ambiguity [is_clause_selectable]
# correctly catches), leaving no safe second side to pair with "100km".
BUNDLE_SECTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("10:vidro-protegido-plus-no-24/1", "10:vidro-protegido-no25/1"),
    (
        "10:vidro-protegido-logomarca-no-150/1",
        "10:vidro-protegido-plus-logomarca-no-151/1",
    ),
    ("10:vidro-protegido-carga-no-83/1", "10:vidro-protegido-plus-carga-no-155/1"),
    (
        "10:assistencia-auto-dia-e-noite-ilimitado-no-108",
        "10:assistencia-auto-dia-e-noite-ilimitado-premium-no-174",
    ),
)


@dataclass(frozen=True)
class NearDuplicatePair:
    """Two clauses that read similarly enough to confuse a retriever."""

    clause_id_a: str
    clause_id_b: str
    ratio: float


@dataclass(frozen=True)
class AdversarialSlot:
    """One candidate question slot for the adversarial drafting script.

    ``secondary_clause_id`` is the paired exclusion for
    ``coverage_with_exclusion`` slots. ``distractor_clause_id`` is the
    confusable clause for ``cross_document``/``hdi_brand_collision``/
    ``bundle_section`` slots -- never intended to land in
    ``reference_clause_ids``, only shown to the LLM as what NOT to answer
    with. Exactly one of the two is set, depending on ``adversarial_category``.
    """

    row_id: str
    adversarial_category: str
    document_id: str
    primary_clause_id: str
    secondary_clause_id: str | None = None
    distractor_clause_id: str | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def find_near_duplicate_clause_pairs(
    clauses_a: list[ParsedClauseRecord],
    clauses_b: list[ParsedClauseRecord],
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float | None = None,
    min_text_chars: int = MIN_TEXT_CHARS_FOR_NEAR_DUP,
    require_same_clause_type: bool = True,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> list[NearDuplicatePair]:
    """Rank clause pairs across two lists by body-text similarity.

    Purely structural: ``difflib.SequenceMatcher`` over whitespace-
    normalized text, no LLM judging relevance -- same philosophy as
    [M2-08]'s ``find_candidates``. Restricted to substantive clause types
    (``CORE_CLAUSE_TYPES``) and to a minimum length, because an unfiltered
    cross product is dominated by short boilerplate lines (measured 500+
    false hits per pair without the floor). ``max_ratio`` exists for the
    ``bundle_section`` caller: the highest-ratio matches there are
    byte-identical shared boilerplate common to every product variant,
    which a real retrieval confusion cannot be built on.
    """
    pool_a = [
        record
        for record in clauses_a
        if record.clause_type in CORE_CLAUSE_TYPES
        and len(record.text.strip()) >= min_text_chars
    ]
    pool_b = [
        record
        for record in clauses_b
        if record.clause_type in CORE_CLAUSE_TYPES
        and len(record.text.strip()) >= min_text_chars
    ]

    pairs: list[NearDuplicatePair] = []
    for record_a in pool_a:
        text_a = _normalize_text(record_a.text)
        for record_b in pool_b:
            if record_a.clause_id == record_b.clause_id:
                continue
            if (
                require_same_clause_type
                and record_a.clause_type != record_b.clause_type
            ):
                continue
            len_a, len_b = len(text_a), len(record_b.text.strip())
            shorter, longer = sorted((len_a, len_b))
            if longer and (longer - shorter) / longer > MAX_LENGTH_RATIO_BAND:
                continue
            text_b = _normalize_text(record_b.text)
            ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
            if ratio < min_ratio or (max_ratio is not None and ratio > max_ratio):
                continue
            pairs.append(
                NearDuplicatePair(
                    clause_id_a=record_a.clause_id,
                    clause_id_b=record_b.clause_id,
                    ratio=ratio,
                )
            )

    pairs.sort(key=lambda pair: pair.ratio, reverse=True)
    return pairs[:max_pairs]


_LEADING_NUMBERING = re.compile(r"^\s*[\divxlcIVXLC]+[.\)]?\s*")


def _normalize_title(title: str) -> str:
    return _LEADING_NUMBERING.sub("", title).strip().casefold()


def find_shared_title_clause_pairs(
    clauses_a: list[ParsedClauseRecord],
    clauses_b: list[ParsedClauseRecord],
) -> list[tuple[ParsedClauseRecord, ParsedClauseRecord]]:
    """Pair clauses whose titles match after stripping numbering and casefolding.

    The reliable signal for ``hdi_brand_collision``: body-text similarity is
    weak between structurally unrelated CASCO and liability documents (e.g.
    the obviously-correct "ÂMBITO GEOGRÁFICO" pair between docs 12 and 21
    scores only 0.134 on body text), but titles match once numbering is
    stripped.
    """
    by_title_b: dict[str, list[ParsedClauseRecord]] = defaultdict(list)
    for record in clauses_b:
        by_title_b[_normalize_title(record.title)].append(record)

    pairs: list[tuple[ParsedClauseRecord, ParsedClauseRecord]] = []
    for record_a in clauses_a:
        key = _normalize_title(record_a.title)
        if not key:
            continue
        for record_b in by_title_b.get(key, []):
            pairs.append((record_a, record_b))
    return pairs


def is_clause_selectable(
    record: ParsedClauseRecord,
    *,
    twins: dict[str, frozenset[str]],
    child_counts: dict[str, int],
) -> bool:
    """Whether a clause may anchor a new adversarial question.

    Checks only the two universal degenerate-content hazards: duplicate
    text (an ambiguous ground truth -- which twin did the retriever mean to
    return?) and near-empty text (nothing to build a question on).
    Deliberately does NOT reuse the rest of
    [casco_clause_selection.find_exclusion_reasons] (``split_mid_sentence``,
    ``container_prefer_child``, ``glossary_container_prefer_entry``,
    ``artifact_title``): those are heuristics tuned against the CASCO
    corpus's own parse-quality signal and measurably over-fire on the
    RCF-A/CARTA VERDE/ASSIST sibling documents this module also reads --
    ``split_mid_sentence`` alone flags the large majority of otherwise
    ordinary numbered-heading clauses in docs 12/21/30 (a heading not ending
    in punctuation, followed by lowercase body text, is normal there, not a
    parse defect). A borderline candidate here still goes to human review in
    the draft CSV, so returning a plausible one is the safer failure mode
    than returning none. ``child_counts`` is accepted for interface
    symmetry with the CASCO helper but unused by this check.
    """
    del child_counts
    twin_ids = twins.get(record.clause_id, frozenset())
    text = record.text.strip()
    return not twin_ids and len(text) >= selection.MIN_CANDIDATE_TEXT_CHARS


def clauses_for_document(
    records: list[ParsedClauseRecord], document_id: str
) -> list[ParsedClauseRecord]:
    """Return every clause belonging to one document."""
    return [record for record in records if record.document_id == document_id]


def build_child_counts(records: list[ParsedClauseRecord]) -> dict[str, int]:
    """Return {clause_id: number of children}, corpus-wide."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.parent_id is not None:
            counts[record.parent_id] += 1
    return counts


def load_existing_coverage_with_exclusion_clause_ids(
    path: Path = Path("data/golden_set/coverage_with_exclusion.jsonl"),
) -> frozenset[str]:
    """Return clause ids referenced by an already-finalized coverage_with_exclusion row.

    Used to avoid re-selecting a coverage/exclusion pair M2-02 already
    anchored a question on.
    """
    if not path.exists():
        return frozenset()
    import json

    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ids.update(json.loads(line)["reference_clause_ids"])
    return frozenset(ids)


def select_coverage_with_exclusion_slots(
    records: list[ParsedClauseRecord],
    manifest_path: Path = Path("data/policies/manifest.csv"),
    *,
    already_used_ids: frozenset[str] = frozenset(),
    max_slots: int = DEFAULT_MAX_PAIRS,
) -> list[AdversarialSlot]:
    """Coverage clauses paired with a nearby limiting exclusion, over the 15 CASCO docs.

    Reuses [casco_clause_selection] in full -- its scope classification is
    valid here, since every candidate is a CASCO document. The exclusion
    partner is found with [find_candidate_clauses.find_candidates] (M2-08),
    the same structural completeness tool M2-02 used: shared parent,
    matching bundle_section, or a textual cross-reference.
    """
    try:
        from find_candidate_clauses import find_candidates
    except ModuleNotFoundError:
        from scripts.find_candidate_clauses import find_candidates

    documents = selection.load_casco_documents(manifest_path)
    ancestor_titles = selection.build_ancestor_titles(records)
    twins = selection.build_duplicate_text_index(records)

    slots: list[AdversarialSlot] = []
    slot_number = 0
    for doc_row in documents:
        doc_id = doc_row["id"]
        candidates = selection.build_candidates_for_document(
            records, doc_id, ancestor_titles, twins
        )
        by_clause_id = {c.clause.clause_id: c for c in candidates}
        coverage_candidates = sorted(
            (
                c
                for c in candidates
                if c.clause.clause_type.value == "coverage"
                and c.is_selectable
                and c.scope == selection.SCOPE_CASCO
                and c.clause.clause_id not in already_used_ids
            ),
            key=lambda c: c.clause.clause_id,
        )
        for coverage in coverage_candidates:
            structural = find_candidates(
                records, coverage.clause.clause_id, max_candidates=10
            )
            exclusion_id = next(
                (
                    cand.clause_id
                    for cand in structural
                    if cand.clause_id in by_clause_id
                    and by_clause_id[cand.clause_id].clause.clause_type.value
                    == "exclusion"
                    and by_clause_id[cand.clause_id].is_selectable
                ),
                None,
            )
            if exclusion_id is None:
                continue
            slots.append(
                AdversarialSlot(
                    row_id=f"cwe-{doc_id}-{slot_number:02d}",
                    adversarial_category="coverage_with_exclusion",
                    document_id=doc_id,
                    primary_clause_id=coverage.clause.clause_id,
                    secondary_clause_id=exclusion_id,
                )
            )
            slot_number += 1
            if len(slots) >= max_slots:
                return slots
    return slots


def select_cross_document_slots(
    records: list[ParsedClauseRecord],
    *,
    pairs_per_document_pair: int = 2,
) -> list[AdversarialSlot]:
    """Near-duplicate clause pairs across the eight same-insurer documents.

    Combines two structural signals, ranked-search results first: body-text
    similarity ([find_near_duplicate_clause_pairs]) and shared, numbering-
    stripped titles ([find_shared_title_clause_pairs]). Measured against the
    real corpus, no single signal covers all eight pairs -- KOVR/AKAD/Porto
    Seguro/HDI-Seguros lean on one or the other, or (for pair (9, 19)) on
    [MANUAL_NEAR_DUPLICATE_PAIRS] when neither returns anything.
    """
    twins = selection.build_duplicate_text_index(records)
    child_counts = build_child_counts(records)

    slots: list[AdversarialSlot] = []
    for doc_a, doc_b in SAME_INSURER_PAIRS:
        clauses_a = clauses_for_document(records, doc_a)
        clauses_b = clauses_for_document(records, doc_b)
        by_id = {r.clause_id: r for r in clauses_a + clauses_b}

        ranked = find_near_duplicate_clause_pairs(clauses_a, clauses_b, max_pairs=30)
        candidate_ids = [(pair.clause_id_a, pair.clause_id_b) for pair in ranked]
        candidate_ids += [
            (record_a.clause_id, record_b.clause_id)
            for record_a, record_b in find_shared_title_clause_pairs(
                clauses_a, clauses_b
            )
        ]
        candidate_ids += MANUAL_NEAR_DUPLICATE_PAIRS.get((doc_a, doc_b), [])

        taken = 0
        seen: set[str] = set()
        for clause_id_a, clause_id_b in candidate_ids:
            if taken >= pairs_per_document_pair or clause_id_a in seen:
                continue
            record_a, record_b = by_id.get(clause_id_a), by_id.get(clause_id_b)
            if record_a is None or record_b is None:
                continue
            if not is_clause_selectable(
                record_a, twins=twins, child_counts=child_counts
            ) or not is_clause_selectable(
                record_b, twins=twins, child_counts=child_counts
            ):
                continue
            seen.add(clause_id_a)
            slots.append(
                AdversarialSlot(
                    row_id=f"xdoc-{doc_a}x{doc_b}-{taken:02d}",
                    adversarial_category="cross_document",
                    document_id=doc_a,
                    primary_clause_id=clause_id_a,
                    distractor_clause_id=clause_id_b,
                )
            )
            taken += 1
    return slots


def select_hdi_brand_collision_slots(
    records: list[ParsedClauseRecord],
    *,
    pairs_per_document_pair: int = 4,
) -> list[AdversarialSlot]:
    """Clauses in an HDI Seguros document paired with a same-titled HDI Global clause.

    Title matching is the primary signal here (see the module docstring):
    body-text similarity is weak between these structurally unrelated
    CASCO/liability/legacy documents. Ranked near-duplicate search and
    [MANUAL_NEAR_DUPLICATE_PAIRS] (needed for 30<->21, whose documents share
    almost no vocabulary) supplement it.
    """
    twins = selection.build_duplicate_text_index(records)
    child_counts = build_child_counts(records)

    slots: list[AdversarialSlot] = []
    for doc_a, doc_b in HDI_BRAND_COLLISION_PAIRS:
        clauses_a = clauses_for_document(records, doc_a)
        clauses_b = clauses_for_document(records, doc_b)
        by_id = {r.clause_id: r for r in clauses_a + clauses_b}

        candidate_ids = [
            (record_a.clause_id, record_b.clause_id)
            for record_a, record_b in find_shared_title_clause_pairs(
                clauses_a, clauses_b
            )
        ]
        candidate_ids += [
            (pair.clause_id_a, pair.clause_id_b)
            for pair in find_near_duplicate_clause_pairs(
                clauses_a, clauses_b, min_ratio=0.5, max_pairs=10
            )
        ]
        candidate_ids += MANUAL_NEAR_DUPLICATE_PAIRS.get((doc_a, doc_b), [])

        taken = 0
        seen: set[str] = set()
        for clause_id_a, clause_id_b in candidate_ids:
            if taken >= pairs_per_document_pair or clause_id_a in seen:
                continue
            record_a, record_b = by_id.get(clause_id_a), by_id.get(clause_id_b)
            if record_a is None or record_b is None:
                continue
            if not is_clause_selectable(
                record_a, twins=twins, child_counts=child_counts
            ) or not is_clause_selectable(
                record_b, twins=twins, child_counts=child_counts
            ):
                continue
            seen.add(clause_id_a)
            slots.append(
                AdversarialSlot(
                    row_id=f"hdi-{doc_a}x{doc_b}-{taken:02d}",
                    adversarial_category="hdi_brand_collision",
                    document_id=doc_a,
                    primary_clause_id=clause_id_a,
                    distractor_clause_id=clause_id_b,
                )
            )
            taken += 1
    return slots


def select_bundle_section_slots(
    records: list[ParsedClauseRecord],
) -> list[AdversarialSlot]:
    """Hand-picked near-duplicate clause pairs across Bradesco bundle sections.

    Unlike the other three categories, this one is not a live search: the
    bundle document's highest-ratio cross-section matches are byte-identical
    shared boilerplate (see [BUNDLE_SECTION_PAIRS]'s comment), so the
    differentiating clause was found by reading the sections directly rather
    than by a bare ratio threshold.
    """
    by_id = {r.clause_id: r for r in records if r.document_id == BUNDLE_DOCUMENT_ID}
    twins = selection.build_duplicate_text_index(records)
    child_counts = build_child_counts(records)

    slots: list[AdversarialSlot] = []
    for index, (clause_id_a, clause_id_b) in enumerate(BUNDLE_SECTION_PAIRS):
        record_a, record_b = by_id.get(clause_id_a), by_id.get(clause_id_b)
        if record_a is None or record_b is None:
            continue
        if not is_clause_selectable(
            record_a, twins=twins, child_counts=child_counts
        ) or not is_clause_selectable(record_b, twins=twins, child_counts=child_counts):
            continue
        slots.append(
            AdversarialSlot(
                row_id=f"bundle-{BUNDLE_DOCUMENT_ID}-{index:02d}",
                adversarial_category="bundle_section",
                document_id=BUNDLE_DOCUMENT_ID,
                primary_clause_id=clause_id_a,
                distractor_clause_id=clause_id_b,
            )
        )
    return slots
