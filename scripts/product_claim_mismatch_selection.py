#!/usr/bin/env python3
"""Deterministic scenario selection for the product/claim mismatch set [M2-05].

Layer 1 of the same three-layer authoring flow used across M2 (see
``docs/EVALUATION.md``): picking *which document and which clause justifies
the label* is a pure function of ``build/parsed_clauses.jsonl`` and
``data/policies/manifest.csv`` -- the LLM phrasing layer lives in
``scripts/draft_product_claim_mismatch.py`` and only writes the narrative
text for a scenario this module has already fully resolved.

Every scenario here targets a document that does NOT belong to CASCO -- a
claim describing damage to the insured's own vehicle is incompatible with
all four other product lines by construction (RCF-A is third-party
liability, ASSIST is roadside assistance, GAR.EST is mechanical/electrical
extended warranty, CARTA VERDE is mandatory cross-border liability cover;
see ``docs/DATA_SOURCES.md``'s "Product/claim mismatch" section). The anchor
clause for each scenario is the document's own scope-defining clause -- the
one whose text states what risk the product actually covers, so the
incompatibility is evidenced by the document's own words rather than merely
asserted. Verified against the real corpus: every non-CASCO document titles
this clause with one of a small, consistent vocabulary (``OBJETIVO DO
SEGURO``, ``COBERTURA DO SEGURO``, ``RISCOS COBERTOS``, ``GARANTIAS DO
SEGURO``) -- [select_anchor_clause] ranks candidates on it, falling back to
the lowest-``clause_id`` candidate among [casco_selection.CORE_CLAUSE_TYPES]
when no title (or, for one document, body-opening text) matches.

Explicitly includes the corpus's two liability-only insurers, HDI Global
(document 21, CNPJ 18096627000153) and ARCA (document 22, CNPJ
50428904000190) -- the DoD's own named "clearest product/claim mismatch
cases in the corpus" (``docs/DATA_SOURCES.md``).

**Scope boundary vs. M2-04**: ``synthetic_claims_selection.py``'s scenarios
always target a document that genuinely belongs to its own product line.
This module is the mirror image: every scenario here is a deliberate
mismatch, targeted at documents chosen specifically because they cannot
cover an own-vehicle claim.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.manifest import read_manifest

try:
    # Direct execution: the script's own directory is sys.path[0].
    import adversarial_clause_selection as adversarial_selection
    import casco_clause_selection as casco_selection
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import adversarial_clause_selection as adversarial_selection
    from scripts import casco_clause_selection as casco_selection

MANIFEST_PATH = Path("data/policies/manifest.csv")

NON_CASCO_PRODUCT_LINES = ("RCF-A", "ASSIST", "GAR.EST", "CARTA VERDE")

# DoD floor: >=8. These targets clear it with margin (11 total) while
# spreading across all four non-CASCO lines, weighted toward RCF-A (8
# documents, the largest non-CASCO pool, and home to the two liability-only
# insurers this scenario set must include).
TARGET_COUNTS: dict[str, int] = {
    "RCF-A": 5,
    "ASSIST": 3,
    "GAR.EST": 2,
    "CARTA VERDE": 1,
}

# Document ids that MUST be represented -- the corpus's own named "clearest
# product/claim mismatch cases" (docs/DATA_SOURCES.md): HDI Global and ARCA,
# both liability-only insurers with no own-damage product at all.
REQUIRED_DOCUMENT_IDS: frozenset[str] = frozenset({"21", "22"})

# Titles a document's own scope-defining clause uses across the corpus
# (verified against build/parsed_clauses.jsonl for every non-CASCO product
# line: doc 21 "1. OBJETIVO DO SEGURO"/"2. COBERTURA DO SEGURO", doc 22
# "2. OBJETIVO DO SEGURO"/"6. RISCO COBERTO", doc 17 "2. Riscos Cobertos",
# doc 30 "1. OBEJTIVO DO SEGURO"/"2. RISCO COBERTO", doc 24 "CLÁUSULA 3.
# GARANTIAS DO SEGURO", doc 28 "3. OBJETIVO DO SEGURO"/"4. GARANTIAS DO
# SEGURO"). These clauses answer "what does this product cover" directly
# and rank first (STRONG_ANCHOR_TITLE_PATTERN). ÂMBITO GEOGRÁFICO clauses
# (geographic scope) mention the covered risk only in passing, so they only
# rank as a second-tier match (WEAK_ANCHOR_TITLE_PATTERN).
#
# Both patterns are anchored (`^`) against a numbering-stripped remainder,
# not searched anywhere in the raw title: a bare substring search matched
# "risco coberto" inside "8.1.1.4. Em caso de agravação do risco coberto, a
# Seguradora poderá..." (a notice-of-aggravation condition clause, not a
# scope statement) and "âmbito" inside "3.33. Foro – O âmbito geográfico da
# jurisdição competente..." (a venue/jurisdiction clause) -- both real
# corpus titles that out-ranked the genuine scope-defining heading in the
# same document purely because their clause_id happened to sort lower.
_LEADING_NUMBERING_PATTERN = re.compile(
    r"^\s*(?:cl[áa]usula\s+)?\d+(?:\.\d+)*[aª]?\.?\s*[-–]?\s*", re.IGNORECASE
)
STRONG_ANCHOR_TITLE_PATTERN = re.compile(
    r"^(?:objetivos? do seguro|coberturas? do seguro|riscos? cobertos?|"
    r"garantias? do seguro)",
    re.IGNORECASE,
)
WEAK_ANCHOR_TITLE_PATTERN = re.compile(r"^[âa]mbito", re.IGNORECASE)


def _strip_leading_numbering(title: str) -> str:
    """Return `title` with a leading numbering token removed, once."""
    return _LEADING_NUMBERING_PATTERN.sub("", title, count=1)


# Some documents attach the actual scope statement's prose to a sub-clause
# under an empty-bodied numbered heading (document 25: "1. OBJETIVO DO
# SEGURO" itself is empty, and the real sentence lives in child "1.1", whose
# own title is just that sentence's opening words, so it never matches
# [STRONG_ANCHOR_TITLE_PATTERN] on title alone). Checked only against the
# first 200 characters of a clause's own text, so a clause that merely
# mentions the phrase deep in unrelated body text is not misidentified as a
# scope statement.
STRONG_ANCHOR_BODY_OPENING_PATTERN = re.compile(
    r"^.{0,40}tem por objetivo garantir", re.IGNORECASE
)
BODY_OPENING_CHARS = 200


@dataclass(frozen=True)
class MismatchSlot:
    """One candidate product/claim mismatch scenario for the drafting script."""

    row_id: str
    product_line: str
    document_id: str
    primary_clause_id: str
    selection_notes: str = ""


def load_non_casco_documents_by_line(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, list[dict[str, str]]]:
    """Return manifest rows for the four non-CASCO lines, grouped and id-sorted."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_manifest(manifest_path):
        if row["product_line"] in NON_CASCO_PRODUCT_LINES:
            grouped[row["product_line"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["id"]))
    return grouped


def _selectable_core_clauses(
    records: list[ParsedClauseRecord],
    document_id: str,
    *,
    twins: dict[str, frozenset[str]],
    child_counts: dict[str, int],
) -> list[ParsedClauseRecord]:
    """Return one document's substantive clauses eligible to anchor a scenario.

    Scoped to [casco_selection.CORE_CLAUSE_TYPES], not ``coverage`` alone:
    measured against the real corpus, the scope-defining "OBJETIVO DO
    SEGURO"/"GARANTIAS DO SEGURO" clause is classified ``definition`` in at
    least one document (28), so a coverage-only filter would silently miss
    it and fall back to a much weaker candidate.
    """
    return [
        record
        for record in records
        if record.document_id == document_id
        and record.clause_type in casco_selection.CORE_CLAUSE_TYPES
        and adversarial_selection.is_clause_selectable(
            record, twins=twins, child_counts=child_counts
        )
    ]


def select_anchor_clause(
    candidates: list[ParsedClauseRecord],
) -> ParsedClauseRecord | None:
    """Pick the best scope-defining clause from one document's candidate pool.

    Ranks on a title/body-match tier, in order: [STRONG_ANCHOR_TITLE_PATTERN]
    on the title, [STRONG_ANCHOR_BODY_OPENING_PATTERN] on the clause's own
    text opening, [WEAK_ANCHOR_TITLE_PATTERN] on the title, then no match --
    clause_id breaks ties within a tier. Mirrors
    ``synthetic_claims_selection.py``'s ``_PERIL_HINT_REGEX`` ranking
    pattern. The body-opening tier exists because at least one document
    (25) attaches the real scope-statement sentence to a sub-clause whose
    own title never spells out "objetivo do seguro" (see
    [STRONG_ANCHOR_BODY_OPENING_PATTERN]'s docstring) -- without it, the
    fallback tier would win on an accidental heading fragment instead.
    """
    if not candidates:
        return None

    def sort_key(record: ParsedClauseRecord) -> tuple[int, str]:
        stripped_title = _strip_leading_numbering(record.title)
        text_opening = record.text.strip()[:BODY_OPENING_CHARS]
        if STRONG_ANCHOR_TITLE_PATTERN.search(
            stripped_title
        ) or STRONG_ANCHOR_BODY_OPENING_PATTERN.search(text_opening):
            tier = 0
        elif WEAK_ANCHOR_TITLE_PATTERN.search(stripped_title):
            tier = 1
        else:
            tier = 2
        return (tier, record.clause_id)

    return min(candidates, key=sort_key)


def select_mismatch_slots(
    records: list[ParsedClauseRecord],
    manifest_path: Path = MANIFEST_PATH,
    *,
    target_counts: dict[str, int] | None = None,
    required_document_ids: frozenset[str] = REQUIRED_DOCUMENT_IDS,
) -> list[MismatchSlot]:
    """Select every product/claim mismatch scenario slot, across all four lines.

    Within each product line, a required document (if any belongs to that
    line) is always taken first, then the remaining documents fill the quota
    in id order -- so [REQUIRED_DOCUMENT_IDS] is never crowded out.
    """
    target_counts = target_counts if target_counts is not None else TARGET_COUNTS
    documents_by_line = load_non_casco_documents_by_line(manifest_path)
    twins = casco_selection.build_duplicate_text_index(records)
    child_counts = adversarial_selection.build_child_counts(records)

    slots: list[MismatchSlot] = []
    for product_line, count in target_counts.items():
        documents = documents_by_line.get(product_line, [])
        required = [d for d in documents if d["id"] in required_document_ids]
        rest = [d for d in documents if d["id"] not in required_document_ids]

        taken = 0
        for document in required + rest:
            if taken >= count:
                break
            document_id = document["id"]
            candidates = _selectable_core_clauses(
                records, document_id, twins=twins, child_counts=child_counts
            )
            anchor = select_anchor_clause(candidates)
            if anchor is None:
                continue
            slots.append(
                MismatchSlot(
                    row_id=f"mismatch-{document_id}-{taken:02d}",
                    product_line=product_line,
                    document_id=document_id,
                    primary_clause_id=anchor.clause_id,
                )
            )
            taken += 1
    return slots
