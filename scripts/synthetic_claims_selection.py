#!/usr/bin/env python3
"""Deterministic scenario selection for the synthetic claim set [M2-04].

Layer 1 of the same three-layer authoring flow M2-02/M2-03 already use (see
``docs/EVALUATION.md``): every decision about *which document, which clause(s),
and which verdict* is a pure function of ``build/parsed_clauses.jsonl`` and
``data/policies/manifest.csv``. Nothing here calls a model -- the LLM phrasing
layer lives in ``scripts/draft_synthetic_claims.py`` and only writes the
narrative text for a scenario this module has already fully resolved.

Three scenario types, one per DoD verdict:

- ``compatible``: a selectable ``coverage`` clause, preferring one that names a
  concrete peril over an abstract framing clause, so there is a real event to
  narrate.
- ``incompatible``: a coverage clause paired with the ``exclusion`` clause that
  limits it, found via [find_candidate_clauses.find_candidates] -- the same
  structural pairing ``casco_clause_selection.select_...`` (by way of
  ``adversarial_clause_selection.select_coverage_with_exclusion_slots``) already
  uses, generalized off CASCO. Where no structural pair exists (more likely in
  the small product lines), falls back to a standalone exclusion clause,
  flagged ``exclusion_only_fallback`` for extra reviewer scrutiny.
- ``insufficient_information``: a clause whose applicability turns on a fact
  the corpus shows up repeatedly as load-bearing (geographic scope, vehicle
  use, event date/vigência, a deductible/limit value, conditional wording) --
  see [MISSING_FACT_PATTERNS]. That clause becomes the scenario's
  ``reference_clause_ids``: the DoD requires "the clauses that justify the
  label" even here, the label being "cannot tell whether this clause applies
  without the omitted fact," not "no clause exists" (contrast with
  ``golden_set_schema.GoldenQuestion``'s ``unanswerable``, which is empty by
  construction).

Selectability is product-line-dependent: CASCO reuses
[casco_clause_selection.classify_scope] and its degenerate-clause filter (both
CASCO-tuned, see ``adversarial_clause_selection.py``'s docstring for why they
are not reused elsewhere); every other product line uses
[adversarial_clause_selection.is_clause_selectable]'s lighter twins+min-length
check.

**Scope boundary vs. M2-05**: every scenario here targets a document that
genuinely belongs to its own product line -- an ``insufficient_information``
or ``incompatible`` scenario about, say, an ASSIST document tests ASSIST's own
clauses, never a CASCO-shaped claim aimed at the wrong document. Cross-product
mismatch claims are M2-05's, not this module's.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from domain.clause_classification import ClauseType
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.manifest import read_manifest

try:
    # Direct execution: the script's own directory is sys.path[0].
    import adversarial_clause_selection as adversarial_selection
    import casco_clause_selection as casco_selection
    from find_candidate_clauses import find_candidates
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import adversarial_clause_selection as adversarial_selection
    from scripts import casco_clause_selection as casco_selection
    from scripts.find_candidate_clauses import find_candidates

MANIFEST_PATH = Path("data/policies/manifest.csv")

# DoD floors: >=30 total, >=10 insufficient_information. These targets clear
# both with margin and keep CASCO from dominating the set despite being half
# the corpus -- "balance across product lines" is a DoD requirement, not just
# a nice-to-have.
TARGET_COUNTS: dict[str, dict[str, int]] = {
    "CASCO": {"compatible": 4, "incompatible": 4, "insufficient_information": 4},
    "RCF-A": {"compatible": 3, "incompatible": 3, "insufficient_information": 3},
    "ASSIST": {"compatible": 3, "incompatible": 3, "insufficient_information": 3},
    "GAR.EST": {"compatible": 2, "incompatible": 2, "insufficient_information": 2},
    "CARTA VERDE": {"compatible": 2, "incompatible": 1, "insufficient_information": 1},
}

# Product lines with a large enough clause pool that a clause used for one
# scenario type should not be reused for another within the same run.
# GAR.EST and CARTA VERDE are small enough (CARTA VERDE: one document, 20
# clauses total) that enforcing this would starve later scenario types, so
# reuse there is allowed -- the review CSV is the safety valve, not this dedup.
DEDUP_PRODUCT_LINES = frozenset({"CASCO", "RCF-A", "ASSIST"})

# A loose "this clause describes a concrete event" signal, used only to rank
# compatible-scenario candidates above abstract framing/objective clauses --
# not a strict filter, since a clause that misses it can still anchor a fine
# scenario.
_PERIL_HINT_PATTERN = (
    r"colis[ãa]o|abalroamento|inc[êe]ndio|roubo|furto|granizo|alagamento|"
    r"quebra de vidro|vidro|reboque|guincho|chaveiro|carro reserva|"
    r"defeito|quebra|pane|assist[êe]ncia|terceiros?|acidente|capotamento|"
    r"colis[ãa]o|desastre"
)
_PERIL_HINT_REGEX = re.compile(_PERIL_HINT_PATTERN, re.IGNORECASE)

# Load-bearing "fact types": a regex matched against a clause's title+text
# (clause_type in {coverage, exclusion, condition}) that, when true, means the
# clause's applicability depends on a fact a narrative can plausibly omit.
# Measured against build/parsed_clauses.jsonl: every product line, including
# the single-document CARTA VERDE, has at least one match for at least one
# fact type.
MISSING_FACT_PATTERNS: dict[str, str] = {
    "ambito_geografico": (
        r"[âa]mbito geogr[áa]fico|territ[óo]rio (?:nacional|brasileiro)|"
        r"mercosul|exterior|fora do (?:pa[íi]s|territ[óo]rio)"
    ),
    "uso_do_veiculo": (
        r"aplicativo|transporte remunerado|uso comercial|t[áa]xi|frete|"
        r"loca[çc][ãa]o|compartilhamento de ve[íi]culo"
    ),
    "data_evento_vigencia": (
        r"vig[êe]ncia|per[íi]odo de car[êe]ncia|per[íi]odo de garantia|"
        r"prazo de garantia"
    ),
    "valor_franquia_limite": r"franquia|limite m[áa]ximo de indeniza[çc][ãa]o|\bLMI\b",
    "tipo_evento_condicao": (
        r"desde que|somente (?:quando|se)|apenas (?:quando|se)|"
        r"condicionad[ao] a"
    ),
}
_MISSING_FACT_REGEXES = {
    fact_type: re.compile(pattern, re.IGNORECASE)
    for fact_type, pattern in MISSING_FACT_PATTERNS.items()
}

# Fed to the LLM drafting layer verbatim (with {clause_id} interpolated) as
# the instruction for what to omit from a insufficient_information narrative.
MISSING_FACT_INSTRUCTIONS: dict[str, str] = {
    "ambito_geografico": (
        "Omita em que país/local o evento ocorreu -- a cláusula {clause_id} só "
        "se aplica dentro de um âmbito geográfico específico, e sem essa "
        "informação não dá para saber se o evento está dentro dele."
    ),
    "uso_do_veiculo": (
        "Omita se o veículo estava sendo usado para transporte remunerado/"
        "aplicativo no momento do evento -- a cláusula {clause_id} condiciona "
        "a cobertura a esse uso."
    ),
    "data_evento_vigencia": (
        "Omita a data do evento de forma que não dê para saber se ele ocorreu "
        "dentro do prazo/vigência que a cláusula {clause_id} exige."
    ),
    "valor_franquia_limite": (
        "Omita o valor estimado do prejuízo -- a cláusula {clause_id} depende "
        "de comparar esse valor com um limite/franquia."
    ),
    "tipo_evento_condicao": (
        "Descreva o evento de forma vaga o suficiente para não dar para saber "
        "qual das hipóteses previstas na cláusula {clause_id} de fato ocorreu."
    ),
}

_RELEVANT_TYPES_FOR_MISSING_FACT = frozenset(
    {ClauseType.COVERAGE, ClauseType.EXCLUSION, ClauseType.CONDITION}
)


@dataclass(frozen=True)
class ScenarioSlot:
    """One candidate claim scenario for the drafting script.

    ``secondary_clause_id`` is the paired exclusion for ``incompatible`` slots
    (``None`` for the ``exclusion_only_fallback`` case, where
    ``primary_clause_id`` is itself the exclusion). ``missing_fact_type`` is
    set only for ``insufficient_information`` slots, keying
    [MISSING_FACT_INSTRUCTIONS].
    """

    row_id: str
    product_line: str
    document_id: str
    scenario_type: str  # "compatible" | "incompatible" | "insufficient_information"
    primary_clause_id: str
    secondary_clause_id: str | None = None
    missing_fact_type: str | None = None
    selection_notes: str = ""


def load_documents_by_product_line(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, list[dict[str, str]]]:
    """Return manifest rows grouped by product_line, each group id-sorted."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_manifest(manifest_path):
        grouped[row["product_line"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["id"]))
    return grouped


def _selectable_clauses_for_document(
    records: list[ParsedClauseRecord],
    document_id: str,
    product_line: str,
    *,
    ancestor_titles: dict[str, tuple[str, ...]],
    twins: dict[str, frozenset[str]],
) -> list[ParsedClauseRecord]:
    """Return the clauses in one document eligible to anchor a new scenario."""
    if product_line == "CASCO":
        candidates = casco_selection.build_candidates_for_document(
            records, document_id, ancestor_titles, twins
        )
        return [
            candidate.clause
            for candidate in candidates
            if candidate.is_selectable
            and candidate.scope == casco_selection.SCOPE_CASCO
        ]
    child_counts = adversarial_selection.build_child_counts(records)
    return [
        record
        for record in records
        if record.document_id == document_id
        and adversarial_selection.is_clause_selectable(
            record, twins=twins, child_counts=child_counts
        )
    ]


def _round_robin_take[T](items: list[tuple[str, T]], n: int) -> list[tuple[str, T]]:
    """Take up to `n` items, spreading picks across distinct group keys.

    Groups (by the tuple's first element) are visited in first-seen order;
    within a group, items keep their original order. This is what keeps a
    product line's scenarios spread across its documents instead of piling up
    on document 1.
    """
    groups: dict[str, list[tuple[str, T]]] = defaultdict(list)
    order: list[str] = []
    for key, item in items:
        if key not in groups:
            order.append(key)
        groups[key].append((key, item))

    taken: list[tuple[str, T]] = []
    index = 0
    while len(taken) < n:
        progressed = False
        for key in order:
            if index < len(groups[key]):
                taken.append(groups[key][index])
                progressed = True
                if len(taken) >= n:
                    break
        if not progressed:
            break
        index += 1
    return taken


def select_compatible_slots(
    records: list[ParsedClauseRecord],
    documents: list[dict[str, str]],
    *,
    product_line: str,
    ancestor_titles: dict[str, tuple[str, ...]],
    twins: dict[str, frozenset[str]],
    target_count: int,
    already_used_ids: frozenset[str] = frozenset(),
) -> list[ScenarioSlot]:
    """Select `target_count` coverage clauses across `documents`, one product line."""
    candidates: list[tuple[str, ParsedClauseRecord]] = []
    for document in documents:
        document_id = document["id"]
        selectable = _selectable_clauses_for_document(
            records,
            document_id,
            product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
        )
        coverage = [
            record
            for record in selectable
            if record.clause_type == ClauseType.COVERAGE
            and record.clause_id not in already_used_ids
        ]
        coverage.sort(
            key=lambda record: (
                0 if _PERIL_HINT_REGEX.search(f"{record.title}\n{record.text}") else 1,
                record.clause_id,
            )
        )
        candidates.extend((document_id, record) for record in coverage)

    picked = _round_robin_take(candidates, target_count)
    return [
        ScenarioSlot(
            row_id=f"sc-compat-{document_id}-{index:02d}",
            product_line=product_line,
            document_id=document_id,
            scenario_type="compatible",
            primary_clause_id=record.clause_id,
        )
        for index, (document_id, record) in enumerate(picked)
    ]


def select_incompatible_slots(
    records: list[ParsedClauseRecord],
    documents: list[dict[str, str]],
    *,
    product_line: str,
    ancestor_titles: dict[str, tuple[str, ...]],
    twins: dict[str, frozenset[str]],
    target_count: int,
    already_used_ids: frozenset[str] = frozenset(),
) -> list[ScenarioSlot]:
    """Select `target_count` coverage+exclusion pairs across `documents`."""
    candidates: list[tuple[str, tuple[str, str | None, str]]] = []
    for document in documents:
        document_id = document["id"]
        selectable = _selectable_clauses_for_document(
            records,
            document_id,
            product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
        )
        by_id = {record.clause_id: record for record in selectable}
        coverage = sorted(
            (
                record
                for record in selectable
                if record.clause_type == ClauseType.COVERAGE
                and record.clause_id not in already_used_ids
            ),
            key=lambda record: record.clause_id,
        )
        exclusions = sorted(
            (
                record
                for record in selectable
                if record.clause_type == ClauseType.EXCLUSION
                and record.clause_id not in already_used_ids
            ),
            key=lambda record: record.clause_id,
        )

        paired_any = False
        for cov in coverage:
            structural = find_candidates(records, cov.clause_id, max_candidates=10)
            exclusion_id = next(
                (
                    candidate.clause_id
                    for candidate in structural
                    if candidate.clause_id in by_id
                    and by_id[candidate.clause_id].clause_type == ClauseType.EXCLUSION
                    and candidate.clause_id not in already_used_ids
                ),
                None,
            )
            if exclusion_id is not None:
                candidates.append((document_id, (cov.clause_id, exclusion_id, "")))
                paired_any = True
        if not paired_any and exclusions:
            candidates.append(
                (
                    document_id,
                    (exclusions[0].clause_id, None, "exclusion_only_fallback"),
                )
            )

    picked = _round_robin_take(candidates, target_count)
    slots: list[ScenarioSlot] = []
    for index, (document_id, (primary_id, secondary_id, notes)) in enumerate(picked):
        slots.append(
            ScenarioSlot(
                row_id=f"sc-incompat-{document_id}-{index:02d}",
                product_line=product_line,
                document_id=document_id,
                scenario_type="incompatible",
                primary_clause_id=primary_id,
                secondary_clause_id=secondary_id,
                selection_notes=notes,
            )
        )
    return slots


def select_insufficient_information_slots(
    records: list[ParsedClauseRecord],
    documents: list[dict[str, str]],
    *,
    product_line: str,
    ancestor_titles: dict[str, tuple[str, ...]],
    twins: dict[str, frozenset[str]],
    target_count: int,
    already_used_ids: frozenset[str] = frozenset(),
) -> list[ScenarioSlot]:
    """Select `target_count` clauses gated on an omittable load-bearing fact."""
    candidates: list[tuple[str, tuple[str, str]]] = []
    for document in documents:
        document_id = document["id"]
        selectable = _selectable_clauses_for_document(
            records,
            document_id,
            product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
        )
        pool = sorted(
            (
                record
                for record in selectable
                if record.clause_type in _RELEVANT_TYPES_FOR_MISSING_FACT
                and record.clause_id not in already_used_ids
            ),
            key=lambda record: record.clause_id,
        )
        for record in pool:
            haystack = f"{record.title}\n{record.text}"
            for fact_type, regex in _MISSING_FACT_REGEXES.items():
                if regex.search(haystack):
                    candidates.append((document_id, (record.clause_id, fact_type)))
                    break

    picked = _round_robin_take(candidates, target_count)
    slots: list[ScenarioSlot] = []
    for index, (document_id, (clause_id, fact_type)) in enumerate(picked):
        slots.append(
            ScenarioSlot(
                row_id=f"sc-insuf-{document_id}-{index:02d}",
                product_line=product_line,
                document_id=document_id,
                scenario_type="insufficient_information",
                primary_clause_id=clause_id,
                missing_fact_type=fact_type,
            )
        )
    return slots


def select_all_slots(
    records: list[ParsedClauseRecord],
    manifest_path: Path = MANIFEST_PATH,
    *,
    target_counts: dict[str, dict[str, int]] | None = None,
) -> list[ScenarioSlot]:
    """Select every scenario slot across all five product lines.

    Order within a product line: compatible, then incompatible, then
    insufficient_information -- so, on lines where dedup is enforced
    ([DEDUP_PRODUCT_LINES]), an incompatible pairing cannot claim a clause a
    compatible scenario already anchored on, and likewise for
    insufficient_information.
    """
    target_counts = target_counts if target_counts is not None else TARGET_COUNTS
    documents_by_line = load_documents_by_product_line(manifest_path)
    ancestor_titles = casco_selection.build_ancestor_titles(records)
    twins = casco_selection.build_duplicate_text_index(records)

    slots: list[ScenarioSlot] = []
    for product_line, counts in target_counts.items():
        documents = documents_by_line.get(product_line, [])
        used: set[str] = set()
        enforce_dedup = product_line in DEDUP_PRODUCT_LINES

        compatible = select_compatible_slots(
            records,
            documents,
            product_line=product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
            target_count=counts.get("compatible", 0),
            already_used_ids=frozenset(used) if enforce_dedup else frozenset(),
        )
        if enforce_dedup:
            used.update(slot.primary_clause_id for slot in compatible)
        slots.extend(compatible)

        incompatible = select_incompatible_slots(
            records,
            documents,
            product_line=product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
            target_count=counts.get("incompatible", 0),
            already_used_ids=frozenset(used) if enforce_dedup else frozenset(),
        )
        if enforce_dedup:
            for slot in incompatible:
                used.add(slot.primary_clause_id)
                if slot.secondary_clause_id:
                    used.add(slot.secondary_clause_id)
        slots.extend(incompatible)

        insufficient = select_insufficient_information_slots(
            records,
            documents,
            product_line=product_line,
            ancestor_titles=ancestor_titles,
            twins=twins,
            target_count=counts.get("insufficient_information", 0),
            already_used_ids=frozenset(used) if enforce_dedup else frozenset(),
        )
        if enforce_dedup:
            used.update(slot.primary_clause_id for slot in insufficient)
        slots.extend(insufficient)

    return slots
