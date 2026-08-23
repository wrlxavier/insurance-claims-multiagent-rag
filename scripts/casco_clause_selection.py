#!/usr/bin/env python3
"""Deterministic source-clause selection for the CASCO golden set [M2-02].

Layer 1 of [M2-01]'s three-layer authoring flow, kept strictly separate from
the LLM phrasing layer in ``scripts/draft_golden_questions_casco.py`` -- the
same split ``scripts/sample_parsing_quality.py`` has from
``scripts/validate_parsing_quality_sample.py``. Nothing here calls a model;
every decision is a pure function of ``build/parsed_clauses.jsonl`` and is
reproducible from it.

Four selection hazards this module exists to prevent, each measured against
the corpus rather than assumed:

1. **Bundled non-own-damage content was selectable.** The 15 CASCO filings
   are multi-product PDFs that also carry residential, personal-accident
   (APP) and institutional/ESG material; 9 of 75 questions landed there.
   ``bundle_section`` turns out to be exactly the root ancestor's title
   (2803/2803 clauses), so [build_ancestor_titles] subsumes it and catches
   more -- but only 4 of the 9. Text-level matching recovers 3 more, and the
   last two are bare numeric injury tables at depth 0 with no lexical marker
   at all, so they need [OUT_OF_SCOPE_CLAUSE_IDS]. Third-party liability
   (RCF) and 24h assistance are deliberately *not* out of scope: RCF is
   required [M2-02] vocabulary, and the assistance questions were accepted
   on review. Assistance is merely tagged and deprioritised so it cannot
   take over a document again (document 12 was 4 of 5 assistance).
2. **Degenerate clauses were selectable.** 382 CASCO clauses share
   byte-identical text with a twin in the same document (document 10 alone:
   254), which makes a single-id ground truth unscoreable -- a retriever
   returning the identical twin is marked wrong by luck of the id. Clause
   boundaries also split some titles mid-sentence, and OCR produced
   spaced-letter artefact titles. See [find_exclusion_reasons].
3. **Coverage is easy to measure on the wrong text.** Scoring vocabulary
   against the *source clause* overstates it badly -- on one 75-question
   draft the clause-side count read incêndio 12 / colisão 12 while the
   question-side figures were 0 and 1, enough to declare a failing DoD met.
   [score_question_vocabulary] scores the question, which is what the DoD is
   about; the clause-side hits stay available as a selection signal but are
   never reported as coverage.
4. **``perda total`` matched injury tables.** "perda total da visão" and
   "perda total do uso de um membro" are APP disability wording, not vehicle
   total loss. The pattern now excludes them and adds the term Brazilian
   motor policies actually use for a written-off vehicle, ``indenização
   integral``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from domain.clause_classification import ClauseType
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.manifest import read_manifest

PRODUCT_LINE = "CASCO"
MIN_QUESTIONS_PER_DOC = 3
MAX_QUESTIONS_PER_DOC = 5
MIN_CANDIDATE_TEXT_CHARS = 30

# DoD floors. The issue says "cover" the vocabulary without giving a number;
# three questions per term is the floor this curation commits to, so a
# single incidental mention cannot be read as coverage.
MIN_VOCAB_QUESTIONS_PER_TERM = 3
MIN_INDEMNITY_BASIS_QUESTIONS = 5
INDEMNITY_BASIS_QUOTA_PER_REGIME: dict[str, int] = {"VD": 2, "VMR": 2, "VD+VMR": 2}

CORE_CLAUSE_TYPES = frozenset(
    {
        ClauseType.COVERAGE,
        ClauseType.EXCLUSION,
        ClauseType.CONDITION,
        ClauseType.DEFINITION,
    }
)

VOCAB_PATTERNS: dict[str, str] = {
    "colisao": r"colis[ãa]o|abalroamento",
    "roubo_furto": r"roubo|furto",
    "incendio": r"inc[êe]ndio",
    # The negative lookahead keeps APP disability wording ("perda total da
    # visão", "perda total do uso de um membro") out; `indenização integral`
    # is what motor policies call a written-off vehicle.
    "perda_total": (
        r"perda total(?!\s+d[aeo]s?\s+(?:vis|audi|fala|movimento|uso|membro|fun[çc]))"
        r"|indeniza[çc][ãa]o integral"
    ),
    "franquia": r"franquia",
    # Adjacency was too rigid: "danos a terceiros" missed three genuine
    # third-party claims that put words in between -- "o veículo de terceiro
    # sofreu danos materiais", "dano moral e estético a um terceiro". The
    # DoD term is third-party damage, so proximity within one sentence is
    # the right test, not word order.
    "rc_facultativa": (
        r"responsabilidade civil facultativa|rcfv?\b|"
        r"(?:danos?|preju[íi]zos?|indeniza[çc][ãa]o)[^.;]{0,80}terceiro|"
        r"terceiro[^.;]{0,80}(?:danos?|preju[íi]zos?|indeniza[çc][ãa]o)"
    ),
    "ambito_geografico": (
        r"[âa]mbito geogr[áa]fico|territ[óo]rio (?:nacional|brasileiro)|mercosul"
    ),
}

INDEMNITY_BASIS_PATTERN = (
    r"valor determinado|valor de mercado referenciado|valor de mercado|"
    r"base de indeniza[çc][ãa]o|apura[çc][ãa]o do (?:valor|preju[íi]zo)|"
    r"c[áa]lculo da indeniza[çc][ãa]o|fator de ajuste"
)

# Matched against a clause's own title, its text, and every ancestor title.
OUT_OF_SCOPE_PATTERNS: dict[str, str] = {
    "residencial": (
        r"residencial|resid[êe]ncia|im[óo]vel|benfeitoria|"
        r"conte[úu]do (?:do|da) (?:im[óo]vel|resid)"
    ),
    "app_acidentes_pessoais": (
        r"acidentes? pessoa(?:l|is)|\bapp\b|invalidez permanente|"
        r"morte acidental|dmha|despesas m[ée]dico"
    ),
    "material_nao_contratual": (
        r"\besg\b|\bsasb\b|sustentabilidade|governan[çc]a corporativa|"
        r"taxonomia|rela[çc][õo]es com investidores"
    ),
}

# Clause-level scope patterns are deliberately loose, because a clause that
# merely mentions "imóvel" is usually a residential clause. A QUESTION is
# different: "bateu contra o muro de uma residência" is a collision scenario
# in which the house is scenery, and flagging it out of scope is a false
# positive. So the question surface tests for the PRODUCT being asked about,
# not for an incidental noun.
QUESTION_OUT_OF_SCOPE_PATTERNS: dict[str, str] = {
    "residencial": (
        r"seguro residencial|cobertura (?:residencial|para a resid[êe]ncia|"
        r"do im[óo]vel)|conte[úu]do do im[óo]vel|danos? ao im[óo]vel|"
        r"benfeitoria"
    ),
    "app_acidentes_pessoais": (
        r"acidentes? pessoa(?:l|is)|\bapp\b|invalidez permanente|"
        r"morte acidental|dmha"
    ),
    "material_nao_contratual": (
        r"\besg\b|\bsasb\b|sustentabilidade|governan[çc]a corporativa|"
        r"taxonomia sasb"
    ),
}

# Bare "assistência" is a reliable section marker on its own -- qualifying
# it (with "24 horas", say) misses titles like "ASSISTÊNCIA AUTO DIA E NOITE
# ILIMITADO" and silently labels those sections `casco`. The service nouns
# catch the sections that never use the word at all.
ASSISTANCE_PATTERN = (
    r"assist[êe]ncia|reboque|guincho|chaveiro|carro reserva|"
    r"servi[çc]os? aos? passageiros?|hospedagem|translado|"
    r"remo[çc][ãa]o|t[áa]xi"
)

# Human-readable Portuguese for each vocabulary key. The prompts used to
# interpolate the internal key, which leaked verbatim into a question as
# "além do caso de roubo_furto total do veículo" (review code E6).
VOCAB_DISPLAY_NAMES: dict[str, str] = {
    "colisao": "colisão",
    "roubo_furto": "roubo ou furto",
    "incendio": "incêndio",
    "perda_total": "perda total / indenização integral",
    "franquia": "franquia",
    "rc_facultativa": "responsabilidade civil facultativa (danos a terceiros)",
    "ambito_geografico": "âmbito geográfico",
}

# A question "covers" a term only when it puts the term in a concrete claim
# situation. Counting mentions credited incêndio three times when two were
# the name of a coverage package ("Compreensiva - Colisão, Incêndio,
# Roubo/Furto...") and the third a contrast inside a definition -- no fire
# claim existed anywhere in the set.
SCENARIO_MARKER_PATTERN = (
    r"\bse\b|\bcaso\b|ap[óo]s\b|durante\b|quando\b|em caso de|"
    r"o segurado\b|um ve[íi]culo\b|o ve[íi]culo\b|houve\b|sofreu\b|"
    r"teve\b|ocorreu\b|colidiu\b|capotou\b|foi (?:roubado|furtado|incendiado)"
)
PACKAGE_LISTING_WINDOW_CHARS = 80
PACKAGE_LISTING_MIN_TERMS = 3
MIN_VOCAB_SCENARIOS_PER_TERM = 2

# Self-referential phrasing: the question assumes the reader already knows
# which clause is on screen, which M2-01's authoring rules forbid.
SELF_REFERENCE_PATTERN = (
    r"\bdeste documento\b|\bdesta ap[óo]lice\b|\bnesta cl[áa]usula\b|"
    r"\bdesta cl[áa]usula\b|\bna cl[áa]usula acima\b|\bcl[áa]usula em quest[ãa]o\b|"
    r"\bdo presente contrato\b|\bacima (?:citad|mencionad)"
)

# APP disability grids, matched against a clause's OWN title and text and
# deliberately NOT inherited down the ancestor chain. Document 11 shows why
# both halves matter: `11:membros-inferiores` is such a grid ("Perda Total do
# uso de uma falange..."), but the parser hung three genuine own-damage
# clauses under it -- BLINDAGEM, carro reserva, danos morais a terceiros.
# Matching own content blocks the grid; not inheriting keeps its
# mis-parented children usable, which is the difference between working
# around the parse defect and being defeated by it.
DISABILITY_TABLE_PATTERN = (
    r"falange|polegar|anquilose|membros? (?:superiores|inferiores)|"
    r"perda (?:total|parcial) do uso de|invalidez (?:permanente|total|parcial)"
)

# Density separates a grid from a glossary that merely has an "invalidez
# permanente" entry: measured over the corpus, the real grids score 6 matches
# while glossaries score 1-3. A leaf clause needs only one match, since there
# the topic is necessarily the subject.
DISABILITY_GRID_MIN_MATCHES = 4

# A glossary entry whose whole title is a regulator's name defines the
# regulator, not anything a claims analyst adjudicates -- `7:susep` and
# `13:susep` both read "a SUSEP é uma autarquia vinculada ao Ministério da
# Fazenda". Matched on the title alone, so the many clauses that merely cite
# a SUSEP process number stay in scope.
REGULATOR_TITLE_PATTERN = (
    r"^(?:susep|cnsp|procon|denatran|detran|ans|cvm|bacen|banco central)\W*$"
)

# A bundled filing's glossary defines terms for every product it covers, so
# its BODY inevitably mentions "acidente pessoal" and "imóvel" no matter how
# own-damage the clause is. Matching scope against that text flagged 13
# perfectly good rows -- `1:glossario` (10.5k chars), `4:3` (36 children),
# `7:susep` (a glossary root that merely starts with the regulator entry).
# So body text is only consulted for clauses small and leaf-like enough that
# the matched topic must actually be their subject; for everything else,
# titles decide. Where a glossary is genuinely misused it shows up in the
# QUESTION, which [question_scope_flag] checks separately.
GLOSSARY_TITLE_PATTERN = r"gloss[áa]rio|defini[çc][õo]es|termos t[ée]cnicos"
MAX_TEXT_CHARS_FOR_SCOPE_MATCH = 2000

# A heading that enumerates several products ("CONDIÇÕES GERAIS DOS SEGUROS
# DE AUTOMÓVEL, RCF-V E APP") is a combined section, not a dedicated
# personal-accident one; its own-damage content stays in scope.
OWN_DAMAGE_TITLE_PATTERN = r"autom[óo]vel|\bauto\b|casco|ve[íi]culo"

# Depth-0 clauses whose content is out of scope but carries no lexical or
# structural marker at all: document 6's four APP disability tables are bare
# numeric percentage grids sitting as root siblings of genuine CASCO
# sections, because the PDF's section nesting was lost at parse time. There
# is no signal to detect them by, so they are named. Parsing is out of scope
# for [M2-02]; this is the honest workaround, not a fix.
OUT_OF_SCOPE_CLAUSE_IDS: frozenset[str] = frozenset(
    {
        "6:t-o-t-a-l",
        "6:p-a-r-c-i-a-l-diversos",
        "6:p-a-r-c-i-a-l-membros-inferiores",
        "6:p-a-r-c-i-a-l-membros-superiores",
    }
)

_SPACED_LETTER_TITLE = re.compile(r"^(?:[A-Za-zÀ-ÿ]\s+){3,}")
_SENTENCE_END = ".?!:;"

_VOCAB_REGEXES = {
    term: re.compile(pattern, re.IGNORECASE) for term, pattern in VOCAB_PATTERNS.items()
}
_BASIS_REGEX = re.compile(INDEMNITY_BASIS_PATTERN, re.IGNORECASE)
_OUT_OF_SCOPE_REGEXES = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in OUT_OF_SCOPE_PATTERNS.items()
}
_ASSISTANCE_REGEX = re.compile(ASSISTANCE_PATTERN, re.IGNORECASE)
_QUESTION_OUT_OF_SCOPE_REGEXES = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in QUESTION_OUT_OF_SCOPE_PATTERNS.items()
}
_SCENARIO_MARKER_REGEX = re.compile(SCENARIO_MARKER_PATTERN, re.IGNORECASE)
_SELF_REFERENCE_REGEX = re.compile(SELF_REFERENCE_PATTERN, re.IGNORECASE)
_DISABILITY_TABLE_REGEX = re.compile(DISABILITY_TABLE_PATTERN, re.IGNORECASE)
_REGULATOR_TITLE_REGEX = re.compile(REGULATOR_TITLE_PATTERN, re.IGNORECASE)
_GLOSSARY_TITLE_REGEX = re.compile(GLOSSARY_TITLE_PATTERN, re.IGNORECASE)
_OWN_DAMAGE_TITLE_REGEX = re.compile(OWN_DAMAGE_TITLE_PATTERN, re.IGNORECASE)

SCOPE_CASCO = "casco"
SCOPE_ASSISTANCE = "periferico:assistencia"


@dataclass(frozen=True)
class ClauseCandidate:
    """One clause considered as a question's source, with why it was kept or not."""

    clause: ParsedClauseRecord
    vocab_hits: frozenset[str]
    is_indemnity_basis: bool
    scope: str
    exclusion_reasons: tuple[str, ...]
    twin_ids: frozenset[str]

    @property
    def is_out_of_scope(self) -> bool:
        """Whether this clause belongs to a bundled non-own-damage section."""
        return self.scope.startswith("fora_escopo:")

    @property
    def is_selectable(self) -> bool:
        """Whether a new question may be anchored on this clause."""
        return not self.is_out_of_scope and not self.exclusion_reasons

    @property
    def is_core_type(self) -> bool:
        """Whether the clause is substantive rather than boilerplate."""
        return self.clause.clause_type in CORE_CLAUSE_TYPES


def load_casco_documents(manifest_path: Path) -> list[dict[str, str]]:
    """Return the 15 CASCO manifest rows, ordered by numeric document id."""
    rows = [
        row
        for row in read_manifest(manifest_path)
        if row["product_line"] == PRODUCT_LINE
    ]
    return sorted(rows, key=lambda row: int(row["id"]))


def build_ancestor_titles(
    records: list[ParsedClauseRecord],
) -> dict[str, tuple[str, ...]]:
    """Map each clause_id to the titles of its ancestors, nearest parent first.

    Strictly more informative than ``bundle_section``, which is exactly the
    root ancestor's title -- this also sees mid-tree markers such as
    document 8's "3. SERVIÇOS DE ASSISTÊNCIA 24 HORAS" that the root title
    misses.
    """
    by_id = {record.clause_id: record for record in records}
    chains: dict[str, tuple[str, ...]] = {}
    for record in records:
        titles: list[str] = []
        seen: set[str] = {record.clause_id}
        parent_id = record.parent_id
        while parent_id is not None and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            parent = by_id[parent_id]
            titles.append(parent.title)
            parent_id = parent.parent_id
        chains[record.clause_id] = tuple(titles)
    return chains


def build_duplicate_text_index(
    records: list[ParsedClauseRecord],
) -> dict[str, frozenset[str]]:
    """Map each clause_id to other clause_ids with byte-identical text.

    Scoped to the same document, since that is where an ambiguous ground
    truth actually bites. Clauses shorter than
    [MIN_CANDIDATE_TEXT_CHARS] are skipped -- a shared one-line stub is not
    the ambiguity this guards against.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        stripped = record.text.strip()
        if len(stripped) < MIN_CANDIDATE_TEXT_CHARS:
            continue
        groups[(record.document_id, stripped)].append(record.clause_id)

    twins: dict[str, frozenset[str]] = {}
    for clause_ids in groups.values():
        if len(clause_ids) < 2:
            continue
        as_set = set(clause_ids)
        for clause_id in clause_ids:
            twins[clause_id] = frozenset(as_set - {clause_id})
    return twins


def classify_scope(
    record: ParsedClauseRecord,
    ancestor_titles: tuple[str, ...],
    child_count: int = 0,
) -> str:
    """Return "casco", the assistance tag, or "fora_escopo:<reason>".

    Titles always decide; body text is consulted only for clauses small and
    leaf-like enough that a matched topic must genuinely be their subject.
    See [GLOSSARY_TITLE_PATTERN] for why -- matching a bundled filing's
    glossary body against the out-of-scope patterns marked 13 sound
    own-damage questions as residential or personal-accident.
    """
    if record.clause_id in OUT_OF_SCOPE_CLAUSE_IDS:
        return "fora_escopo:app_acidentes_pessoais"

    titles = "\n".join([record.title, *ancestor_titles])
    text = record.text.strip()
    is_glossary_like = (
        child_count > 0
        or len(text) > MAX_TEXT_CHARS_FOR_SCOPE_MATCH
        or bool(_GLOSSARY_TITLE_REGEX.search(record.title))
    )

    if not is_glossary_like and _REGULATOR_TITLE_REGEX.match(record.title.strip()):
        return "fora_escopo:material_nao_contratual"

    disability_hits = len(_DISABILITY_TABLE_REGEX.findall(f"{record.title}\n{text}"))
    if disability_hits >= DISABILITY_GRID_MIN_MATCHES or (
        disability_hits and not is_glossary_like
    ):
        return "fora_escopo:app_acidentes_pessoais"

    haystack = titles if is_glossary_like else f"{titles}\n{text}"
    for label, regex in _OUT_OF_SCOPE_REGEXES.items():
        if regex.search(haystack):
            # A heading naming several products alongside the vehicle is a
            # combined section, not a dedicated non-own-damage one.
            if _OWN_DAMAGE_TITLE_REGEX.search(titles):
                continue
            return f"fora_escopo:{label}"
    # Assistance is matched on the same surface as the out-of-scope
    # patterns: `11:membros-inferiores/3-6` names no service in its title but
    # its 176-character body is about "Serviços aos passageiros", and it was
    # classified `casco` and then selected (review code E3).
    if _ASSISTANCE_REGEX.search(haystack):
        return SCOPE_ASSISTANCE
    return SCOPE_CASCO


def question_scope_flag(question: str) -> str:
    """Flag a QUESTION that asks about a bundled non-own-damage topic.

    The clause-level filter deliberately keeps glossaries in scope, because a
    glossary is a legitimate source for own-damage definitions. What the
    review actually objected to in rows like 4-00 and 12-00 was the *question*
    -- asking a glossary to define "Acidente Pessoal". That is a property of
    the question, so it is checked here rather than by excluding the clause.
    """
    for label, regex in _QUESTION_OUT_OF_SCOPE_REGEXES.items():
        if regex.search(question):
            return f"fora_escopo:{label}"
    return ""


def is_glossary_title(title: str) -> bool:
    """Whether a clause title marks it as a glossary or definitions section."""
    return bool(_GLOSSARY_TITLE_REGEX.search(title))


def is_glossary_container(record: ParsedClauseRecord, child_count: int) -> bool:
    """Whether this is a glossary that has sub-entries to point at instead.

    The single granularity rule: a glossary WITH children is never an anchor,
    because the entry the question is about exists as its own clause. A
    glossary with no children (document 1's is 10.5k characters of inline
    text) stays a valid leaf -- there is nothing more specific to reference.
    Document 6 previously used both rules at once, anchoring 6-00 on the
    sub-entry `6:2/2.45` and 6-05 on the whole 9,865-character `6:2`.
    """
    return child_count > 0 and bool(_GLOSSARY_TITLE_REGEX.search(record.title))


def find_exclusion_reasons(
    record: ParsedClauseRecord,
    twin_ids: frozenset[str],
    child_count: int,
) -> tuple[str, ...]:
    """Return why this clause must not anchor a NEW question, if it must not.

    Empty means selectable. These are parse defects worked around rather
    than fixed -- parsing is out of scope for [M2-02].
    """
    reasons: list[str] = []
    title = record.title.strip()
    text = record.text.strip()

    if twin_ids:
        reasons.append("duplicate_text")
    if len(text) < MIN_CANDIDATE_TEXT_CHARS:
        reasons.append("text_too_short")
    if _SPACED_LETTER_TITLE.match(title):
        reasons.append("artifact_title")
    continues_title = bool(text[:1]) and (
        text[0].islower() or text[0] in "(" or text[0].isdigit()
    )
    if title.endswith((",", ";")) or (
        title and title[-1] not in _SENTENCE_END and continues_title
    ):
        reasons.append("split_mid_sentence")
    if child_count and len(text) < MIN_CANDIDATE_TEXT_CHARS * 4:
        reasons.append("container_prefer_child")
    if is_glossary_container(record, child_count):
        reasons.append("glossary_container_prefer_entry")
    return tuple(reasons)


def coverage_with_exclusion_gap(
    reference_clause_ids: list[str], by_id: dict[str, ParsedClauseRecord]
) -> str:
    """Return what a coverage_with_exclusion ground truth is missing, or "".

    The single standard: such a question must reference at least one
    COVERAGE clause and at least one EXCLUSION clause, because that pairing
    is the whole thing [M3-06] measures. Enforced structurally rather than
    left to per-question judgement -- "exclusion only" and "exclusion plus
    deductible" are each defensible in isolation, and a category built from
    a handful of questions cannot afford one rule per question.
    """
    types = {by_id[cid].clause_type for cid in reference_clause_ids if cid in by_id}
    missing = [
        name
        for name, clause_type in (
            ("coverage", ClauseType.COVERAGE),
            ("exclusion", ClauseType.EXCLUSION),
        )
        if clause_type not in types
    ]
    return "missing_" + "_and_".join(missing) if missing else ""


def build_candidates_for_document(
    records: list[ParsedClauseRecord],
    document_id: str,
    ancestor_titles: dict[str, tuple[str, ...]],
    twins: dict[str, frozenset[str]],
) -> list[ClauseCandidate]:
    """Tag every clause in one document with scope, quality and vocabulary hits.

    Returns all of them, selectable or not -- the rejected ones are the
    audit trail written by [build_selection_audit_rows], which is what makes
    the pre-selection pool evidence rather than an assertion.
    """
    child_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.parent_id is not None:
            child_counts[record.parent_id] += 1

    candidates: list[ClauseCandidate] = []
    for record in records:
        if record.document_id != document_id:
            continue
        haystack = f"{record.title}\n{record.text}"
        chain = ancestor_titles.get(record.clause_id, ())
        twin_ids = twins.get(record.clause_id, frozenset())
        candidates.append(
            ClauseCandidate(
                clause=record,
                vocab_hits=frozenset(
                    term
                    for term, regex in _VOCAB_REGEXES.items()
                    if regex.search(haystack)
                ),
                is_indemnity_basis=bool(_BASIS_REGEX.search(haystack)),
                scope=classify_scope(
                    record, chain, child_counts.get(record.clause_id, 0)
                ),
                exclusion_reasons=find_exclusion_reasons(
                    record, twin_ids, child_counts.get(record.clause_id, 0)
                ),
                twin_ids=twin_ids,
            )
        )
    return candidates


def score_question_vocabulary(question: str) -> frozenset[str]:
    """Return the vocabulary terms the QUESTION text uses.

    The DoD is about what the questions ask, not what their source clauses
    happen to contain. Scoring the clause measures the wrong surface and
    reports coverage the questions do not actually have: a fire clause can
    be the source of a question that never mentions fire.
    """
    return frozenset(
        term for term, regex in _VOCAB_REGEXES.items() if regex.search(question)
    )


def question_is_indemnity_basis(question: str) -> bool:
    """Whether the QUESTION itself asks about how indemnity is calculated."""
    return bool(_BASIS_REGEX.search(question))


def _package_listing_spans(question: str) -> list[tuple[int, int]]:
    """Return character spans where vocabulary terms are merely enumerated.

    A run of three or more vocabulary terms inside a short window is the name
    of a coverage package, not a claim -- "Compreensiva (Colisão, Incêndio,
    Roubo/Furto e Alagamento)" mentions three terms and describes no event.
    """
    hits: list[tuple[int, int]] = []
    for regex in _VOCAB_REGEXES.values():
        hits.extend((m.start(), m.end()) for m in regex.finditer(question))
    hits.sort()

    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(hits):
        cluster = [hits[index]]
        while (
            index + 1 < len(hits)
            and hits[index + 1][0] - cluster[0][0] <= PACKAGE_LISTING_WINDOW_CHARS
        ):
            index += 1
            cluster.append(hits[index])
        if len(cluster) >= PACKAGE_LISTING_MIN_TERMS:
            spans.append((cluster[0][0], cluster[-1][1]))
        index += 1
    return spans


def score_question_scenarios(question: str, question_type: str) -> frozenset[str]:
    """Return the terms this question actually exercises as a claim situation.

    Narrower than [score_question_vocabulary] on purpose: a glossary
    `definition` question, and any term appearing only inside a coverage
    package listing, does not put the term into a claim. See
    [SCENARIO_MARKER_PATTERN] for what counts as posing a situation.
    """
    if question_type == "definition":
        return frozenset()
    if not _SCENARIO_MARKER_REGEX.search(question):
        return frozenset()

    listings = _package_listing_spans(question)

    def outside_listing(start: int, end: int) -> bool:
        return not any(s <= start and end <= e for s, e in listings)

    return frozenset(
        term
        for term, regex in _VOCAB_REGEXES.items()
        if any(outside_listing(m.start(), m.end()) for m in regex.finditer(question))
    )


def question_self_reference_flag(question: str) -> str:
    """Return the self-referential phrase a question uses, or "".

    M2-01 requires a question to stand on its own; "Na cobertura Vidro
    Protegido Moto Plus deste documento" assumes the reader can already see
    which document is meant (review code E6).
    """
    match = _SELF_REFERENCE_REGEX.search(question)
    return match.group(0) if match else ""


def term_in_title(term: str, title: str) -> bool:
    """Whether a vocabulary term appears in a clause's own title.

    Used to rank source clauses when a replacement slot is targeted at a
    specific term: a clause titled "6. ÂMBITO GEOGRÁFICO" is genuinely about
    geographic scope, whereas one that merely says "no território brasileiro"
    while assigning the payout to a lender is not, and forcing a question
    about the term out of it would produce a bad question.
    """
    return bool(_VOCAB_REGEXES[term].search(title))


def select_indemnity_basis_documents(
    candidates_by_doc: dict[str, list[ClauseCandidate]],
    regime_by_doc: dict[str, str],
) -> set[str]:
    """Choose which documents carry an indemnity-basis slot, spread by regime.

    Takes up to [INDEMNITY_BASIS_QUOTA_PER_REGIME] documents per regime in
    document-id order, then tops up so the DoD's >=5 total holds even where
    one regime's corpus is thin.
    """
    eligible_by_regime: dict[str, list[str]] = defaultdict(list)
    for doc_id, candidates in candidates_by_doc.items():
        if any(c.is_indemnity_basis and c.is_selectable for c in candidates):
            eligible_by_regime[regime_by_doc[doc_id]].append(doc_id)
    for docs in eligible_by_regime.values():
        docs.sort(key=int)

    chosen: set[str] = set()
    for regime, quota in INDEMNITY_BASIS_QUOTA_PER_REGIME.items():
        chosen.update(eligible_by_regime.get(regime, [])[:quota])

    if len(chosen) < MIN_INDEMNITY_BASIS_QUESTIONS:
        remaining = sorted(
            (
                doc_id
                for docs in eligible_by_regime.values()
                for doc_id in docs
                if doc_id not in chosen
            ),
            key=int,
        )
        for doc_id in remaining:
            if len(chosen) >= MIN_INDEMNITY_BASIS_QUESTIONS:
                break
            chosen.add(doc_id)
    return chosen


def pick_slots(
    candidates: list[ClauseCandidate],
    vocab_running_counts: dict[str, int],
    *,
    want_indemnity_basis: bool,
    limit: int = MAX_QUESTIONS_PER_DOC,
    already_used_ids: frozenset[str] = frozenset(),
) -> list[ClauseCandidate]:
    """Pick up to `limit` source clauses for one document.

    Only selectable candidates are eligible. Priority: one DEFINITION clause,
    then an indemnity-basis clause where the document was designated to carry
    one, then whichever clause serves the vocabulary term with the lowest
    running count so far, with assistance-tagged clauses deprioritised so a
    bundled section cannot dominate a document.
    """
    pool = [
        c
        for c in candidates
        if c.is_selectable and c.clause.clause_id not in already_used_ids
    ]
    chosen: list[ClauseCandidate] = []
    used_ids: set[str] = set(already_used_ids)

    def take(candidate: ClauseCandidate) -> None:
        chosen.append(candidate)
        used_ids.add(candidate.clause.clause_id)
        for term in candidate.vocab_hits:
            vocab_running_counts[term] = vocab_running_counts.get(term, 0) + 1

    definition = next(
        (
            c
            for c in pool
            if c.clause.clause_type == ClauseType.DEFINITION and c.scope == SCOPE_CASCO
        ),
        None,
    )
    if definition is not None and limit > 0:
        take(definition)

    if want_indemnity_basis:
        basis = next(
            (
                c
                for c in pool
                if c.is_indemnity_basis and c.clause.clause_id not in used_ids
            ),
            None,
        )
        if basis is not None and len(chosen) < limit:
            take(basis)

    def sort_key(candidate: ClauseCandidate) -> tuple[int, int, int, str]:
        if candidate.vocab_hits:
            lowest = min(
                vocab_running_counts.get(term, 0) for term in candidate.vocab_hits
            )
        else:
            lowest = len(VOCAB_PATTERNS)
        return (
            lowest,
            1 if candidate.scope == SCOPE_ASSISTANCE else 0,
            0 if candidate.is_core_type else 1,
            candidate.clause.clause_id,
        )

    while len(chosen) < limit:
        remaining = [c for c in pool if c.clause.clause_id not in used_ids]
        if not remaining:
            break
        take(min(remaining, key=sort_key))
    return chosen


def build_selection_audit_rows(
    candidates_by_doc: dict[str, list[ClauseCandidate]],
    selected_ids: frozenset[str],
) -> list[dict[str, str]]:
    """Flatten every considered clause into audit rows, selected or not."""
    rows: list[dict[str, str]] = []
    for doc_id in sorted(candidates_by_doc, key=int):
        for candidate in candidates_by_doc[doc_id]:
            rows.append(
                {
                    "document_id": doc_id,
                    "clause_id": candidate.clause.clause_id,
                    "title": candidate.clause.title,
                    "clause_type": candidate.clause.clause_type.value,
                    "scope": candidate.scope,
                    "selectable": "Y" if candidate.is_selectable else "",
                    "exclusion_reasons": ",".join(candidate.exclusion_reasons),
                    "vocabulary_terms_hit": ",".join(sorted(candidate.vocab_hits)),
                    "indemnity_basis_clause": "Y"
                    if candidate.is_indemnity_basis
                    else "",
                    "twin_clause_ids": ";".join(sorted(candidate.twin_ids)),
                    "selected": "Y"
                    if candidate.clause.clause_id in selected_ids
                    else "",
                }
            )
    return rows


SELECTION_AUDIT_FIELDNAMES = [
    "document_id",
    "clause_id",
    "title",
    "clause_type",
    "scope",
    "selectable",
    "exclusion_reasons",
    "vocabulary_terms_hit",
    "indemnity_basis_clause",
    "twin_clause_ids",
    "selected",
]
