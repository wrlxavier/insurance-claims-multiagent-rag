#!/usr/bin/env python3
"""Deterministic candidate selection for the unanswerable question set [M2-05].

Layer 1 of the same three-layer authoring flow used across M2 (see
``docs/EVALUATION.md``), but shaped differently from every other question
type: there is no source clause to select, because the entire point of an
``unanswerable`` question is that nothing answers it. This module's job is
instead to run a **deterministic textual search** over
``build/parsed_clauses.jsonl`` for each ``(document, fact_type)`` candidate
and record what it finds -- this is the search *tool*, never the
confirmation itself. ``docs/EVALUATION.md``'s own rule (from M2-01) is that
absence must be confirmed by the author's own direct textual search, never
by asking an LLM to prove a negative; this module is what makes that search
direct and reproducible instead of an unrecorded manual grep, and the
author's review of its output in the draft CSV is what actually confirms
absence, exactly as [find_candidate_clauses] never decides relevance on its
own.

Five fact types the corpus structurally cannot answer for any document, per
the project's own scope statement (``docs/SCOPE.md``/M0-06): these are
registered product conditions, not individual policy certificates, so none
of them carries a deductible amount, an insured sum, a premium, a concrete
policy period, or an endorsement number for a specific insured. Measured
against the real corpus: ``premium``, ``policy_period`` and ``endorsement``
return zero hits across all 30 documents; ``deductible`` and
``insured_amount`` return a handful of hits, but every one of them is a
sub-coverage-specific limit, a floor/minimum, or a discount -- never the
policy-level value a question would ask for -- which is exactly the
"similar number a careless system would return instead" case the DoD's
decoy bullet asks for. Those specific hits are hardcoded in
[MANUAL_DECOY_SPECS], found by reading the matched clauses directly (the
same "found by reading the source documents directly" precedent
``adversarial_clause_selection.py``'s ``MANUAL_NEAR_DUPLICATE_PAIRS``
already sets), not by an automated relevance judgment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.manifest import read_manifest

MANIFEST_PATH = Path("data/policies/manifest.csv")

FACT_TYPES: tuple[str, ...] = (
    "deductible",
    "insured_amount",
    "premium",
    "policy_period",
    "endorsement",
)

# DoD floor: >=15 clean_absent + >=3 decoy. These targets clear both with
# margin: 4 clean_absent per fact type (20 total) plus every verified decoy
# in MANUAL_DECOY_SPECS (3).
CLEAN_ABSENT_TARGET_PER_FACT_TYPE = 4

FACT_TYPE_LABELS_PT: dict[str, str] = {
    "deductible": "franquia",
    "insured_amount": "importância segurada / limite máximo de indenização",
    "premium": "prêmio",
    "policy_period": "vigência (datas específicas desta apólice)",
    "endorsement": "endosso",
}

# Why the value is absent BY CONSTRUCTION -- not a parsing gap, a property
# of the corpus itself (docs/SCOPE.md/M0-06): every document here is a
# registered product's general conditions, never an individual policy
# certificate, so none of these five facts is ever fixed to a specific
# insured. Pre-populated into the draft CSV's `notes` column as the DoD's
# required "why absent" record, for the author to confirm or amend.
ABSENCE_REASON_PT: dict[str, str] = {
    "deductible": (
        "Este documento é uma condição geral de produto registrado na "
        "SUSEP, não uma apólice individual -- não fixa o valor da franquia "
        "para um segurado específico, apenas (quando presente) a fórmula "
        "de cálculo."
    ),
    "insured_amount": (
        "A importância segurada é individual de cada apólice (depende do "
        "veículo segurado) e nunca aparece fixada nesta condição geral de "
        "produto registrado."
    ),
    "premium": (
        "O prêmio é calculado por apólice (perfil do segurado, veículo, "
        "praça) e nunca é fixado nesta condição geral de produto "
        "registrado."
    ),
    "policy_period": (
        "Este documento define regras gerais sobre vigência, nunca as "
        "datas de início/fim de uma apólice específica -- essas datas só "
        "existem no bilhete/apólice individual, que não faz parte deste "
        "corpus."
    ),
    "endorsement": (
        "Endossos são alterações registradas em uma apólice individual "
        "específica; este documento é a condição geral do produto e não "
        "referencia nenhum endosso concreto."
    ),
}

# Currency/date/number patterns tied to a CONCRETE value for the fact type,
# not the mere existence of a rule about it -- e.g. "franquia" alone is
# expected everywhere (every product's conditions discuss the deductible
# concept); what must be absent is a policy-level number.
ABSENCE_SEARCH_PATTERNS: dict[str, re.Pattern[str]] = {
    "deductible": re.compile(
        r"franquia[^.;]{0,60}?R\$\s?[\d.,]+|R\$\s?[\d.,]+[^.;]{0,60}?franquia",
        re.IGNORECASE,
    ),
    "insured_amount": re.compile(
        r"(import[âa]ncia segurada|valor segurado|"
        r"limite m[áa]ximo de indeniza[çc][ãa]o)[^.;]{0,60}?R\$\s?[\d.,]+",
        re.IGNORECASE,
    ),
    "premium": re.compile(r"pr[êe]mio[^.;]{0,60}?R\$\s?[\d.,]+", re.IGNORECASE),
    "policy_period": re.compile(
        r"vig[êe]ncia[^.;]{0,80}?\d{2}/\d{2}/\d{4}", re.IGNORECASE
    ),
    "endorsement": re.compile(r"endosso\s+n[ºo°]?\.?\s*\d+", re.IGNORECASE),
}


@dataclass(frozen=True)
class DecoySpec:
    """One hand-verified "similar number" case.

    A hit that looks relevant but does not answer the general policy-level
    question. ``why_it_does_not_answer`` is the author's own textual-search
    finding, written out so the draft CSV's ``notes`` column carries the
    reasoning rather than just a clause id.
    """

    document_id: str
    fact_type: str
    decoy_clause_id: str
    why_it_does_not_answer: str


# Every entry verified against build/parsed_clauses.jsonl by reading the
# full matched clause, not just the regex hit substring (the hit substring
# alone was misleading for document 12: the bare match "Franquia será de
# R$ 285,00" turned out, in full context, to be the monetary limit of the
# "Reparo Abaixo da Franquia" ancillary benefit, not a franchise value at
# all).
MANUAL_DECOY_SPECS: tuple[DecoySpec, ...] = (
    DecoySpec(
        document_id="1",
        fact_type="insured_amount",
        decoy_clause_id=(
            "1:condicoes-contratuais-gerais-para-as-coberturas-de-automovel-"
            "rcfv-disposicoes-preliminares/17"
        ),
        why_it_does_not_answer=(
            "R$ 1.000,00 é apenas o PISO mínimo do Limite Máximo de "
            "Indenização contratado (usado, nesta mesma cláusula, para "
            "calcular a franquia como 8% desse limite) -- não é a "
            "importância segurada do veículo deste segurado, que esta "
            "condição geral nunca fixa."
        ),
    ),
    DecoySpec(
        document_id="12",
        fact_type="deductible",
        decoy_clause_id="12:assitencia-a-vidros-blindados/7/7.2",
        why_it_does_not_answer=(
            "R$ 285,00 é o limite monetário do benefício acessório 'Reparo "
            "Abaixo da Franquia' (quando o serviço é feito em oficina não "
            "referenciada) -- não é o valor da franquia cobrada para a "
            "substituição de vidros, faróis, lanternas ou retrovisores, que "
            "esta cláusula só menciona de forma genérica ('as franquias "
            "correspondentes'), sem informar o valor."
        ),
    ),
    DecoySpec(
        document_id="14",
        fact_type="deductible",
        decoy_clause_id="14:beneficios-e-descontos",
        why_it_does_not_answer=(
            "R$ 400,00 é um DESCONTO aplicado sobre a franquia, não o valor "
            "final da franquia em si -- a cláusula nunca informa esse valor "
            "resultante."
        ),
    ),
)


@dataclass(frozen=True)
class UnansweredSlot:
    """One candidate unanswerable-question scenario for the drafting script."""

    row_id: str
    document_id: str
    fact_type: str
    slot_kind: str  # "clean_absent" | "decoy"
    search_evidence: str
    decoy_clause_id: str | None = None
    decoy_snippet: str | None = None


def load_all_documents_by_line(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, list[dict[str, str]]]:
    """Return every manifest row grouped by product_line, each group id-sorted."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_manifest(manifest_path):
        grouped[row["product_line"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["id"]))
    return grouped


def _round_robin_all_documents(
    grouped: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Return every document, ordered by round-robin across product lines.

    Spreads picks across product lines instead of exhausting one line's
    documents before moving to the next -- the same shape as
    ``synthetic_claims_selection._round_robin_take``, generalized to a full
    ordering rather than a capped take.
    """
    order = list(grouped.keys())
    ordered: list[dict[str, str]] = []
    index = 0
    while True:
        progressed = False
        for line in order:
            docs = grouped[line]
            if index < len(docs):
                ordered.append(docs[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return ordered


SNIPPET_CONTEXT_CHARS = 300


def search_document_for_fact(
    records: list[ParsedClauseRecord], document_id: str, fact_type: str
) -> list[tuple[str, str]]:
    """Return every (clause_id, snippet) hit for one (document, fact_type).

    The snippet is a window centered on the match ([SNIPPET_CONTEXT_CHARS]
    on each side), not just the bare matched substring -- a long clause's
    surrounding context is what tells a human (or a decoy prompt) what the
    matched number actually refers to. Truncating from the clause's start
    instead of around the match is exactly the bug that shipped a decoy
    whose displayed text never reached its own cited value (documents 12
    and 13's real matches sit at characters ~1350 and ~10,800 of clauses
    1,703 and 11,655 characters long).

    Purely structural regex search -- no LLM, no relevance judgment. An
    empty result is the evidence a `clean_absent` slot cites; a non-empty
    result means the document is not safe to use for that fact type
    (unless it is one of the hand-verified [MANUAL_DECOY_SPECS] cases).
    """
    pattern = ABSENCE_SEARCH_PATTERNS[fact_type]
    hits: list[tuple[str, str]] = []
    for record in records:
        if record.document_id != document_id:
            continue
        match = pattern.search(record.text)
        if match:
            start = max(0, match.start() - SNIPPET_CONTEXT_CHARS)
            end = min(len(record.text), match.end() + SNIPPET_CONTEXT_CHARS)
            snippet = record.text[start:end].strip()
            hits.append((record.clause_id, snippet))
    return hits


def select_clean_absent_slots(
    records: list[ParsedClauseRecord],
    manifest_path: Path = MANIFEST_PATH,
    *,
    target_per_fact_type: int = CLEAN_ABSENT_TARGET_PER_FACT_TYPE,
) -> list[UnansweredSlot]:
    """Select documents with zero search hits for each fact type, spread by line.

    Prefers a document not already used by an earlier fact type in this
    same call, falling back to reuse only if the corpus can't supply enough
    unused ones. Without this preference, every fact type's round-robin
    search lands on the same first few documents -- three of the five fact
    types (premium, policy_period, endorsement) have zero hits anywhere in
    the corpus, so nothing else would disambiguate them, and 20 slots would
    concentrate on 4-5 documents instead of a broad, varied sample.
    """
    grouped = load_all_documents_by_line(manifest_path)
    documents = _round_robin_all_documents(grouped)

    slots: list[UnansweredSlot] = []
    used_document_ids: set[str] = set()
    for fact_type in FACT_TYPES:
        chosen: list[str] = []
        for prefer_unused in (True, False):
            if len(chosen) >= target_per_fact_type:
                break
            for document in documents:
                if len(chosen) >= target_per_fact_type:
                    break
                document_id = document["id"]
                if document_id in chosen:
                    continue
                if prefer_unused and document_id in used_document_ids:
                    continue
                if search_document_for_fact(records, document_id, fact_type):
                    continue
                chosen.append(document_id)

        for document_id in chosen:
            used_document_ids.add(document_id)
            slots.append(
                UnansweredSlot(
                    row_id=f"unans-{fact_type}-{document_id}",
                    document_id=document_id,
                    fact_type=fact_type,
                    slot_kind="clean_absent",
                    search_evidence=(
                        f"Busca textual por padrão de {FACT_TYPE_LABELS_PT[fact_type]} "
                        f"sobre o documento {document_id}: 0 ocorrências. "
                        f"{ABSENCE_REASON_PT[fact_type]}"
                    ),
                )
            )
    return slots


def select_decoy_slots(
    records: list[ParsedClauseRecord],
    *,
    decoy_specs: tuple[DecoySpec, ...] = MANUAL_DECOY_SPECS,
) -> list[UnansweredSlot]:
    """Build decoy slots from [MANUAL_DECOY_SPECS], re-verifying each hit still fires.

    Re-running the structural search (rather than trusting the hardcoded
    snippet) means a corpus re-parse that changes the clause's text is
    caught here instead of silently shipping a stale decoy.
    """
    slots: list[UnansweredSlot] = []
    for spec in decoy_specs:
        hits = search_document_for_fact(records, spec.document_id, spec.fact_type)
        hit_by_id = dict(hits)
        snippet = hit_by_id.get(spec.decoy_clause_id)
        if snippet is None:
            continue
        slots.append(
            UnansweredSlot(
                row_id=f"unans-{spec.fact_type}-{spec.document_id}-decoy",
                document_id=spec.document_id,
                fact_type=spec.fact_type,
                slot_kind="decoy",
                search_evidence=(
                    f"Busca textual encontrou um valor numérico relacionado em "
                    f"{spec.decoy_clause_id!r} ({snippet!r}), mas ele NÃO responde "
                    f"à pergunta: {spec.why_it_does_not_answer}"
                ),
                decoy_clause_id=spec.decoy_clause_id,
                decoy_snippet=snippet,
            )
        )
    return slots


def select_unanswerable_slots(
    records: list[ParsedClauseRecord],
    manifest_path: Path = MANIFEST_PATH,
    *,
    target_per_fact_type: int = CLEAN_ABSENT_TARGET_PER_FACT_TYPE,
) -> list[UnansweredSlot]:
    """Select every unanswerable-question candidate slot, clean and decoy."""
    clean = select_clean_absent_slots(
        records, manifest_path, target_per_fact_type=target_per_fact_type
    )
    decoy = select_decoy_slots(records)
    return clean + decoy
