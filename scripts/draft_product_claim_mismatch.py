#!/usr/bin/env python3
"""Draft product/claim mismatch narratives [M2-05].

The LLM phrasing layer of M2-05's three-layer authoring flow: scenario
selection (which document, which anchor clause) lives in
``scripts/product_claim_mismatch_selection.py`` (deterministic, no model) --
this script's only job is to phrase a claim narrative for a scenario that
module has already fully resolved. Every scenario's ``expected_verdict`` is
``incompatible`` by construction (an own-vehicle-damage claim against a
document that structurally cannot cover it), so unlike
``draft_synthetic_claims.py`` there is only one narrative shape to draft: a
policyholder describing everyday damage to their OWN vehicle, submitted
against a product that was never meant to answer it. The model never sees
or decides the verdict, and ``reference_clause_ids`` is fixed to the
Layer-1 anchor clause -- both fixed structurally, never the model's call.

LLM: OpenRouter, ``google/gemini-3.7-flash``, pinned to the
``google-vertex/global`` provider route via the shared
[infrastructure.config.llm_client_factory.build_chat_model] -- single
provider, no fallback, matching every other M2 drafting script's pin for
the same model.

A regex PII safety net ([pii_safety_net.scan_narrative_for_pii]) runs over
every drafted narrative before it is written to the CSV; hits populate
``pii_flag`` for extra human review, never a silent drop.

Run ``PYTHONPATH=app/src uv run python scripts/draft_product_claim_mismatch.py``
(``--dry-run`` prints the Layer-1 slot counts against the DoD floor, no LLM
calls, no CSV written).
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl

try:
    # Direct execution: the script's own directory is sys.path[0].
    import pii_safety_net
    import product_claim_mismatch_selection as selection
    from product_claim_mismatch_selection import MismatchSlot
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import pii_safety_net
    from scripts import product_claim_mismatch_selection as selection
    from scripts.product_claim_mismatch_selection import MismatchSlot

MANIFEST_PATH = Path("data/policies/manifest.csv")
DRAFT_CSV_PATH = Path("eval/product_claim_mismatch_draft.csv")

DRAFT_MODEL = "google/gemini-3.7-flash"
DRAFT_PROVIDER_ORDER = ["google-vertex/global"]

SOURCE_TEXT_CAP = 2000
MAX_SLOTS_PER_CALL = 6

DOD_FLOOR_TOTAL = 8

# Everyday own-vehicle perils to vary across drafted narratives, so the set
# doesn't read as eleven copies of the same collision story.
PERILS = (
    "colisão com outro veículo",
    "furto do veículo",
    "roubo do veículo",
    "incêndio no veículo",
    "quebra do para-brisa/vidro",
    "colisão contra um poste ou muro",
    "granizo que danificou a lataria",
    "alagamento que atingiu o carro",
)


class DraftedClaim(BaseModel):
    """One drafted claim narrative, anchored on a resolved scenario slot."""

    row_id: str = Field(
        ...,
        description="The row_id of the slot this narrative answers, copied verbatim.",
    )
    narrative: str = Field(
        ...,
        description=(
            "Relato do sinistro em português do Brasil, primeira pessoa, "
            "registro informal, como um segurado leigo escreveria -- nunca "
            "cita número de cláusula ou termos jurídicos, e nunca menciona "
            "que o dano foi a terceiros. Descreve dano ao PRÓPRIO veículo "
            "do segurado. Contém pelo menos um detalhe irrelevante/"
            "tangencial e uma data imprecisa (nunca uma data exata "
            "DD/MM/AAAA). NUNCA contém nome de pessoa, CPF/RG, placa de "
            "veículo ou endereço específico."
        ),
    )
    draft_notes: str = Field(
        ..., description="Uma linha: qual sinistro foi descrito e que ruído foi usado."
    )


class DraftedClaimsBatch(BaseModel):
    """A batch of drafted narratives for one group of slots."""

    claims: list[DraftedClaim]


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the built corpus, failing loudly if `make parse` hasn't run yet."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


def call_llm(prompt: str, output_model: type[BaseModel]) -> BaseModel:
    """Invoke a structured-output chain against the pinned OpenRouter/Gemini model.

    Single provider, no fallback. Retries transient failures; any other
    failure re-raises, since there is no sane fallback for a drafting
    judgment.
    """
    llm = build_chat_model(
        get_llm_settings(),
        DRAFT_MODEL,
        provider_order=DRAFT_PROVIDER_ORDER,
        allow_fallbacks=False,
    )
    chain = cast(Runnable[str, BaseModel], llm.with_structured_output(output_model))
    last_exc: Exception | None = None
    for attempt in range(1, DEFAULT_LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            return chain.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last_exc = exc
            if attempt < DEFAULT_LLM_RETRY_MAX_ATTEMPTS:
                time.sleep(DEFAULT_LLM_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


# --- prompts ------------------------------------------------------------


def format_clause_block(clause_id: str, by_id: dict[str, ParsedClauseRecord]) -> str:
    """Render one clause as a labeled block for a prompt."""
    record = by_id[clause_id]
    text = record.text.strip()[:SOURCE_TEXT_CAP]
    return (
        f"[{clause_id}]\n"
        f"título: {record.title}\n"
        f"tipo: {record.clause_type.value}\n"
        f"texto: {text}"
    )


_SHARED_RULES = (
    "Regras de autoria (M2-05):\n"
    "- Você é o SEGURADO, escrevendo em primeira pessoa para relatar um "
    "sinistro à sua seguradora -- não um analista, não um advogado. Registro "
    "informal, como alguém realmente escreveria: frases mais soltas, 'acho "
    "que', 'não lembro certo', hesitação.\n"
    "- O relato descreve dano ao PRÓPRIO veículo do segurado (colisão, "
    "furto, roubo, incêndio, quebra de vidro, granizo, alagamento) -- NUNCA "
    "dano causado pelo segurado a terceiros, e nunca menciona assistência "
    "24h, guincho ou reboque como o motivo do acionamento.\n"
    "- A data do evento é sempre IMPRECISA: 'faz umas duas semanas', 'no fim "
    "do mês passado', 'não anotei o dia certo' -- NUNCA uma data exata.\n"
    "- Inclua pelo menos um detalhe irrelevante/tangencial (o trânsito, o "
    "tempo, uma reclamação sobre a espera no telefone, um comentário sobre a "
    "rotina do dia) -- como um relato real, não um resumo limpo dos fatos.\n"
    "- PROIBIDO, mesmo fictício: nome de pessoa (refira-se a 'eu', 'minha "
    "esposa', 'o outro motorista'), CPF/RG/qualquer número de documento, "
    "placa de veículo (refira-se a 'meu carro', 'o carro branco'), endereço "
    "específico (cidade tudo bem; rua e número, não).\n"
    "- Nunca cite número de cláusula, 'esta apólice', 'conforme o contrato' "
    "-- um segurado real não fala assim.\n"
    "- Use o row_id exatamente como informado."
)


def build_draft_prompt(
    *,
    product_line: str,
    group_label: str,
    slots: list[MismatchSlot],
    peril_by_row_id: dict[str, str],
    by_id: dict[str, ParsedClauseRecord],
) -> str:
    """Build the drafting prompt for one group of slots (one document)."""
    slot_blocks: list[str] = []
    for slot in slots:
        peril = peril_by_row_id[slot.row_id]
        slot_blocks.append(
            f"row_id: {slot.row_id}  (devolva este row_id exatamente)\n"
            f"SINISTRO A DESCREVER: {peril}.\n"
            "CLÁUSULA DE ESCOPO DO PRODUTO (contexto de curadoria -- NUNCA "
            "cite ou parafraseie esta cláusula no relato; ela só existe para "
            "você entender que produto é este, não para aparecer no "
            "texto):\n" + format_clause_block(slot.primary_clause_id, by_id)
        )

    return (
        "Você redige relatos de sinistro (claim narratives) sintéticos para "
        "testar um sistema de triagem de sinistros de seguro auto "
        "brasileiro. Cada relato é o texto que um segurado leigo enviaria "
        "para abrir um sinistro -- não uma pergunta, não um resumo "
        "jurídico. Este lote testa especificamente o caso em que um "
        "segurado relata dano ao PRÓPRIO veículo contra uma apólice que "
        "não é de seguro de danos ao veículo (é responsabilidade civil, "
        "assistência, garantia estendida ou carta verde) -- o relato em si "
        "deve soar como um sinistro de danos ao veículo perfeitamente "
        "normal, sem qualquer sinal de que o segurado sabe que está no "
        "produto errado.\n\n"
        f"Linha de produto (do documento, NÃO mencionar no relato): "
        f"{product_line}.\n"
        f"Grupo: {group_label}.\n\n"
        f"{_SHARED_RULES}\n\n"
        f"CENÁRIOS ({len(slots)} relato(s), um por row_id):\n\n"
        + "\n\n".join(slot_blocks)
        + f"\n\nRetorne exatamente {len(slots)} relato(s), um por row_id."
    )


# --- CSV ------------------------------------------------------------------

CSV_FIELDNAMES = [
    "row_id",
    "product_line",
    "document_id",
    "insurer",
    "susep_process",
    "primary_clause_id",
    "primary_clause_title",
    "primary_clause_text",
    "narrative",
    "expected_verdict",
    "reference_clause_ids",
    "reference_clause_texts",
    "selection_notes",
    "draft_notes",
    "pii_flag",
    "authored_at",
    "review_verdict",
    "review_correction",
    "approved",
    "finalized_claim_id",
]


def sort_key_for_row_id(row_id: str) -> tuple[int, int]:
    """Order rows by document id, then slot number."""
    _, doc_part, slot_part = row_id.split("-", 2)
    return (int(doc_part), int(slot_part))


def write_csv(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    """Write the merged row set, sorted by (document, slot)."""
    ordered = [rows_by_id[key] for key in sorted(rows_by_id, key=sort_key_for_row_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def _static_row(
    slot: MismatchSlot, by_id: dict[str, ParsedClauseRecord]
) -> dict[str, str]:
    """Build the non-LLM CSV columns for one slot."""
    primary = by_id[slot.primary_clause_id]
    row = dict.fromkeys(CSV_FIELDNAMES, "")
    row.update(
        {
            "row_id": slot.row_id,
            "product_line": slot.product_line,
            "document_id": slot.document_id,
            "insurer": primary.insurer,
            "susep_process": primary.susep_process,
            "primary_clause_id": slot.primary_clause_id,
            "primary_clause_title": primary.title,
            "primary_clause_text": primary.text.strip()[:SOURCE_TEXT_CAP],
            # expected_verdict is forced structurally -- every mismatch
            # scenario is incompatible by construction, never the LLM's call.
            "expected_verdict": "incompatible",
            "reference_clause_ids": slot.primary_clause_id,
            "reference_clause_texts": (
                f"[{slot.primary_clause_id}] {primary.title}: "
                f"{primary.text.strip()[:200]}"
            ),
            "selection_notes": slot.selection_notes,
        }
    )
    return row


def group_slots_by_document(slots: list[MismatchSlot]) -> dict[str, list[MismatchSlot]]:
    """Group slots into per-document batches capped at MAX_SLOTS_PER_CALL."""
    by_document: dict[str, list[MismatchSlot]] = defaultdict(list)
    for slot in slots:
        by_document[slot.document_id].append(slot)

    groups: dict[str, list[MismatchSlot]] = {}
    for document_id, document_slots in by_document.items():
        for chunk_index in range(0, len(document_slots), MAX_SLOTS_PER_CALL):
            chunk = document_slots[chunk_index : chunk_index + MAX_SLOTS_PER_CALL]
            key = f"{document_id}-{chunk_index // MAX_SLOTS_PER_CALL}"
            groups[key] = chunk
    return groups


# --- reporting --------------------------------------------------------------


def print_coverage_report(rows: list[dict[str, str]]) -> bool:
    """Print the DoD tally. Returns True if it passes."""
    total = len(rows)
    print(f"\nTotal drafted rows: {total} (DoD floor: {DOD_FLOOR_TOTAL})")
    by_line: dict[str, int] = defaultdict(int)
    for row in rows:
        by_line[row["product_line"]] += 1
    for product_line in selection.TARGET_COUNTS:
        print(f"  {product_line:14s} {by_line.get(product_line, 0)}")
    passed = total >= DOD_FLOOR_TOTAL
    if not passed:
        print(f"  <-- BELOW DoD FLOOR (need >= {DOD_FLOOR_TOTAL} total)")
    return passed


def print_slot_counts(slots: list[MismatchSlot]) -> None:
    """Print per-product-line counts, for --dry-run."""
    counts: dict[str, int] = defaultdict(int)
    for slot in slots:
        counts[slot.product_line] += 1
    for product_line in selection.TARGET_COUNTS:
        print(f"  {product_line:14s} {counts.get(product_line, 0)}")


# --- main -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection and slot counts; never call an LLM.",
    )
    return parser.parse_args()


def main() -> None:
    """Select scenarios and draft their narratives."""
    load_dotenv()
    args = _parse_args()

    records = load_corpus()
    by_id = {record.clause_id: record for record in records}
    slots = selection.select_mismatch_slots(records, MANIFEST_PATH)

    if args.dry_run:
        print(
            f"\n--dry-run: {len(slots)} candidate slot(s) selected. "
            "No LLM calls, no CSV written."
        )
        print_slot_counts(slots)
        print(f"\nTotal: {len(slots)} (floor {DOD_FLOOR_TOTAL})")
        return

    rows: dict[str, dict[str, str]] = {}
    groups = group_slots_by_document(slots)
    # Cycled globally across every slot (not per-group): most documents
    # contribute exactly one slot each, so a per-group cycle would restart
    # at index 0 every time and every narrative would describe the same
    # peril.
    peril_by_row_id = {
        slot.row_id: PERILS[index % len(PERILS)] for index, slot in enumerate(slots)
    }

    for group_key, group_slots in sorted(groups.items()):
        product_line = group_slots[0].product_line
        print(f"Drafting {len(group_slots)} claim(s) for group {group_key}...")
        prompt = build_draft_prompt(
            product_line=product_line,
            group_label=group_key,
            slots=group_slots,
            peril_by_row_id=peril_by_row_id,
            by_id=by_id,
        )
        drafted = cast(DraftedClaimsBatch, call_llm(prompt, DraftedClaimsBatch))
        drafted_by_row = {claim.row_id: claim for claim in drafted.claims}

        for slot in group_slots:
            row = _static_row(slot, by_id)
            claim = drafted_by_row.get(slot.row_id)
            if claim is None:
                print(f"  WARNING: model returned no narrative for {slot.row_id}")
                rows[slot.row_id] = row
                continue
            pii_hits = pii_safety_net.scan_narrative_for_pii(claim.narrative)
            row.update(
                {
                    "narrative": claim.narrative,
                    "draft_notes": claim.draft_notes,
                    "pii_flag": ";".join(pii_hits),
                }
            )
            rows[slot.row_id] = row

    write_csv(DRAFT_CSV_PATH, rows)
    print(f"\nWrote {len(rows)} row(s) to {DRAFT_CSV_PATH}")
    print_coverage_report(list(rows.values()))
    pii_hits_total = sum(1 for row in rows.values() if row["pii_flag"])
    if pii_hits_total:
        print(
            f"\nWARNING: {pii_hits_total} row(s) flagged by the PII safety net "
            "-- review the 'pii_flag' column before approving."
        )


if __name__ == "__main__":
    main()
