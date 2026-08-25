#!/usr/bin/env python3
"""Draft synthetic claim narratives [M2-04].

The LLM phrasing layer of M2-04's three-layer authoring flow: scenario
selection (which document, which clause(s), which verdict) lives in
``scripts/synthetic_claims_selection.py`` (deterministic, no model) -- this
script's only job is to phrase a claim narrative for a scenario that module
has already fully resolved. Unlike the golden-question drafters, there is no
"completeness pass": ``reference_clause_ids`` is fixed by Layer 1, so the
model never chooses which clauses matter, only how a policyholder would
describe the event.

LLM: OpenRouter, ``google/gemini-3.7-flash``, pinned to the
``google-vertex/global`` provider route via the shared
[infrastructure.config.llm_client_factory.build_chat_model] -- single
provider, no fallback, matching ``draft_golden_questions_adversarial.py``'s
existing pin for the same model.

A regex PII safety net ([pii_safety_net.scan_narrative_for_pii]) runs over
every drafted narrative before it is written to the CSV; hits populate
``pii_flag`` for extra human review, never a silent drop.

Run ``PYTHONPATH=app/src uv run python scripts/draft_synthetic_claims.py``
(``--dry-run`` prints the Layer-1 slot counts against the DoD floors, no LLM
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
    import synthetic_claims_selection as selection
    from synthetic_claims_selection import MISSING_FACT_INSTRUCTIONS, ScenarioSlot
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import pii_safety_net
    from scripts import synthetic_claims_selection as selection
    from scripts.synthetic_claims_selection import (
        MISSING_FACT_INSTRUCTIONS,
        ScenarioSlot,
    )

MANIFEST_PATH = Path("data/policies/manifest.csv")
DRAFT_CSV_PATH = Path("eval/synthetic_claims_draft.csv")

DRAFT_MODEL = "google/gemini-3.7-flash"
DRAFT_PROVIDER_ORDER = ["google-vertex/global"]

SOURCE_TEXT_CAP = 2000
MAX_SLOTS_PER_CALL = 6

DOD_FLOOR_TOTAL = 30
DOD_FLOOR_INSUFFICIENT = 10


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
            "cita número de cláusula ou termos jurídicos. Contém pelo menos "
            "um detalhe irrelevante/tangencial e uma data imprecisa (nunca "
            "uma data exata DD/MM/AAAA). NUNCA contém nome de pessoa, "
            "CPF/RG, placa de veículo ou endereço específico."
        ),
    )
    draft_notes: str = Field(
        ...,
        description=(
            "Uma linha: que técnica(s) de ruído foram usadas e, apenas para "
            "insufficient_information, qual fato foi omitido."
        ),
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
    "Regras de autoria (M2-04):\n"
    "- Você é o SEGURADO, escrevendo em primeira pessoa para relatar um "
    "sinistro à sua seguradora -- não um analista, não um advogado. Registro "
    "informal, como alguém realmente escreveria: frases mais soltas, 'acho "
    "que', 'não lembro certo', hesitação.\n"
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


def build_scenario_instruction(slot: ScenarioSlot) -> str:
    """Return the per-scenario-type instruction block for one slot."""
    if slot.scenario_type == "compatible":
        return (
            "TIPO: compatible. O evento deve ser uma instância limpa do "
            f"risco coberto pela cláusula {slot.primary_clause_id}, sem "
            "nenhum detalhe que ative uma exclusão típica (embriaguez, uso "
            "em competição, condutor sem habilitação, uso comercial não "
            "informado, fora do prazo de vigência)."
        )
    if slot.scenario_type == "incompatible":
        trigger_id = slot.secondary_clause_id or slot.primary_clause_id
        return (
            "TIPO: incompatible. O relato deve mencionar, de forma natural "
            "e sem que o segurado pareça perceber a implicação, o fato "
            f"descrito na cláusula de exclusão {trigger_id} -- como um "
            "detalhe qualquer do relato, não uma confissão."
        )
    assert slot.scenario_type == "insufficient_information"
    assert slot.missing_fact_type is not None
    instruction = MISSING_FACT_INSTRUCTIONS[slot.missing_fact_type].format(
        clause_id=slot.primary_clause_id
    )
    return (
        f"TIPO: insufficient_information. {instruction} Omita esse fato SEM "
        "chamar atenção para a omissão -- não diga 'não vou informar X', "
        "apenas não mencione."
    )


def build_draft_prompt(
    *,
    product_line: str,
    group_label: str,
    slots: list[ScenarioSlot],
    by_id: dict[str, ParsedClauseRecord],
) -> str:
    """Build the drafting prompt for one group of slots (one document)."""
    slot_blocks: list[str] = []
    for slot in slots:
        clause_blocks = [format_clause_block(slot.primary_clause_id, by_id)]
        if slot.secondary_clause_id:
            clause_blocks.append(format_clause_block(slot.secondary_clause_id, by_id))
        slot_blocks.append(
            f"row_id: {slot.row_id}  (devolva este row_id exatamente)\n"
            f"{build_scenario_instruction(slot)}\n"
            "CLÁUSULA(S):\n" + "\n---\n".join(clause_blocks)
        )

    return (
        "Você redige relatos de sinistro (claim narratives) sintéticos para "
        "testar um sistema de triagem de sinistros de seguro auto "
        "brasileiro. Cada relato é o texto que um segurado leigo enviaria "
        "para abrir um sinistro -- não uma pergunta, não um resumo "
        "jurídico.\n\n"
        f"Linha de produto: {product_line}.\n"
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
    "scenario_type",
    "document_id",
    "insurer",
    "susep_process",
    "primary_clause_id",
    "primary_clause_title",
    "primary_clause_text",
    "secondary_clause_id",
    "secondary_clause_title",
    "secondary_clause_text",
    "missing_fact_type",
    "missing_fact_instruction",
    "narrative",
    "expected_verdict",
    "reference_clause_ids",
    "reference_clause_texts",
    "selection_notes",
    "draft_notes",
    "pii_flag",
    "authored_at",
    "provider_used",
    "review_verdict",
    "review_correction",
    "approved",
    "finalized_claim_id",
]


def sort_key_for_row_id(row_id: str) -> tuple[str, int, int]:
    """Order rows by scenario prefix, then document id, then slot number."""
    prefix, doc_part, slot_part = row_id.rsplit("-", 2)
    try:
        doc_key = int(doc_part)
    except ValueError:
        doc_key = 0
    return (prefix, doc_key, int(slot_part))


def write_csv(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    """Write the merged row set, sorted by (scenario prefix, document, slot)."""
    ordered = [rows_by_id[key] for key in sorted(rows_by_id, key=sort_key_for_row_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def load_existing_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read a draft CSV if it exists, indexed by row_id."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle)}


def reference_clause_ids_for(slot: ScenarioSlot) -> list[str]:
    """The exhaustive reference set, fixed by Layer 1 -- never the LLM's call."""
    ids = [slot.primary_clause_id]
    if slot.secondary_clause_id:
        ids.append(slot.secondary_clause_id)
    return ids


def _static_row(
    slot: ScenarioSlot, by_id: dict[str, ParsedClauseRecord]
) -> dict[str, str]:
    """Build the non-LLM CSV columns for one slot."""
    primary = by_id[slot.primary_clause_id]
    row = dict.fromkeys(CSV_FIELDNAMES, "")
    reference_ids = reference_clause_ids_for(slot)
    row.update(
        {
            "row_id": slot.row_id,
            "product_line": slot.product_line,
            "scenario_type": slot.scenario_type,
            "document_id": slot.document_id,
            "insurer": primary.insurer,
            "susep_process": primary.susep_process,
            "primary_clause_id": slot.primary_clause_id,
            "primary_clause_title": primary.title,
            "primary_clause_text": primary.text.strip()[:SOURCE_TEXT_CAP],
            "missing_fact_type": slot.missing_fact_type or "",
            # expected_verdict is forced structurally from scenario_type --
            # the two use the identical vocabulary by design, never the LLM's.
            "expected_verdict": slot.scenario_type,
            "reference_clause_ids": ";".join(reference_ids),
            "reference_clause_texts": " || ".join(
                f"[{cid}] {by_id[cid].title}: {by_id[cid].text.strip()[:200]}"
                for cid in reference_ids
                if cid in by_id
            ),
            "selection_notes": slot.selection_notes,
        }
    )
    if slot.secondary_clause_id and slot.secondary_clause_id in by_id:
        secondary = by_id[slot.secondary_clause_id]
        row.update(
            {
                "secondary_clause_id": slot.secondary_clause_id,
                "secondary_clause_title": secondary.title,
                "secondary_clause_text": secondary.text.strip()[:SOURCE_TEXT_CAP],
            }
        )
    if slot.missing_fact_type:
        row["missing_fact_instruction"] = MISSING_FACT_INSTRUCTIONS[
            slot.missing_fact_type
        ].format(clause_id=slot.primary_clause_id)
    return row


def group_slots_by_document(slots: list[ScenarioSlot]) -> dict[str, list[ScenarioSlot]]:
    """Group slots into per-document batches capped at MAX_SLOTS_PER_CALL."""
    by_document: dict[str, list[ScenarioSlot]] = defaultdict(list)
    for slot in slots:
        by_document[slot.document_id].append(slot)

    groups: dict[str, list[ScenarioSlot]] = {}
    for document_id, document_slots in by_document.items():
        for chunk_index in range(0, len(document_slots), MAX_SLOTS_PER_CALL):
            chunk = document_slots[chunk_index : chunk_index + MAX_SLOTS_PER_CALL]
            key = f"{document_id}-{chunk_index // MAX_SLOTS_PER_CALL}"
            groups[key] = chunk
    return groups


# --- reporting --------------------------------------------------------------


def print_coverage_report(rows: list[dict[str, str]]) -> bool:
    """Print the DoD tally by scenario_type. Returns True if it passes."""
    by_type: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[row["scenario_type"]] += 1

    total = len(rows)
    insufficient = by_type.get("insufficient_information", 0)
    print(f"\nTotal drafted rows: {total} (DoD floor: {DOD_FLOOR_TOTAL})")
    for scenario_type in ("compatible", "incompatible", "insufficient_information"):
        print(f"  {scenario_type:26s} {by_type.get(scenario_type, 0)}")
    passed = total >= DOD_FLOOR_TOTAL and insufficient >= DOD_FLOOR_INSUFFICIENT
    if not passed:
        print(
            f"  <-- BELOW DoD FLOOR (need >= {DOD_FLOOR_TOTAL} total, "
            f">= {DOD_FLOOR_INSUFFICIENT} insufficient_information)"
        )
    return passed


def print_slot_counts(slots: list[ScenarioSlot]) -> None:
    """Print per-(product_line, scenario_type) counts, for --dry-run."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for slot in slots:
        counts[(slot.product_line, slot.scenario_type)] += 1
    for product_line in selection.TARGET_COUNTS:
        for scenario_type in ("compatible", "incompatible", "insufficient_information"):
            count = counts.get((product_line, scenario_type), 0)
            print(f"  {product_line:14s} {scenario_type:26s} {count}")


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
    slots = selection.select_all_slots(records, MANIFEST_PATH)

    if args.dry_run:
        print(
            f"\n--dry-run: {len(slots)} candidate slot(s) selected. "
            "No LLM calls, no CSV written."
        )
        print_slot_counts(slots)
        insufficient = sum(
            1 for slot in slots if slot.scenario_type == "insufficient_information"
        )
        print(
            f"\nTotal: {len(slots)} (floor {DOD_FLOOR_TOTAL}), "
            f"insufficient_information: {insufficient} "
            f"(floor {DOD_FLOOR_INSUFFICIENT})"
        )
        return

    rows: dict[str, dict[str, str]] = {}
    groups = group_slots_by_document(slots)

    for group_key, group_slots in sorted(groups.items()):
        product_line = group_slots[0].product_line
        print(f"Drafting {len(group_slots)} claim(s) for group {group_key}...")
        prompt = build_draft_prompt(
            product_line=product_line,
            group_label=group_key,
            slots=group_slots,
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
