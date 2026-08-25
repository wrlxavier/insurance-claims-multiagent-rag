#!/usr/bin/env python3
"""Draft unanswerable golden-set questions [M2-05].

The LLM phrasing layer of M2-05's three-layer authoring flow for the
``unanswerable`` question_type: candidate selection (which document, which
absent fact type, and the deterministic textual-search evidence for its
absence) lives in ``scripts/unanswerable_question_selection.py``
(deterministic, no model) -- this script's only job is to phrase a
question asking for that fact, the way a claims analyst would ask it. The
model never confirms or judges absence (``docs/EVALUATION.md``'s own rule:
an LLM cannot reliably prove a negative), and it never sees a source
clause to answer from -- ``question_type``, ``reference_clause_ids``
(always empty) and ``expected_verdict`` are all forced structurally, never
the model's call.

For ``decoy`` slots, the model is shown the near-miss clause the selection
layer found and instructed to phrase a question specific enough that this
clause does not actually answer it -- the same "distractor must not answer
the question" framing ``draft_golden_questions_adversarial.py`` already
uses for its ``cross_document``/``hdi_brand_collision`` slots.

LLM: OpenRouter, ``google/gemini-3.7-flash``, pinned to the
``google-vertex/global`` provider route via the shared
[infrastructure.config.llm_client_factory.build_chat_model] -- single
provider, no fallback, matching every other M2 drafting script's pin for
the same model.

Run ``PYTHONPATH=app/src uv run python scripts/draft_unanswerable_questions.py``
(``--dry-run`` prints the Layer-1 slot counts against the DoD floors, no
LLM calls, no CSV written).
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
from infrastructure.evaluation.golden_set_schema import Difficulty
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

try:
    # Direct execution: the script's own directory is sys.path[0].
    import unanswerable_question_selection as selection
    from unanswerable_question_selection import (
        FACT_TYPE_LABELS_PT,
        UnansweredSlot,
    )
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import unanswerable_question_selection as selection
    from scripts.unanswerable_question_selection import (
        FACT_TYPE_LABELS_PT,
        UnansweredSlot,
    )

MANIFEST_PATH = Path("data/policies/manifest.csv")
DRAFT_CSV_PATH = Path("eval/unanswerable_draft.csv")

DRAFT_MODEL = "google/gemini-3.7-flash"
DRAFT_PROVIDER_ORDER = ["google-vertex/global"]

MAX_SLOTS_PER_CALL = 8

DOD_FLOOR_CLEAN_ABSENT = 15
DOD_FLOOR_DECOY = 3


class DraftedQuestion(BaseModel):
    """One drafted unanswerable question, anchored on a resolved slot."""

    row_id: str = Field(
        ...,
        description="The row_id of the slot this question answers, copied verbatim.",
    )
    question: str = Field(
        ...,
        description=(
            "A pergunta em português do Brasil, como um analista de "
            "sinistros a faria, autossuficiente e nunca ecoando o texto "
            "de uma cláusula. Pede o valor/data/número concreto do fato, "
            "nunca pergunta 'esta informação existe no documento?'."
        ),
    )
    difficulty: Difficulty
    draft_notes: str = Field(
        ..., description="Uma linha: o que a pergunta pede e por que não é respondível."
    )


class DraftedQuestionsBatch(BaseModel):
    """A batch of drafted questions for one group of slots."""

    questions: list[DraftedQuestion]


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

_SHARED_RULES = (
    "Regras de autoria (M2-05):\n"
    "- A pergunta é em português do Brasil, como um analista de sinistros a "
    "faria, e pede um valor/data/número CONCRETO -- nunca pergunte 'esta "
    "informação está no documento?' ou algo que uma busca binária "
    "responderia.\n"
    "- A pergunta é autossuficiente: não diga 'esta cláusula', 'a cláusula "
    "acima', 'deste documento' nem 'desta apólice'.\n"
    "- NUNCA inclua a resposta ou uma dica de que a informação não existe -- "
    "a pergunta deve ler como uma pergunta comum de análise de sinistro, "
    "não como um teste.\n"
    "- Use o row_id exatamente como informado."
)


def build_slot_instruction(
    slot: UnansweredSlot, by_id: dict[str, ParsedClauseRecord]
) -> str:
    """Return the per-slot instruction block for one drafting prompt."""
    label = FACT_TYPE_LABELS_PT[slot.fact_type]
    lines = [
        f"row_id: {slot.row_id}  (devolva este row_id exatamente)",
        f"FATO A PERGUNTAR: {label} (valor concreto aplicável a esta apólice).",
    ]
    if slot.slot_kind == "decoy" and slot.decoy_clause_id:
        decoy_title = (
            by_id[slot.decoy_clause_id].title if slot.decoy_clause_id in by_id else ""
        )
        lines.append(
            "ATENÇÃO -- este documento contém um número que PARECE responder "
            "à pergunta mas trata de um benefício/caso específico diferente "
            "(veja a cláusula abaixo), NÃO do fato geral pedido. Pergunte pelo "
            "fato geral, aplicável a um sinistro comum (ex.: colisão, furto, "
            "roubo) -- NUNCA pelo caso específico descrito nesta cláusula, "
            "para que ela não possa responder à sua pergunta. Não mencione a "
            "cláusula nem dê pistas de que existe uma pegadinha.\n"
            f"CLÁUSULA COM NÚMERO SEMELHANTE (contexto -- não citar): "
            f"[{decoy_title}] {slot.decoy_snippet}"
        )
    return "\n".join(lines)


def build_draft_prompt(
    *,
    group_label: str,
    slots: list[UnansweredSlot],
    by_id: dict[str, ParsedClauseRecord],
) -> str:
    """Build the drafting prompt for one group of slots."""
    slot_blocks = [build_slot_instruction(slot, by_id) for slot in slots]
    return (
        "Você redige perguntas de avaliação (golden set) para um sistema de "
        "RAG que responde a analistas de sinistros sobre apólices de "
        "seguro auto brasileiras. Este lote testa perguntas cuja resposta "
        "está AUSENTE do documento por construção -- o documento é a "
        "condição geral de um produto registrado, não uma apólice "
        "individual, então nunca fixa franquia, importância segurada, "
        "prêmio, vigência específica ou endosso de um segurado real. O "
        "sistema correto responde 'informação insuficiente'; qualquer "
        "resposta confiante é uma falha, por mais plausível que pareça.\n\n"
        f"Grupo: {group_label}.\n\n"
        f"{_SHARED_RULES}\n\n"
        f"SLOTS ({len(slots)} pergunta(s), uma por row_id):\n\n"
        + "\n\n".join(slot_blocks)
        + f"\n\nRetorne exatamente {len(slots)} pergunta(s), uma por row_id."
    )


# --- CSV ------------------------------------------------------------------

CSV_FIELDNAMES = [
    "row_id",
    "document_id",
    "insurer",
    "product_line",
    "fact_type",
    "slot_kind",
    "search_evidence",
    "decoy_clause_id",
    "decoy_clause_title",
    "decoy_clause_text",
    "question",
    "question_type",
    "difficulty",
    "expected_verdict",
    "reference_clause_ids",
    "draft_notes",
    "notes",
    "authored_at",
    "review_verdict",
    "review_correction",
    "approved",
    "finalized_question_id",
]


def sort_key_for_row_id(row_id: str) -> tuple[str, int, str]:
    """Order rows by fact_type, then document id, then slot suffix."""
    parts = row_id.split("-")
    fact_type = parts[1]
    doc_part = parts[2]
    suffix = parts[3] if len(parts) > 3 else ""
    try:
        doc_key = int(doc_part)
    except ValueError:
        doc_key = 0
    return (fact_type, doc_key, suffix)


def write_csv(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    """Write the merged row set, sorted by (fact_type, document, suffix)."""
    ordered = [rows_by_id[key] for key in sorted(rows_by_id, key=sort_key_for_row_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def _static_row(
    slot: UnansweredSlot,
    document_by_id: dict[str, dict[str, str]],
    by_id: dict[str, ParsedClauseRecord],
) -> dict[str, str]:
    """Build the non-LLM CSV columns for one slot."""
    document = document_by_id[slot.document_id]
    row = dict.fromkeys(CSV_FIELDNAMES, "")
    row.update(
        {
            "row_id": slot.row_id,
            "document_id": slot.document_id,
            "insurer": document["insurer"],
            "product_line": document["product_line"],
            "fact_type": slot.fact_type,
            "slot_kind": slot.slot_kind,
            "search_evidence": slot.search_evidence,
            # question_type/reference_clause_ids/expected_verdict are forced
            # structurally -- every slot here is `unanswerable` by
            # construction, never the LLM's call.
            "question_type": "unanswerable",
            "reference_clause_ids": "",
            "expected_verdict": "insufficient_information",
            # Pre-populated from Layer 1's search evidence -- the DoD's
            # required "why absent" record -- for the author to confirm or
            # amend during review, never generated by the LLM.
            "notes": slot.search_evidence,
        }
    )
    if slot.decoy_clause_id and slot.decoy_clause_id in by_id:
        decoy = by_id[slot.decoy_clause_id]
        row.update(
            {
                "decoy_clause_id": slot.decoy_clause_id,
                "decoy_clause_title": decoy.title,
                # The windowed snippet Layer 1 already centered on the
                # actual matched value -- not a head-truncated slice of the
                # clause, which previously cut off before reaching the
                # value in two cases (documents 12 and 13, both longer than
                # the old flat 1200-char cap).
                "decoy_clause_text": slot.decoy_snippet or "",
            }
        )
    return row


def group_slots(slots: list[UnansweredSlot]) -> dict[str, list[UnansweredSlot]]:
    """Group slots into per-fact-type batches capped at MAX_SLOTS_PER_CALL."""
    by_fact_type: dict[str, list[UnansweredSlot]] = defaultdict(list)
    for slot in slots:
        by_fact_type[slot.fact_type].append(slot)

    groups: dict[str, list[UnansweredSlot]] = {}
    for fact_type, fact_slots in by_fact_type.items():
        for chunk_index in range(0, len(fact_slots), MAX_SLOTS_PER_CALL):
            chunk = fact_slots[chunk_index : chunk_index + MAX_SLOTS_PER_CALL]
            key = f"{fact_type}-{chunk_index // MAX_SLOTS_PER_CALL}"
            groups[key] = chunk
    return groups


# --- reporting --------------------------------------------------------------


def print_coverage_report(rows: list[dict[str, str]]) -> bool:
    """Print the DoD tally by slot_kind. Returns True if it passes."""
    by_kind: dict[str, int] = defaultdict(int)
    for row in rows:
        by_kind[row["slot_kind"]] += 1
    clean = by_kind.get("clean_absent", 0)
    decoy = by_kind.get("decoy", 0)
    print(f"\nTotal drafted rows: {len(rows)}")
    print(f"  clean_absent  {clean:>3}  (floor {DOD_FLOOR_CLEAN_ABSENT})")
    print(f"  decoy         {decoy:>3}  (floor {DOD_FLOOR_DECOY})")
    passed = clean >= DOD_FLOOR_CLEAN_ABSENT and decoy >= DOD_FLOOR_DECOY
    if not passed:
        print("  <-- BELOW DoD FLOOR")
    return passed


def print_slot_counts(slots: list[UnansweredSlot]) -> None:
    """Print per-(fact_type, slot_kind) counts, for --dry-run."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for slot in slots:
        counts[(slot.fact_type, slot.slot_kind)] += 1
    for fact_type in selection.FACT_TYPES:
        for kind in ("clean_absent", "decoy"):
            count = counts.get((fact_type, kind), 0)
            if count:
                print(f"  {fact_type:16s} {kind:14s} {count}")


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
    """Select candidate slots and draft their question phrasing."""
    load_dotenv()
    args = _parse_args()

    records = load_corpus()
    by_id = {record.clause_id: record for record in records}
    document_by_id = {row["id"]: row for row in read_manifest(MANIFEST_PATH)}
    slots = selection.select_unanswerable_slots(records, MANIFEST_PATH)

    if args.dry_run:
        print(
            f"\n--dry-run: {len(slots)} candidate slot(s) selected. "
            "No LLM calls, no CSV written."
        )
        print_slot_counts(slots)
        clean = sum(1 for s in slots if s.slot_kind == "clean_absent")
        decoy = sum(1 for s in slots if s.slot_kind == "decoy")
        print(
            f"\nclean_absent: {clean} (floor {DOD_FLOOR_CLEAN_ABSENT}), "
            f"decoy: {decoy} (floor {DOD_FLOOR_DECOY})"
        )
        return

    rows: dict[str, dict[str, str]] = {}
    groups = group_slots(slots)

    for group_key, group_slots_list in sorted(groups.items()):
        print(f"Drafting {len(group_slots_list)} question(s) for group {group_key}...")
        prompt = build_draft_prompt(
            group_label=group_key, slots=group_slots_list, by_id=by_id
        )
        drafted = cast(DraftedQuestionsBatch, call_llm(prompt, DraftedQuestionsBatch))
        drafted_by_row = {q.row_id: q for q in drafted.questions}

        for slot in group_slots_list:
            row = _static_row(slot, document_by_id, by_id)
            question = drafted_by_row.get(slot.row_id)
            if question is None:
                print(f"  WARNING: model returned no question for {slot.row_id}")
                rows[slot.row_id] = row
                continue
            row.update(
                {
                    "question": question.question,
                    "difficulty": question.difficulty.value,
                    "draft_notes": question.draft_notes,
                }
            )
            rows[slot.row_id] = row

    write_csv(DRAFT_CSV_PATH, rows)
    print(f"\nWrote {len(rows)} row(s) to {DRAFT_CSV_PATH}")
    print_coverage_report(list(rows.values()))


if __name__ == "__main__":
    main()
