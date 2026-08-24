#!/usr/bin/env python3
"""Draft and repair adversarial golden-set questions [M2-03].

The LLM phrasing layer of [M2-01]'s three-layer authoring flow, for the four
adversarial categories: source-pair selection lives in
``scripts/adversarial_clause_selection.py`` (deterministic, no model);
verification stays the reviewing author's, and nothing here writes to
``data/golden_set/`` -- only ``scripts/finalize_golden_set_from_review.py``
(``--csv eval/golden_set_draft_adversarial.csv``) promotes rows the author
has approved.

Two passes per slot group, mirroring ``scripts/draft_golden_questions_casco.py``
[M2-02]:

1. **Drafting** -- the model sees a *context bundle*: the anchor clause plus
   its [M2-08] structural candidates, its parent, children and byte-identical
   twins. For ``cross_document``/``hdi_brand_collision``/``bundle_section``
   slots it additionally sees the *distractor* clause, explicitly labeled as
   a clause the question must NOT be answerable from -- the distractor's id
   is never a legal ``reference_clause_ids`` entry (including it would
   corrupt Recall/MRR: it is a wrong answer, not a partial one).
2. **Completeness** -- every drafted question is re-presented against its
   group's clause library and the model must return the minimal exhaustive
   reference set (still excluding the distractor) and a reason for each
   candidate it rejected.

LLM: OpenRouter, ``google/gemini-3.7-flash``, pinned to the
``google-vertex/global`` provider route via the shared
[infrastructure.config.llm_client_factory.build_chat_model] -- single
provider, no fallback (``allow_fallbacks=False``), matching this repo's
existing vision-model pin. Unlike ``draft_golden_questions_casco.py``, this
script does not need a bespoke client: M2-03 uses exactly one provider
throughout.

Repair mode (``--review-csv``) is deliberately simpler than CASCO's: a row
marked ``approved`` truthy is kept verbatim; a row with a non-empty
``review_correction`` is redrafted with that text injected as an explicit
instruction; every other row is dropped (its slot is not auto-replaced --
the adversarial slots are hand-curated pairs, not a vocabulary quota a
replacement algorithm can restock from).

Run ``make draft-golden-questions-adversarial`` (fresh draft) or
``make repair-golden-questions-adversarial REVIEW=<csv>``. ``--dry-run``
prints the selection and slot counts without spending a single API call.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from enum import StrEnum
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
from infrastructure.evaluation.golden_set_schema import Difficulty, ExpectedVerdict
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl

try:
    # Direct execution: the script's own directory is sys.path[0].
    import adversarial_clause_selection as selection
    import casco_clause_selection as casco_selection
    from adversarial_clause_selection import AdversarialSlot
    from find_candidate_clauses import find_candidates
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import adversarial_clause_selection as selection
    from scripts import casco_clause_selection as casco_selection
    from scripts.adversarial_clause_selection import AdversarialSlot
    from scripts.find_candidate_clauses import find_candidates

MANIFEST_PATH = Path("data/policies/manifest.csv")
DRAFT_CSV_PATH = Path("eval/golden_set_draft_adversarial.csv")
EXISTING_COVERAGE_WITH_EXCLUSION_PATH = Path(
    "data/golden_set/coverage_with_exclusion.jsonl"
)

DRAFT_MODEL = "google/gemini-3.7-flash"
DRAFT_PROVIDER_ORDER = ["google-vertex/global"]

SOURCE_TEXT_CAP = 4000
BUNDLE_TEXT_CAP = 900
MAX_BUNDLE_CLAUSES = 12
MAX_LIBRARY_CLAUSES = 45

# Draft-count caps per category: at or above the DoD floors within this CSV
# alone (coverage_with_exclusion also has 4 already-finalized rows from
# M2-02 in data/golden_set/coverage_with_exclusion.jsonl, so 15 here gives
# 19 total), following M2-02's "draft close to target, review is the safety
# valve" pattern rather than over-drafting a large surplus.
MAX_SLOTS_PER_CATEGORY: dict[str, int] = {
    "coverage_with_exclusion": 15,
    "cross_document": 16,
    "hdi_brand_collision": 7,
    "bundle_section": 4,
}

CATEGORY_LABELS: dict[str, str] = {
    "coverage_with_exclusion": "cobertura + exclusão que a limita",
    "cross_document": "quase-duplicata entre documentos da mesma seguradora (CNPJ)",
    "hdi_brand_collision": "colisão de marca HDI (duas entidades legais distintas)",
    "bundle_section": "seção correta dentro do documento multi-produto Bradesco",
}


class DraftableQuestionType(StrEnum):
    """The only types this script authors.

    ``question_type`` per row is ultimately forced structurally from the
    slot's ``adversarial_category`` (see ``CATEGORY_QUESTION_TYPE`` and
    ``main()``'s post-draft override), not left to the model's discretion --
    the selection layer already decided which category a slot belongs to.
    This enum only bounds what the model may legally return so a
    close-but-wrong guess still validates. ``unanswerable``/``definition``
    stay out of scope (other issues).
    """

    DIRECT_LOOKUP = "direct_lookup"
    COVERAGE_WITH_EXCLUSION = "coverage_with_exclusion"
    CROSS_DOCUMENT = "cross_document"


# The schema-correct QuestionType per adversarial category, applied
# structurally after drafting (see main()) regardless of what the model
# returned in DraftedQuestion.question_type. hdi_brand_collision and
# bundle_section have no dedicated QuestionType value (closed 5-value enum,
# frozen at M2-01), so direct_lookup -- "the answer is one clause, no
# cross-document framing" -- is the least-wrong existing value for them; the
# real category lives in the CSV's adversarial_category column and, once
# finalized, in notes.
CATEGORY_QUESTION_TYPE: dict[str, DraftableQuestionType] = {
    "coverage_with_exclusion": DraftableQuestionType.COVERAGE_WITH_EXCLUSION,
    "cross_document": DraftableQuestionType.CROSS_DOCUMENT,
    "hdi_brand_collision": DraftableQuestionType.DIRECT_LOOKUP,
    "bundle_section": DraftableQuestionType.DIRECT_LOOKUP,
}


class DraftedQuestion(BaseModel):
    """One drafted question, anchored on a slot's clause(s)."""

    row_id: str = Field(
        ...,
        description="The row_id of the slot this question answers, copied verbatim.",
    )
    question: str = Field(
        ...,
        description=(
            "The question in Brazilian Portuguese, phrased as a claims analyst "
            "would ask it, self-contained (never 'esta cláusula'), and never "
            "echoing the clause's literal wording."
        ),
    )
    question_type: DraftableQuestionType
    difficulty: Difficulty
    reference_clause_ids: list[str] = Field(
        ...,
        description=(
            "Every clause_id needed to answer the question exhaustively, all "
            "drawn from that slot's allowed ids. NEVER include the "
            "distractor id, even if it is shown as context."
        ),
    )
    expected_verdict: ExpectedVerdict | None = Field(
        None,
        description="Null for direct_lookup. Required for coverage_with_exclusion.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "What the question tests and, specifically, what a retriever "
            "without the right metadata filter would surface instead "
            "(name the distractor clause_id/document when the slot has one) "
            "-- this becomes the failure-catalogue note the author copies "
            "into the final golden set."
        ),
    )


class DraftedQuestionsBatch(BaseModel):
    """A batch of drafted questions for one group of slots."""

    questions: list[DraftedQuestion]


class RejectedCandidate(BaseModel):
    """A candidate clause the completeness check considered and left out."""

    clause_id: str
    reason: str = Field(..., description="One line: why it is not needed.")


class IncludedClause(BaseModel):
    """A clause the completeness check kept, and why it is indispensable."""

    clause_id: str
    reason: str = Field(..., description="One line: what the question needs it for.")


class CompletenessVerdict(BaseModel):
    """The exhaustive reference set for one existing question."""

    row_id: str
    included: list[IncludedClause] = Field(default_factory=list)
    reference_clause_ids: list[str] = Field(
        ...,
        description=(
            "The minimal set of clause_ids that TOGETHER answer the "
            "question exhaustively. NEVER the distractor id."
        ),
    )
    rejected: list[RejectedCandidate] = Field(default_factory=list)


class CompletenessBatch(BaseModel):
    """Completeness verdicts for one group's questions."""

    verdicts: list[CompletenessVerdict]


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the built corpus, failing loudly if `make parse` hasn't run yet."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


# --- LLM call ---------------------------------------------------------------


def call_llm(prompt: str, output_model: type[BaseModel]) -> BaseModel:
    """Invoke a structured-output chain against the pinned OpenRouter/Gemini model.

    Single provider, no fallback -- matching this repo's existing
    ``LLM_VISION_PROVIDER_ORDER`` convention (see
    [infrastructure.config.llm_client_factory]). Retries transient failures;
    any other failure re-raises, since there is no sane fallback for a
    drafting judgment.
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


# --- context bundles ---------------------------------------------------------


def build_context_bundle(
    records: list[ParsedClauseRecord],
    by_id: dict[str, ParsedClauseRecord],
    children_by_parent: dict[str, list[str]],
    twins: dict[str, frozenset[str]],
    primary_clause_id: str,
    *,
    extra_ids: list[str] | None = None,
) -> list[str]:
    """Return the clause_ids the model may reference for one slot.

    Mirrors ``draft_golden_questions_casco.py``'s bundle builder: primary
    first, then [M2-08]'s structural candidates, then children, parent and
    twins -- the cap bites from the end. ``extra_ids`` (a
    ``coverage_with_exclusion`` slot's paired exclusion clause) are pinned
    right after the primary so they always survive the cap.
    """
    ordered: list[str] = [primary_clause_id]

    def add(clause_id: str) -> None:
        if clause_id in by_id and clause_id not in ordered:
            ordered.append(clause_id)

    for extra_id in extra_ids or []:
        add(extra_id)
    for candidate in find_candidates(records, primary_clause_id, max_candidates=8):
        add(candidate.clause_id)
    for child_id in children_by_parent.get(primary_clause_id, []):
        add(child_id)
    primary = by_id[primary_clause_id]
    if primary.parent_id is not None:
        add(primary.parent_id)
    for twin_id in sorted(twins.get(primary_clause_id, frozenset())):
        add(twin_id)
    return ordered[:MAX_BUNDLE_CLAUSES]


def format_clause_block(clause_id: str, by_id: dict[str, ParsedClauseRecord]) -> str:
    """Render one clause as a labeled block for a prompt's clause library."""
    record = by_id[clause_id]
    text = record.text.strip()[:BUNDLE_TEXT_CAP]
    return (
        f"[{clause_id}]\n"
        f"título: {record.title}\n"
        f"tipo: {record.clause_type.value}\n"
        f"documento: {record.document_id} ({record.insurer})\n"
        f"texto: {text}"
    )


def format_clause_library(
    clause_ids: list[str], by_id: dict[str, ParsedClauseRecord]
) -> str:
    """Render a deduplicated clause library for one group's prompt."""
    blocks = [
        format_clause_block(clause_id, by_id)
        for clause_id in clause_ids[:MAX_LIBRARY_CLAUSES]
    ]
    return "\n---\n".join(blocks)


# --- prompts ------------------------------------------------------------

_SHARED_RULES = (
    "Regras de autoria (M2-01):\n"
    "- A pergunta é em português do Brasil, como um analista de sinistros a "
    "faria, e NUNCA repete o texto literal da cláusula.\n"
    "- A pergunta é autossuficiente: não diga 'esta cláusula', 'a cláusula "
    "acima', 'deste documento' nem 'desta apólice'; quem lê a pergunta não "
    "sabe qual cláusula você viu.\n"
    "- A pergunta NUNCA faz referência à existência de outro documento, "
    "seção ou cláusula 'concorrente' -- ela deve ler como uma pergunta comum "
    "sobre UMA apólice. A cláusula concorrente, quando mostrada, é contexto "
    "de curadoria, nunca material de fraseado.\n"
    "- reference_clause_ids deve ser EXAUSTIVO mas nunca deve incluir o id "
    "da cláusula concorrente/distratora, mesmo que ela seja mostrada como "
    "contexto.\n"
    "- expected_verdict é obrigatório para 'coverage_with_exclusion' e nulo "
    "para 'direct_lookup'/'cross_document'.\n"
    "- Use apenas clause_ids da biblioteca oferecida (ids permitidos)."
)


def build_draft_prompt(
    *,
    category: str,
    group_label: str,
    library_ids: list[str],
    by_id: dict[str, ParsedClauseRecord],
    slots: list[dict[str, object]],
) -> str:
    """Build the drafting prompt for one group of slots (one adversarial category)."""
    slot_blocks: list[str] = []
    for slot in slots:
        lines = [
            f"row_id: {slot['row_id']}  (devolva este row_id exatamente)",
            f"ids permitidos: {', '.join(cast(list[str], slot['allowed_ids']))}",
        ]
        if category == "coverage_with_exclusion":
            lines.append(
                "TIPO OBRIGATÓRIO: 'coverage_with_exclusion'. "
                f"Cláusula de COBERTURA: {slot['primary_clause_id']}. "
                f"Cláusula de EXCLUSÃO que a limita: {slot['secondary_clause_id']}. "
                "reference_clause_ids TEM de conter as duas. Preencha "
                "expected_verdict (normalmente 'incompatible', já que a "
                "exclusão limita a cobertura perguntada)."
            )
        else:
            type_label = (
                "cross_document" if category == "cross_document" else "direct_lookup"
            )
            lines.append(
                f"TIPO OBRIGATÓRIO: '{type_label}'. "
                f"Cláusula CORRETA (âncora da pergunta): {slot['primary_clause_id']}.\n"
                f"CLÁUSULA CONCORRENTE (NÃO deve responder à pergunta, "
                f"NUNCA em reference_clause_ids): {slot['distractor_clause_id']}.\n"
                "Encontre um detalhe factual que aparece na cláusula CORRETA "
                "e está ausente ou é diferente na cláusula CONCORRENTE -- um "
                "valor numérico, o nome de uma variante/produto específico, "
                "um procedimento específico. A pergunta deve depender desse "
                "detalhe, de forma que a cláusula concorrente NÃO a "
                "responda igualmente bem."
            )
            if category == "hdi_brand_collision":
                lines.append(
                    "ATENÇÃO -- COLISÃO DE MARCA: as duas cláusulas pertencem a "
                    "entidades legais DIFERENTES que compartilham a marca "
                    f"'HDI' ({slot['insurer_a']}, CNPJ {slot['cnpj_a']} vs. "
                    f"{slot['insurer_b']}, CNPJ {slot['cnpj_b']}). A pergunta "
                    "nunca deve mencionar CNPJ nem 'entidade' -- apenas "
                    "descrever um cenário de sinistro cuja resposta correta "
                    "está na cláusula CORRETA."
                )
        slot_blocks.append("\n".join(lines))

    return (
        "Você redige perguntas ADVERSARIAIS de avaliação (golden set) para "
        "um sistema de RAG que responde a analistas de sinistros sobre "
        "apólices de seguro auto brasileiras. Perguntas adversariais são "
        "desenhadas para expor onde um retriever mal filtrado erra -- não "
        "são hostis, são específicas o bastante para que só a cláusula "
        "correta as responda.\n\n"
        f"Categoria: {category} ({CATEGORY_LABELS[category]}).\n"
        f"Grupo: {group_label}.\n\n"
        f"{_SHARED_RULES}\n\n"
        f"BIBLIOTECA DE CLÁUSULAS:\n{format_clause_library(library_ids, by_id)}\n\n"
        f"SLOTS ({len(slots)} perguntas, uma por slot):\n"
        + "\n\n".join(slot_blocks)
        + f"\n\nRetorne exatamente {len(slots)} pergunta(s), uma por slot."
    )


def build_completeness_prompt(
    *,
    group_label: str,
    library_ids: list[str],
    by_id: dict[str, ParsedClauseRecord],
    questions: list[dict[str, object]],
) -> str:
    """Build the completeness prompt: exhaustive refs plus rejection reasons."""
    question_blocks = [
        f"row_id: {item['row_id']}\n"
        f"pergunta: {item['question']}\n"
        f"referências atuais: {', '.join(cast(list[str], item['current_refs']))}\n"
        f"ids permitidos: {', '.join(cast(list[str], item['allowed_ids']))}"
        for item in questions
    ]
    return (
        "Você executa a verificação de COMPLETUDE de um golden set "
        "ADVERSARIAL de RAG sobre apólices de seguro auto brasileiras. NÃO "
        "reescreva nenhuma pergunta: seu trabalho é apenas decidir, para "
        "cada uma, quais cláusulas são necessárias para respondê-la de "
        "forma exaustiva -- NUNCA inclua um id que não esteja nos ids "
        "permitidos (a cláusula concorrente/distratora nunca está entre "
        "eles).\n\n"
        f"Grupo: {group_label}.\n\n"
        "Para cada pergunta devolva:\n"
        "- reference_clause_ids: o conjunto MÍNIMO de clause_ids que, "
        "juntos, respondem à pergunta por completo.\n"
        "- included: para CADA id incluído, uma linha dizendo o que a "
        "pergunta ficaria sem responder caso ele faltasse.\n"
        "- rejected: para CADA id permitido não incluído, uma linha dizendo "
        "por quê.\n\n"
        f"BIBLIOTECA DE CLÁUSULAS:\n{format_clause_library(library_ids, by_id)}\n\n"
        f"PERGUNTAS ({len(questions)}):\n" + "\n\n".join(question_blocks)
    )


# --- CSV ----------------------------------------------------------------

CSV_FIELDNAMES = [
    "row_id",
    "adversarial_category",
    "document_id",
    "insurer",
    "filename",
    "primary_clause_id",
    "primary_clause_title",
    "primary_clause_text",
    "secondary_clause_id",
    "distractor_clause_id",
    "distractor_document_id",
    "distractor_clause_title",
    "distractor_clause_text",
    "question",
    "question_type",
    "difficulty",
    "expected_verdict",
    "reference_clause_ids",
    "reference_clause_texts",
    "completeness_pool_size",
    "completeness_considered_ids",
    "completeness_included_reasons",
    "completeness_rejected_reasons",
    "draft_notes",
    "notes",
    "authored_at",
    "review_verdict",
    "review_correction",
    "approved",
    "finalized_question_id",
]


def sort_key_for_row_id(row_id: str) -> tuple[str, str, int]:
    """Order rows by category, then doc part, then slot number."""
    category, doc_part, slot_part = row_id.rsplit("-", 2)
    return (category, doc_part, int(slot_part))


def write_csv(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    """Write the merged row set, sorted by (category, doc part, slot)."""
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


def load_review(path: Path) -> dict[str, dict[str, str]]:
    """Read the author's review verdicts, indexed by row_id."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Pass --review-csv <path>.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle)}


# --- slot grouping --------------------------------------------------------


def group_slots(slots: list[AdversarialSlot]) -> dict[str, list[AdversarialSlot]]:
    """Group slots into batches small enough for one prompt each.

    coverage_with_exclusion groups by document (the slot's natural context
    unit); the other three categories group by the two-document (or, for
    bundle_section, single-document) row_id prefix -- ``xdoc-4x18-00`` and
    ``xdoc-4x18-01`` share group key ``xdoc-4x18``.
    """
    groups: dict[str, list[AdversarialSlot]] = defaultdict(list)
    for slot in slots:
        if slot.adversarial_category == "coverage_with_exclusion":
            key = f"cwe-{slot.document_id}"
        else:
            key = slot.row_id.rsplit("-", 1)[0]
        groups[key].append(slot)
    return groups


def build_slot_context(
    slot: AdversarialSlot,
    *,
    records: list[ParsedClauseRecord],
    by_id: dict[str, ParsedClauseRecord],
    children_by_parent: dict[str, list[str]],
    twins: dict[str, frozenset[str]],
) -> dict[str, object]:
    """Return the allowed-ids bundle and prompt metadata for one slot."""
    extra = [slot.secondary_clause_id] if slot.secondary_clause_id else []
    bundle_ids = build_context_bundle(
        records,
        by_id,
        children_by_parent,
        twins,
        slot.primary_clause_id,
        extra_ids=extra,
    )
    library_ids = list(bundle_ids)
    if slot.distractor_clause_id and slot.distractor_clause_id in by_id:
        library_ids.append(slot.distractor_clause_id)

    context: dict[str, object] = {
        "slot": slot,
        "allowed_ids": bundle_ids,
        "library_ids": library_ids,
    }
    if slot.adversarial_category == "hdi_brand_collision" and slot.distractor_clause_id:
        primary = by_id[slot.primary_clause_id]
        distractor = by_id[slot.distractor_clause_id]
        context["insurer_a"] = primary.insurer
        context["cnpj_a"] = primary.cnpj
        context["insurer_b"] = distractor.insurer
        context["cnpj_b"] = distractor.cnpj
    return context


# --- reporting ------------------------------------------------------------


def print_coverage_report(rows: list[dict[str, str]]) -> bool:
    """Print the DoD tally by adversarial_category. Returns True if it passes."""
    dod_floors = {
        "coverage_with_exclusion": 15,
        "cross_document": 10,
        "hdi_brand_collision": 5,
        "bundle_section": 3,
    }
    by_category: dict[str, int] = defaultdict(int)
    for row in rows:
        by_category[row["adversarial_category"]] += 1

    existing_cwe = 0
    if EXISTING_COVERAGE_WITH_EXCLUSION_PATH.exists():
        with EXISTING_COVERAGE_WITH_EXCLUSION_PATH.open(encoding="utf-8") as handle:
            existing_cwe = sum(1 for line in handle if line.strip())

    print(f"\nTotal drafted rows: {len(rows)}")
    passed = True
    for category, floor in dod_floors.items():
        drafted = by_category.get(category, 0)
        already_final = existing_cwe if category == "coverage_with_exclusion" else 0
        total = drafted + already_final
        flag = "" if total >= floor else "  <-- BELOW DoD FLOOR"
        if total < floor:
            passed = False
        print(
            f"  {category:24s} drafted={drafted:>3}  "
            f"already_finalized={already_final:>3}  "
            f"total={total:>3}  floor={floor}{flag}"
        )
    return passed


# --- main -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Author's review CSV; enables repair mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection and slot counts; never call an LLM.",
    )
    return parser.parse_args()


def _rows_from_slots(
    slots: list[AdversarialSlot],
    by_id: dict[str, ParsedClauseRecord],
) -> dict[str, dict[str, str]]:
    """Build the static (non-LLM) CSV columns for a batch of slots."""
    rows: dict[str, dict[str, str]] = {}
    for slot in slots:
        primary = by_id[slot.primary_clause_id]
        row = dict.fromkeys(CSV_FIELDNAMES, "")
        row.update(
            {
                "row_id": slot.row_id,
                "adversarial_category": slot.adversarial_category,
                "document_id": slot.document_id,
                "insurer": primary.insurer,
                "filename": "",
                "primary_clause_id": slot.primary_clause_id,
                "primary_clause_title": primary.title,
                "primary_clause_text": primary.text.strip()[:SOURCE_TEXT_CAP],
                "secondary_clause_id": slot.secondary_clause_id or "",
            }
        )
        if slot.distractor_clause_id and slot.distractor_clause_id in by_id:
            distractor = by_id[slot.distractor_clause_id]
            row.update(
                {
                    "distractor_clause_id": slot.distractor_clause_id,
                    "distractor_document_id": distractor.document_id,
                    "distractor_clause_title": distractor.title,
                    "distractor_clause_text": distractor.text.strip()[:SOURCE_TEXT_CAP],
                }
            )
        rows[slot.row_id] = row
    return rows


def main() -> None:  # noqa: C901 - one linear pipeline, documented in sections
    """Draft or repair the adversarial golden-set questions."""
    load_dotenv()
    args = _parse_args()

    records = load_corpus()
    by_id = {record.clause_id: record for record in records}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.parent_id is not None:
            children_by_parent[record.parent_id].append(record.clause_id)
    twins = casco_selection.build_duplicate_text_index(records)

    existing_cwe_ids = selection.load_existing_coverage_with_exclusion_clause_ids(
        EXISTING_COVERAGE_WITH_EXCLUSION_PATH
    )

    all_slots: list[AdversarialSlot] = []
    all_slots += selection.select_coverage_with_exclusion_slots(
        records,
        MANIFEST_PATH,
        already_used_ids=existing_cwe_ids,
        max_slots=MAX_SLOTS_PER_CATEGORY["coverage_with_exclusion"],
    )
    all_slots += selection.select_cross_document_slots(records)[
        : MAX_SLOTS_PER_CATEGORY["cross_document"]
    ]
    all_slots += selection.select_hdi_brand_collision_slots(records)[
        : MAX_SLOTS_PER_CATEGORY["hdi_brand_collision"]
    ]
    all_slots += selection.select_bundle_section_slots(records)[
        : MAX_SLOTS_PER_CATEGORY["bundle_section"]
    ]

    existing = load_existing_csv(DRAFT_CSV_PATH)
    review = load_review(args.review_csv) if args.review_csv else {}

    if review:
        # Repair mode: the review file defines the universe of rows in play.
        # `approved` truthy -> keep verbatim. A non-empty `review_correction`
        # -> redraft with that text injected. Anything else -> dropped.
        by_row_id = {slot.row_id: slot for slot in all_slots}
        kept: dict[str, dict[str, str]] = {}
        to_redraft: list[AdversarialSlot] = []
        corrections: dict[str, str] = {}
        for row_id, review_row in review.items():
            base = existing.get(row_id)
            if base is None:
                print(
                    f"WARNING: {row_id} is in the review file but not in "
                    f"{DRAFT_CSV_PATH}; skipping."
                )
                continue
            approved = review_row.get("approved", "").strip().lower() in {
                "y",
                "yes",
                "true",
                "1",
            }
            correction = review_row.get("review_correction", "").strip()
            if approved and not correction:
                kept[row_id] = base
                continue
            if correction:
                slot = by_row_id.get(row_id)
                if slot is None:
                    print(
                        f"WARNING: {row_id} has a correction but its slot no "
                        "longer exists in the current selection; skipping."
                    )
                    continue
                to_redraft.append(slot)
                corrections[row_id] = correction
                continue
            print(f"Dropping {row_id} (not approved, no correction).")
        print(
            f"Repair mode: {len(kept)} kept verbatim, {len(to_redraft)} to redraft, "
            f"{len(review) - len(kept) - len(to_redraft)} dropped."
        )
        slots_to_draft = to_redraft
    else:
        corrections = {}
        kept = {}
        slots_to_draft = all_slots

    if args.dry_run:
        print(
            f"\n--dry-run: {len(all_slots)} candidate slots found, "
            f"{len(slots_to_draft)} would be drafted. No LLM calls, no CSV written."
        )
        for category in MAX_SLOTS_PER_CATEGORY:
            count = sum(1 for s in all_slots if s.adversarial_category == category)
            print(f"  {category:24s} {count}")
        return

    rows: dict[str, dict[str, str]] = dict(kept)
    static_rows = _rows_from_slots(slots_to_draft, by_id)
    groups = group_slots(slots_to_draft)

    for group_key, group_slot_list in sorted(groups.items()):
        contexts = {
            slot.row_id: build_slot_context(
                slot,
                records=records,
                by_id=by_id,
                children_by_parent=children_by_parent,
                twins=twins,
            )
            for slot in group_slot_list
        }
        library_ids: list[str] = []
        for ctx in contexts.values():
            for clause_id in cast(list[str], ctx["library_ids"]):
                if clause_id not in library_ids:
                    library_ids.append(clause_id)

        category = group_slot_list[0].adversarial_category
        draft_slots_payload = []
        for slot in group_slot_list:
            ctx = contexts[slot.row_id]
            payload: dict[str, object] = {
                "row_id": slot.row_id,
                "allowed_ids": ctx["allowed_ids"],
                "primary_clause_id": slot.primary_clause_id,
                "secondary_clause_id": slot.secondary_clause_id,
                "distractor_clause_id": slot.distractor_clause_id,
            }
            if "insurer_a" in ctx:
                payload.update(
                    insurer_a=ctx["insurer_a"],
                    cnpj_a=ctx["cnpj_a"],
                    insurer_b=ctx["insurer_b"],
                    cnpj_b=ctx["cnpj_b"],
                )
            if slot.row_id in corrections:
                payload["correction"] = corrections[slot.row_id]
            draft_slots_payload.append(payload)

        prompt = build_draft_prompt(
            category=category,
            group_label=group_key,
            library_ids=library_ids,
            by_id=by_id,
            slots=draft_slots_payload,
        )
        print(f"Drafting {len(group_slot_list)} question(s) for group {group_key}...")
        drafted = cast(DraftedQuestionsBatch, call_llm(prompt, DraftedQuestionsBatch))
        drafted_by_row = {q.row_id: q for q in drafted.questions}

        # --- completeness pass, same group -----------------------------
        completeness_payload = [
            {
                "row_id": slot.row_id,
                "question": drafted_by_row[slot.row_id].question,
                "current_refs": drafted_by_row[slot.row_id].reference_clause_ids,
                "allowed_ids": contexts[slot.row_id]["allowed_ids"],
            }
            for slot in group_slot_list
            if slot.row_id in drafted_by_row
        ]
        completeness_by_row: dict[str, CompletenessVerdict] = {}
        if completeness_payload:
            completeness_prompt = build_completeness_prompt(
                group_label=group_key,
                library_ids=library_ids,
                by_id=by_id,
                questions=completeness_payload,
            )
            completeness = cast(
                CompletenessBatch, call_llm(completeness_prompt, CompletenessBatch)
            )
            completeness_by_row = {v.row_id: v for v in completeness.verdicts}

        for slot in group_slot_list:
            question = drafted_by_row.get(slot.row_id)
            row = static_rows[slot.row_id]
            if question is None:
                print(f"  WARNING: model returned no question for {slot.row_id}")
                rows[slot.row_id] = row
                continue
            verdict = completeness_by_row.get(slot.row_id)
            reference_ids = (
                verdict.reference_clause_ids
                if verdict is not None
                else question.reference_clause_ids
            )
            # Structural guardrail: the distractor must never survive into
            # the reference set, whatever the model returned.
            if slot.distractor_clause_id:
                reference_ids = [
                    cid for cid in reference_ids if cid != slot.distractor_clause_id
                ]
            # Structural guardrail: question_type is forced from the slot's
            # adversarial_category, not trusted from the model's field --
            # see CATEGORY_QUESTION_TYPE.
            forced_type = CATEGORY_QUESTION_TYPE[slot.adversarial_category]
            row.update(
                {
                    "question": question.question,
                    "question_type": forced_type.value,
                    "difficulty": question.difficulty.value,
                    "expected_verdict": (
                        question.expected_verdict.value
                        if question.expected_verdict
                        else ""
                    ),
                    "reference_clause_ids": ";".join(reference_ids),
                    "reference_clause_texts": " || ".join(
                        f"[{cid}] {by_id[cid].title}: {by_id[cid].text.strip()[:200]}"
                        for cid in reference_ids
                        if cid in by_id
                    ),
                    "draft_notes": question.reasoning,
                }
            )
            if verdict is not None:
                row.update(
                    {
                        "completeness_pool_size": str(
                            len(cast(list[str], contexts[slot.row_id]["allowed_ids"]))
                        ),
                        "completeness_considered_ids": ";".join(
                            cast(list[str], contexts[slot.row_id]["allowed_ids"])
                        ),
                        "completeness_included_reasons": " || ".join(
                            f"{item.clause_id}: {item.reason}"
                            for item in verdict.included
                        ),
                        "completeness_rejected_reasons": " || ".join(
                            f"{item.clause_id}: {item.reason}"
                            for item in verdict.rejected
                        ),
                    }
                )
            rows[slot.row_id] = row

    write_csv(DRAFT_CSV_PATH, rows)
    print(f"\nWrote {len(rows)} row(s) to {DRAFT_CSV_PATH}")
    print_coverage_report(list(rows.values()))


if __name__ == "__main__":
    main()
