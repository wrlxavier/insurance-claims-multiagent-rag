#!/usr/bin/env python3
"""Draft and repair CASCO golden-set questions [M2-02].

The LLM phrasing layer of [M2-01]'s three-layer authoring flow. Source-clause
selection lives in ``scripts/casco_clause_selection.py`` (deterministic, no
model); verification stays the reviewing author's, and nothing here writes to
``data/golden_set/`` -- only ``scripts/finalize_golden_set_from_review.py``
promotes rows the author has approved.

The central constraint: a model can only cite clauses it has been shown. Ask
for a question about one isolated clause and the answer will reference that
clause and nothing else -- not because no limiting exclusion exists, but
because none was on offer. A completeness check run over such a prompt cannot
change a single answer, which makes layer 3 of the flow theatre.

Two passes avoid that, both batched per document:

1. **Drafting** -- for each slot the model sees a *context bundle*: the
   source clause plus its [M2-08] structural candidates, its parent, its
   children and its byte-identical twins, each with text. Every id in the
   bundle is a legal reference, and the instruction now asks for every clause
   needed rather than discouraging a second one.
2. **Completeness** -- every surviving question is re-presented against the
   document's clause library and the model must return the minimal
   exhaustive reference set *and a reason for each candidate it rejected*.
   Those reasons land in the CSV, so "the completeness check ran" is
   evidence rather than an assertion. Byte-identical twins of any referenced
   clause are then unioned in deterministically, because a retriever that
   returns the twin of the right clause is not wrong.

Coverage is scored on the **question** text, not the source clause: the DoD
vocabulary is a claim asked about what the questions say, and a clause-side
count will report terms the questions never use.

Repair mode (``--review-csv``) routes each reviewed row by the author's
verdict: ``sim`` keeps the question text and runs completeness only;
``revisar`` redrafts with that row's ``correcao_sugerida`` injected as an
explicit instruction; ``nao`` drops the row and frees its slot for a
replacement, targeted at whichever vocabulary term is furthest below the DoD
floor. Surviving ``row_id``s are never renumbered -- the review references
them. The review CSV, not this script's own previous output, defines the
reviewed universe, so repeated runs against one review file are idempotent.

Run ``make draft-golden-questions-casco`` (fresh draft) or
``make repair-golden-questions-casco`` (apply a review). ``--dry-run`` prints
the selection, the routing plan and the coverage gap analysis without
spending a single API call.

Fresh-draft mode **overwrites** ``eval/golden_set_draft_casco.csv``, including
the ``finalized_question_id`` column linking rows to questions already
promoted into ``data/golden_set/``. Once rows have been finalized, that file
is provenance: reach for repair mode, or recover the overwrite from git.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from infrastructure.evaluation.golden_set_schema import (
    Difficulty,
    ExpectedVerdict,
)
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl

try:
    # Direct execution: the script's own directory is sys.path[0].
    import casco_clause_selection as selection
    from casco_clause_selection import ClauseCandidate
    from find_candidate_clauses import find_candidates
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import casco_clause_selection as selection
    from scripts.casco_clause_selection import ClauseCandidate
    from scripts.find_candidate_clauses import find_candidates

# Bound once from the module rather than duplicated across both import
# branches: keeping two `from ... import (...)` lists in sync silently left
# the package-import branch missing five names, which would have raised
# NameError only at runtime.
MAX_QUESTIONS_PER_DOC = selection.MAX_QUESTIONS_PER_DOC
MIN_QUESTIONS_PER_DOC = selection.MIN_QUESTIONS_PER_DOC
MIN_VOCAB_QUESTIONS_PER_TERM = selection.MIN_VOCAB_QUESTIONS_PER_TERM
MIN_VOCAB_SCENARIOS_PER_TERM = selection.MIN_VOCAB_SCENARIOS_PER_TERM
MIN_INDEMNITY_BASIS_QUESTIONS = selection.MIN_INDEMNITY_BASIS_QUESTIONS
SELECTION_AUDIT_FIELDNAMES = selection.SELECTION_AUDIT_FIELDNAMES
VOCAB_DISPLAY_NAMES = selection.VOCAB_DISPLAY_NAMES
VOCAB_PATTERNS = selection.VOCAB_PATTERNS
build_ancestor_titles = selection.build_ancestor_titles
build_candidates_for_document = selection.build_candidates_for_document
build_duplicate_text_index = selection.build_duplicate_text_index
build_selection_audit_rows = selection.build_selection_audit_rows
coverage_with_exclusion_gap = selection.coverage_with_exclusion_gap
load_casco_documents = selection.load_casco_documents
pick_slots = selection.pick_slots
question_is_indemnity_basis = selection.question_is_indemnity_basis
question_scope_flag = selection.question_scope_flag
question_self_reference_flag = selection.question_self_reference_flag
score_question_scenarios = selection.score_question_scenarios
score_question_vocabulary = selection.score_question_vocabulary
select_indemnity_basis_documents = selection.select_indemnity_basis_documents
term_in_title = selection.term_in_title

MANIFEST_PATH = Path("data/policies/manifest.csv")
DRAFT_CSV_PATH = Path("eval/golden_set_draft_casco.csv")
SELECTION_AUDIT_PATH = Path("eval/golden_set_casco_selection_audit.csv")
CACHE_DIR = Path("eval/temp/golden_draft_casco")
PROVIDER_STATE_PATH = CACHE_DIR / "provider_state.json"

ANTHROPIC_MODEL = "claude-sonnet-5"
OPENROUTER_MODEL = "anthropic/claude-sonnet-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Generous by design: the reviewer checks correctness against this excerpt,
# and a cap tight enough to cut the sub-item holding the answer turns the
# review into guesswork. It exists only to stop a pathological clause from
# bloating the CSV.
SOURCE_TEXT_CAP = 4000
BUNDLE_TEXT_CAP = 900
MAX_BUNDLE_CLAUSES = 12
MAX_LIBRARY_CLAUSES = 45
NEAR_DUPLICATE_RATIO = 0.85
# Topic overlap on content words, which catches paraphrases the ratio misses.
NEAR_DUPLICATE_JACCARD = 0.6
# Containment catches the other shape: the same question with extra
# qualifiers bolted on, where Jaccard is dragged down by the longer side.
NEAR_DUPLICATE_CONTAINMENT = 0.8
NEAR_DUPLICATE_MIN_WORDS = 4

# Rows the author has typed coverage_with_exclusion: questions that already
# turn on a coverage/exclusion pair and were drafted under the wrong label.
# Correcting a label is in scope here; authoring NEW adversarial questions is
# [M2-03]'s job (its DoD asks for >=15), so every other slot is steered to a
# question a single coverage clause can answer rather than being allowed to
# drift into that category and pre-empt that issue's sample.
EXCLUSION_RETYPE_ROW_IDS = frozenset({"1-03", "2-04", "15-01", "15-02"})

# Per-row instructions from the author that the review CSV has no column for.
# Injected into the redraft prompt verbatim, and they override the verdict:
# a row named here is redrafted even if its verdict alone would not ask for
# it. Keyed by row_id, in the same Portuguese the prompts use.
AUTHOR_DECISIONS: dict[str, str] = {
    "15-02": (
        "DECISÃO DO AUTOR: volte à pergunta original sobre vandalismo/tumultos "
        "na cobertura de vidros, tipada como coverage_with_exclusion, com a "
        "cláusula de COBERTURA de vidros E a exclusão no reference_clause_ids. "
        "Não use a pergunta sobre 'outras exclusões previstas em outra "
        "cláusula', cujo gabarito não continha a resposta."
    ),
}

ORIGIN_KEPT = "kept"
ORIGIN_CORRECTED = "corrected"
ORIGIN_REPLACEMENT = "replacement"
ORIGIN_NEW = "new"


class DraftableQuestionType(StrEnum):
    """The only types M2-02 authors.

    ``unanswerable`` and ``cross_document`` belong to other issues, and
    handing the model the full [QuestionType] enum let it emit two
    ``unanswerable`` rows that still carried reference ids -- a combination
    ``GoldenQuestion`` rejects outright, so they would have failed at
    finalize rather than at draft time.
    """

    DIRECT_LOOKUP = "direct_lookup"
    DEFINITION = "definition"
    COVERAGE_WITH_EXCLUSION = "coverage_with_exclusion"


class DraftedQuestion(BaseModel):
    """One drafted question, anchored on a slot's source clause."""

    row_id: str = Field(
        ...,
        description="The row_id of the slot this question answers, copied verbatim.",
    )
    slot_clause_id: str = Field(
        ...,
        description=(
            "clause_id this question is anchored on. Normally the slot's own "
            "clause, but when the correction asks for a more specific clause, "
            "return that one instead -- it must still be in the slot's "
            "allowed ids."
        ),
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
            "drawn from that slot's offered bundle. Include a limiting "
            "exclusion when one applies; prefer a specific child clause over "
            "its container."
        ),
    )
    expected_verdict: ExpectedVerdict | None = Field(
        None,
        description=(
            "Null for direct_lookup/definition. Required for coverage_with_exclusion."
        ),
    )
    reasoning: str = Field(
        ..., description="What the question tests and why those clauses answer it."
    )


class DraftedQuestionsForDocument(BaseModel):
    """A batch of drafted questions for one document."""

    questions: list[DraftedQuestion]


class RejectedCandidate(BaseModel):
    """A candidate clause the completeness check considered and left out."""

    clause_id: str
    reason: str = Field(..., description="One line: why it is not needed.")


class IncludedClause(BaseModel):
    """A clause the completeness check kept, and why it is indispensable."""

    clause_id: str
    reason: str = Field(
        ...,
        description=(
            "One line: what the question could NOT answer without this "
            "clause. If the clause only adds context, drop it instead."
        ),
    )


class CompletenessVerdict(BaseModel):
    """The exhaustive reference set for one existing question."""

    row_id: str
    included: list[IncludedClause] = Field(
        default_factory=list,
        description=(
            "One entry per clause in reference_clause_ids, justifying why it "
            "is required. This is what stops the set inflating with nearby "
            "but unnecessary clauses."
        ),
    )
    reference_clause_ids: list[str] = Field(
        ...,
        description=(
            "The minimal set of clause_ids that TOGETHER answer the question "
            "exhaustively. Drawn only from that question's allowed ids."
        ),
    )
    rejected: list[RejectedCandidate] = Field(
        default_factory=list,
        description=(
            "Every allowed candidate NOT included, each with a one-line "
            "reason. This is the completeness audit trail."
        ),
    )


class CompletenessForDocument(BaseModel):
    """Completeness verdicts for one document's questions."""

    verdicts: list[CompletenessVerdict]


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the built corpus, failing loudly if `make parse` hasn't run yet."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


# --- provider handling ----------------------------------------------------


def looks_like_quota_exhaustion(exc: Exception) -> bool:
    """Conservative heuristic for 'this API key is out of credits'.

    Deliberately narrow: misreading a transient failure as exhaustion would
    switch providers -- and spend -- for no reason.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("credit balance", "insufficient_quota", "quota exceeded", "billing")
    return any(marker in text for marker in markers)


def load_provider_state() -> str:
    """Return the persisted active provider ('anthropic' or 'openrouter')."""
    if PROVIDER_STATE_PATH.exists():
        state = json.loads(PROVIDER_STATE_PATH.read_text(encoding="utf-8"))
        return cast(str, state["provider"])
    return "anthropic"


def save_provider_state(provider: str) -> None:
    """Persist the active provider so a restart resumes on it directly."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROVIDER_STATE_PATH.write_text(json.dumps({"provider": provider}), encoding="utf-8")


def build_chat_model(provider: str) -> BaseChatModel:
    """Instantiate the primary (Anthropic) or fallback (OpenRouter) chat model.

    Deliberately bypasses [infrastructure.config.llm_client_factory]: that
    factory backs the single shared production LLM surface and has no
    precedent for a second provider or mid-run fallback, which only this
    one-off curation script needs.
    """
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=api_key, timeout=180)
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. The Anthropic key looks "
                "exhausted -- add OPENROUTER_API_KEY to .env to resume."
            )
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=180,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def call_with_provider_fallback(
    prompt: str, provider: str, output_model: type[BaseModel]
) -> tuple[BaseModel, str]:
    """Invoke a structured-output chain, retrying and falling back on exhaustion.

    Retries transient failures on the active provider; on a
    quota-exhaustion-looking failure persists the switch to OpenRouter and
    retries there, so the rest of the run continues. Any other failure
    re-raises -- there is no sane fallback for a drafting judgment.
    """
    llm = build_chat_model(provider)
    chain = cast(Runnable[str, BaseModel], llm.with_structured_output(output_model))
    last_exc: Exception | None = None
    for attempt in range(1, DEFAULT_LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            return chain.invoke(prompt), provider
        except Exception as exc:
            last_exc = exc
            if provider == "anthropic" and looks_like_quota_exhaustion(exc):
                print(
                    f"WARNING: Anthropic call failed ({exc}) -- looks like "
                    "quota/credit exhaustion. Switching to OpenRouter for the "
                    "rest of this run.",
                    file=sys.stderr,
                )
                save_provider_state("openrouter")
                return call_with_provider_fallback(prompt, "openrouter", output_model)
            if attempt < DEFAULT_LLM_RETRY_MAX_ATTEMPTS:
                time.sleep(DEFAULT_LLM_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


# --- context bundles ------------------------------------------------------


def build_context_bundle(
    records: list[ParsedClauseRecord],
    by_id: dict[str, ParsedClauseRecord],
    children_by_parent: dict[str, list[str]],
    twins: dict[str, frozenset[str]],
    primary_clause_id: str,
) -> list[str]:
    """Return the clause_ids the model may reference for one question.

    What lands here bounds how exhaustive the references can be: the model
    cannot cite a limiting exclusion, a more specific child, or an identical
    twin unless it is shown them. Order is deliberate -- primary first, then
    the structural
    neighbours [M2-08] surfaced, then family -- because the cap bites from
    the end.
    """
    ordered: list[str] = [primary_clause_id]

    def add(clause_id: str) -> None:
        if clause_id in by_id and clause_id not in ordered:
            ordered.append(clause_id)

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


def format_clause_library(
    clause_ids: list[str], by_id: dict[str, ParsedClauseRecord]
) -> str:
    """Render a deduplicated clause library for one document's prompt."""
    blocks: list[str] = []
    for clause_id in clause_ids[:MAX_LIBRARY_CLAUSES]:
        record = by_id[clause_id]
        text = record.text.strip()[:BUNDLE_TEXT_CAP]
        blocks.append(
            f"[{clause_id}]\n"
            f"título: {record.title}\n"
            f"tipo: {record.clause_type.value}\n"
            f"texto: {text}"
        )
    return "\n---\n".join(blocks)


# --- prompts --------------------------------------------------------------

_SHARED_RULES = (
    "Regras de autoria (M2-01):\n"
    "- A pergunta é em português do Brasil, como um analista de sinistros a "
    "faria, e NUNCA repete o texto literal da cláusula.\n"
    "- A pergunta é autossuficiente: não diga 'esta cláusula' nem 'a cláusula "
    "acima'; quem lê a pergunta não sabe qual cláusula você viu.\n"
    "- reference_clause_ids deve ser EXAUSTIVO: inclua TODAS as cláusulas "
    "necessárias para responder, não apenas a principal. Se uma exclusão "
    "limita a cobertura perguntada, ela entra. Se a franquia faz parte da "
    "resposta, ela entra.\n"
    "- Prefira a cláusula-filha específica ao contêiner genérico quando a "
    "resposta estiver na filha.\n"
    "- question_type: 'definition' para verbete de glossário; "
    "'direct_lookup' no resto. Só use 'coverage_with_exclusion' nos slots "
    "explicitamente marcados como tal (nesses, expected_verdict é "
    "obrigatório). Nos demais slots, formule uma pergunta que uma única "
    "cláusula de cobertura responda -- não force o par cobertura+exclusão.\n"
    "- expected_verdict deve ser nulo para direct_lookup e definition.\n"
    "- Em 'coverage_with_exclusion', reference_clause_ids TEM de conter ao "
    "menos uma cláusula de COBERTURA e ao menos uma de EXCLUSÃO. A cobertura "
    "deve ser a garantia especificamente atingida pela exclusão, nunca o "
    "'objeto do seguro' genérico.\n"
    "- Não inclua cláusula que só dá contexto. Inclua uma cláusula apenas se, "
    "sem ela, a pergunta ficar sem resposta completa.\n"
    "- A pergunta não pode dizer 'deste documento', 'desta apólice' nem "
    "'nesta cláusula': quem lê não vê o documento.\n"
    "- Quando o cenário for um sinistro, descreva o evento concreto em vez de "
    "listar o nome do pacote de coberturas.\n"
    "- Use apenas clause_ids da biblioteca oferecida."
)


def display_term(term: object) -> str:
    """Portuguese wording for a vocabulary key, never the internal key."""
    key = str(term)
    return str(VOCAB_DISPLAY_NAMES.get(key, key))


def build_draft_prompt(
    *,
    insurer: str,
    indemnity_regime: str,
    document_id: str,
    library_ids: list[str],
    by_id: dict[str, ParsedClauseRecord],
    slots: list[dict[str, object]],
) -> str:
    """Build the drafting prompt for one document's new/corrected slots."""
    slot_blocks: list[str] = []
    for slot in slots:
        lines = [
            f"row_id: {slot['row_id']}  (devolva este row_id exatamente)",
            f"slot_clause_id: {slot['clause_id']}",
            f"ids permitidos: {', '.join(cast(list[str], slot['bundle_ids']))}",
        ]
        if slot.get("allow_coverage_with_exclusion"):
            lines.append(
                "TIPO PERMITIDO: este slot PODE ser 'coverage_with_exclusion' "
                "(a revisão identificou que a pergunta já é de cobertura + "
                "exclusão). Preencha expected_verdict."
            )
        else:
            lines.append("TIPO: use apenas 'direct_lookup' ou 'definition' neste slot.")
        anchor = by_id.get(str(slot["clause_id"]))
        if anchor is not None and selection.is_glossary_title(anchor.title):
            # A bundled filing's glossary defines terms for every product it
            # covers, so left to itself the model reaches for "Acidente
            # Pessoal". 12-00 was rejected for exactly that and its
            # replacement 12-05 repeated it.
            lines.append(
                "ATENÇÃO — GLOSSÁRIO MULTIPRODUTO: esta cláusula define termos "
                "de vários produtos. A pergunta TEM de tratar de um verbete de "
                "dano próprio ao veículo (por exemplo: indenização integral, "
                "salvado, avaria, franquia, valor de mercado referenciado, "
                "sinistro, colisão, incêndio, roubo/furto). NUNCA use verbetes "
                "de acidentes pessoais/APP, invalidez, residência ou material "
                "institucional."
            )
        if slot.get("decision"):
            lines.append(str(slot["decision"]))
        if slot.get("correction"):
            lines.append(
                "CORREÇÃO EXIGIDA PELO AUTOR (siga à risca, esta pergunta "
                f"está sendo refeita): {slot['correction']}"
            )
        if slot.get("previous_question"):
            lines.append(
                f"pergunta anterior (a substituir): {slot['previous_question']}"
            )
        if slot.get("target_term"):
            lines.append(
                "TERMO OBRIGATÓRIO: a pergunta deve descrever um CENÁRIO "
                "DE SINISTRO concreto envolvendo "
                f"'{display_term(slot['target_term'])}' "
                "e usar esse vocabulário no enunciado em português corrente. "
                "Não basta citar o termo dentro do nome de um pacote de "
                "coberturas."
            )
        if slot.get("soft_target_term"):
            if slot.get("needs_scenario"):
                prefix = (
                    "CENÁRIO OBRIGATÓRIO: descreva um sinistro concreto (o que "
                    "aconteceu com o veículo) envolvendo "
                )
            else:
                prefix = (
                    "TERMO PREFERENCIAL: se — e somente se — couber "
                    "naturalmente nesta cláusula, trate de "
                )
            lines.append(
                prefix
                + f"'{display_term(slot['soft_target_term'])}' "
                + "em português corrente. Nunca distorça a pergunta nem a "
                "correção exigida para encaixar o termo, e nunca escreva o "
                "identificador interno do termo."
            )
        slot_blocks.append("\n".join(lines))

    return (
        "Você redige perguntas de avaliação (golden set) para um sistema de "
        "RAG que responde a analistas de sinistros sobre apólices de seguro "
        "auto (CASCO) brasileiras.\n\n"
        f"Documento: document_id={document_id}, seguradora={insurer}, "
        f"regime de indenização={indemnity_regime}.\n\n"
        f"{_SHARED_RULES}\n\n"
        f"BIBLIOTECA DE CLÁUSULAS:\n{format_clause_library(library_ids, by_id)}\n\n"
        f"SLOTS ({len(slots)} perguntas, uma por slot):\n"
        + "\n\n".join(slot_blocks)
        + f"\n\nRetorne exatamente {len(slots)} pergunta(s), uma por slot."
    )


def build_completeness_prompt(
    *,
    document_id: str,
    library_ids: list[str],
    by_id: dict[str, ParsedClauseRecord],
    questions: list[dict[str, object]],
) -> str:
    """Build the completeness prompt: exhaustive refs plus rejection reasons."""
    question_blocks = []
    for item in questions:
        question_blocks.append(
            f"row_id: {item['row_id']}\n"
            f"pergunta: {item['question']}\n"
            f"referências atuais: {', '.join(cast(list[str], item['current_refs']))}\n"
            f"ids permitidos: {', '.join(cast(list[str], item['allowed_ids']))}"
        )

    return (
        "Você executa a verificação de COMPLETUDE de um golden set de RAG "
        "sobre apólices de seguro auto (CASCO) brasileiras. NÃO reescreva "
        "nenhuma pergunta: seu trabalho é apenas decidir, para cada uma, "
        "quais cláusulas são necessárias para respondê-la de forma "
        "exaustiva.\n\n"
        f"Documento: document_id={document_id}.\n\n"
        "Para cada pergunta devolva:\n"
        "- reference_clause_ids: o conjunto MÍNIMO de clause_ids que, juntos, "
        "respondem à pergunta por completo. Um recuperador que devolva todos "
        "esses ids deve conseguir responder; se faltar algum, o gabarito está "
        "errado. Inclua exclusões que limitem a cobertura perguntada e a "
        "franquia quando ela faça parte da resposta.\n"
        "- included: para CADA id que você INCLUIU, uma linha dizendo o que a "
        "pergunta ficaria sem responder caso ele faltasse. Se a cláusula "
        "apenas dá contexto, tire-a da resposta. Exaustivo significa 'toda "
        "cláusula necessária', não 'toda cláusula do entorno'.\n"
        "- rejected: para CADA id permitido que você NÃO incluiu, uma linha "
        "dizendo por quê. Esta lista é a trilha de auditoria da verificação "
        "de completude; não a deixe vazia se houver ids permitidos fora da "
        "resposta.\n\n"
        f"BIBLIOTECA DE CLÁUSULAS:\n{format_clause_library(library_ids, by_id)}\n\n"
        f"PERGUNTAS ({len(questions)}):\n" + "\n\n".join(question_blocks)
    )


# --- caching --------------------------------------------------------------


def cache_key(payload: str) -> str:
    """Content hash used to skip an unchanged LLM call on a rerun."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cached(name: str, fingerprint: str) -> dict[str, object] | None:
    """Return a cached LLM response if its fingerprint still matches."""
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    cached = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if cached.get("fingerprint") != fingerprint:
        return None
    return cached


def save_cached(
    name: str, fingerprint: str, provider: str, payload: dict[str, object]
) -> None:
    """Persist an LLM response keyed by its prompt fingerprint."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(
        json.dumps(
            {"fingerprint": fingerprint, "provider": provider, "payload": payload},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# --- CSV ------------------------------------------------------------------

CSV_FIELDNAMES = [
    "row_id",
    "document_id",
    "insurer",
    "indemnity_regime",
    "filename",
    "origin",
    "review_verdict",
    "review_codes",
    "review_correction",
    "primary_source_clause_id",
    "primary_source_clause_title",
    "source_clause_text",
    "source_clause_text_chars",
    "scope",
    "bundle_section",
    "question_scope_flag",
    "self_reference_flag",
    "coverage_exclusion_gap",
    "source_clause_flags",
    "twin_clause_ids",
    "question",
    "question_type",
    "difficulty",
    "expected_verdict",
    "reference_clause_ids",
    "reference_clause_texts",
    "vocabulary_terms_hit_question",
    "scenario_terms",
    "indemnity_basis_question",
    "near_duplicate_of",
    "type_coerced_from",
    "completeness_pool_size",
    "completeness_considered_ids",
    "completeness_included_reasons",
    "completeness_rejected_reasons",
    "draft_notes",
    "notes",
    "authored_at",
    "provider_used",
    "approved",
    "finalized_question_id",
]


def load_existing_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read a draft CSV if it exists, indexed by row_id."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle)}


def sort_key_for_row_id(row_id: str) -> tuple[int, int]:
    """Order rows by document then slot, so the CSV reads document by document."""
    document_part, slot_part = row_id.rsplit("-", 1)
    return (int(document_part), int(slot_part))


def write_csv(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    """Write the merged row set, sorted by (document_id, slot)."""
    ordered = [rows_by_id[key] for key in sorted(rows_by_id, key=sort_key_for_row_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)


def write_selection_audit(path: Path, rows: list[dict[str, str]]) -> None:
    """Write every considered clause with its include/exclude reason."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_AUDIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# Each review round names its columns differently (`aprovado` /
# `veredito_rodada2`, `correcao_sugerida` / `correcao_rodada2`). Normalising
# on read keeps the routing logic indifferent to which round's file it is
# handed, so a later round needs no code change.
_REVIEW_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "verdict": ("aprovado", "veredito_rodada2", "veredito", "veredito_rodada3"),
    "correction": ("correcao_sugerida", "correcao_rodada2", "correcao_rodada3"),
    "codes": ("codigos_problema", "codigos_rodada2", "codigos_rodada3"),
    "question": ("question",),
    "question_type": ("question_type_atual", "question_type"),
    "reference_clause_ids": ("reference_clause_ids_atual", "reference_clause_ids"),
    "document_id": ("document_id",),
}


def normalise_review_row(row: dict[str, str]) -> dict[str, str]:
    """Map one review row onto canonical keys, whichever round wrote it."""
    normalised: dict[str, str] = {}
    for canonical, aliases in _REVIEW_COLUMN_ALIASES.items():
        for alias in aliases:
            value = row.get(alias)
            if value is not None and value.strip():
                normalised[canonical] = value
                break
        normalised.setdefault(canonical, "")
    return normalised


def load_review(path: Path) -> dict[str, dict[str, str]]:
    """Read the author's review verdicts, indexed by row_id and normalised."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Pass --review-csv <path>.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["row_id"]: normalise_review_row(row) for row in csv.DictReader(handle)
        }


# --- reporting ------------------------------------------------------------


_STOPWORDS = frozenset(
    """a o as os um uma de do da dos das em no na nos nas por para com e ou que
    qual quais como se ao aos à às é são ser foi sobre entre seu sua seus suas
    esse essa este esta isso apólice seguro segurado veículo""".split()
)


def content_words(question: str) -> frozenset[str]:
    """Significant words of a question, for topic comparison."""
    words = re.findall(r"[a-zà-ÿ]{3,}", question.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def find_near_duplicates(questions_by_row: dict[str, str]) -> dict[str, str]:
    """Map row_id to another row_id asking substantially the same thing.

    Lexical similarity alone is the wrong instrument: "Como a apólice define
    o Limite Máximo de Indenização (LMI)?" and the same question with a
    trailing clause scored 0.648 and slipped under a 0.85 threshold while
    being the same question (review code E7). Topic overlap on content words
    catches the paraphrase; the sequence ratio still catches near-verbatim
    repeats that share few distinctive words.
    """
    matches: dict[str, str] = {}
    items = sorted(questions_by_row.items(), key=lambda kv: sort_key_for_row_id(kv[0]))
    for index, (row_id, question) in enumerate(items):
        normalised = " ".join(question.lower().split())
        words = content_words(question)
        for other_id, other in items[index + 1 :]:
            other_words = content_words(other)
            union = words | other_words
            overlap = len(words & other_words)
            jaccard = overlap / len(union) if union else 0.0
            smaller = min(len(words), len(other_words))
            containment = overlap / smaller if smaller else 0.0
            ratio = difflib.SequenceMatcher(
                None, normalised, " ".join(other.lower().split())
            ).ratio()
            if (
                jaccard >= NEAR_DUPLICATE_JACCARD
                or ratio >= NEAR_DUPLICATE_RATIO
                or (
                    smaller >= NEAR_DUPLICATE_MIN_WORDS
                    and containment >= NEAR_DUPLICATE_CONTAINMENT
                )
            ):
                matches.setdefault(other_id, row_id)
    return matches


def print_coverage_report(rows: list[dict[str, str]]) -> bool:
    """Print the DoD report scored on QUESTION text. Returns True if it passes."""
    passed = True
    per_doc: dict[str, int] = defaultdict(int)
    for row in rows:
        per_doc[row["document_id"]] += 1

    print(f"\nTotal questions: {len(rows)} across {len(per_doc)} documents")
    for doc_id in sorted(per_doc, key=int):
        count = per_doc[doc_id]
        flag = (
            ""
            if MIN_QUESTIONS_PER_DOC <= count <= MAX_QUESTIONS_PER_DOC
            else "  <-- OUT OF RANGE"
        )
        if flag:
            passed = False
        print(f"  document {doc_id}: {count}{flag}")

    print("\nVocabulary coverage (DoD surface): mentions / claim scenarios")
    vocab_counts: dict[str, int] = defaultdict(int)
    scenario_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for term in row["vocabulary_terms_hit_question"].split(","):
            if term:
                vocab_counts[term] += 1
        for term in row["scenario_terms"].split(","):
            if term:
                scenario_counts[term] += 1
    for term in sorted(VOCAB_PATTERNS):
        mentions = vocab_counts.get(term, 0)
        scenarios = scenario_counts.get(term, 0)
        problems = []
        if mentions < MIN_VOCAB_QUESTIONS_PER_TERM:
            problems.append(f"mentions<{MIN_VOCAB_QUESTIONS_PER_TERM}")
        if scenarios < MIN_VOCAB_SCENARIOS_PER_TERM:
            problems.append(f"scenarios<{MIN_VOCAB_SCENARIOS_PER_TERM}")
        if problems:
            passed = False
        flag = f"  <-- {', '.join(problems)}" if problems else ""
        print(f"  {term:20s} {mentions:>2} mentions / {scenarios:>2} scenarios{flag}")

    basis_rows = [r for r in rows if r["indemnity_basis_question"] == "Y"]
    by_regime: dict[str, int] = defaultdict(int)
    for row in basis_rows:
        by_regime[row["indemnity_regime"]] += 1
    print(
        f"\nIndemnity-basis questions (question text): {len(basis_rows)}, "
        f"by regime: {dict(sorted(by_regime.items()))}"
    )
    if len(basis_rows) < MIN_INDEMNITY_BASIS_QUESTIONS or len(by_regime) < 3:
        passed = False
        print("  <-- BELOW FLOOR (need >=5 spanning VD, VMR and VD+VMR)")

    exhaustive = sum(1 for r in rows if len(r["reference_clause_ids"].split(";")) > 1)
    print(f"\nMulti-clause reference sets: {exhaustive}/{len(rows)}")
    audited = sum(1 for r in rows if r["completeness_rejected_reasons"].strip())
    print(f"Rows with a recorded completeness audit trail: {audited}/{len(rows)}")

    # Structural violations fail the gate rather than merely annotating the
    # row: a flag nobody is forced to act on gets shipped past.
    for column, label in (
        ("question_scope_flag", "question out of scope"),
        ("self_reference_flag", "self-referential question"),
        ("coverage_exclusion_gap", "coverage_with_exclusion missing a side"),
        ("near_duplicate_of", "near-duplicate question"),
    ):
        offenders = [r["row_id"] for r in rows if r[column].strip()]
        if offenders:
            passed = False
            print(f"\nFAIL - {label}: {offenders}")

    shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        shared[(row["document_id"], row["primary_source_clause_id"])].append(
            row["row_id"]
        )
    collisions = [ids for ids in shared.values() if len(ids) > 1]
    if collisions:
        passed = False
        print(f"\nFAIL - rows sharing one source clause in a document: {collisions}")

    by_type: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[row["question_type"]] += 1
    print(f"\nquestion_type: {dict(sorted(by_type.items()))}")
    return passed


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
        help="Print selection, routing and gap analysis; never call an LLM.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901 - one linear pipeline, documented in sections
    """Draft or repair the CASCO golden-set questions."""
    load_dotenv()
    args = _parse_args()

    records = load_corpus()
    documents = load_casco_documents(MANIFEST_PATH)
    if len(documents) != 15:
        raise ValueError(f"Expected 15 CASCO documents, found {len(documents)}")

    by_id = {record.clause_id: record for record in records}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.parent_id is not None:
            children_by_parent[record.parent_id].append(record.clause_id)
    ancestor_titles = build_ancestor_titles(records)
    twins = build_duplicate_text_index(records)

    meta_by_doc = {row["id"]: row for row in documents}
    regime_by_doc = {row["id"]: row["indemnity_regime"] for row in documents}
    candidates_by_doc = {
        row["id"]: build_candidates_for_document(
            records, row["id"], ancestor_titles, twins
        )
        for row in documents
    }
    candidate_by_clause_id = {
        candidate.clause.clause_id: candidate
        for candidates in candidates_by_doc.values()
        for candidate in candidates
    }

    existing = load_existing_csv(DRAFT_CSV_PATH)
    review = load_review(args.review_csv) if args.review_csv else {}
    if review:
        tally = Counter(
            r["verdict"].strip().lower() or "(blank)" for r in review.values()
        )
        print(f"Review file: {args.review_csv}  ({len(review)} rows) {dict(tally)}")

    # The review file, not the draft CSV, is the authority on which rows the
    # author actually saw. A reviewed row missing from the CSV is rebuilt
    # from the review's own columns rather than silently refilled with a new
    # replacement -- that is what happened to 2-01 and 3-00, whose
    # corrections asked for a re-anchor the earlier code could not express.
    for row_id, review_row in review.items():
        if review_row.get("verdict", "").strip() == "nao":
            continue
        if row_id in existing:
            for column, key in (
                ("question", "question"),
                ("question_type", "question_type"),
                ("reference_clause_ids", "reference_clause_ids"),
            ):
                value = review_row.get(key, "").strip()
                if value:
                    existing[row_id][column] = value
            continue
        refs = [
            cid
            for cid in review_row.get("reference_clause_ids", "")
            .replace(",", ";")
            .split(";")
            if cid.strip()
        ]
        if not refs:
            continue
        rebuilt = dict.fromkeys(CSV_FIELDNAMES, "")
        rebuilt.update(
            {
                "row_id": row_id,
                "document_id": review_row.get("document_id", row_id.split("-")[0]),
                "question": review_row.get("question", ""),
                "question_type": review_row.get("question_type") or "direct_lookup",
                "difficulty": "medium",
                "reference_clause_ids": ";".join(cid.strip() for cid in refs),
                "primary_source_clause_id": refs[0].strip(),
            }
        )
        existing[row_id] = rebuilt
        print(f"Rebuilt reviewed row {row_id} from the review file")

    # --- route every reviewed row by the author's verdict ------------------
    # In repair mode the review file defines the universe of reviewed rows.
    # Any row in the CSV that the author never saw is a previous repair run's
    # own output, and must be regenerated rather than inherited: reading it
    # back as an existing row makes the run non-idempotent, which is how a
    # replacement drafted under the earlier, unrestricted type enum survived
    # a rerun labelled `kept` and carrying `coverage_with_exclusion` with a
    # single reference.
    kept: dict[str, dict[str, str]] = {}
    corrected: dict[str, dict[str, str]] = {}
    dropped: list[str] = []
    for row_id, row in existing.items():
        if review and row_id not in review:
            continue
        author_verdict = (review.get(row_id, {}).get("verdict") or "").strip().lower()
        if row_id in AUTHOR_DECISIONS:
            # An explicit instruction overrides a rejection: the author said
            # what this row should become, so it is redrafted rather than
            # dropped and refilled by an unrelated replacement.
            corrected[row_id] = row
        elif author_verdict == "nao":
            dropped.append(row_id)
        elif author_verdict == "revisar":
            corrected[row_id] = row
        else:
            kept[row_id] = row

    # A structural violation is not a matter of taste, so an approved row
    # carrying one is redrafted rather than shipped -- a coverage_with_exclusion
    # missing its coverage clause measures nothing [M3-06] cares about, however
    # good the question reads.
    auto_corrections: dict[str, str] = {}
    surviving_questions = {
        row_id: existing[row_id]["question"] for row_id in list(kept) + list(corrected)
    }
    surviving_duplicates = find_near_duplicates(surviving_questions)
    for row_id in list(kept):
        row = kept[row_id]
        refs = [c for c in row.get("reference_clause_ids", "").split(";") if c]
        problems: list[str] = []
        if question_scope_flag(row["question"]):
            problems.append(
                "a pergunta trata de produto fora do escopo (dano próprio/CASCO); "
                "reformule sobre um tema de casco da mesma cláusula"
            )
        phrase = question_self_reference_flag(row["question"])
        if phrase:
            problems.append(
                f"a pergunta é auto-referencial ('{phrase}'); reescreva de forma "
                "autossuficiente"
            )
        if row.get("question_type") == "coverage_with_exclusion":
            gap = coverage_with_exclusion_gap(refs, by_id)
            if gap:
                problems.append(
                    f"gabarito de coverage_with_exclusion incompleto ({gap}): "
                    "inclua a cláusula de cobertura especificamente atingida "
                    "pela exclusão E a exclusão"
                )
        if row_id in surviving_duplicates:
            problems.append(
                f"pergunta quase idêntica à {surviving_duplicates[row_id]}; "
                "diversifique o tema mantendo a mesma cláusula"
            )
        if problems:
            auto_corrections[row_id] = "CORREÇÃO ESTRUTURAL: " + "; ".join(problems)
            corrected[row_id] = kept.pop(row_id)

    if auto_corrections:
        print(
            f"Escalated {len(auto_corrections)} approved row(s) on structural grounds:"
        )
        for row_id in sorted(auto_corrections, key=sort_key_for_row_id):
            print(f"  ! {row_id}: {auto_corrections[row_id][:110]}")

    surviving_by_doc: dict[str, int] = defaultdict(int)
    for row_id in list(kept) + list(corrected):
        surviving_by_doc[existing[row_id]["document_id"]] += 1

    # --- close vocabulary gaps where it costs the least --------------------
    # Only `kept` rows have text that is already fixed, so only they count
    # toward the floor. The 32 rows being redrafted anyway are the cheapest
    # place to close a gap: attaching a soft target there spreads coverage
    # across many documents, whereas loading it onto the 9 freed slots would
    # force three incêndio questions out of document 6, which has exactly one
    # clause actually about fire.
    vocab_counts: dict[str, int] = dict.fromkeys(VOCAB_PATTERNS, 0)
    scenario_counts: dict[str, int] = dict.fromkeys(VOCAB_PATTERNS, 0)
    for row_id in kept:
        kept_question = existing[row_id]["question"]
        for term in score_question_vocabulary(kept_question):
            vocab_counts[term] += 1
        for term in score_question_scenarios(
            kept_question, existing[row_id].get("question_type", "")
        ):
            scenario_counts[term] += 1

    # Two slots of headroom above the floor. A soft target only lands if the
    # term fits the clause naturally, so some always fall away: with one slot
    # of headroom ambito_geografico finished at 2, and a later redraft of
    # document 11 dropped rc_facultativa to 2 the same way. Over-assigning
    # costs nothing -- a term already at its floor stops being a deficit.
    soft_target_ceiling = MIN_VOCAB_QUESTIONS_PER_TERM + 2

    scenario_ceiling = MIN_VOCAB_SCENARIOS_PER_TERM + 1

    def deficit_terms(ceiling: int = MIN_VOCAB_QUESTIONS_PER_TERM) -> list[str]:
        """Terms below a floor, scenario shortfalls first.

        The gate now requires claim scenarios as well as mentions, and
        incêndio failed at 2 mentions / 1 scenario precisely because targets
        were assigned on mention counts alone -- the redrafts named the
        coverage and never described a fire.
        """
        scenario_short = sorted(
            (
                t
                for t in VOCAB_PATTERNS
                if scenario_counts[t] < MIN_VOCAB_SCENARIOS_PER_TERM
            ),
            key=lambda t: scenario_counts[t],
        )
        mention_short = sorted(
            (
                t
                for t in VOCAB_PATTERNS
                if vocab_counts[t] < ceiling and t not in scenario_short
            ),
            key=lambda t: vocab_counts[t],
        )
        return scenario_short + mention_short

    # Scenarios get headroom for the same reason mentions do: a soft target
    # lands only where the clause supports it, so aiming exactly at the floor
    # leaves nothing for attrition -- franquia fell from 2 scenarios to 1 when
    # a document-11 redraft moved.
    def needs_scenario(term: str) -> bool:
        return bool(scenario_counts[term] < scenario_ceiling)

    soft_targets: dict[str, str] = {}
    scenario_targets: set[str] = set()
    for row_id in sorted(corrected, key=sort_key_for_row_id):
        pending_terms = deficit_terms(soft_target_ceiling)
        if not pending_terms:
            break
        clause = by_id.get(existing[row_id]["primary_source_clause_id"])
        if clause is None:
            continue
        candidate = candidate_by_clause_id.get(clause.clause_id)
        hits = candidate.vocab_hits if candidate else frozenset()
        choice = next(
            (t for t in pending_terms if term_in_title(t, clause.title)), None
        ) or next((t for t in pending_terms if t in hits), None)
        if choice is not None:
            soft_targets[row_id] = choice
            if needs_scenario(choice):
                scenario_targets.add(row_id)
                scenario_counts[choice] += 1
            vocab_counts[choice] += 1

    used_clause_ids = {
        existing[row_id]["primary_source_clause_id"]
        for row_id in list(kept) + list(corrected)
    }
    indemnity_docs = select_indemnity_basis_documents(candidates_by_doc, regime_by_doc)

    replacements: list[dict[str, object]] = []
    for row in documents:
        doc_id = row["id"]
        take = MAX_QUESTIONS_PER_DOC - surviving_by_doc[doc_id]
        if take <= 0:
            continue
        pool = [
            c
            for c in candidates_by_doc[doc_id]
            if c.is_selectable and c.clause.clause_id not in used_clause_ids
        ]
        for _ in range(take):
            # A hard target is only assigned when the document has a clause
            # whose TITLE carries the term. Anything weaker manufactures a
            # question the clause cannot actually answer.
            target = next(
                (
                    term
                    for term in deficit_terms()
                    if any(term_in_title(term, c.clause.title) for c in pool)
                ),
                None,
            )
            if target is not None:
                target_term = target
                chosen = min(
                    (c for c in pool if term_in_title(target_term, c.clause.title)),
                    key=lambda c: c.clause.clause_id,
                )
            else:
                picked = pick_slots(
                    pool,
                    vocab_counts,
                    want_indemnity_basis=doc_id in indemnity_docs,
                    limit=1,
                )
                if not picked:
                    break
                chosen = picked[0]
            pool = [c for c in pool if c.clause.clause_id != chosen.clause.clause_id]
            used_clause_ids.add(chosen.clause.clause_id)
            if target is not None:
                vocab_counts[target] += 1
                if needs_scenario(target):
                    scenario_counts[target] += 1
            replacements.append(
                {
                    "document_id": doc_id,
                    "candidate": chosen,
                    "target_term": target,
                }
            )

    # --- assign row_ids to replacements, continuing each document's slots -
    next_slot: dict[str, int] = defaultdict(int)
    for row_id in review or existing:
        doc_part, slot_part = row_id.rsplit("-", 1)
        next_slot[doc_part] = max(next_slot[doc_part], int(slot_part) + 1)
    for replacement in replacements:
        doc_id = cast(str, replacement["document_id"])
        replacement["row_id"] = f"{doc_id}-{next_slot[doc_id]:02d}"
        next_slot[doc_id] += 1

    print(
        f"Routing: {len(kept)} kept, {len(corrected)} to correct, "
        f"{len(dropped)} dropped, {len(replacements)} replacements"
    )
    for replacement in replacements:
        candidate = cast(ClauseCandidate, replacement["candidate"])
        print(
            f"  + {replacement['row_id']:6s} doc {replacement['document_id']:>2} "
            f"hard-term={replacement['target_term'] or '-':18s} "
            f"{candidate.clause.clause_id}"
        )
    for row_id, term in sorted(
        soft_targets.items(), key=lambda kv: sort_key_for_row_id(kv[0])
    ):
        print(f"  ~ {row_id:6s} redraft soft-term={term}")

    selected_ids = frozenset(used_clause_ids)
    write_selection_audit(
        SELECTION_AUDIT_PATH,
        build_selection_audit_rows(candidates_by_doc, selected_ids),
    )
    print(f"Wrote selection audit to {SELECTION_AUDIT_PATH}")

    if args.dry_run:
        print("\n--dry-run: no LLM calls, no CSV written.")
        print("Vocabulary counts (question text) before replacements are drafted:")
        for term in sorted(VOCAB_PATTERNS):
            print(f"  {term:20s} {vocab_counts[term]}")
        return

    provider = load_provider_state()
    rows: dict[str, dict[str, str]] = {}

    bundles: dict[str, list[str]] = {}

    def bundle_for(clause_id: str) -> list[str]:
        if clause_id not in bundles:
            bundles[clause_id] = build_context_bundle(
                records, by_id, children_by_parent, twins, clause_id
            )
        return bundles[clause_id]

    # --- pass A: draft corrected rows and replacements --------------------
    drafted: dict[str, DraftedQuestion] = {}
    coerced_types: set[str] = set()
    for doc_row in documents:
        doc_id = doc_row["id"]
        slots: list[dict[str, object]] = []
        for row_id, row in corrected.items():
            if row["document_id"] != doc_id:
                continue
            clause_id = row["primary_source_clause_id"]
            slots.append(
                {
                    "row_id": row_id,
                    "clause_id": clause_id,
                    "bundle_ids": bundle_for(clause_id),
                    "correction": " | ".join(
                        part
                        for part in (
                            review.get(row_id, {}).get("correction", ""),
                            auto_corrections.get(row_id, ""),
                        )
                        if part
                    ),
                    "previous_question": (
                        review.get(row_id, {}).get("question") or row["question"]
                    ),
                    "soft_target_term": soft_targets.get(row_id),
                    "needs_scenario": row_id in scenario_targets,
                    "decision": AUTHOR_DECISIONS.get(row_id, ""),
                    "allow_coverage_with_exclusion": row_id in EXCLUSION_RETYPE_ROW_IDS,
                }
            )
        for replacement in replacements:
            if replacement["document_id"] != doc_id:
                continue
            candidate = cast(ClauseCandidate, replacement["candidate"])
            clause_id = candidate.clause.clause_id
            slots.append(
                {
                    "row_id": replacement["row_id"],
                    "clause_id": clause_id,
                    "bundle_ids": bundle_for(clause_id),
                    "target_term": replacement["target_term"],
                }
            )
        if not slots:
            continue

        library_ids: list[str] = []
        for slot in slots:
            for clause_id in cast(list[str], slot["bundle_ids"]):
                if clause_id not in library_ids:
                    library_ids.append(clause_id)

        prompt = build_draft_prompt(
            insurer=doc_row["insurer"],
            indemnity_regime=doc_row["indemnity_regime"],
            document_id=doc_id,
            library_ids=library_ids,
            by_id=by_id,
            slots=slots,
        )
        fingerprint = cache_key(prompt)
        cached = load_cached(f"draft_document_{doc_id}", fingerprint)
        if cached is not None:
            batch = DraftedQuestionsForDocument.model_validate(cached["payload"])
            used_provider = cast(str, cached["provider"])
            print(f"document {doc_id}: reusing cached draft ({used_provider})")
        else:
            print(
                f"document {doc_id}: drafting {len(slots)} question(s) "
                f"via {provider}..."
            )
            result, used_provider = call_with_provider_fallback(
                prompt, provider, DraftedQuestionsForDocument
            )
            batch = cast(DraftedQuestionsForDocument, result)
            provider = used_provider
            save_cached(
                f"draft_document_{doc_id}",
                fingerprint,
                used_provider,
                batch.model_dump(mode="json"),
            )

        by_row = {q.row_id: q for q in batch.questions}
        by_clause = {q.slot_clause_id: q for q in batch.questions}
        for slot in slots:
            slot_row_id = cast(str, slot["row_id"])
            slot_clause_id = cast(str, slot["clause_id"])
            # Keyed on row_id, not clause_id: a correction may legitimately
            # tell the model to re-anchor onto a more specific child clause
            # (review code E10), and keying on the original clause silently
            # dropped exactly those rows -- 2-01 and 3-00 were lost that way.
            question = by_row.get(slot_row_id) or by_clause.get(slot_clause_id)
            if question is None:
                print(
                    f"WARNING: no drafted question for slot "
                    f"{slot_row_id} ({slot_clause_id})",
                    file=sys.stderr,
                )
                continue
            allowed_ids = set(cast(list[str], slot["bundle_ids"]))
            if question.slot_clause_id not in allowed_ids:
                question.slot_clause_id = slot_clause_id
            question.reference_clause_ids = [
                cid for cid in question.reference_clause_ids if cid in allowed_ids
            ] or [question.slot_clause_id]
            if (
                question.question_type == DraftableQuestionType.COVERAGE_WITH_EXCLUSION
                and slot_row_id not in EXCLUSION_RETYPE_ROW_IDS
            ):
                # Scope guard, not a relabel of the content: M2-02 only
                # retypes the three rows the review identified. Flagged so
                # the reviewer can send a genuine pair to [M2-03].
                question.question_type = DraftableQuestionType.DIRECT_LOOKUP
                coerced_types.add(slot_row_id)
            drafted[slot_row_id] = question

    # --- assemble every surviving row before the completeness pass --------
    def base_row(
        row_id: str, document_id: str, clause_id: str, origin: str
    ) -> dict[str, str]:
        meta = meta_by_doc[document_id]
        candidate = candidate_by_clause_id.get(clause_id)
        clause = by_id[clause_id]
        text = clause.text.strip()
        review_row = review.get(row_id, {})
        return {
            "row_id": row_id,
            "document_id": document_id,
            "insurer": meta["insurer"],
            "indemnity_regime": meta["indemnity_regime"],
            "filename": meta["filename"],
            "origin": origin,
            "review_verdict": review_row.get("verdict", ""),
            "review_codes": review_row.get("codes", ""),
            "review_correction": review_row.get("correction", ""),
            "primary_source_clause_id": clause_id,
            "primary_source_clause_title": clause.title,
            "source_clause_text": text[:SOURCE_TEXT_CAP],
            "source_clause_text_chars": str(len(text)),
            "scope": candidate.scope if candidate else "",
            "source_clause_flags": ",".join(candidate.exclusion_reasons)
            if candidate
            else "",
            "twin_clause_ids": ";".join(sorted(twins.get(clause_id, frozenset()))),
            "authored_at": "",
            "approved": "",
            "finalized_question_id": "",
            "notes": "",
        }

    pending: dict[str, dict[str, object]] = {}
    for row_id, row in kept.items():
        clause_id = row["primary_source_clause_id"]
        pending[row_id] = {
            "row": base_row(row_id, row["document_id"], clause_id, ORIGIN_KEPT),
            "question": row["question"],
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "expected_verdict": row["expected_verdict"],
            "refs": [c for c in row["reference_clause_ids"].split(";") if c],
            "draft_notes": row.get("draft_notes", ""),
            "provider": row.get("provider_used", ""),
            "clause_id": clause_id,
        }
    for row_id, question in drafted.items():
        origin = ORIGIN_CORRECTED if row_id in corrected else ORIGIN_REPLACEMENT
        document_id = (
            corrected[row_id]["document_id"]
            if row_id in corrected
            else next(
                cast(str, r["document_id"])
                for r in replacements
                if r["row_id"] == row_id
            )
        )
        if question.question_type != DraftableQuestionType.COVERAGE_WITH_EXCLUSION:
            question.expected_verdict = None
        pending[row_id] = {
            "row": base_row(row_id, document_id, question.slot_clause_id, origin),
            "question": question.question,
            "question_type": question.question_type.value,
            "difficulty": question.difficulty.value,
            "expected_verdict": (
                question.expected_verdict.value if question.expected_verdict else ""
            ),
            "refs": question.reference_clause_ids,
            "draft_notes": question.reasoning,
            "provider": provider,
            "clause_id": question.slot_clause_id,
        }

    # --- pass B: completeness over every surviving row --------------------
    for doc_row in documents:
        doc_id = doc_row["id"]
        items = [
            (row_id, item)
            for row_id, item in sorted(
                pending.items(), key=lambda kv: sort_key_for_row_id(kv[0])
            )
            if cast(dict[str, str], item["row"])["document_id"] == doc_id
        ]
        if not items:
            continue

        questions: list[dict[str, object]] = []
        library_ids = []
        for row_id, item in items:
            bundle_ids = bundle_for(cast(str, item["clause_id"]))
            for clause_id in bundle_ids:
                if clause_id not in library_ids:
                    library_ids.append(clause_id)
            questions.append(
                {
                    "row_id": row_id,
                    "question": item["question"],
                    "current_refs": item["refs"],
                    "allowed_ids": bundle_ids,
                }
            )

        prompt = build_completeness_prompt(
            document_id=doc_id,
            library_ids=library_ids,
            by_id=by_id,
            questions=questions,
        )
        fingerprint = cache_key(prompt)
        cached = load_cached(f"completeness_document_{doc_id}", fingerprint)
        if cached is not None:
            batch_c = CompletenessForDocument.model_validate(cached["payload"])
            used_provider = cast(str, cached["provider"])
            print(f"document {doc_id}: reusing cached completeness ({used_provider})")
        else:
            print(
                f"document {doc_id}: completeness over {len(questions)} "
                f"question(s) via {provider}..."
            )
            result, used_provider = call_with_provider_fallback(
                prompt, provider, CompletenessForDocument
            )
            batch_c = cast(CompletenessForDocument, result)
            provider = used_provider
            save_cached(
                f"completeness_document_{doc_id}",
                fingerprint,
                used_provider,
                batch_c.model_dump(mode="json"),
            )

        verdicts = {v.row_id: v for v in batch_c.verdicts}
        for row_id, item in items:
            completeness = verdicts.get(row_id)
            allowed_ids = set(bundle_for(cast(str, item["clause_id"])))
            if completeness is None:
                print(f"WARNING: no completeness verdict for {row_id}", file=sys.stderr)
                item["completeness"] = None
                continue
            refs = [
                cid for cid in completeness.reference_clause_ids if cid in allowed_ids
            ]
            if not refs:
                refs = [cast(str, item["clause_id"])]
            # A retriever returning a byte-identical twin of a referenced
            # clause is not wrong; the twin ids are indistinguishable by
            # content, so they belong in the ground truth together.
            expanded = list(refs)
            for clause_id in refs:
                for twin in sorted(twins.get(clause_id, frozenset())):
                    if twin not in expanded:
                        expanded.append(twin)
            item["refs"] = expanded
            item["completeness"] = completeness

    # --- finalise rows ----------------------------------------------------
    questions_by_row = {
        row_id: cast(str, i["question"]) for row_id, i in pending.items()
    }
    near_duplicates = find_near_duplicates(questions_by_row)

    for row_id, item in pending.items():
        row = cast(dict[str, str], item["row"])
        refs = cast(list[str], item["refs"])
        question_text = cast(str, item["question"])
        completeness = cast(CompletenessVerdict | None, item.get("completeness"))
        bundle_ids = bundle_for(cast(str, item["clause_id"]))

        row["question"] = question_text
        row["question_type"] = cast(str, item["question_type"])
        row["difficulty"] = cast(str, item["difficulty"])
        row["expected_verdict"] = cast(str, item["expected_verdict"])
        row["reference_clause_ids"] = ";".join(refs)
        row["reference_clause_texts"] = " || ".join(
            f"[{cid}] {by_id[cid].title}: {by_id[cid].text.strip()[:BUNDLE_TEXT_CAP]}"
            for cid in refs
            if cid in by_id
        )
        row["vocabulary_terms_hit_question"] = ",".join(
            sorted(score_question_vocabulary(question_text))
        )
        row["indemnity_basis_question"] = (
            "Y" if question_is_indemnity_basis(question_text) else ""
        )
        row["scenario_terms"] = ",".join(
            sorted(score_question_scenarios(question_text, row["question_type"]))
        )
        row["bundle_section"] = by_id[refs[0]].bundle_section or "" if refs else ""
        row["question_scope_flag"] = question_scope_flag(question_text)
        row["self_reference_flag"] = question_self_reference_flag(question_text)
        row["coverage_exclusion_gap"] = (
            coverage_with_exclusion_gap(refs, by_id)
            if row["question_type"] == "coverage_with_exclusion"
            else ""
        )
        row["near_duplicate_of"] = near_duplicates.get(row_id, "")
        row["type_coerced_from"] = (
            "coverage_with_exclusion" if row_id in coerced_types else ""
        )
        row["completeness_pool_size"] = str(len(bundle_ids))
        row["completeness_considered_ids"] = ";".join(bundle_ids)
        row["completeness_included_reasons"] = (
            " || ".join(f"{c.clause_id}: {c.reason}" for c in completeness.included)
            if completeness
            else ""
        )
        row["completeness_rejected_reasons"] = (
            " || ".join(f"{r.clause_id}: {r.reason}" for r in completeness.rejected)
            if completeness
            else ""
        )
        row["draft_notes"] = cast(str, item["draft_notes"])
        row["provider_used"] = cast(str, item["provider"])
        rows[row_id] = row

    write_csv(DRAFT_CSV_PATH, rows)
    print(f"\nWrote {len(rows)} row(s) to {DRAFT_CSV_PATH}")
    passed = print_coverage_report(list(rows.values()))
    print(f"\nDoD gate: {'PASS' if passed else 'FAIL -- see flags above'}")
    print(
        "Next: review the CSV (focus on origin=corrected/replacement and the "
        "reference_clause_ids on origin=kept), set approved=Y, then run "
        "`make finalize-golden-set-casco`."
    )


if __name__ == "__main__":
    main()
