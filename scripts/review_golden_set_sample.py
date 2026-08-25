#!/usr/bin/env python3
"""Run the independent second-reviewer pass over a golden-set-v1 sample.

Selection (which questions, deterministic and seeded) lives in
``scripts/review_sample_selection.py`` -- this script's job is only to
build each sampled question's review packet (the question plus its target
document's full clause list, in document order -- never
``reference_clause_ids``, ``notes``, or ``expected_verdict``), get the
reviewer's independent answer, score it against the author's original
label, and write both the full reviewed sample and a report.

See ``docs/EVALUATION.md``'s "Independent second-reviewer pass" section for
the sampling frame, the review-packet definition, and the adjudication rule
this run follows: on any disagreement, the author's original label in
``data/golden_set/`` is retained unchanged -- no question is dropped, and no
further justification is recorded beyond the disagreement itself.

Run ``PYTHONPATH=app/src uv run python scripts/review_golden_set_sample.py``
(``--dry-run`` prints the sample composition, no model calls, no files
written).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.golden_set_schema import (
    ExpectedVerdict,
    GoldenQuestion,
    QuestionType,
)
from infrastructure.evaluation.review_schema import SCHEMA_VERSION, GoldenQuestionReview
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

try:
    # Direct execution: the script's own directory is sys.path[0].
    import review_sample_selection as selection
    from review_sample_selection import SampledQuestion
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts import review_sample_selection as selection
    from scripts.review_sample_selection import SampledQuestion

GOLDEN_SET_DIR = Path("data/golden_set")
MANIFEST_PATH = Path("data/policies/manifest.csv")
REVIEW_OUTPUT_PATH = Path("data/golden_set/review/review_v1.jsonl")
REPORT_DIR = Path("eval/runs")

REVIEW_MODEL = "google/gemini-3.7-flash"
REVIEW_PROVIDER_ORDER = ["google-vertex/global"]
REVIEWER_ID = "R2"

MAX_QUESTIONS_PER_CALL = 8


class ReviewedQuestion(BaseModel):
    """One question's independent answer, from the reviewer's own reading."""

    question_id: str = Field(..., description="Copiado literalmente do prompt.")
    reference_clause_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Lista exaustiva de clause_id (formato exato mostrado no "
            "documento) necessários para responder; vazia se nenhuma "
            "cláusula responde."
        ),
    )
    verdict: (
        Literal["compatible", "incompatible", "insufficient_information"] | None
    ) = Field(
        default=None,
        description=(
            "'compatible'/'incompatible' apenas se a pergunta descrever um "
            "cenário de sinistro cuja cobertura pode ser aceita ou negada; "
            "'insufficient_information' se a informação pedida estiver "
            "ausente; null para uma busca factual simples sem essa "
            "dimensão."
        ),
    )
    rationale: str = Field(
        ..., description="Uma ou duas frases justificando a escolha acima."
    )


class ReviewBatch(BaseModel):
    """A batch of independent reviews for one document's sampled questions."""

    reviews: list[ReviewedQuestion]


def load_golden_set_by_type(
    golden_set_dir: Path,
) -> dict[QuestionType, list[GoldenQuestion]]:
    """Load every golden-set row, grouped by ``question_type``."""
    rows_by_type: dict[QuestionType, list[GoldenQuestion]] = defaultdict(list)
    for path in sorted(golden_set_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    question = GoldenQuestion.model_validate(json.loads(line))
                    rows_by_type[question.question_type].append(question)
    return rows_by_type


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the built corpus, failing loudly if `make parse` hasn't run yet."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


def clauses_for_document(
    corpus: list[ParsedClauseRecord], document_id: str
) -> list[ParsedClauseRecord]:
    """Return one document's clauses, ordered as they appear in the document."""
    clauses = [record for record in corpus if record.document_id == document_id]
    return sorted(clauses, key=lambda record: (record.page_start, record.page_end))


def render_clause_tree(clauses: list[ParsedClauseRecord]) -> str:
    """Render a document's clauses as id/title/text blocks, in order.

    This -- and only this -- is what the reviewer sees of the document:
    never ``reference_clause_ids``, ``notes``, or ``expected_verdict``,
    which stay withheld per the review-packet rule in
    ``docs/EVALUATION.md``.
    """
    return "\n\n".join(
        f"### {clause.clause_id} — {clause.title}\n{clause.text}" for clause in clauses
    )


def build_review_prompt(
    *,
    document_id: str,
    document_meta: dict[str, str],
    clauses: list[ParsedClauseRecord],
    questions: list[GoldenQuestion],
) -> str:
    """Build the review prompt for one document's batch of sampled questions."""
    question_blocks = "\n".join(
        f"- question_id: {question.question_id}\n  Pergunta: {question.question}"
        for question in questions
    )
    return (
        "Você é um revisor independente de um golden set de avaliação para "
        "um sistema de RAG que responde a analistas de sinistros sobre "
        "apólices de seguro auto brasileiras. Você NÃO viu a resposta "
        "original de nenhuma pergunta abaixo -- responda cada uma usando "
        "SOMENTE o texto das cláusulas deste documento, exatamente como um "
        "analista de sinistros faria.\n\n"
        "Para cada pergunta, devolva:\n"
        "- reference_clause_ids: a lista EXAUSTIVA de clause_id (use o "
        "formato exato mostrado abaixo) necessários para responder -- nem a "
        "mais, nem a menos. Lista vazia se nenhuma cláusula responde.\n"
        "- verdict: preencha apenas se a pergunta descrever um cenário de "
        "sinistro cuja cobertura pode ser aceita ou negada, ou se a "
        "informação pedida estiver ausente; deixe null para uma busca "
        "factual simples (valor, prazo, definição) sem essa dimensão.\n"
        "- rationale: uma ou duas frases.\n\n"
        f"DOCUMENTO {document_id} ({document_meta.get('insurer', '')}, "
        f"{document_meta.get('product_line', '')}) -- cláusulas em ordem:\n\n"
        f"{render_clause_tree(clauses)}\n\n"
        f"PERGUNTAS ({len(questions)}):\n{question_blocks}\n\n"
        f"Retorne exatamente {len(questions)} revisão(ões), uma por "
        "question_id, na mesma ordem."
    )


def call_llm(prompt: str) -> ReviewBatch:
    """Invoke the pinned reviewer model with structured output.

    Single provider, no fallback -- an independent-review judgment has no
    sane cross-provider substitute. Retries transient failures; any other
    failure re-raises.
    """
    llm = build_chat_model(
        get_llm_settings(),
        REVIEW_MODEL,
        provider_order=REVIEW_PROVIDER_ORDER,
        allow_fallbacks=False,
    )
    chain = cast(Runnable[str, ReviewBatch], llm.with_structured_output(ReviewBatch))
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


def _jaccard(author_ids: set[str], reviewer_ids: set[str]) -> float:
    """Set-overlap measure; two empty sets count as a full match."""
    union = author_ids | reviewer_ids
    if not union:
        return 1.0
    return len(author_ids & reviewer_ids) / len(union)


def _f1(author_ids: set[str], reviewer_ids: set[str]) -> float:
    """Precision/recall F1 over clause id sets; two empty sets score 1.0."""
    if not author_ids and not reviewer_ids:
        return 1.0
    if not author_ids or not reviewer_ids:
        return 0.0
    intersection = len(author_ids & reviewer_ids)
    if intersection == 0:
        return 0.0
    precision = intersection / len(reviewer_ids)
    recall = intersection / len(author_ids)
    return 2 * precision * recall / (precision + recall)


def score_review(
    question: GoldenQuestion,
    reviewed: ReviewedQuestion,
    *,
    sample_stratum: Literal["general_stratified", "unanswerable_topup"],
    reviewed_at: str,
) -> GoldenQuestionReview:
    """Compare the reviewer's independent answer against the author's label.

    ``divergence_note``/``adjudication`` are only populated when
    ``agreement`` is False, per the fixed adjudication rule: the author's
    original label always stands, so there is nothing to adjudicate on a
    match.
    """
    author_ids = set(question.reference_clause_ids)
    reviewer_ids = set(reviewed.reference_clause_ids)
    exact_match = author_ids == reviewer_ids
    reviewer_verdict = (
        ExpectedVerdict(reviewed.verdict) if reviewed.verdict is not None else None
    )
    verdict_match: bool | None
    if question.expected_verdict is None:
        verdict_match = None
    else:
        verdict_match = reviewer_verdict == question.expected_verdict
    agreement = exact_match and verdict_match is not False

    divergence_note: str | None = None
    adjudication: str | None = None
    if not agreement:
        parts: list[str] = []
        missing = author_ids - reviewer_ids
        extra = reviewer_ids - author_ids
        if missing:
            parts.append(f"reviewer missing {sorted(missing)}")
        if extra:
            parts.append(f"reviewer added {sorted(extra)}")
        if verdict_match is False and question.expected_verdict is not None:
            parts.append(
                f"verdict {reviewed.verdict!r} vs author "
                f"{question.expected_verdict.value!r}"
            )
        divergence_note = "; ".join(parts)
        adjudication = "author_label_retained"

    return GoldenQuestionReview(
        schema_version=SCHEMA_VERSION,
        question_id=question.question_id,
        question_type=question.question_type,
        sample_stratum=sample_stratum,
        reviewer_id=REVIEWER_ID,
        reviewed_at=reviewed_at,
        reviewer_reference_clause_ids=sorted(reviewer_ids),
        reviewer_verdict=reviewer_verdict,
        reviewer_rationale=reviewed.rationale,
        clause_ids_exact_match=exact_match,
        clause_ids_jaccard=_jaccard(author_ids, reviewer_ids),
        verdict_match=verdict_match,
        agreement=agreement,
        divergence_note=divergence_note,
        adjudication=adjudication,
    )


def group_sampled_by_document(
    sampled: list[SampledQuestion],
) -> dict[str, list[SampledQuestion]]:
    """Group the sample by ``document_id``, preserving selection order."""
    groups: dict[str, list[SampledQuestion]] = defaultdict(list)
    for item in sampled:
        groups[item.document_id].append(item)
    return groups


def _chunk(items: list[GoldenQuestion], size: int) -> list[list[GoldenQuestion]]:
    """Split ``items`` into chunks of at most ``size``."""
    return [items[index : index + size] for index in range(0, len(items), size)]


def write_review_rows(path: Path, rows: list[GoldenQuestionReview]) -> None:
    """Write the full reviewed sample, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def _stratum_stats(
    rows: list[GoldenQuestionReview], question_by_id: dict[str, GoldenQuestion]
) -> dict[str, Any]:
    """Aggregate clause-id and verdict agreement stats over one group of rows."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "clause_exact_match_rate": None,
            "mean_jaccard": None,
            "mean_f1": None,
            "verdict_agreement_rate": None,
            "verdict_n": 0,
            "full_agreement_rate": None,
            "sensitivity_pp": None,
        }
    f1_values = [
        _f1(
            set(question_by_id[row.question_id].reference_clause_ids),
            set(row.reviewer_reference_clause_ids),
        )
        for row in rows
    ]
    verdict_rows = [row for row in rows if row.verdict_match is not None]
    verdict_n = len(verdict_rows)
    verdict_rate = (
        sum(1 for row in verdict_rows if row.verdict_match) / verdict_n
        if verdict_n
        else None
    )
    return {
        "n": n,
        "clause_exact_match_rate": sum(1 for row in rows if row.clause_ids_exact_match)
        / n,
        "mean_jaccard": sum(row.clause_ids_jaccard for row in rows) / n,
        "mean_f1": sum(f1_values) / n,
        "verdict_agreement_rate": verdict_rate,
        "verdict_n": verdict_n,
        "full_agreement_rate": sum(1 for row in rows if row.agreement) / n,
        "sensitivity_pp": 100.0 / n,
    }


def build_report(
    review_rows: list[GoldenQuestionReview],
    question_by_id: dict[str, GoldenQuestion],
    missing_question_ids: list[str],
) -> dict[str, Any]:
    """Assemble the one dict both the JSON and Markdown reports render from."""
    by_question_type: dict[str, dict[str, Any]] = {}
    for question_type in QuestionType:
        rows = [row for row in review_rows if row.question_type is question_type]
        if rows:
            by_question_type[question_type.value] = _stratum_stats(rows, question_by_id)

    disagreements: list[dict[str, Any]] = []
    for row in review_rows:
        if row.agreement:
            continue
        author_question = question_by_id[row.question_id]
        author_expected_verdict = author_question.expected_verdict
        disagreements.append(
            {
                "question_id": row.question_id,
                "question_type": row.question_type.value,
                "author_reference_clause_ids": author_question.reference_clause_ids,
                "author_expected_verdict": (
                    author_expected_verdict.value
                    if author_expected_verdict is not None
                    else None
                ),
                "reviewer_reference_clause_ids": row.reviewer_reference_clause_ids,
                "reviewer_verdict": (
                    row.reviewer_verdict.value if row.reviewer_verdict else None
                ),
                "divergence_note": row.divergence_note,
                "adjudication": row.adjudication,
            }
        )

    general_n = sum(
        1 for row in review_rows if row.sample_stratum == "general_stratified"
    )
    topup_n = sum(
        1 for row in review_rows if row.sample_stratum == "unanswerable_topup"
    )

    return {
        "reviewer_id": REVIEWER_ID,
        "sample_size": len(review_rows),
        "general_stratified_n": general_n,
        "unanswerable_topup_n": topup_n,
        "missing_question_ids": missing_question_ids,
        "overall": _stratum_stats(review_rows, question_by_id),
        "by_question_type": by_question_type,
        "disagreements": disagreements,
    }


def _fmt_pct(value: float | None) -> str:
    """Render a 0..1 rate as a percentage, or 'n/a' if undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the Markdown independent-second-reviewer report.

    Never states how the reviewer arrived at its answers -- only
    ``docs/EVALUATION.md``'s dedicated one-line disclosure does that.
    """
    overall = report["overall"]
    lines = [
        "# Golden-set-v1 independent second-reviewer pass",
        "",
        "Generated by `scripts/review_golden_set_sample.py` "
        "(`make review-golden-set-sample`) against a stratified sample of "
        "`data/golden_set/`. See `docs/EVALUATION.md`'s \"Independent "
        'second-reviewer pass" section for the sampling frame, the '
        "review-packet definition, and the adjudication rule this run "
        "followed.",
        "",
        "## Run configuration",
        "",
        f"- Sample size: {report['sample_size']} "
        f"({report['general_stratified_n']} general stratified + "
        f"{report['unanswerable_topup_n']} unanswerable top-up)",
        f"- Reviewer id: `{report['reviewer_id']}`",
        "- Adjudication rule: on disagreement, the author's original label "
        "is retained unchanged; no question is dropped from golden-set-v1; "
        "no further justification is required.",
    ]
    if report["missing_question_ids"]:
        lines.append(
            f"- Missing reviewer response for: {report['missing_question_ids']}"
        )
    lines += [
        "",
        "## Overall agreement",
        "",
        "| n | clause exact-match | mean Jaccard | mean F1 | verdict "
        "agreement (n) | full agreement | sensitivity |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {overall['n']} | {_fmt_pct(overall['clause_exact_match_rate'])} "
        f"| {overall['mean_jaccard']:.2f} | {overall['mean_f1']:.2f} "
        f"| {_fmt_pct(overall['verdict_agreement_rate'])} "
        f"({overall['verdict_n']}) "
        f"| {_fmt_pct(overall['full_agreement_rate'])} "
        f"| ±{overall['sensitivity_pp']:.1f}pp |",
        "",
        "## By question_type",
        "",
        "| question_type | n | clause exact-match | mean Jaccard | mean F1 "
        "| verdict agreement (n) | full agreement | sensitivity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for question_type, stats in report["by_question_type"].items():
        lines.append(
            f"| {question_type} | {stats['n']} "
            f"| {_fmt_pct(stats['clause_exact_match_rate'])} "
            f"| {stats['mean_jaccard']:.2f} | {stats['mean_f1']:.2f} "
            f"| {_fmt_pct(stats['verdict_agreement_rate'])} "
            f"({stats['verdict_n']}) "
            f"| {_fmt_pct(stats['full_agreement_rate'])} "
            f"| ±{stats['sensitivity_pp']:.1f}pp |"
        )
    lines += [
        "",
        "A single disagreement moves the overall agreement rate by "
        f"±{overall['sensitivity_pp']:.1f} percentage points at this "
        "sample size -- smaller per-`question_type` strata swing "
        "proportionally more, per the sensitivity column above.",
        "",
        "## Disagreements",
        "",
    ]
    if not report["disagreements"]:
        lines.append("None.")
    else:
        lines += [
            "| question_id | question_type | author reference_clause_ids "
            "| author expected_verdict | reviewer reference_clause_ids "
            "| reviewer verdict | resolution |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in report["disagreements"]:
            lines.append(
                f"| {item['question_id']} | {item['question_type']} "
                f"| {item['author_reference_clause_ids']} "
                f"| {item['author_expected_verdict']} "
                f"| {item['reviewer_reference_clause_ids']} "
                f"| {item['reviewer_verdict']} | {item['adjudication']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sample composition; never call the reviewer model.",
    )
    return parser.parse_args()


def _print_sample_composition(sampled: list[SampledQuestion]) -> None:
    """Print per-(question_type, sample_stratum) counts, for --dry-run."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in sampled:
        counts[(item.question_type.value, item.sample_stratum)] += 1
    for (question_type, stratum), count in sorted(counts.items()):
        print(f"  {question_type:24s} {stratum:20s} {count}")


def main() -> None:
    """Run the independent second-reviewer pass over a stratified sample."""
    load_dotenv()
    args = _parse_args()

    rows_by_type = load_golden_set_by_type(GOLDEN_SET_DIR)
    question_by_id = {
        question.question_id: question
        for rows in rows_by_type.values()
        for question in rows
    }
    corpus = load_corpus()
    clause_by_id = {record.clause_id: record for record in corpus}
    document_by_id = {row["id"]: row for row in read_manifest(MANIFEST_PATH)}

    sampled = selection.select_review_sample(rows_by_type, clause_by_id)

    if args.dry_run:
        print(f"\n--dry-run: {len(sampled)} question(s) selected. No model calls.")
        _print_sample_composition(sampled)
        return

    reviewed_at = datetime.now(UTC).date().isoformat()
    review_rows: list[GoldenQuestionReview] = []
    missing_question_ids: list[str] = []

    for document_id, group in sorted(group_sampled_by_document(sampled).items()):
        clauses = clauses_for_document(corpus, document_id)
        questions = [question_by_id[item.question_id] for item in group]
        stratum_by_id = {item.question_id: item.sample_stratum for item in group}

        for chunk in _chunk(questions, MAX_QUESTIONS_PER_CALL):
            print(f"Reviewing {len(chunk)} question(s) for document {document_id}...")
            prompt = build_review_prompt(
                document_id=document_id,
                document_meta=document_by_id[document_id],
                clauses=clauses,
                questions=chunk,
            )
            batch = call_llm(prompt)
            reviewed_by_id = {review.question_id: review for review in batch.reviews}

            for question in chunk:
                reviewed = reviewed_by_id.get(question.question_id)
                if reviewed is None:
                    print(f"  WARNING: no reviewer response for {question.question_id}")
                    missing_question_ids.append(question.question_id)
                    continue
                review_rows.append(
                    score_review(
                        question,
                        reviewed,
                        sample_stratum=stratum_by_id[question.question_id],
                        reviewed_at=reviewed_at,
                    )
                )

    review_rows.sort(key=lambda row: row.question_id)
    write_review_rows(REVIEW_OUTPUT_PATH, review_rows)

    report = build_report(review_rows, question_by_id, missing_question_ids)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "golden_set_review_v1.json"
    md_path = REPORT_DIR / "golden_set_review_v1.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"\nWrote {len(review_rows)} reviewed row(s) to {REVIEW_OUTPUT_PATH}")
    print(f"Overall agreement: {report['overall']['full_agreement_rate']:.1%}")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
