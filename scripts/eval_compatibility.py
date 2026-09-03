#!/usr/bin/env python3
"""Measure the [M4-05] compatibility node's verdicts against ``golden-set-v1``.

The [M4-05] DoD: "test on the golden set; report verdict accuracy against the
labels". Only two golden ``question_type``s carry a verdict label --
``coverage_with_exclusion`` (19, all ``incompatible``) and ``unanswerable`` (23,
all ``insufficient_information``). There is **no ``compatible``-labelled golden
question**; full three-class accuracy over the synthetic claims is [M4-10]'s.

Per labelled question this script runs the **real** retrieval node
(``GraphRetrievalAdapter`` -- hybrid RRF + rerank + exclusion co-retrieval, the
[M3-08] best config, same as ``scripts/eval_retrieval_node.py``) to populate
``citations`` and the [M3-07] gate flag, then the **real** compatibility node
(``infrastructure.graph.nodes.compatibility.compatibility``) on the reasoning
model. The compatibility node is invoked directly -- the ``route_after_retrieval``
gate is bypassed -- so the node's *own* ``insufficient_information`` judgement is
measured even on the ``unanswerable`` rows the gate would otherwise divert.

Reported, broken down by ``question_type``: the 3x3 verdict confusion matrix,
overall accuracy and per-class precision/recall; the citation-grounding rate
(every assertion cites a retrieved clause -- 100% by construction, so the
degrade-to-insufficient count is the real signal); reference-clause overlap for
``coverage_with_exclusion``; and the gate's ``context_sufficient`` alongside the
node verdict.

Needs a running Postgres with loaded + embedded chunks, the optional ``embed``
uv group, and ``LLM_*`` in ``.env`` (the reasoning model). Run via
``make eval-compatibility``. Writes ``eval/runs/compatibility.{md,json}`` and a
per-question ``eval/runs/compatibility_predictions.jsonl``; the committed
analysis lives in ``docs/COMPATIBILITY_ASSESSMENT.md``.
"""

from __future__ import annotations

import json
import platform
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langgraph.runtime import Runtime

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion
from infrastructure.evaluation.verdict_metrics import (
    VerdictMetrics,
    confusion_table_lines,
    metrics_json,
    verdict_metrics,
)
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.compatibility import compatibility
from infrastructure.graph.nodes.retrieval import retrieval
from infrastructure.graph.state import (
    Citation,
    ClaimState,
    CompatibilityAssessment,
    ExtractedEntities,
)
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    MANIFEST_PATH,
    load_chunk_corpus,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)
from scripts.eval_retrieval_node import _build_adapter

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "compatibility.json"
MD_PATH = OUTPUT_DIR / "compatibility.md"
PREDICTIONS_PATH = OUTPUT_DIR / "compatibility_predictions.jsonl"

_DEGRADED_MARKER = "ungrounded_after"


@dataclass(frozen=True)
class _QuestionResult:
    question_id: str
    question_type: str
    document_id: str
    expected_verdict: str
    predicted_verdict: str | None
    context_sufficient: bool | None
    n_citations: int
    cited_clause_ids: tuple[str, ...]
    reference_clause_ids: tuple[str, ...]
    grounding_degraded: bool
    every_assertion_grounded: bool
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted_verdict == self.expected_verdict


@dataclass(frozen=True)
class CompatibilityEvalResult:
    """Everything ``make eval-compatibility`` produces, for the report + the test."""

    meta: dict[str, Any]
    overall: VerdictMetrics
    by_question_type: dict[str, VerdictMetrics]
    grounding_degraded_ids: list[str]
    reference_overlap: dict[str, float]
    gate_by_question_type: dict[str, dict[str, int]]
    results: list[_QuestionResult]
    error_question_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``eval/runs/compatibility.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "overall": metrics_json(self.overall),
            "by_question_type": {
                name: metrics_json(m) for name, m in self.by_question_type.items()
            },
            "grounding_degraded_ids": self.grounding_degraded_ids,
            "reference_overlap": self.reference_overlap,
            "gate_by_question_type": self.gate_by_question_type,
            "error_question_ids": self.error_question_ids,
        }


def _labelled_questions(questions: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
    """The golden questions that carry a verdict label -- the scorable subset."""
    return [q for q in questions if q.expected_verdict is not None]


def _run_pipeline(
    question: GoldenQuestion,
    manifest_row: dict[str, str],
    context: GraphContext,
) -> tuple[CompatibilityAssessment, bool | None, list[Citation], str]:
    entities = ExtractedEntities(
        description=question.question,
        susep_process=manifest_row["susep_process"],
        product_line=manifest_row["product_line"],
    )
    state: dict[str, object] = {
        "claim_id": question.question_id,
        "raw_claim_text": question.question,
        "entities": entities,
    }
    retr = retrieval(cast(ClaimState, state), Runtime(context=context))
    citations = cast(list[Citation], retr["citations"])
    context_sufficient = cast("bool | None", retr["context_sufficient"])

    assess_state: dict[str, object] = {
        **state,
        "citations": citations,
        "context_sufficient": context_sufficient,
    }
    comp = compatibility(cast(ClaimState, assess_state), Runtime(context=context))
    assessment = cast(CompatibilityAssessment, comp["compatibility"])
    audit = cast(list[Any], comp["audit_trail"])[0]
    return assessment, context_sufficient, citations, audit.node_input or ""


def _score_question(
    question: GoldenQuestion,
    assessment: CompatibilityAssessment,
    context_sufficient: bool | None,
    citations: list[Citation],
    assessment_node_input: str,
) -> _QuestionResult:
    cited_ids = tuple(c.clause_id for c in assessment.citations)
    reference = tuple(question.reference_clause_ids)
    degraded = _DEGRADED_MARKER in assessment_node_input
    # By construction the node only emits an assessment whose every assertion is
    # grounded (or it degrades to insufficient_information). "grounded" here means
    # the emitted citations are a subset of what retrieval returned.
    retrieved = {c.clause_id for c in citations}
    grounded = set(cited_ids) <= retrieved
    return _QuestionResult(
        question_id=question.question_id,
        question_type=question.question_type.value,
        document_id=question.document_id,
        expected_verdict=cast(Any, question.expected_verdict).value,
        predicted_verdict=assessment.verdict.value,
        context_sufficient=context_sufficient,
        n_citations=len(assessment.citations),
        cited_clause_ids=cited_ids,
        reference_clause_ids=reference,
        grounding_degraded=degraded,
        every_assertion_grounded=grounded,
    )


def _verdict_metrics(rows: Sequence[_QuestionResult]) -> VerdictMetrics:
    """Score the rows through the shared [evaluation.verdict_metrics] arithmetic."""
    return verdict_metrics([(r.expected_verdict, r.predicted_verdict) for r in rows])


def _reference_overlap(rows: Sequence[_QuestionResult]) -> dict[str, float]:
    """Share of ``coverage_with_exclusion`` rows whose citations cover every ref id."""
    subset = [
        r
        for r in rows
        if r.question_type == "coverage_with_exclusion" and r.reference_clause_ids
    ]
    if not subset:
        return {"n": 0.0, "all_references_cited": 0.0, "any_reference_cited": 0.0}
    full = sum(
        1 for r in subset if set(r.reference_clause_ids) <= set(r.cited_clause_ids)
    )
    partial = sum(
        1 for r in subset if set(r.reference_clause_ids) & set(r.cited_clause_ids)
    )
    return {
        "n": float(len(subset)),
        "all_references_cited": full / len(subset),
        "any_reference_cited": partial / len(subset),
    }


def run_compatibility_eval() -> CompatibilityEvalResult:
    """Run retrieval + the compatibility node over every labelled golden question."""
    settings = get_llm_settings()
    reasoning_model = build_chat_model(
        settings,
        settings.llm_model_reasoning,
        provider_order=settings.llm_reasoning_provider_order,
        allow_fallbacks=settings.llm_reasoning_allow_fallbacks,
    )
    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = _labelled_questions(load_golden_questions(GOLDEN_SET_DIR))
    chunks = load_chunk_corpus()
    corpus = load_corpus(JSONL_PATH)

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    rows: list[_QuestionResult] = []
    errors: list[str] = []
    try:
        assert_chunk_table_ready(session)
        adapter = _build_adapter(session, chunks, corpus)
        context = GraphContext(
            fast_model=reasoning_model,
            reasoning_model=reasoning_model,
            retriever=adapter,
            llm_settings=settings,
        )
        for question in questions:
            try:
                (
                    assessment,
                    context_sufficient,
                    citations,
                    node_input,
                ) = _run_pipeline(
                    question, document_meta[question.document_id], context
                )
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                errors.append(question.question_id)
                rows.append(
                    _QuestionResult(
                        question_id=question.question_id,
                        question_type=question.question_type.value,
                        document_id=question.document_id,
                        expected_verdict=cast(Any, question.expected_verdict).value,
                        predicted_verdict=None,
                        context_sufficient=None,
                        n_citations=0,
                        cited_clause_ids=(),
                        reference_clause_ids=tuple(question.reference_clause_ids),
                        grounding_degraded=False,
                        every_assertion_grounded=False,
                        error=repr(exc),
                    )
                )
                continue
            row = _score_question(
                question, assessment, context_sufficient, citations, node_input
            )
            rows.append(row)
            print(
                f"{row.question_id:<28} {row.question_type:<22} "
                f"expected={row.expected_verdict:<24} got={row.predicted_verdict} "
                f"{'OK' if row.correct else 'X'}",
                flush=True,
            )
    finally:
        session.close()
        engine.dispose()

    by_type: dict[str, VerdictMetrics] = {}
    for question_type in sorted({r.question_type for r in rows}):
        subset = [r for r in rows if r.question_type == question_type]
        by_type[question_type] = _verdict_metrics(subset)

    gate_by_type: dict[str, dict[str, int]] = {}
    for question_type in sorted({r.question_type for r in rows}):
        subset = [r for r in rows if r.question_type == question_type]
        gate_by_type[question_type] = dict(
            Counter(str(r.context_sufficient) for r in subset)
        )

    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.llm_model_reasoning,
        "provider_order": settings.llm_reasoning_provider_order,
        "golden_set_dir": str(GOLDEN_SET_DIR),
        "labelled_question_count": len(questions),
        "retriever": "hybrid RRF + cross-encoder rerank + exclusion co-retrieval",
        "platform": platform.platform(),
        "dod_note": (
            "The golden set carries verdict labels only for coverage_with_exclusion "
            "(all incompatible) and unanswerable (all insufficient_information) -- no "
            "compatible-labelled question exists. Full three-class accuracy over the "
            "synthetic claims is [M4-10]."
        ),
    }
    return CompatibilityEvalResult(
        meta=meta,
        overall=_verdict_metrics(rows),
        by_question_type=by_type,
        grounding_degraded_ids=[r.question_id for r in rows if r.grounding_degraded],
        reference_overlap=_reference_overlap(rows),
        gate_by_question_type=gate_by_type,
        results=rows,
        error_question_ids=errors,
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(result: CompatibilityEvalResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    overall = result.overall
    lines = [
        "# Compatibility assessment node -- measurement ([M4-05])",
        "",
        "Generated by `scripts/eval_compatibility.py` (`make eval-compatibility`): "
        "the real retrieval node + the real "
        "`infrastructure.graph.nodes.compatibility.compatibility` node over the "
        f"verdict-labelled subset of `{result.meta['golden_set_dir']}` "
        f"({result.meta['labelled_question_count']} questions), retriever = "
        f"{result.meta['retriever']}, reasoning model = "
        f"`{result.meta['model']}` (provider order "
        f"`{result.meta['provider_order']}`). Regenerable; committed analysis in "
        "`docs/COMPATIBILITY_ASSESSMENT.md`.",
        "",
        f"- Generated (UTC): {result.meta['generated_at_utc']}",
        f"- Platform: {result.meta['platform']}",
        f"- Errors: {result.error_question_ids or 'none'}",
        f"- Note: {result.meta['dod_note']}",
        "",
        "## Verdict accuracy",
        "",
        f"- Overall accuracy: **{_pct(overall.accuracy)}** ({overall.n} questions)",
        "",
        "Per class (over the labelled subset):",
        "",
        "| verdict | support | precision | recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for verdict, stats in overall.per_class.items():
        lines.append(
            f"| {verdict} | {int(stats['support'])} | "
            f"{_pct(stats['precision'])} | {_pct(stats['recall'])} |"
        )
    lines += ["", "Confusion matrix (overall):", ""]
    lines += confusion_table_lines(overall)

    lines += ["", "### By question type", ""]
    for name, metrics in result.by_question_type.items():
        lines += [
            f"**{name}** — accuracy **{_pct(metrics.accuracy)}** ({metrics.n})",
            "",
        ]
        lines += confusion_table_lines(metrics)
        lines.append("")

    overlap = result.reference_overlap
    lines += [
        "## Citation grounding",
        "",
        f"- Grounding degraded to insufficient_information: "
        f"**{len(result.grounding_degraded_ids)}** "
        f"{result.grounding_degraded_ids or ''}",
        "- Every emitted assertion cites a retrieved clause: **by construction** "
        "(the node rejects and retries otherwise, then degrades).",
        "",
        "## Reference-clause overlap (`coverage_with_exclusion`)",
        "",
        f"- All labelled reference clauses cited: "
        f"**{_pct(overlap['all_references_cited'])}** ({int(overlap['n'])} questions)",
        f"- At least one reference clause cited: "
        f"**{_pct(overlap['any_reference_cited'])}**",
        "- Bounded above by M4-04's Recall@10 for this type (84.2%): the node "
        "can only cite an exclusion retrieval surfaced.",
        "",
        "## Gate (`context_sufficient`) vs the node verdict",
        "",
        "```",
        json.dumps(result.gate_by_question_type, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def _prediction_row(r: _QuestionResult) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "question_type": r.question_type,
        "document_id": r.document_id,
        "expected_verdict": r.expected_verdict,
        "predicted_verdict": r.predicted_verdict,
        "correct": r.correct,
        "context_sufficient": r.context_sufficient,
        "n_citations": r.n_citations,
        "cited_clause_ids": list(r.cited_clause_ids),
        "reference_clause_ids": list(r.reference_clause_ids),
        "grounding_degraded": r.grounding_degraded,
        "error": r.error,
    }


def main() -> None:
    """Run the eval and write ``eval/runs/compatibility.{md,json}`` + predictions."""
    result = run_compatibility_eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in result.results:
            handle.write(json.dumps(_prediction_row(row), ensure_ascii=False) + "\n")
    print("")
    print(
        f"verdict accuracy {_pct(result.overall.accuracy)} "
        f"({result.overall.n} questions) | "
        f"degraded {len(result.grounding_degraded_ids)} | "
        f"errors {result.error_question_ids or 'none'}"
    )
    print(f"Wrote {JSON_PATH}, {MD_PATH} and {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
