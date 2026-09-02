#!/usr/bin/env python3
"""Measure the [M4-08] recommendation node over ``golden-set-v1``.

The unit tests (``tests/unit/infrastructure/graph/test_recommendation.py``) cover
the node's shape and its two structural guarantees with a fake LLM. This script
is the on-demand live check: per verdict-labelled golden question it runs the
**real** pipeline -- the retrieval node (``GraphRetrievalAdapter``: hybrid RRF +
rerank + exclusion co-retrieval, the [M3-08] best config), then the **real**
compatibility node (reasoning model) and consistency node (fast model), then the
**real** recommendation node (fast model) -- and reports what the consolidation
produced.

The two guarantees the DoD makes structural are re-checked here on live output:

- **Citation grounding.** Every clause id the recommendation carries must be one
  retrieval returned. By construction the node only copies
  ``compatibility.citations`` (already a retrieved subset) -- so the expected
  rate is 100%, and anything less is a real regression.
- **An abstaining verdict is never confident.** For every row whose effective
  verdict is ``insufficient_information`` (compatibility abstained, retrieval
  missed), ``recommendation.confidence`` must stay at or below
  ``_INSUFFICIENT_CONFIDENCE_CEILING`` (0.3).

Also reported: the posture-derivation wiring check (the posture the node recorded
is the mapping of the compatibility verdict -- expected 100%), the confidence
distribution by posture, the consistency-flag pass-through (the node must carry
``consistency.signals`` verbatim), and a light justification check (does the
prose name a clause id when the verdict is settled). End-to-end verdict accuracy
over the synthetic claims and a faithfulness judge are [M4-10]'s.

Needs a running Postgres with loaded + embedded chunks, the optional ``embed`` uv
group, and ``LLM_*`` in ``.env``. Run via ``make eval-recommendation``. Writes
``eval/runs/recommendation.{md,json}`` and a per-question
``eval/runs/recommendation_predictions.jsonl``; the committed analysis lives in
``docs/RECOMMENDATION_NODE.md``.
"""

from __future__ import annotations

import json
import platform
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langgraph.runtime import Runtime

from domain.verdict import Verdict
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.compatibility import compatibility
from infrastructure.graph.nodes.consistency import consistency
from infrastructure.graph.nodes.recommendation import (
    _INSUFFICIENT_CONFIDENCE_CEILING,
    _VERDICT_POSTURE,
    recommendation,
)
from infrastructure.graph.nodes.retrieval import retrieval
from infrastructure.graph.state import (
    Citation,
    ClaimState,
    CompatibilityAssessment,
    ConsistencyReport,
    ExtractedEntities,
    Recommendation,
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
JSON_PATH = OUTPUT_DIR / "recommendation.json"
MD_PATH = OUTPUT_DIR / "recommendation.md"
PREDICTIONS_PATH = OUTPUT_DIR / "recommendation_predictions.jsonl"

_INSUFFICIENT_POSTURES = frozenset(
    {"claimant_gaps", "retrieval_miss", "inconclusive", "no_assessment"}
)
_POSTURE_RE = re.compile(r"posture=(\S+)")
_CLAUSE_ID_RE = re.compile(r"[A-Za-z0-9][\w./:-]*:[\w.-]+")


@dataclass(frozen=True)
class _QuestionResult:
    question_id: str
    question_type: str
    document_id: str
    expected_verdict: str
    compatibility_verdict: str | None
    posture: str | None
    recommendation_confidence: float | None
    recommendation_citation_ids: tuple[str, ...]
    retrieved_clause_ids: tuple[str, ...]
    citations_grounded: bool
    n_consistency_signals: int
    n_recommendation_flags: int
    flags_passed_through: bool
    justification_len: int
    justification_names_a_clause: bool
    llm_failed: bool
    error: str | None = None

    @property
    def is_insufficient_posture(self) -> bool:
        return self.posture in _INSUFFICIENT_POSTURES


@dataclass(frozen=True)
class _PostureConfidence:
    n: int
    min: float
    mean: float
    max: float


@dataclass(frozen=True)
class RecommendationEvalResult:
    """Everything ``make eval-recommendation`` produces, for the report + the test."""

    meta: dict[str, Any]
    n_scored: int
    citation_grounding_rate: float
    insufficient_posture_max_confidence: float | None
    flag_pass_through_rate: float
    posture_derivation_rate: float
    justification_names_a_clause_rate: float
    confidence_by_posture: dict[str, _PostureConfidence]
    posture_counts: dict[str, int]
    llm_failed_ids: list[str]
    results: list[_QuestionResult]
    error_question_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``eval/runs/recommendation.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "n_scored": self.n_scored,
            "citation_grounding_rate": self.citation_grounding_rate,
            "insufficient_posture_max_confidence": (
                self.insufficient_posture_max_confidence
            ),
            "flag_pass_through_rate": self.flag_pass_through_rate,
            "posture_derivation_rate": self.posture_derivation_rate,
            "justification_names_a_clause_rate": self.justification_names_a_clause_rate,
            "confidence_by_posture": {
                posture: vars(stats)
                for posture, stats in self.confidence_by_posture.items()
            },
            "posture_counts": self.posture_counts,
            "llm_failed_ids": self.llm_failed_ids,
            "error_question_ids": self.error_question_ids,
        }


def _labelled_questions(questions: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
    """The golden questions that carry a verdict label -- the scorable subset."""
    return [q for q in questions if q.expected_verdict is not None]


def _run_pipeline(
    question: GoldenQuestion,
    manifest_row: dict[str, str],
    context: GraphContext,
) -> tuple[
    CompatibilityAssessment, ConsistencyReport, Recommendation, list[Citation], str
]:
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

    # The compatibility node is run directly (like scripts/eval_compatibility.py):
    # its own insufficient_information judgement is measured even on the
    # unanswerable rows the [M3-07] gate would otherwise divert. The
    # recommendation node then consolidates that assessment -- the assessed
    # path, posture from the compatibility verdict. The retrieval-miss and
    # claimant-gaps postures are exercised by the unit tests and by [M4-10].
    assess_state: dict[str, object] = {**state, "citations": citations}
    comp = compatibility(cast(ClaimState, assess_state), Runtime(context=context))
    assessment = cast(CompatibilityAssessment, comp["compatibility"])
    cons = consistency(cast(ClaimState, assess_state), Runtime(context=context))
    report = cast(ConsistencyReport, cons["consistency"])

    rec_state: dict[str, object] = {
        **assess_state,
        "compatibility": assessment,
        "consistency": report,
    }
    rec_update = recommendation(cast(ClaimState, rec_state), Runtime(context=context))
    rec = cast(Recommendation, rec_update["recommendation"])
    audit = cast(list[Any], rec_update["audit_trail"])[0]
    return assessment, report, rec, citations, audit.node_input or ""


def _score_question(
    question: GoldenQuestion,
    assessment: CompatibilityAssessment,
    report: ConsistencyReport,
    rec: Recommendation,
    retrieved: list[Citation],
    node_input: str,
) -> _QuestionResult:
    retrieved_ids = {c.clause_id for c in retrieved}
    rec_ids = tuple(c.clause_id for c in rec.citations)
    posture_match = _POSTURE_RE.search(node_input)
    posture = posture_match.group(1) if posture_match else None
    justification_ids = set(_CLAUSE_ID_RE.findall(rec.justification))
    return _QuestionResult(
        question_id=question.question_id,
        question_type=question.question_type.value,
        document_id=question.document_id,
        expected_verdict=cast(Any, question.expected_verdict).value,
        compatibility_verdict=assessment.verdict.value,
        posture=posture,
        recommendation_confidence=rec.confidence,
        recommendation_citation_ids=rec_ids,
        retrieved_clause_ids=tuple(sorted(retrieved_ids)),
        citations_grounded=set(rec_ids) <= retrieved_ids,
        n_consistency_signals=len(report.signals),
        n_recommendation_flags=len(rec.consistency_flags),
        flags_passed_through=[s.detail for s in report.signals]
        == [f.detail for f in rec.consistency_flags],
        justification_len=len(rec.justification),
        justification_names_a_clause=bool(justification_ids & retrieved_ids),
        llm_failed="llm_failed=True" in node_input,
    )


def _posture_is_the_verdict_mapping(row: _QuestionResult) -> bool:
    """The posture the node recorded is ``_VERDICT_POSTURE`` of the assessed verdict.

    On the assessed path (every row this eval scores) the posture is a pure
    function of the compatibility verdict -- ``inconclusive`` for
    ``insufficient_information``, else the verdict name. This re-checks that
    mapping on the live audit string; anything but 100% is a ``_posture``
    regression or a broken ``node_input`` format.
    """
    if row.posture is None or row.compatibility_verdict is None:
        return False
    return row.posture == _VERDICT_POSTURE[Verdict(row.compatibility_verdict)]


def _confidence_by_posture(
    rows: Sequence[_QuestionResult],
) -> dict[str, _PostureConfidence]:
    by_posture: dict[str, list[float]] = {}
    for row in rows:
        if row.posture is None or row.recommendation_confidence is None:
            continue
        by_posture.setdefault(row.posture, []).append(row.recommendation_confidence)
    return {
        posture: _PostureConfidence(
            n=len(values),
            min=round(min(values), 3),
            mean=round(sum(values) / len(values), 3),
            max=round(max(values), 3),
        )
        for posture, values in sorted(by_posture.items())
    }


def run_recommendation_eval() -> RecommendationEvalResult:
    """Run the full pipeline + the recommendation node over every labelled question."""
    settings = get_llm_settings()
    reasoning_model = build_chat_model(
        settings,
        settings.llm_model_reasoning,
        provider_order=settings.llm_reasoning_provider_order,
        allow_fallbacks=settings.llm_reasoning_allow_fallbacks,
    )
    fast_model = build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=settings.llm_fast_provider_order,
        allow_fallbacks=settings.llm_fast_allow_fallbacks,
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
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            retriever=adapter,
            llm_settings=settings,
        )
        for question in questions:
            try:
                assessment, report, rec, citations, node_input = _run_pipeline(
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
                        compatibility_verdict=None,
                        posture=None,
                        recommendation_confidence=None,
                        recommendation_citation_ids=(),
                        retrieved_clause_ids=(),
                        citations_grounded=False,
                        n_consistency_signals=0,
                        n_recommendation_flags=0,
                        flags_passed_through=False,
                        justification_len=0,
                        justification_names_a_clause=False,
                        llm_failed=False,
                        error=repr(exc),
                    )
                )
                continue
            row = _score_question(
                question, assessment, report, rec, citations, node_input
            )
            rows.append(row)
            print(
                f"{row.question_id:<28} posture={row.posture or '-':<14} "
                f"conf={row.recommendation_confidence} "
                f"grounded={'OK' if row.citations_grounded else 'X'} "
                f"flags={row.n_recommendation_flags}",
                flush=True,
            )
    finally:
        session.close()
        engine.dispose()

    scored = [r for r in rows if r.error is None]
    insufficient = [r for r in scored if r.is_insufficient_posture]
    insufficient_max = (
        max(r.recommendation_confidence or 0.0 for r in insufficient)
        if insufficient
        else None
    )
    settled = [r for r in scored if r.posture in {"compatible", "incompatible"}]
    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reasoning_model": settings.llm_model_reasoning,
        "fast_model": settings.llm_model_fast,
        "golden_set_dir": str(GOLDEN_SET_DIR),
        "labelled_question_count": len(questions),
        "retriever": "hybrid RRF + cross-encoder rerank + exclusion co-retrieval",
        "platform": platform.platform(),
        "insufficient_confidence_ceiling": _INSUFFICIENT_CONFIDENCE_CEILING,
        "dod_note": (
            "Citation grounding and the insufficient-verdict confidence ceiling "
            "are structural node guarantees re-checked here on live output. "
            "End-to-end verdict accuracy over the synthetic claims and a "
            "faithfulness judge are [M4-10]."
        ),
    }
    return RecommendationEvalResult(
        meta=meta,
        n_scored=len(scored),
        citation_grounding_rate=(
            sum(r.citations_grounded for r in scored) / len(scored) if scored else 0.0
        ),
        insufficient_posture_max_confidence=insufficient_max,
        flag_pass_through_rate=(
            sum(r.flags_passed_through for r in scored) / len(scored) if scored else 0.0
        ),
        posture_derivation_rate=(
            sum(_posture_is_the_verdict_mapping(r) for r in scored) / len(scored)
            if scored
            else 0.0
        ),
        justification_names_a_clause_rate=(
            sum(r.justification_names_a_clause for r in settled) / len(settled)
            if settled
            else 0.0
        ),
        confidence_by_posture=_confidence_by_posture(scored),
        posture_counts=dict(Counter(r.posture or "error" for r in rows)),
        llm_failed_ids=[r.question_id for r in scored if r.llm_failed],
        results=rows,
        error_question_ids=errors,
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(result: RecommendationEvalResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    lines = [
        "# Recommendation node -- measurement ([M4-08])",
        "",
        "Generated by `scripts/eval_recommendation.py` (`make eval-recommendation`): "
        "the real retrieval + compatibility + consistency + recommendation nodes "
        f"over the verdict-labelled subset of `{result.meta['golden_set_dir']}` "
        f"({result.meta['labelled_question_count']} questions), reasoning model "
        f"`{result.meta['reasoning_model']}`, fast model "
        f"`{result.meta['fast_model']}`. Regenerable; committed analysis in "
        "`docs/RECOMMENDATION_NODE.md`.",
        "",
        f"- Generated (UTC): {result.meta['generated_at_utc']}",
        f"- Platform: {result.meta['platform']}",
        f"- Scored: {result.n_scored}",
        f"- Errors: {result.error_question_ids or 'none'}",
        f"- LLM justification degraded to the template: "
        f"{result.llm_failed_ids or 'none'}",
        f"- Note: {result.meta['dod_note']}",
        "",
        "## Structural guarantees (re-checked live)",
        "",
        f"- **Citation grounding rate: {_pct(result.citation_grounding_rate)}** "
        "(every recommendation clause id was returned by retrieval)",
        f"- **Max confidence on an `insufficient_information` posture: "
        f"{result.insufficient_posture_max_confidence}** "
        f"(ceiling {result.meta['insufficient_confidence_ceiling']})",
        f"- **Consistency-flag pass-through: {_pct(result.flag_pass_through_rate)}** "
        "(`consistency.signals` carried verbatim)",
        "",
        "## Consolidation quality",
        "",
        f"- Posture is the compatibility verdict's mapping "
        f"(wiring check, expect 100%): {_pct(result.posture_derivation_rate)}",
        f"- Justification names a retrieved clause id on a settled verdict: "
        f"{_pct(result.justification_names_a_clause_rate)}",
        "",
        "### Confidence by posture",
        "",
        "| posture | n | min | mean | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for posture, stats in result.confidence_by_posture.items():
        lines.append(
            f"| {posture} | {stats.n} | {stats.min} | {stats.mean} | {stats.max} |"
        )
    lines += [
        "",
        "### Posture counts",
        "",
        "```",
        json.dumps(result.posture_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Findings",
        "",
        "_To be written from the first run._",
        "",
    ]
    return "\n".join(lines) + "\n"


def _prediction_row(r: _QuestionResult) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "question_type": r.question_type,
        "document_id": r.document_id,
        "expected_verdict": r.expected_verdict,
        "compatibility_verdict": r.compatibility_verdict,
        "posture": r.posture,
        "recommendation_confidence": r.recommendation_confidence,
        "recommendation_citation_ids": list(r.recommendation_citation_ids),
        "citations_grounded": r.citations_grounded,
        "n_consistency_signals": r.n_consistency_signals,
        "n_recommendation_flags": r.n_recommendation_flags,
        "flags_passed_through": r.flags_passed_through,
        "justification_len": r.justification_len,
        "justification_names_a_clause": r.justification_names_a_clause,
        "llm_failed": r.llm_failed,
        "error": r.error,
    }


def main() -> None:
    """Run the eval and write ``eval/runs/recommendation.{md,json}`` + predictions."""
    result = run_recommendation_eval()
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
        f"grounding {_pct(result.citation_grounding_rate)} | "
        f"insufficient max conf {result.insufficient_posture_max_confidence} | "
        f"flag pass-through {_pct(result.flag_pass_through_rate)} | "
        f"errors {result.error_question_ids or 'none'}"
    )
    print(f"Wrote {JSON_PATH}, {MD_PATH} and {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
