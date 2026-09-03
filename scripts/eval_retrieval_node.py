#!/usr/bin/env python3
"""Measure the [M4-04] retrieval node against ``golden-set-v1``.

The [M4-04] DoD asks the node to build the query from the extracted entities,
apply the classification's metadata pre-filter, return typed citations, and set
``context_sufficient`` from the [M3-07] gate. This script runs the **real node**
(``infrastructure.graph.nodes.retrieval.retrieval``) over a
``GraphContext`` whose ``retriever`` is the production
``infrastructure.rag.graph_retrieval_adapter.GraphRetrievalAdapter`` -- hybrid
RRF + cross-encoder rerank + exclusion co-retrieval, the [M3-08] best config.

It is LLM-free: each golden question is fed as the node's ``entities.description``
with the SUSEP process + product line from ``data/policies/manifest.csv`` for the
document the question targets, so ``_build_query`` / ``_build_filter`` get
realistic inputs without an intake call. The entity->query composition itself is
unit-tested (``tests/unit/infrastructure/graph/test_retrieval.py``); this
measures retrieval quality and the gate wiring end to end.

Reported: Recall@10 / MRR / nDCG@10 / foreign-document rate over the scorable
questions, and the gate's recall over the 23 ``unanswerable`` questions plus its
false-abstention rate over the scorable ones -- the [M3-07] numbers, re-measured
through the graph node. All broken down by ``question_type``.

Needs a running Postgres with loaded + embedded chunks and the optional ``embed``
uv group. Run via ``make eval-retrieval-node``. Writes
``eval/runs/retrieval_node.{md,json}``; the committed analysis lives in
``docs/RETRIEVAL_NODE.md``.
"""

from __future__ import annotations

import json
import platform
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.retrieval import RETRIEVAL_K, retrieval
from infrastructure.graph.state import Citation, ClaimState, ExtractedEntities
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from infrastructure.rag.graph_retrieval_adapter import GraphRetrievalAdapter
from infrastructure.rag.retriever_factory import (
    build_graph_retriever,
    retriever_components_from_corpora,
)
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    MANIFEST_PATH,
    load_chunk_corpus,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "retrieval_node.json"
MD_PATH = OUTPUT_DIR / "retrieval_node.md"

_NDCG_K = 10


@dataclass(frozen=True)
class _QuestionResult:
    question_id: str
    question_type: str
    document_id: str
    is_unanswerable: bool
    citation_ids: tuple[str, ...]
    foreign_citation_ids: tuple[str, ...]
    context_sufficient: bool | None
    recall_at_10: float | None
    mrr: float | None
    ndcg_at_10: float | None
    error: str | None = None


@dataclass(frozen=True)
class _Metrics:
    n: int
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    foreign_document_rate: float
    mean_citations: float


@dataclass(frozen=True)
class _GateMetrics:
    unanswerable_n: int
    gate_recall: float
    scorable_n: int
    gate_false_abstention_rate: float
    unanswerable_missed: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalNodeEvalResult:
    """Everything ``make eval-retrieval-node`` produces, for the report + the test."""

    meta: dict[str, Any]
    overall: _Metrics
    by_question_type: dict[str, _Metrics]
    gate: _GateMetrics
    results: list[_QuestionResult]
    error_question_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``eval/runs/retrieval_node.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "overall": _metrics_json(self.overall),
            "by_question_type": {
                name: _metrics_json(m) for name, m in self.by_question_type.items()
            },
            "gate": {
                "unanswerable_n": self.gate.unanswerable_n,
                "gate_recall": self.gate.gate_recall,
                "scorable_n": self.gate.scorable_n,
                "gate_false_abstention_rate": self.gate.gate_false_abstention_rate,
                "unanswerable_missed": list(self.gate.unanswerable_missed),
            },
            "error_question_ids": self.error_question_ids,
        }


def _metrics_json(m: _Metrics) -> dict[str, float | int]:
    return {
        "n": m.n,
        "recall_at_10": m.recall_at_10,
        "mrr": m.mrr,
        "ndcg_at_10": m.ndcg_at_10,
        "foreign_document_rate": m.foreign_document_rate,
        "mean_citations": m.mean_citations,
    }


class _UnusedModel:
    """The retrieval node never calls a chat model; ``GraphContext`` still needs one."""


def _stub_llm_settings() -> LlmSettings:
    """A placeholder ``LlmSettings``: the retrieval node reads none of its fields.

    Built directly rather than via ``get_llm_settings()`` so this eval runs with
    no ``LLM_*`` in ``.env`` -- it is LLM-free by design.
    """
    return LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY="unused",
        LLM_MODEL_FAST="unused",
        LLM_MODEL_REASONING="unused",
        EMBEDDING_MODEL="unused",
        RERANKER_MODEL="unused",
        _env_file=None,
    )


def _build_adapter(
    session: Any, chunks: Sequence[Any], corpus: Sequence[Any]
) -> GraphRetrievalAdapter:
    """Compose the [M3-08] retrieval stack (shared with the [M5-04] API root).

    A thin wrapper over ``infrastructure.rag.retriever_factory`` kept because
    ``eval_compatibility`` / ``eval_recommendation`` import it by this name.
    """
    return build_graph_retriever(
        session, retriever_components_from_corpora(chunks, corpus)
    )


def _run_node(
    question: GoldenQuestion,
    manifest_row: dict[str, str],
    context: GraphContext,
) -> tuple[list[Citation], bool | None]:
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
    update = retrieval(cast(ClaimState, state), Runtime(context=context))
    citations = cast(list[Citation], update["citations"])
    context_sufficient = cast("bool | None", update["context_sufficient"])
    return citations, context_sufficient


def _score_question(
    question: GoldenQuestion,
    citations: Sequence[Citation],
    context_sufficient: bool | None,
) -> _QuestionResult:
    is_unanswerable = question.question_type is QuestionType.UNANSWERABLE
    citation_ids = tuple(c.clause_id for c in citations)
    foreign = tuple(
        c.clause_id for c in citations if c.document_id != question.document_id
    )

    recall = mrr = ndcg = None
    if not is_unanswerable and question.reference_clause_ids:
        reference = question.reference_clause_ids
        recall = recall_at_k(list(citation_ids), reference, RETRIEVAL_K)
        mrr = reciprocal_rank(list(citation_ids), reference)
        ndcg = ndcg_at_k(list(citation_ids), reference, _NDCG_K)

    return _QuestionResult(
        question_id=question.question_id,
        question_type=question.question_type.value,
        document_id=question.document_id,
        is_unanswerable=is_unanswerable,
        citation_ids=citation_ids,
        foreign_citation_ids=foreign,
        context_sufficient=context_sufficient,
        recall_at_10=recall,
        mrr=mrr,
        ndcg_at_10=ndcg,
    )


def _aggregate(rows: Sequence[_QuestionResult]) -> _Metrics:
    scorable = [r for r in rows if not r.is_unanswerable and r.recall_at_10 is not None]
    if not scorable:
        return _Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_citations = sum(len(r.citation_ids) for r in scorable)
    foreign = sum(len(r.foreign_citation_ids) for r in scorable)
    recalls = [r.recall_at_10 for r in scorable if r.recall_at_10 is not None]
    mrrs = [r.mrr for r in scorable if r.mrr is not None]
    ndcgs = [r.ndcg_at_10 for r in scorable if r.ndcg_at_10 is not None]
    return _Metrics(
        n=len(scorable),
        recall_at_10=statistics.fmean(recalls),
        mrr=statistics.fmean(mrrs),
        ndcg_at_10=statistics.fmean(ndcgs),
        foreign_document_rate=foreign / total_citations if total_citations else 0.0,
        mean_citations=total_citations / len(scorable),
    )


def _gate_metrics(rows: Sequence[_QuestionResult]) -> _GateMetrics:
    unanswerable = [r for r in rows if r.is_unanswerable]
    scorable = [r for r in rows if not r.is_unanswerable]
    missed = tuple(
        r.question_id for r in unanswerable if r.context_sufficient is not False
    )
    gate_recall = 1.0 - len(missed) / len(unanswerable) if unanswerable else 0.0
    false_abstentions = sum(1 for r in scorable if r.context_sufficient is False)
    return _GateMetrics(
        unanswerable_n=len(unanswerable),
        gate_recall=gate_recall,
        scorable_n=len(scorable),
        gate_false_abstention_rate=(
            false_abstentions / len(scorable) if scorable else 0.0
        ),
        unanswerable_missed=missed,
    )


def run_retrieval_node_eval() -> RetrievalNodeEvalResult:
    """Run the retrieval node over every golden question and score the output."""
    document_meta = load_document_metadata(MANIFEST_PATH)
    questions = load_golden_questions(GOLDEN_SET_DIR)
    chunks = load_chunk_corpus()
    corpus = load_corpus(JSONL_PATH)
    settings = _stub_llm_settings()

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    rows: list[_QuestionResult] = []
    errors: list[str] = []
    try:
        assert_chunk_table_ready(session)
        adapter = _build_adapter(session, chunks, corpus)
        unused = cast(BaseChatModel, _UnusedModel())
        context = GraphContext(
            fast_model=unused,
            reasoning_model=unused,
            retriever=adapter,
            llm_settings=settings,
        )
        for question in questions:
            try:
                citations, sufficient = _run_node(
                    question, document_meta[question.document_id], context
                )
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                errors.append(question.question_id)
                rows.append(
                    _QuestionResult(
                        question_id=question.question_id,
                        question_type=question.question_type.value,
                        document_id=question.document_id,
                        is_unanswerable=(
                            question.question_type is QuestionType.UNANSWERABLE
                        ),
                        citation_ids=(),
                        foreign_citation_ids=(),
                        context_sufficient=None,
                        recall_at_10=None,
                        mrr=None,
                        ndcg_at_10=None,
                        error=repr(exc),
                    )
                )
                continue
            rows.append(_score_question(question, citations, sufficient))
            print(
                f"{question.question_id:<28} {question.question_type.value:<22} "
                f"cited={len(rows[-1].citation_ids)} "
                f"sufficient={rows[-1].context_sufficient}",
                flush=True,
            )
    finally:
        session.close()
        engine.dispose()

    by_type: dict[str, _Metrics] = {}
    for question_type in sorted({r.question_type for r in rows}):
        subset = [r for r in rows if r.question_type == question_type]
        metrics = _aggregate(subset)
        if metrics.n:
            by_type[question_type] = metrics

    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "golden_set_dir": str(GOLDEN_SET_DIR),
        "question_count": len(questions),
        "corpus_path": str(JSONL_PATH),
        "retrieval_k": RETRIEVAL_K,
        "config": "hybrid RRF + cross-encoder rerank + exclusion co-retrieval",
        "platform": platform.platform(),
    }
    return RetrievalNodeEvalResult(
        meta=meta,
        overall=_aggregate(rows),
        by_question_type=by_type,
        gate=_gate_metrics(rows),
        results=rows,
        error_question_ids=errors,
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(result: RetrievalNodeEvalResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    overall = result.overall
    gate = result.gate
    lines = [
        "# Retrieval node -- measurement ([M4-04])",
        "",
        "Generated by `scripts/eval_retrieval_node.py` (`make eval-retrieval-node`): "
        "the real `infrastructure.graph.nodes.retrieval.retrieval` node over "
        f"`{result.meta['golden_set_dir']}` ({result.meta['question_count']} "
        "questions), retriever = "
        f"{result.meta['config']}. LLM-free. Regenerable; committed analysis in "
        "`docs/RETRIEVAL_NODE.md`.",
        "",
        f"- Generated (UTC): {result.meta['generated_at_utc']}",
        f"- Retrieval k: {result.meta['retrieval_k']}",
        f"- Platform: {result.meta['platform']}",
        f"- Errors: {result.error_question_ids or 'none'}",
        "",
        "## Retrieval quality (scorable questions)",
        "",
        f"- Recall@10: **{_pct(overall.recall_at_10)}** ({overall.n} questions)",
        f"- MRR: **{overall.mrr:.3f}**",
        f"- nDCG@10: **{overall.ndcg_at_10:.3f}**",
        f"- Foreign-document rate: **{_pct(overall.foreign_document_rate)}**",
        f"- Mean citations returned: {overall.mean_citations:.1f}",
        "",
        "| question_type | n | Recall@10 | MRR | nDCG@10 | foreign-doc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in result.by_question_type.items():
        lines.append(
            f"| {name} | {m.n} | {_pct(m.recall_at_10)} | {m.mrr:.3f} "
            f"| {m.ndcg_at_10:.3f} | {_pct(m.foreign_document_rate)} |"
        )
    lines += [
        "",
        "## Insufficient-context gate (via the node's `context_sufficient`)",
        "",
        f"- Gate recall over `unanswerable`: **{_pct(gate.gate_recall)}** "
        f"({gate.unanswerable_n} questions)",
        f"- Missed (must be empty): {list(gate.unanswerable_missed) or 'none'}",
        f"- False-abstention rate over scorable: "
        f"**{_pct(gate.gate_false_abstention_rate)}** ({gate.scorable_n} questions)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the eval and write ``eval/runs/retrieval_node.{md,json}``."""
    result = run_retrieval_node_eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    print("")
    print(
        f"Recall@10 {_pct(result.overall.recall_at_10)} | "
        f"foreign-doc {_pct(result.overall.foreign_document_rate)} | "
        f"gate recall {_pct(result.gate.gate_recall)} "
        f"(missed {list(result.gate.unanswerable_missed) or 'none'})"
    )
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
