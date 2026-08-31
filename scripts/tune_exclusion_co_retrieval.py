#!/usr/bin/env python3
"""Sweep the [M3-06] reserved exclusion-slot count -- record the curve.

The M3-06 DoD asks to "measure the effect on the ``coverage_with_exclusion``
subset specifically, before and after". This does that across the one knob that
moves what the golden-set metrics see: ``RESERVED_EXCLUSION_SLOTS`` -- how many
of the final top-k slots are held for the exclusion clauses linked to a retrieved
coverage clause.

**One expensive pass, then pure-Python replay.** The base ranking (hybrid RRF +
cross-encoder rerank, filtered to each question's SUSEP process + CNPJ -- the
same configuration as ``make eval-retrieval-rerank``) is computed once per
scorable question. Exclusion co-retrieval at each slot count is then a
deterministic structural post-process over the parsed corpus
(``ExclusionCoRetrievalRetriever``) -- no model calls, so the sweep itself costs
seconds.

Reports, for the no-co-retrieval baseline and each slot count: overall
Recall@{1,5,10} / MRR / nDCG@10 and pooled exclusion-clause recall over all 117
scorable ``golden-set-v1`` questions, plus the ``coverage_with_exclusion`` subset
(n = 19) broken out on its own -- the number M3-06 exists to move.

Needs a running Postgres with loaded + embedded chunks and the optional ``embed``
uv group (for the single base pass only). Run via ``make
tune-exclusion-co-retrieval``. Writes ``eval/runs/exclusion_co_retrieval_tuning.
{md,json}`` (gitignored); the committed curve and the chosen slot count live in
``docs/EXCLUSION_CO_RETRIEVAL.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.evaluation.retriever import FilterableRetriever
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from infrastructure.rag.exclusion_co_retrieval import (
    ClauseGraph,
    ExclusionCoRetrievalRetriever,
)
from infrastructure.rag.exclusion_co_retrieval_config import (
    ADJACENT_SECTION_MAX_PAGE_GAP,
    RESERVED_EXCLUSION_SLOTS,
)
from infrastructure.rag.exclusion_co_retrieval_config import (
    config_fingerprint as co_retrieval_config_fingerprint,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    K_VALUES,
    MANIFEST_PATH,
    NDCG_K,
    RETRIEVE_K,
    _build_filter_for,
    _open_retriever,
    aggregate,
    compute_exclusion_clause_recall,
    evaluate_questions,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "exclusion_co_retrieval_tuning.json"
MD_PATH = OUTPUT_DIR / "exclusion_co_retrieval_tuning.md"

# The reserved-slot grid. 0 is not in it -- the "no co-retrieval" row is the raw
# base ranking, reported separately. The golden ``coverage_with_exclusion``
# reference sets carry one limiting exclusion each, so 1-3 brackets the useful
# range; a deeper sweep would only trade more displacement risk for nothing.
SLOT_COUNTS: tuple[int, ...] = (1, 2, 3)

_SUBSET = QuestionType.COVERAGE_WITH_EXCLUSION.value


class _ReplayRetriever:
    """Returns a precomputed base ranking per question text, cut to k."""

    def __init__(self, by_question: Mapping[str, list[str]]) -> None:
        self._by_question = by_question

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        del metadata_filter
        return self._by_question[question][:k]


class ExclusionCoRetrievalTuningConfig(BaseModel):
    """The reproducibility stamp for one ``make tune-exclusion-co-retrieval`` run."""

    schema_version: str
    run_at_utc: datetime
    golden_set_dir: str
    golden_set_question_count: int
    corpus_path: str
    corpus_clause_count: int
    slot_counts: list[int]
    chosen_reserved_exclusion_slots: int
    adjacent_section_max_page_gap: int
    co_retrieval_config_fingerprint: str
    base_reranker_config_fingerprint: str
    base_hybrid_config_fingerprint: str
    platform: str


def _scorable(questions: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
    return [q for q in questions if q.question_type is not QuestionType.UNANSWERABLE]


def _metric_row(
    retriever: Any,
    questions: Sequence[GoldenQuestion],
    document_meta: dict[str, dict[str, str]],
    clause_by_id: dict[str, ParsedClauseRecord],
) -> dict[str, Any]:
    """Score one retriever over the filtered golden set; overall + subset row."""
    filter_for = _build_filter_for("default", document_meta)
    rows, _ = evaluate_questions(
        questions, retriever, document_meta, filter_for=filter_for
    )
    overall = aggregate(rows, K_VALUES, NDCG_K)
    exclusion = compute_exclusion_clause_recall(rows, clause_by_id, k=max(K_VALUES))
    subset = aggregate(
        [row for row in rows if row.question_type == _SUBSET], K_VALUES, NDCG_K
    )
    return {
        "recall@1": overall["recall@1"],
        "recall@5": overall["recall@5"],
        "recall@10": overall["recall@10"],
        "mrr": overall["mrr"],
        f"ndcg@{NDCG_K}": overall[f"ndcg@{NDCG_K}"],
        "exclusion_recall": exclusion["recall"],
        "exclusion_hits": exclusion["hits"],
        "exclusion_total": exclusion["total"],
        "n": int(overall["n"]),
        "subset_n": int(subset["n"]),
        "subset_recall@10": subset["recall@10"],
        "subset_mrr": subset["mrr"],
        f"subset_ndcg@{NDCG_K}": subset[f"ndcg@{NDCG_K}"],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the sweep as Markdown; numbers are copied into the topic doc."""
    config = report["config"]
    lines = [
        "# Reserved exclusion-slot sweep",
        "",
        "Generated by `scripts/tune_exclusion_co_retrieval.py` "
        "(`make tune-exclusion-co-retrieval`) against the golden set in "
        "`data/golden_set/` under the `--filter default` SUSEP-process + CNPJ "
        "pre-filter. Regenerable; the committed curve and the chosen slot count "
        "live in `docs/EXCLUSION_CO_RETRIEVAL.md`.",
        "",
        "## Run configuration",
        "",
        f"- Golden set: `{config['golden_set_dir']}` "
        f"({config['golden_set_question_count']} questions)",
        f"- Corpus: `{config['corpus_path']}` "
        f"({config['corpus_clause_count']} clauses)",
        f"- Base ranking: hybrid RRF + cross-encoder rerank "
        f"(reranker fingerprint `{config['base_reranker_config_fingerprint']}`, "
        f"hybrid `{config['base_hybrid_config_fingerprint']}`) -- one pass, then "
        f"pure-Python co-retrieval replay",
        f"- Co-retrieval: adjacent-section page gap "
        f"{config['adjacent_section_max_page_gap']}; config fingerprint "
        f"`{config['co_retrieval_config_fingerprint']}`",
        f"- Reserved-slot counts swept: {config['slot_counts']}; chosen "
        f"(in `exclusion_co_retrieval_config.py`): "
        f"**{config['chosen_reserved_exclusion_slots']}**",
        f"- Run at (UTC): {config['run_at_utc']}",
        f"- Platform: {config['platform']}",
        "",
        "## Curve",
        "",
        "`coverage_with_exclusion` is the subset M3-06 exists to move "
        f"(n = {report['rows'][0]['subset_n']}); the overall columns watch for "
        "collateral regression on the other 98 questions.",
        "",
        "| reserved slots | overall R@1 | overall R@5 | overall R@10 | overall "
        f"MRR | overall nDCG@{NDCG_K} | exclusion recall | cov+excl R@10 | "
        "cov+excl MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        label = (
            "0 (no co-retrieval)"
            if row["reserved_slots"] is None
            else str(row["reserved_slots"])
        )
        exclusion = row["exclusion_recall"]
        exclusion_cell = (
            f"{exclusion:.1%} ({row['exclusion_hits']}/{row['exclusion_total']})"
            if exclusion is not None
            else "n/a"
        )
        lines.append(
            f"| {label} | {row['recall@1']:.1%} | {row['recall@5']:.1%} "
            f"| {row['recall@10']:.1%} | {row['mrr']:.1%} "
            f"| {row[f'ndcg@{NDCG_K}']:.1%} | {exclusion_cell} "
            f"| {row['subset_recall@10']:.1%} | {row['subset_mrr']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the sweep against the golden set and write the report."""
    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)
    graph = ClauseGraph(corpus)

    base_args = argparse.Namespace(
        retriever="hybrid",
        fusion="rrf",
        filter_mode="default",
        seed=42,
        rerank=True,
        co_retrieval=False,
    )
    base_by_question: dict[str, list[str]] = {}
    with _open_retriever(base_args, corpus) as (retriever, base_config):
        filterable = cast(FilterableRetriever, retriever)
        for question in _scorable(questions):
            metadata_filter = RetrievalFilter.from_manifest_row(
                document_meta[question.document_id]
            )
            if question.question in base_by_question:
                raise ValueError(
                    f"duplicate golden question text: {question.question!r}"
                )
            base_by_question[question.question] = filterable.retrieve(
                question.question, k=RETRIEVE_K, metadata_filter=metadata_filter
            )
    print(f"cached the base ranking for {len(base_by_question)} questions", flush=True)

    rows: list[dict[str, Any]] = []
    baseline = _metric_row(
        _ReplayRetriever(base_by_question), questions, document_meta, clause_by_id
    )
    baseline["reserved_slots"] = None
    rows.append(baseline)
    print(
        f"baseline (no co-retrieval): overall R@10={baseline['recall@10']:.1%} "
        f"excl={baseline['exclusion_recall']} "
        f"cov+excl R@10={baseline['subset_recall@10']:.1%}",
        flush=True,
    )

    for slots in SLOT_COUNTS:
        retriever = ExclusionCoRetrievalRetriever(
            _ReplayRetriever(base_by_question), graph, reserved_slots=slots
        )
        row = _metric_row(retriever, questions, document_meta, clause_by_id)
        row["reserved_slots"] = slots
        rows.append(row)
        print(
            f"slots {slots}: overall R@10={row['recall@10']:.1%} "
            f"excl={row['exclusion_recall']} "
            f"cov+excl R@10={row['subset_recall@10']:.1%} "
            f"MRR={row['subset_mrr']:.1%}",
            flush=True,
        )

    config = ExclusionCoRetrievalTuningConfig(
        schema_version=SCHEMA_VERSION,
        run_at_utc=datetime.now(UTC),
        golden_set_dir=str(GOLDEN_SET_DIR),
        golden_set_question_count=len(questions),
        corpus_path=str(JSONL_PATH),
        corpus_clause_count=len(corpus),
        slot_counts=list(SLOT_COUNTS),
        chosen_reserved_exclusion_slots=RESERVED_EXCLUSION_SLOTS,
        adjacent_section_max_page_gap=ADJACENT_SECTION_MAX_PAGE_GAP,
        co_retrieval_config_fingerprint=co_retrieval_config_fingerprint(),
        base_reranker_config_fingerprint=base_config["reranker_config_fingerprint"],
        base_hybrid_config_fingerprint=base_config["hybrid_config_fingerprint"],
        platform=platform.platform(),
    )

    report = {"config": config.model_dump(mode="json"), "rows": rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
