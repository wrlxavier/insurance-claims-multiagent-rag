#!/usr/bin/env python3
"""Run the [M3-08] benchmark matrix: every retrieval configuration, one table.

The M3-08 DoD asks to "run all four configurations on ``golden-set-v1``: lexical
only, dense only, hybrid, hybrid + rerank (+ exclusion co-retrieval)" and
"report Recall@1/5/10, MRR, nDCG@10, exclusion recall, mean latency and cost per
query for each", broken down by question type. This script is the single driver
that produces that comparison in one pass, so ``make build-index &&
make eval-retrieval-matrix`` reproduces every committed number.

It does **not** re-implement scoring -- it drives the [M2-06] harness
(``scripts/eval_retrieval.py``) exactly as ``scripts/tune_reranking.py`` and
``scripts/tune_exclusion_co_retrieval.py`` do: build an ``argparse.Namespace``
per configuration, open the retriever via ``_open_retriever``, wrap it in a
timing proxy, run ``evaluate_questions``. Every configuration runs on the
**filtered default path** (SUSEP process + CNPJ), the system's real retrieval
path (``docs/HYBRID_RETRIEVAL.md``); the unfiltered degradation numbers stay in
that document.

Two configurations beyond the DoD's four are included, both settled with the
project owner:

* ``hybrid RRF + rerank`` and ``hybrid RRF + rerank + co-retrieval`` are both
  shown so the co-retrieval delta is visible; the second is the best
  configuration.
* ``hybrid weighted + rerank + co-retrieval`` re-opens the RRF-vs-weighted
  fusion call ``docs/HYBRID_RETRIEVAL.md`` deferred here, now with the reranker
  in the loop.

A ``random`` reference row is included as in ``docs/HYBRID_RETRIEVAL.md``.

**Latency + cost.** Per-query wall-clock of ``retrieve()`` over the 117 scorable
questions, on this machine, reported p50 / p95 / mean (the
``scripts/tune_reranking.summarise_latency`` shape). The embedding and reranker
caches are warm, so the number is the warm-path cost; the doc decomposes the
cold components (the query embedder, ``docs/EMBEDDINGS.md``; the CPU
cross-encoder, ``docs/RERANKING.md``). Both models run locally, so the dollar
cost per query is **$0.00** and no price constant is introduced -- the [M1-09]
stale-pricing rule.

Needs a running Postgres with loaded + embedded chunks and the optional ``embed``
uv group (same as ``make eval-retrieval-hybrid``). Run via
``make eval-retrieval-matrix``. Writes ``eval/runs/retrieval_benchmark_matrix.
{md,json}`` (gitignored); the committed table and the verdict live in
``docs/RETRIEVAL_BENCHMARK.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from infrastructure.evaluation.golden_set_schema import GoldenQuestion
from infrastructure.evaluation.random_retriever import DEFAULT_SEED
from infrastructure.evaluation.retrieval_run_schema import (
    SCHEMA_VERSION as RUN_SCHEMA_VERSION,
)
from infrastructure.evaluation.retrieval_run_schema import RetrievalRunConfig
from infrastructure.evaluation.retriever import FilterableRetriever, Retriever
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from infrastructure.rag.retrieval_filter import RetrievalFilter
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    K_VALUES,
    MANIFEST_PATH,
    NDCG_K,
    _build_filter_for,
    _open_retriever,
    build_report,
    evaluate_questions,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)
from scripts.tune_reranking import summarise_latency

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "retrieval_benchmark_matrix.json"
MD_PATH = OUTPUT_DIR / "retrieval_benchmark_matrix.md"

_COST_NOTE = (
    "Dollar cost per query: **$0.00**. Both the embedder "
    "(`Alibaba-NLP/gte-multilingual-base`) and the cross-encoder "
    "(`Alibaba-NLP/gte-multilingual-reranker-base`) run locally via "
    "sentence-transformers -- no API, no per-token charge -- so no price "
    "constant is introduced ([M1-09] stale-pricing rule). The reproducible "
    "cost is machine time, in the latency column."
)


@dataclass(frozen=True)
class MatrixConfig:
    """One row of the benchmark matrix and the harness Namespace that runs it."""

    key: str
    label: str
    retriever: str
    fusion: str
    rerank: bool
    co_retrieval: bool
    filter_mode: str

    def namespace(self) -> argparse.Namespace:
        """The ``argparse.Namespace`` the harness helpers read."""
        return argparse.Namespace(
            retriever=self.retriever,
            seed=DEFAULT_SEED,
            fusion=self.fusion,
            filter_mode=self.filter_mode,
            rerank=self.rerank,
            co_retrieval=self.co_retrieval,
        )


# Every real configuration runs on the filtered default path (SUSEP process +
# CNPJ). `random` is the unfiltered self-test floor, as in docs/HYBRID_RETRIEVAL.md.
MATRIX: tuple[MatrixConfig, ...] = (
    MatrixConfig("random", "random (self-test)", "random", "rrf", False, False, "none"),
    MatrixConfig("lexical", "lexical", "lexical", "rrf", False, False, "default"),
    MatrixConfig("dense", "dense", "dense", "rrf", False, False, "default"),
    MatrixConfig("hybrid_rrf", "hybrid RRF", "hybrid", "rrf", False, False, "default"),
    MatrixConfig(
        "hybrid_rrf_rerank",
        "hybrid RRF + rerank",
        "hybrid",
        "rrf",
        True,
        False,
        "default",
    ),
    MatrixConfig(
        "hybrid_rrf_rerank_co_retrieval",
        "hybrid RRF + rerank + co-retrieval",
        "hybrid",
        "rrf",
        True,
        True,
        "default",
    ),
    MatrixConfig(
        "hybrid_weighted_rerank_co_retrieval",
        "hybrid weighted + rerank + co-retrieval",
        "hybrid",
        "weighted",
        True,
        True,
        "default",
    ),
)


class _TimingRetriever:
    """Wraps a retriever, recording per-call ``retrieve`` wall-clock in ms.

    Satisfies [infrastructure.evaluation.retriever.FilterableRetriever]: the
    ``metadata_filter`` default lets ``evaluate_questions`` call it with or
    without a filter (``random`` is scored unfiltered).
    """

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.samples_ms: list[float] = []

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        start = time.perf_counter()
        try:
            if metadata_filter is None:
                return self._inner.retrieve(question, k=k)
            filterable = cast(FilterableRetriever, self._inner)
            return filterable.retrieve(question, k=k, metadata_filter=metadata_filter)
        finally:
            self.samples_ms.append((time.perf_counter() - start) * 1000.0)


class MatrixRunConfig(BaseModel):
    """The reproducibility stamp for one ``make eval-retrieval-matrix`` run."""

    schema_version: str
    run_at_utc: datetime
    golden_set_dir: str
    golden_set_question_count: int
    corpus_path: str
    corpus_clause_count: int
    filter_mode: str
    configurations: list[str]
    platform: str


def score_configuration(
    matrix_config: MatrixConfig,
    questions: Sequence[GoldenQuestion],
    corpus: Sequence[ParsedClauseRecord],
    document_meta: dict[str, dict[str, str]],
    clause_by_id: dict[str, ParsedClauseRecord],
) -> dict[str, Any]:
    """Run one configuration through the harness and return its full report dict."""
    namespace = matrix_config.namespace()
    filter_for = _build_filter_for(namespace.filter_mode, document_meta)
    with _open_retriever(namespace, corpus) as (retriever, extra_config):
        timed = _TimingRetriever(retriever)
        rows, unanswerable_count = evaluate_questions(
            questions, timed, document_meta, filter_for=filter_for
        )
        run_config = RetrievalRunConfig(
            schema_version=RUN_SCHEMA_VERSION,
            retriever_name=namespace.retriever,
            k_values=list(K_VALUES),
            ndcg_k=NDCG_K,
            golden_set_dir=str(GOLDEN_SET_DIR),
            golden_set_question_count=len(questions),
            corpus_path=str(JSONL_PATH),
            corpus_clause_count=len(corpus),
            run_at_utc=datetime.now(UTC),
            filter_mode=namespace.filter_mode,
            **extra_config,
        )
    report = build_report(run_config, rows, unanswerable_count, clause_by_id)
    report["matrix_key"] = matrix_config.key
    report["matrix_label"] = matrix_config.label
    report["latency_ms"] = summarise_latency(timed.samples_ms)
    return report


def _fmt(value: float) -> str:
    return f"{value:.1%}"


def _matrix_table(reports: Sequence[dict[str, Any]]) -> list[str]:
    """The headline comparison table: one row per configuration."""
    lines = [
        "## The matrix",
        "",
        "| configuration | n | R@1 | R@5 | R@10 | MRR | nDCG@10 | "
        "exclusion recall | foreign-doc | latency ms (p50 / mean) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        overall = report["overall"]
        exclusion = report["exclusion_clause_recall"]
        exclusion_cell = (
            f"{_fmt(exclusion['recall'])} ({exclusion['hits']}/{exclusion['total']})"
            if exclusion["recall"] is not None
            else "n/a"
        )
        foreign = report["foreign_document_rate"]["rate"]
        foreign_cell = _fmt(foreign) if foreign is not None else "n/a"
        latency = report["latency_ms"]
        lines.append(
            f"| {report['matrix_label']} | {int(overall['n'])} "
            f"| {_fmt(overall['recall@1'])} | {_fmt(overall['recall@5'])} "
            f"| {_fmt(overall['recall@10'])} | {_fmt(overall['mrr'])} "
            f"| {_fmt(overall[f'ndcg@{NDCG_K}'])} | {exclusion_cell} "
            f"| {foreign_cell} "
            f"| {latency['p50']:.1f} / {latency['mean']:.1f} |"
        )
    lines.append("")
    return lines


def _by_question_type_table(reports: Sequence[dict[str, Any]]) -> list[str]:
    """Recall@10 and MRR per question_type, configurations as columns."""
    types = ("direct_lookup", "coverage_with_exclusion", "cross_document", "definition")
    scored = [r for r in reports if r["matrix_key"] != "random"]
    header = (
        "| question_type | metric | "
        + " | ".join(r["matrix_label"] for r in scored)
        + " |"
    )
    sep = "| --- | --- | " + " | ".join("---:" for _ in scored) + " |"
    lines = ["## By question type", "", header, sep]
    for question_type in types:
        groups = [r["by_question_type"].get(question_type, {}) for r in scored]
        n = int(groups[0].get("n", 0)) if groups else 0
        for metric, key in (("R@10", "recall@10"), ("MRR", "mrr")):
            cells = [_fmt(g[key]) if g.get(key) is not None else "—" for g in groups]
            label = f"`{question_type}` (n={n})" if metric == "R@10" else ""
            lines.append(f"| {label} | {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _latency_table(reports: Sequence[dict[str, Any]]) -> list[str]:
    """Per-configuration latency detail plus the cost note."""
    lines = [
        "## Latency and cost per query",
        "",
        "Per-query `retrieve()` wall-clock over the 117 scorable questions on "
        "this machine, embedding + reranker caches warm.",
        "",
        "| configuration | n | p50 ms | p95 ms | mean ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        latency = report["latency_ms"]
        lines.append(
            f"| {report['matrix_label']} | {int(latency['n'])} "
            f"| {latency['p50']:.1f} | {latency['p95']:.1f} "
            f"| {latency['mean']:.1f} |"
        )
    lines += ["", _COST_NOTE, ""]
    return lines


def _config_stamp(reports: Sequence[dict[str, Any]]) -> list[str]:
    """The exact config fingerprints behind every row (M3-08 DoD item 4)."""
    lines = ["## Exact configuration per run", ""]
    for report in reports:
        config = report["config"]
        parts = [
            f"- **{report['matrix_label']}** (`{report['matrix_key']}`): "
            f"retriever `{config['retriever_name']}`, "
            f"filter `{config.get('filter_mode', 'none')}`"
        ]
        if config.get("fusion_strategy"):
            parts.append(f", fusion `{config['fusion_strategy']}`")
        if config.get("hybrid_config_fingerprint"):
            parts.append(f", hybrid `{config['hybrid_config_fingerprint']}`")
        if config.get("lexical_config_fingerprint"):
            parts.append(f", lexical `{config['lexical_config_fingerprint']}`")
        if config.get("embedding_config_fingerprint"):
            parts.append(f", embedding `{config['embedding_config_fingerprint']}`")
        if config.get("reranker_config_fingerprint"):
            parts.append(
                f", reranker `{config['reranker_config_fingerprint']}` "
                f"(depth {config['rerank_candidate_depth']})"
            )
        if config.get("co_retrieval_config_fingerprint"):
            parts.append(
                f", co-retrieval `{config['co_retrieval_config_fingerprint']}` "
                f"({config['reserved_exclusion_slots']} slot)"
            )
        lines.append("".join(parts))
    lines.append("")
    return lines


def render_matrix_report(report: dict[str, Any]) -> str:
    """Render the whole matrix as Markdown; numbers copied into the committed doc."""
    config = report["run_config"]
    reports = report["configurations"]
    scorable = next(
        int(r["overall"]["n"]) for r in reports if r["matrix_key"] != "random"
    )
    lines = [
        "# Retrieval benchmark matrix",
        "",
        "Generated by `scripts/benchmark_retrieval_matrix.py` "
        "(`make eval-retrieval-matrix`) against the golden set in "
        "`data/golden_set/`. Regenerable; the committed table, the "
        "per-question-type analysis and the verdict live in "
        "`docs/RETRIEVAL_BENCHMARK.md`. Every real configuration runs on the "
        "`--filter default` SUSEP-process + CNPJ path; the 23 `unanswerable` "
        "questions are excluded from every metric (empty reference set).",
        "",
        "## Run configuration",
        "",
        f"- Golden set: `{config['golden_set_dir']}` "
        f"({config['golden_set_question_count']} questions, {scorable} scorable)",
        f"- Corpus: `{config['corpus_path']}` "
        f"({config['corpus_clause_count']} clauses)",
        f"- Filter: `{config['filter_mode']}` (per-question SUSEP process + CNPJ)",
        f"- Platform: {config['platform']}",
        f"- Run at (UTC): {config['run_at_utc']}",
        "",
    ]
    lines += _matrix_table(reports)
    lines += _by_question_type_table(reports)
    lines += _latency_table(reports)
    lines += _config_stamp(reports)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    """Run every matrix configuration and write the report."""
    _parse_args()
    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)

    reports: list[dict[str, Any]] = []
    for matrix_config in MATRIX:
        print(f"running {matrix_config.key} ...", flush=True)
        report = score_configuration(
            matrix_config, questions, corpus, document_meta, clause_by_id
        )
        overall = report["overall"]
        print(
            f"  R@10={_fmt(overall['recall@10'])} MRR={_fmt(overall['mrr'])} "
            f"nDCG={_fmt(overall[f'ndcg@{NDCG_K}'])} "
            f"latency p50={report['latency_ms']['p50']:.1f}ms",
            flush=True,
        )
        reports.append(report)

    run_config = MatrixRunConfig(
        schema_version=SCHEMA_VERSION,
        run_at_utc=datetime.now(UTC),
        golden_set_dir=str(GOLDEN_SET_DIR),
        golden_set_question_count=len(questions),
        corpus_path=str(JSONL_PATH),
        corpus_clause_count=len(corpus),
        filter_mode="default",
        configurations=[config.key for config in MATRIX],
        platform=platform.platform(),
    )
    document = {
        "run_config": run_config.model_dump(mode="json"),
        "configurations": reports,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_PATH.write_text(render_matrix_report(document), encoding="utf-8")
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
