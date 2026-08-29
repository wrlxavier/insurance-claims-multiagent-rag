#!/usr/bin/env python3
"""Sweep the [M3-05] cross-encoder reranker's candidate depth -- record the curve.

The M3-05 DoD asks to "tune retrieval k and rerank n on the golden set; record
the curve, not just the chosen point", and to "measure the added latency and
state the trade-off explicitly". This does both for the one knob that moves what
the golden-set metrics see: ``RERANK_CANDIDATE_DEPTH`` -- how many of the fused
hybrid candidates the cross-encoder re-scores before the top-k cut.

**Metrics** (Recall@{1,5,10} / MRR / nDCG@10 / exclusion-clause recall, over all
117 scorable ``golden-set-v1`` questions under the ``--filter default``
SUSEP-process + CNPJ pre-filter) come from a single rerank pass at the deepest
depth: ``fused[:30]`` is a prefix of ``fused[:50]``, so every shallower depth's
ranking is that prefix re-sorted by the same scores -- the cross-encoder runs
once per question, not once per (question, depth). That pass runs on the GPU
when one is present; the scores (and so every metric) are device-independent.

**Latency** is measured separately, on a ``device="cpu"`` reranker, as a probe
over ``LATENCY_PROBE_QUESTIONS`` questions at each depth. CPU is the number the
``docs/RERANKING.md`` trade-off turns on -- a reranker in an interactive path,
no GPU serving infra. The query embedder is lazy (every golden query vector is
already cached), so it never competes with the reranker for the dev GPU.

The **rerank-n** dimension (final context size) is read straight off the
Recall@1 / @5 / @10 columns of each row -- n = 1 / 5 / 10 -- so there is no
separate n sweep here; ``docs/RERANKING.md`` states this.

Needs a running Postgres with loaded + embedded chunks and the optional ``embed``
uv group (same as ``make eval-retrieval-hybrid``). Run via ``make
tune-reranking``. Writes ``eval/runs/rerank_tuning.{md,json}`` (gitignored); the
committed curve and the chosen depth live in ``docs/RERANKING.md``.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from infrastructure.rag.dense_retriever import DenseRetriever
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.hybrid_retriever import HybridRetriever
from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_config import (
    RERANK_CANDIDATE_DEPTH,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)
from infrastructure.rag.reranker_config import (
    config_fingerprint as reranker_config_fingerprint,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    K_VALUES,
    MANIFEST_PATH,
    NDCG_K,
    _build_filter_for,
    _hybrid_config_fields,
    _load_query_embedder,
    _load_reranker,
    aggregate,
    build_clause_text_map,
    build_lexical_retriever,
    compute_exclusion_clause_recall,
    evaluate_questions,
    load_chunk_corpus,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "rerank_tuning.json"
MD_PATH = OUTPUT_DIR / "rerank_tuning.md"

# The candidate-depth grid. 10 is a pure reorder of the top-10 (Recall@10 cannot
# move); the deeper points test whether the cross-encoder rescues a relevant
# clause the fusion ranked 11..N, and at what latency. Capped at 50: a
# process+CNPJ-filtered partition on `golden-set-v1` rarely holds more than that
# many candidates, so a deeper sweep would only repeat the depth-50 row.
RERANK_CANDIDATE_DEPTHS: tuple[int, ...] = (10, 20, 30, 50)

# How many questions the latency probe re-times at each shallower depth. The
# deepest depth's latency is the full main pass (n = every scorable question);
# the shallower rows only need a stable p50/mean, so a small sample keeps the
# extra CPU cost modest.
LATENCY_PROBE_QUESTIONS = 20


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, same formula as ``scripts/benchmark_ann_index.py``."""
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


def summarise_latency(samples_ms: list[float]) -> dict[str, float]:
    """Summarise per-query rerank latency samples (milliseconds)."""
    if not samples_ms:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(samples_ms)
    return {
        "n": len(ordered),
        "p50": round(_percentile(ordered, 0.50), 1),
        "p95": round(_percentile(ordered, 0.95), 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


class _ReplayRetriever:
    """Returns a precomputed reranked clause-id list per question text, cut to k."""

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


def _rerank_order(candidates: Sequence[str], scores: Sequence[float]) -> list[str]:
    """The candidate ids sorted by score desc; equal scores keep candidate order."""
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order]


def _score_rows(
    retriever: Any,
    questions: Sequence[GoldenQuestion],
    document_meta: dict[str, dict[str, str]],
    clause_by_id: dict[str, ParsedClauseRecord],
) -> dict[str, Any]:
    """Run one retriever over the filtered golden set; return its metric row."""
    filter_for = _build_filter_for("default", document_meta)
    rows, _ = evaluate_questions(
        questions, retriever, document_meta, filter_for=filter_for
    )
    overall = aggregate(rows, K_VALUES, NDCG_K)
    exclusion = compute_exclusion_clause_recall(rows, clause_by_id, k=max(K_VALUES))
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
    }


class RerankTuningConfig(BaseModel):
    """The reproducibility stamp for one ``make tune-reranking`` run."""

    schema_version: str
    run_at_utc: datetime
    golden_set_dir: str
    golden_set_question_count: int
    corpus_path: str
    corpus_clause_count: int
    chunk_corpus_chunk_count: int
    candidate_depths: list[int]
    max_depth: int
    latency_probe_questions: int
    chosen_candidate_depth: int
    reranker_model_id: str
    reranker_model_revision: str
    reranker_config_fingerprint: str
    embedding_config_fingerprint: str
    lexical_config_fingerprint: str
    hybrid_config_fingerprint: str
    scoring_device: str
    reranker_device: str
    platform: str


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the sweep as Markdown (numbers copied into docs/RERANKING.md)."""
    config = report["config"]
    lines = [
        "# Reranker candidate-depth sweep",
        "",
        "Generated by `scripts/tune_reranking.py` (`make tune-reranking`) against "
        "the golden set in `data/golden_set/` under the `--filter default` "
        "SUSEP-process + CNPJ pre-filter. Regenerable; the committed curve and "
        "the chosen depth live in `docs/RERANKING.md`.",
        "",
        "## Run configuration",
        "",
        f"- Golden set: `{config['golden_set_dir']}` "
        f"({config['golden_set_question_count']} questions)",
        f"- Corpus: `{config['corpus_path']}` "
        f"({config['corpus_clause_count']} clauses); "
        f"{config['chunk_corpus_chunk_count']} chunks",
        f"- Reranker: `{config['reranker_model_id']}` @ "
        f"`{config['reranker_model_revision']}` "
        f"(fingerprint `{config['reranker_config_fingerprint']}`)",
        f"- Hybrid RRF fingerprint: `{config['hybrid_config_fingerprint']}`; "
        f"embedding `{config['embedding_config_fingerprint']}`; "
        f"lexical `{config['lexical_config_fingerprint']}`",
        f"- Reranker device: `{config['scoring_device']}` (scoring) / "
        f"`{config['reranker_device']}` (latency probe); "
        f"platform: {config['platform']}",
        f"- Metrics: all {config['golden_set_question_count']} golden questions, "
        f"one rerank pass at depth {config['max_depth']} (shallower depths are "
        f"that prefix re-sorted -- scores are device-independent). Latency: a "
        f"separate CPU probe over {config['latency_probe_questions']} questions "
        f"per depth.",
        f"- Candidate depths swept: {config['candidate_depths']}; "
        f"chosen (in `reranker_config.py`): "
        f"**{config['chosen_candidate_depth']}**",
        f"- Run at (UTC): {config['run_at_utc']}",
        "",
        "## Curve",
        "",
        "The `n` dimension (final context cut) is the Recall@1 / @5 / @10 "
        "columns: n = 1 / 5 / 10.",
        "",
        "| candidate depth | Recall@1 | Recall@5 | Recall@10 | MRR | "
        f"nDCG@{NDCG_K} | exclusion recall | CPU rerank ms/query "
        "(p50 / p95 / mean) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        latency = row["latency_ms"]
        if row["candidate_depth"] is None:
            depth_label = "hybrid RRF (no rerank)"
            latency_cell = "—"
        else:
            depth_label = str(row["candidate_depth"])
            latency_cell = (
                f"{latency['p50']:.1f} / {latency['p95']:.1f} / {latency['mean']:.1f}"
            )
        exclusion = row["exclusion_recall"]
        exclusion_cell = (
            f"{exclusion:.1%} ({row['exclusion_hits']}/{row['exclusion_total']})"
            if exclusion is not None
            else "n/a"
        )
        lines.append(
            f"| {depth_label} "
            f"| {row['recall@1']:.1%} | {row['recall@5']:.1%} "
            f"| {row['recall@10']:.1%} | {row['mrr']:.1%} "
            f"| {row[f'ndcg@{NDCG_K}']:.1%} | {exclusion_cell} | {latency_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def _scorable(questions: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
    return [q for q in questions if q.question_type is not QuestionType.UNANSWERABLE]


def _rerank_at_max_depth(
    hybrid: HybridRetriever,
    reranker: Reranker,
    questions: Sequence[GoldenQuestion],
    document_meta: dict[str, dict[str, str]],
    text_map: Mapping[str, str],
    max_depth: int,
) -> dict[str, tuple[list[str], list[float]]]:
    """Per scorable question: its top-``max_depth`` hybrid candidates and rerank scores.

    Keyed by question text -- ``evaluate_questions`` only ever hands the retriever
    the question string.
    """
    per_question: dict[str, tuple[list[str], list[float]]] = {}
    for question in _scorable(questions):
        metadata_filter = RetrievalFilter.from_manifest_row(
            document_meta[question.document_id]
        )
        candidates = hybrid.retrieve(
            question.question, k=max_depth, metadata_filter=metadata_filter
        )
        passages = [text_map.get(clause_id, "") for clause_id in candidates]
        scores = reranker.rerank(question.question, passages) if passages else []
        if question.question in per_question:
            raise ValueError(f"duplicate golden question text: {question.question!r}")
        per_question[question.question] = (candidates, scores)
    return per_question


def _probe_latency(
    reranker: Reranker,
    questions: Sequence[GoldenQuestion],
    per_question: dict[str, tuple[list[str], list[float]]],
    text_map: Mapping[str, str],
    depth: int,
    sample_size: int,
) -> dict[str, float]:
    """Time ``reranker`` re-scoring the top-``depth`` passages for a question sample."""
    samples_ms: list[float] = []
    for question in _scorable(questions)[:sample_size]:
        candidates, _ = per_question[question.question]
        passages = [text_map.get(c, "") for c in candidates[:depth]]
        if not passages:
            continue
        start = time.perf_counter()
        reranker.rerank(question.question, passages)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return summarise_latency(samples_ms)


def main() -> None:
    """Run the sweep against the golden set and write the report."""
    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)
    chunks = load_chunk_corpus()
    text_map = build_clause_text_map(chunks)
    hybrid_fields = _hybrid_config_fields(chunks, "rrf")
    max_depth = max(RERANK_CANDIDATE_DEPTHS)

    # Scoring on the GPU when present (device-independent scores, ~10x faster);
    # the latency the docs/RERANKING.md trade-off reports is a CPU number, probed
    # on its own instance.
    score_reranker = _load_reranker()
    scoring_device = str(getattr(score_reranker, "device", "unknown"))
    cpu_reranker = _load_reranker(device="cpu")
    reranker_device = str(getattr(cpu_reranker, "device", "unknown"))

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    rows: list[dict[str, Any]] = []
    try:
        assert_chunk_table_ready(session)
        embedder = CachingEmbedder(_load_query_embedder())
        dense = DenseRetriever(session, embedder)
        hybrid = HybridRetriever(build_lexical_retriever(chunks), dense)

        baseline = _score_rows(hybrid, questions, document_meta, clause_by_id)
        baseline |= {"candidate_depth": None, "latency_ms": summarise_latency([])}
        rows.append(baseline)
        print(
            f"baseline hybrid RRF: R@10={baseline['recall@10']:.1%} "
            f"MRR={baseline['mrr']:.1%}",
            flush=True,
        )

        # One expensive pass: rerank every scorable question's top-`max_depth`
        # candidates. Every shallower depth is a prefix of this, re-sorted.
        per_question = _rerank_at_max_depth(
            hybrid, score_reranker, questions, document_meta, text_map, max_depth
        )
        print(
            f"reranked {len(per_question)} questions at depth {max_depth} "
            f"on {scoring_device}",
            flush=True,
        )

        for depth in RERANK_CANDIDATE_DEPTHS:
            replay = {
                text: _rerank_order(candidates[:depth], scores[:depth])
                for text, (candidates, scores) in per_question.items()
            }
            row = _score_rows(
                _ReplayRetriever(replay), questions, document_meta, clause_by_id
            )
            row["latency_ms"] = _probe_latency(
                cpu_reranker,
                questions,
                per_question,
                text_map,
                depth,
                LATENCY_PROBE_QUESTIONS,
            )
            row["candidate_depth"] = depth
            rows.append(row)
            print(
                f"depth {depth:>3}: R@10={row['recall@10']:.1%} "
                f"MRR={row['mrr']:.1%} nDCG={row[f'ndcg@{NDCG_K}']:.1%} "
                f"excl={row['exclusion_recall']} "
                f"cpu p50={row['latency_ms']['p50']:.0f}ms",
                flush=True,
            )

        config = RerankTuningConfig(
            schema_version=SCHEMA_VERSION,
            run_at_utc=datetime.now(UTC),
            golden_set_dir=str(GOLDEN_SET_DIR),
            golden_set_question_count=len(questions),
            corpus_path=str(JSONL_PATH),
            corpus_clause_count=len(corpus),
            chunk_corpus_chunk_count=len(chunks),
            candidate_depths=list(RERANK_CANDIDATE_DEPTHS),
            max_depth=max_depth,
            latency_probe_questions=LATENCY_PROBE_QUESTIONS,
            chosen_candidate_depth=RERANK_CANDIDATE_DEPTH,
            reranker_model_id=RERANKER_MODEL_ID,
            reranker_model_revision=RERANKER_MODEL_REVISION,
            reranker_config_fingerprint=reranker_config_fingerprint(),
            embedding_config_fingerprint=hybrid_fields["embedding_config_fingerprint"],
            lexical_config_fingerprint=hybrid_fields["lexical_config_fingerprint"],
            hybrid_config_fingerprint=hybrid_fields["hybrid_config_fingerprint"],
            scoring_device=scoring_device,
            reranker_device=reranker_device,
            platform=platform.platform(),
        )
    finally:
        session.close()
        engine.dispose()

    report = {"config": config.model_dump(mode="json"), "rows": rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
