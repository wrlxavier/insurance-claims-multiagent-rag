#!/usr/bin/env python3
"""Score a retriever against the golden set [M2-06].

``--retriever random`` (default) runs the built-in [infrastructure.
evaluation.random_retriever.RandomRetriever] -- a retriever that ignores
the question and returns random clause ids -- the [M2-06] self-test that
proves the harness itself is correct (every metric collapses toward zero
against garbage retrieval).

``--retriever lexical`` [M3-03] runs [infrastructure.rag.lexical_retriever.
LexicalRetriever]: hand-rolled Okapi BM25 over ``build/chunks.jsonl`` with
Portuguese tokenisation and Snowball stemming, each chunk hit rolled up to
its ``source_clause_ids``. The M3-03 standalone baseline; committed numbers
and the per-question-type verdict live in ``docs/LEXICAL_RETRIEVAL.md``.

``--retriever dense`` / ``--retriever hybrid`` [M3-04] run
[infrastructure.rag.dense_retriever.DenseRetriever] (exact ``<=>`` cosine
search over the pgvector chunk table) and
[infrastructure.rag.hybrid_retriever.HybridRetriever] (RRF or weighted
fusion of the lexical and dense legs). Both need a running Postgres with
loaded + embedded chunks and the local embedder from the optional ``embed``
uv group. ``--filter default`` cuts each question to its document's SUSEP
process + insurer CNPJ (the default retrieval path); ``--filter none`` is
the unknown-process degradation case. Comparison and verdict:
``docs/HYBRID_RETRIEVAL.md``.

``--rerank`` [M3-05] wraps the chosen retriever in
[infrastructure.rag.reranking_retriever.RerankingRetriever]: it fetches
[infrastructure.rag.reranker_config.RERANK_CANDIDATE_DEPTH] candidates from the
base retriever, re-scores every ``(question, clause text)`` pair with the pinned
cross-encoder (``Alibaba-NLP/gte-multilingual-reranker-base``, also from the
``embed`` group), and returns the reordered top-k. Needs the chunk corpus for
the clause-text lookup. Curve, chosen operating point and latency trade-off:
``docs/RERANKING.md``.

For every golden question except ``unanswerable`` ones (which carry no
``reference_clause_ids`` by schema construction, so Recall/MRR/nDCG are
undefined for them -- their count is still reported, not silently
dropped), retrieves the top 10 clause ids and computes Recall@{1,5,10},
MRR and nDCG@10 against ``reference_clause_ids``, each broken down by
``question_type``, ``product_line`` and extraction mode (joined onto each
question via its ``document_id`` against ``data/policies/manifest.csv``),
plus a separate exclusion-clause recall and a foreign-document rate
([M3-04] item 4) pooled across scored questions.

Writes ``eval/runs/retrieval_eval_<retriever>[...].json`` (machine-readable)
and ``.md`` (human-readable), both stamped with the
[infrastructure.evaluation.retrieval_run_schema.RetrievalRunConfig] that
produced them. Run via ``make eval-retrieval`` (and ``-lexical`` /
``-dense`` / ``-hybrid``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from domain.clause_classification import ClauseType
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.evaluation.random_retriever import DEFAULT_SEED, RandomRetriever
from infrastructure.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from infrastructure.evaluation.retrieval_run_schema import (
    SCHEMA_VERSION,
    RetrievalRunConfig,
)
from infrastructure.evaluation.retriever import FilterableRetriever, Retriever
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.dense_retriever import DenseRetriever
from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.embedding_config import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)
from infrastructure.rag.embedding_config import (
    config_fingerprint as embedding_config_fingerprint,
)
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
from infrastructure.rag.hybrid_config import (
    CANDIDATE_DEPTH,
    FUSION_WEIGHTS,
    RRF_K,
    FusionStrategy,
)
from infrastructure.rag.hybrid_config import (
    config_fingerprint as hybrid_config_fingerprint,
)
from infrastructure.rag.hybrid_retriever import HybridRetriever
from infrastructure.rag.lexical_analyzer import build_analyzer
from infrastructure.rag.lexical_config import (
    BM25_B,
    BM25_K1,
    IDF_VARIANT,
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_INDEX_TEXT_FIELD,
    LEXICAL_STEMMING_EXCEPTIONS_PATH,
)
from infrastructure.rag.lexical_config import (
    config_fingerprint as lexical_config_fingerprint,
)
from infrastructure.rag.lexical_retriever import LexicalRetriever
from infrastructure.rag.lexical_stemming_exceptions import load_stemming_exceptions
from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_cache import CachingReranker
from infrastructure.rag.reranker_config import (
    RERANK_CANDIDATE_DEPTH,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)
from infrastructure.rag.reranker_config import (
    config_fingerprint as reranker_config_fingerprint,
)
from infrastructure.rag.reranking_retriever import RerankingRetriever
from infrastructure.rag.retrieval_filter import RetrievalFilter

GOLDEN_SET_DIR = Path("data/golden_set")
MANIFEST_PATH = Path("data/policies/manifest.csv")
CHUNKS_PATH = CHUNKS_JSONL_PATH
OUTPUT_DIR = Path("eval/runs")
K_VALUES: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10
RETRIEVE_K = max(*K_VALUES, NDCG_K)
RETRIEVER_NAMES = ("random", "lexical", "dense", "hybrid")
FILTER_MODES = ("none", "default")
FUSION_STRATEGIES = tuple(strategy.value for strategy in FusionStrategy)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="random",
        help="Which retriever to score (default: random, the M2-06 self-test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the built-in random retriever (default: 42).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_mode",
        choices=FILTER_MODES,
        default="none",
        help=(
            "Metadata pre-filter [M3-04] (dense/hybrid only). `default` filters "
            "each question to its document's SUSEP process + insurer CNPJ (the "
            "default retrieval path); `none` is the unknown-process degradation "
            "case. Default: none."
        ),
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_STRATEGIES,
        default=FusionStrategy.RRF.value,
        help="Fusion strategy for --retriever hybrid [M3-04] (default: rrf).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Rerank the base retriever's top candidates with the [M3-05] "
            "cross-encoder (lexical/dense/hybrid only; needs the `embed` uv "
            "group and the chunk corpus)."
        ),
    )
    parser.add_argument(
        "--co-retrieval",
        dest="co_retrieval",
        action="store_true",
        help=(
            "After ranking, reserve top-k slots for the [M3-06] exclusion "
            "clauses linked to each retrieved coverage clause (lexical/dense/"
            "hybrid only; applied outermost, after --rerank). Pure structural "
            "post-processing over the parsed corpus -- no model calls."
        ),
    )
    args = parser.parse_args()
    if args.filter_mode == "default" and args.retriever == "random":
        parser.error("--filter default is not supported by --retriever random")
    if args.rerank and args.retriever == "random":
        parser.error("--rerank is not supported by --retriever random")
    if args.co_retrieval and args.retriever == "random":
        parser.error("--co-retrieval is not supported by --retriever random")
    return args


def resolve_source(extraction_mode: str) -> Literal["text", "ocr"]:
    """Map manifest.csv's extraction_mode to the corpus's source vocabulary.

    Mirrors ``scripts/build_corpus.py``'s ``resolve_source`` exactly, so
    this harness's extraction-mode buckets match ``source`` elsewhere in
    the repo (e.g. ``scripts/score_parsing_quality.py``'s by-extraction-
    mode split).
    """
    return "ocr" if extraction_mode == "ocr_required" else "text"


def load_golden_questions(golden_set_dir: Path) -> list[GoldenQuestion]:
    """Parse every ``*.jsonl`` file under golden_set_dir into GoldenQuestion rows.

    No cross-file consistency checks here (question_type-matches-filename,
    duplicate ids, dangling clause ids) -- that's `make validate-golden-set`'s
    job, expected to have already passed.
    """
    questions: list[GoldenQuestion] = []
    for path in sorted(golden_set_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    questions.append(GoldenQuestion.model_validate(json.loads(line)))
    return questions


def load_document_metadata(manifest_path: Path) -> dict[str, dict[str, str]]:
    """document_id -> manifest row, for the product_line/extraction_mode join."""
    return {row["id"]: row for row in read_manifest(manifest_path)}


def load_corpus(jsonl_path: Path) -> list[ParsedClauseRecord]:
    """Load the parsed corpus, failing loudly if `make parse` hasn't run."""
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(jsonl_path)


def load_chunk_corpus(jsonl_path: Path = CHUNKS_PATH) -> list[ChunkRecord]:
    """Load the chunk corpus the lexical retriever indexes, failing loudly."""
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} does not exist. Run `make build-chunks` (after "
            "`make parse`) first."
        )
    return read_chunks_jsonl(jsonl_path)


def build_lexical_retriever(chunks: Sequence[ChunkRecord]) -> LexicalRetriever:
    """Build the [M3-03] BM25 lexical retriever from the chunk corpus."""
    return LexicalRetriever.from_chunks(list(chunks), build_analyzer())


def build_clause_text_map(chunks: Sequence[ChunkRecord]) -> dict[str, str]:
    """``clause_id`` -> the breadcrumb-prefixed chunk text the [M3-05] reranker scores.

    Uses ``ChunkRecord.text`` -- the same field BM25 indexes and the embedder
    embeds -- so all three legs judge the candidate on the same representation
    (the reasoning ``docs/LEXICAL_RETRIEVAL.md`` gives for indexing ``text`` not
    ``display_text``). A clause split across several chunks is rejoined in
    ``chunk_index`` order; a short clause merged into a neighbour's chunk gets
    that chunk's text via ``source_clause_ids``.
    """
    pieces: dict[str, list[tuple[int, str]]] = {}
    for chunk in chunks:
        for clause_id in chunk.source_clause_ids:
            pieces.setdefault(clause_id, []).append((chunk.chunk_index, chunk.text))
    return {
        clause_id: "\n\n".join(text for _, text in sorted(parts))
        for clause_id, parts in pieces.items()
    }


@dataclass(frozen=True)
class ScoredQuestion:
    """One golden question's retrieval scores plus its breakdown metadata."""

    question_id: str
    question_type: str
    document_id: str
    product_line: str
    extraction_mode: str
    reference_clause_ids: tuple[str, ...]
    retrieved: tuple[str, ...]
    recall: dict[int, float]
    mrr: float
    ndcg: float


# A per-question metadata filter, or None (this question is not filtered).
FilterFor = Callable[[GoldenQuestion], RetrievalFilter | None]


def _retrieve(
    retriever: Retriever,
    question: str,
    *,
    k: int,
    metadata_filter: RetrievalFilter | None,
) -> list[str]:
    """Call ``retriever.retrieve``, threading a [M3-04] filter when there is one.

    ``random`` is scored without a filter (the bare [Retriever], and
    ``--filter default`` is rejected for it); the filtered path only ever runs
    against ``lexical``/``dense``/``hybrid``, which satisfy [FilterableRetriever].
    """
    if metadata_filter is None:
        return retriever.retrieve(question, k=k)
    return cast(FilterableRetriever, retriever).retrieve(
        question, k=k, metadata_filter=metadata_filter
    )


def evaluate_questions(
    questions: Sequence[GoldenQuestion],
    retriever: Retriever,
    document_meta: dict[str, dict[str, str]],
    *,
    filter_for: FilterFor | None = None,
    k_values: Sequence[int] = K_VALUES,
    ndcg_k: int = NDCG_K,
    retrieve_k: int = RETRIEVE_K,
) -> tuple[list[ScoredQuestion], int]:
    """Score every non-unanswerable question; return (rows, unanswerable_count).

    ``unanswerable`` questions are skipped entirely -- not retrieved
    against, not scored, since Recall/MRR/nDCG are undefined for an empty
    reference set -- and only counted, so callers can report that count
    instead of silently dropping it.

    ``filter_for`` ([M3-04]) supplies a per-question [RetrievalFilter]; the
    retriever must then satisfy [FilterableRetriever]. ``None`` (the default)
    keeps the pre-M3-04 unfiltered call path.
    """
    rows: list[ScoredQuestion] = []
    unanswerable_count = 0
    for question in questions:
        if question.question_type is QuestionType.UNANSWERABLE:
            unanswerable_count += 1
            continue
        meta = document_meta[question.document_id]
        metadata_filter = filter_for(question) if filter_for is not None else None
        retrieved = _retrieve(
            retriever, question.question, k=retrieve_k, metadata_filter=metadata_filter
        )
        recall = {
            k: recall_at_k(retrieved, question.reference_clause_ids, k)
            for k in k_values
        }
        rows.append(
            ScoredQuestion(
                question_id=question.question_id,
                question_type=question.question_type.value,
                document_id=question.document_id,
                product_line=meta["product_line"],
                extraction_mode=resolve_source(meta["extraction_mode"]),
                reference_clause_ids=tuple(question.reference_clause_ids),
                retrieved=tuple(retrieved),
                recall=recall,
                mrr=reciprocal_rank(retrieved, question.reference_clause_ids),
                ndcg=ndcg_at_k(retrieved, question.reference_clause_ids, ndcg_k),
            )
        )
    return rows, unanswerable_count


def aggregate(
    rows: Sequence[ScoredQuestion], k_values: Sequence[int], ndcg_k: int
) -> dict[str, float]:
    """Mean Recall@k (per k), MRR and nDCG@ndcg_k across rows, plus n.

    An empty ``rows`` returns zeroed metrics rather than raising -- an
    empty breakdown cell (e.g. a product line with no scorable questions)
    is a valid, reportable state, not an error.
    """
    n = len(rows)
    if n == 0:
        result = {f"recall@{k}": 0.0 for k in k_values}
        result["mrr"] = 0.0
        result[f"ndcg@{ndcg_k}"] = 0.0
        result["n"] = 0.0
        return result
    result = {f"recall@{k}": sum(row.recall[k] for row in rows) / n for k in k_values}
    result["mrr"] = sum(row.mrr for row in rows) / n
    result[f"ndcg@{ndcg_k}"] = sum(row.ndcg for row in rows) / n
    result["n"] = float(n)
    return result


def group_by(
    rows: Sequence[ScoredQuestion], key_fn: Callable[[ScoredQuestion], str]
) -> dict[str, list[ScoredQuestion]]:
    """Group scored rows by an arbitrary key, preserving first-seen order."""
    groups: dict[str, list[ScoredQuestion]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def compute_exclusion_clause_recall(
    rows: Sequence[ScoredQuestion],
    clause_by_id: dict[str, ParsedClauseRecord],
    *,
    k: int = RETRIEVE_K,
) -> dict[str, float | int | None]:
    """Micro-averaged recall over every reference clause typed ``exclusion``.

    Pools every (question, reference clause) pair across all scored
    questions -- not just ``coverage_with_exclusion`` ones, since any
    question type could incidentally reference an exclusion clause -- and
    counts how many of those exclusion-typed reference clauses appear in
    that question's top-k retrieved list.
    """
    hits = 0
    total = 0
    for row in rows:
        retrieved_top_k = set(row.retrieved[:k])
        for clause_id in row.reference_clause_ids:
            if clause_by_id[clause_id].clause_type is ClauseType.EXCLUSION:
                total += 1
                if clause_id in retrieved_top_k:
                    hits += 1
    recall = hits / total if total > 0 else None
    return {"k": k, "hits": hits, "total": total, "recall": recall}


def _document_of(clause_id: str) -> str:
    """The document id prefix of a ``{document_id}:{path}`` clause id."""
    return clause_id.split(":", 1)[0]


def compute_foreign_document_rate(
    rows: Sequence[ScoredQuestion], *, k: int = RETRIEVE_K
) -> dict[str, float | int | None]:
    """Fraction of retrieved top-k clause ids from a document other than target.

    [M3-04] DoD item 4: the metadata pre-filter's job is to keep retrieval
    inside the question's own SUSEP process. Pooled over every scored question:
    ``hits`` is the count of retrieved clauses whose document != the question's
    ``document_id``, ``total`` the count of retrieved clauses. **0 under
    ``--filter default``**; a non-trivial number under ``--filter none`` is the
    cross-document leakage the filter removes.
    """
    hits = 0
    total = 0
    for row in rows:
        for clause_id in row.retrieved[:k]:
            total += 1
            if _document_of(clause_id) != row.document_id:
                hits += 1
    rate = hits / total if total > 0 else None
    return {"k": k, "hits": hits, "total": total, "rate": rate}


def build_report(
    config: RetrievalRunConfig,
    rows: Sequence[ScoredQuestion],
    unanswerable_count: int,
    clause_by_id: dict[str, ParsedClauseRecord],
    *,
    exclusion_recall_k: int = RETRIEVE_K,
) -> dict[str, Any]:
    """Assemble the one dict both the JSON and Markdown reports render from."""
    k_values, ndcg_k = config.k_values, config.ndcg_k

    by_question_type: dict[str, dict[str, Any]] = {
        question_type: aggregate(group_rows, k_values, ndcg_k)
        for question_type, group_rows in group_by(
            rows, lambda r: r.question_type
        ).items()
    }
    by_question_type[QuestionType.UNANSWERABLE.value] = {
        "n": unanswerable_count,
        "excluded_from_scoring": True,
    }
    by_product_line = {
        product_line: aggregate(group_rows, k_values, ndcg_k)
        for product_line, group_rows in group_by(rows, lambda r: r.product_line).items()
    }
    by_extraction_mode = {
        mode: aggregate(group_rows, k_values, ndcg_k)
        for mode, group_rows in group_by(rows, lambda r: r.extraction_mode).items()
    }

    return {
        "config": config.model_dump(mode="json"),
        "overall": aggregate(rows, k_values, ndcg_k),
        "by_question_type": by_question_type,
        "by_product_line": by_product_line,
        "by_extraction_mode": by_extraction_mode,
        "exclusion_clause_recall": compute_exclusion_clause_recall(
            rows, clause_by_id, k=exclusion_recall_k
        ),
        "foreign_document_rate": compute_foreign_document_rate(
            rows, k=exclusion_recall_k
        ),
    }


def _fmt(value: float) -> str:
    """Render a 0..1 metric as a percentage, e.g. '12.3%'."""
    return f"{value:.1%}"


def _metric_table(
    title: str,
    header_label: str,
    groups: dict[str, dict[str, Any]],
    k_values: Sequence[int],
    ndcg_k: int,
) -> list[str]:
    """Render one Markdown table of n/Recall@k/MRR/nDCG@k rows, one per group."""
    header = (
        [header_label, "n"]
        + [f"recall@{k}" for k in k_values]
        + ["mrr", f"ndcg@{ndcg_k}"]
    )
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for key, metrics in groups.items():
        if metrics.get("excluded_from_scoring"):
            cells = [key, str(int(metrics["n"]))] + ["—"] * (len(header) - 2)
        else:
            cells = [key, str(int(metrics["n"]))]
            cells += [_fmt(metrics[f"recall@{k}"]) for k in k_values]
            cells.append(_fmt(metrics["mrr"]))
            cells.append(_fmt(metrics[f"ndcg@{ndcg_k}"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the Markdown retrieval-evaluation report."""
    config = report["config"]
    k_values: list[int] = config["k_values"]
    ndcg_k: int = config["ndcg_k"]

    lines = [
        "# Retrieval evaluation",
        "",
        "Generated by `scripts/eval_retrieval.py` (`make eval-retrieval` and "
        "`-lexical` / `-dense` / `-hybrid`) against the golden set in "
        "`data/golden_set/` and the parsed corpus in "
        "`build/parsed_clauses.jsonl`. Recall@k, MRR and nDCG@10 are all "
        "computed over each question's top-10 retrieved clause ids, so MRR "
        "and nDCG@10 are also effectively capped at that depth. "
        "`unanswerable` questions carry no reference clauses by "
        "construction, so every metric below excludes them; their count is "
        "still reported in the question_type table.",
        "",
        "## Run configuration",
        "",
        f"- Retriever: `{config['retriever_name']}`",
        f"- k values: {config['k_values']}",
        f"- nDCG depth: {config['ndcg_k']}",
        f"- Golden set: `{config['golden_set_dir']}` "
        f"({config['golden_set_question_count']} questions)",
        f"- Corpus: `{config['corpus_path']}` "
        f"({config['corpus_clause_count']} clauses)",
    ]
    if config.get("chunk_corpus_path"):
        lines += [
            f"- Chunk corpus (BM25-indexed): `{config['chunk_corpus_path']}` "
            f"({config['chunk_corpus_chunk_count']} chunks)",
            f"- Analyzer version: `{config['lexical_analyzer_version']}`; "
            f"BM25 k1={config['bm25_k1']}, b={config['bm25_b']}; "
            f"IDF `{config['lexical_idf_variant']}`; "
            f"indexed field `{config['lexical_index_text_field']}`; "
            f"{config['stemming_exception_count']} stemming exceptions",
            f"- Lexical config fingerprint: `{config['lexical_config_fingerprint']}`",
        ]
    if config.get("dense_model_id"):
        lines.append(
            f"- Dense model: `{config['dense_model_id']}` @ "
            f"`{config['dense_model_revision']}`; embedding config fingerprint "
            f"`{config['embedding_config_fingerprint']}`"
        )
    if config.get("fusion_strategy"):
        lines += [
            f"- Fusion: `{config['fusion_strategy']}` (RRF k={config['rrf_k']}, "
            f"weights {config['fusion_weights']}, candidate depth "
            f"{config['candidate_depth']})",
            f"- Hybrid config fingerprint: `{config['hybrid_config_fingerprint']}`",
        ]
    if config.get("reranker_model_id"):
        lines += [
            f"- Reranker: `{config['reranker_model_id']}` @ "
            f"`{config['reranker_model_revision']}`; candidate depth "
            f"{config['rerank_candidate_depth']}",
            f"- Reranker config fingerprint: `{config['reranker_config_fingerprint']}`",
        ]
    if config.get("reserved_exclusion_slots") is not None:
        lines += [
            f"- Exclusion co-retrieval ([M3-06]): {config['reserved_exclusion_slots']} "
            f"reserved slot(s), adjacent-section page gap "
            f"{config['adjacent_section_max_page_gap']}",
            "- Co-retrieval config fingerprint: "
            f"`{config['co_retrieval_config_fingerprint']}`",
        ]
    if config.get("filter_mode"):
        lines.append(f"- Metadata filter: `{config['filter_mode']}`")
    lines += [
        f"- Seed: {config['seed']}",
        f"- Run at (UTC): {config['run_at_utc']}",
        "",
    ]
    lines += _metric_table(
        "Overall metrics", "", {"overall": report["overall"]}, k_values, ndcg_k
    )
    lines += _metric_table(
        "By question_type",
        "question_type",
        report["by_question_type"],
        k_values,
        ndcg_k,
    )
    lines += _metric_table(
        "By product line", "product_line", report["by_product_line"], k_values, ndcg_k
    )
    lines += _metric_table(
        "By extraction mode",
        "extraction_mode",
        report["by_extraction_mode"],
        k_values,
        ndcg_k,
    )

    exclusion = report["exclusion_clause_recall"]
    exclusion_recall = exclusion["recall"]
    exclusion_line = (
        _fmt(exclusion_recall)
        if exclusion_recall is not None
        else "n/a (no exclusion clauses referenced)"
    )
    foreign = report["foreign_document_rate"]
    foreign_rate = foreign["rate"]
    foreign_line = (
        _fmt(foreign_rate) if foreign_rate is not None else "n/a (nothing retrieved)"
    )
    lines += [
        "## Exclusion-clause recall",
        "",
        "Of every reference clause across all scored questions whose "
        f"`clause_type == exclusion`, the fraction retrieved in the top-"
        f"{exclusion['k']}: **{exclusion_line}** "
        f"({exclusion['hits']}/{exclusion['total']}).",
        "",
        "## Foreign-document rate",
        "",
        f"Of every clause retrieved in the top-{foreign['k']} across all scored "
        "questions, the fraction from a document other than the question's "
        f"target: **{foreign_line}** ({foreign['hits']}/{foreign['total']}). "
        "[M3-04] item 4: this is **0** under the SUSEP process + CNPJ "
        "pre-filter, and the cross-document leakage the filter removes when "
        "unfiltered.",
        "",
        "## Summary",
        "",
        f"- Retriever: `{config['retriever_name']}`",
        f"- Metadata filter: `{config.get('filter_mode', 'none')}`",
        f"- Questions scored: {int(report['overall']['n'])} "
        f"(of {config['golden_set_question_count']} total; "
        f"{report['by_question_type']['unanswerable']['n']} unanswerable excluded)",
        f"- Overall Recall@{max(k_values)}: "
        f"{_fmt(report['overall'][f'recall@{max(k_values)}'])}",
        f"- Overall MRR: {_fmt(report['overall']['mrr'])}",
        f"- Overall nDCG@{ndcg_k}: {_fmt(report['overall'][f'ndcg@{ndcg_k}'])}",
        f"- Exclusion-clause recall: {exclusion_line}",
        f"- Foreign-document rate: {foreign_line}",
        "",
    ]
    return "\n".join(lines)


def _random_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The `--retriever random` slice of RetrievalRunConfig: just the seed."""
    return {"seed": args.seed}


def _lexical_config_fields(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """The lexical-leg slice of RetrievalRunConfig: the BM25 contract."""
    exception_tokens = load_stemming_exceptions(LEXICAL_STEMMING_EXCEPTIONS_PATH)
    return {
        "seed": None,
        "chunk_corpus_path": str(CHUNKS_PATH),
        "chunk_corpus_chunk_count": len(chunks),
        "lexical_analyzer_version": LEXICAL_ANALYZER_VERSION,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "lexical_idf_variant": IDF_VARIANT,
        "lexical_index_text_field": LEXICAL_INDEX_TEXT_FIELD,
        "stemming_exception_count": len(exception_tokens),
        "lexical_config_fingerprint": lexical_config_fingerprint(
            exception_tokens=exception_tokens
        ),
    }


def _dense_config_fields() -> dict[str, Any]:
    """The dense-leg slice of RetrievalRunConfig: the pinned embedding contract."""
    return {
        "seed": None,
        "dense_model_id": EMBEDDING_MODEL_ID,
        "dense_model_revision": EMBEDDING_MODEL_REVISION,
        "embedding_config_fingerprint": embedding_config_fingerprint(),
    }


def _hybrid_config_fields(chunks: Sequence[ChunkRecord], fusion: str) -> dict[str, Any]:
    """The `--retriever hybrid` slice: both legs plus the fusion contract."""
    lexical = _lexical_config_fields(chunks)
    return {
        **lexical,
        **_dense_config_fields(),
        "fusion_strategy": fusion,
        "rrf_k": RRF_K,
        "fusion_weights": list(FUSION_WEIGHTS),
        "candidate_depth": CANDIDATE_DEPTH,
        "hybrid_config_fingerprint": hybrid_config_fingerprint(
            lexical_config_fingerprint=lexical["lexical_config_fingerprint"]
        ),
    }


def _rerank_config_fields() -> dict[str, Any]:
    """The `--rerank` slice of RetrievalRunConfig: the pinned cross-encoder contract."""
    return {
        "reranker_model_id": RERANKER_MODEL_ID,
        "reranker_model_revision": RERANKER_MODEL_REVISION,
        "rerank_candidate_depth": RERANK_CANDIDATE_DEPTH,
        "reranker_config_fingerprint": reranker_config_fingerprint(),
    }


def _apply_rerank(
    args: argparse.Namespace,
    base: FilterableRetriever,
    chunks: Sequence[ChunkRecord],
    extra_config: dict[str, Any],
) -> tuple[Retriever, dict[str, Any]]:
    """Wrap ``base`` in a [M3-05] RerankingRetriever when ``--rerank`` is set.

    ``chunks`` supplies the clause-text lookup; it is loaded by the caller only
    on the ``--rerank`` path for ``dense`` (lexical/hybrid already have it).
    ``base`` is always a filterable retriever here -- ``random`` (the one
    non-filterable retriever) rejects ``--rerank`` in ``_parse_args``.
    """
    if not args.rerank:
        return base, extra_config
    reranker = CachingReranker(_load_reranker())
    wrapped = RerankingRetriever(base, reranker, build_clause_text_map(chunks))
    return wrapped, {**extra_config, **_rerank_config_fields()}


def _co_retrieval_config_fields() -> dict[str, Any]:
    """The `--co-retrieval` slice of RetrievalRunConfig: the [M3-06] contract."""
    return {
        "reserved_exclusion_slots": RESERVED_EXCLUSION_SLOTS,
        "adjacent_section_max_page_gap": ADJACENT_SECTION_MAX_PAGE_GAP,
        "co_retrieval_config_fingerprint": co_retrieval_config_fingerprint(),
    }


def _apply_co_retrieval(
    args: argparse.Namespace,
    base: Retriever,
    corpus: Sequence[ParsedClauseRecord],
    extra_config: dict[str, Any],
) -> tuple[Retriever, dict[str, Any]]:
    """Wrap ``base`` in a [M3-06] ExclusionCoRetrievalRetriever for ``--co-retrieval``.

    Applied outermost -- after ``--rerank`` -- over the same parsed corpus the
    harness already loaded. Pure structural post-processing, no model calls, so
    (unlike ``--rerank``) it needs nothing from the ``embed`` uv group. ``base``
    is always a filterable retriever here (``random`` rejects ``--co-retrieval``
    in ``_parse_args``).
    """
    if not args.co_retrieval:
        return base, extra_config
    wrapped = ExclusionCoRetrievalRetriever(
        cast(FilterableRetriever, base), ClauseGraph(corpus)
    )
    return wrapped, {**extra_config, **_co_retrieval_config_fields()}


@contextmanager
def _open_retriever(
    args: argparse.Namespace, corpus: Sequence[ParsedClauseRecord]
) -> Iterator[tuple[Retriever, dict[str, Any]]]:
    """Yield ``(retriever, config fields)``; ``dense``/``hybrid`` hold a DB session.

    ``dense``/``hybrid`` (and any ``--rerank`` run) need Postgres (loaded +
    embedded chunks) and the local models from the optional ``embed`` uv group,
    so ``make eval-retrieval-hybrid`` / ``-rerank`` run under ``uv run --group
    embed`` -- mirroring ``make embed-chunks``. The query embedder is wrapped in
    a [infrastructure.rag.embedding_cache.CachingEmbedder] and the reranker in a
    [infrastructure.rag.reranker_cache.CachingReranker] so a re-run over the 117
    golden queries costs nothing.
    """
    name = args.retriever
    if name == "random":
        yield (
            RandomRetriever([r.clause_id for r in corpus], seed=args.seed),
            _random_config_fields(args),
        )
        return
    if name == "lexical":
        chunks = load_chunk_corpus()
        lexical = build_lexical_retriever(chunks)
        base, config = _apply_rerank(
            args, lexical, chunks, _lexical_config_fields(chunks)
        )
        yield _apply_co_retrieval(args, base, corpus, config)
        return

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    try:
        assert_chunk_table_ready(session)
        embedder = CachingEmbedder(_load_query_embedder())
        dense = DenseRetriever(session, embedder)
        if name == "dense":
            chunks = load_chunk_corpus() if args.rerank else []
            base, config = _apply_rerank(args, dense, chunks, _dense_config_fields())
            yield _apply_co_retrieval(args, base, corpus, config)
        else:
            chunks = load_chunk_corpus()
            hybrid = HybridRetriever(
                build_lexical_retriever(chunks),
                dense,
                fusion=FusionStrategy(args.fusion),
            )
            base, config = _apply_rerank(
                args, hybrid, chunks, _hybrid_config_fields(chunks, args.fusion)
            )
            yield _apply_co_retrieval(args, base, corpus, config)
    finally:
        session.close()
        engine.dispose()


class _LazyEmbedder:
    """A query [Embedder] that builds the real model only on a cache miss.

    [M3-05]: every ``golden-set-v1`` query vector is normally already in the
    [infrastructure.rag.embedding_cache.CachingEmbedder] file from a prior
    ``dense`` / ``hybrid`` run, so a run that wraps this in a ``CachingEmbedder``
    never calls ``embed`` -- and eagerly loading ``SentenceTransformerEmbedder``
    anyway pins ~1 GB of the small dev GPU that the ``--rerank`` cross-encoder now
    wants. On an actual miss it constructs the real embedder once (same model,
    same vectors) and delegates from then on.
    """

    def __init__(self) -> None:
        self._real: Embedder | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._real is None:
            from infrastructure.rag.sentence_transformer_embedder import (
                SentenceTransformerEmbedder,
            )

            self._real = SentenceTransformerEmbedder()
        return self._real.embed(texts)


def _load_query_embedder() -> Embedder:
    """A lazily-constructed real embedder (see [_LazyEmbedder]).

    Deferred import so the heavy ``embed`` group is only needed for
    ``dense``/``hybrid`` runs (mirrors ``scripts/embed_chunks.py``).
    """
    return _LazyEmbedder()


def _load_reranker(*, device: str | None = None) -> Reranker:
    """Load the real local cross-encoder ([M3-05]).

    ``device`` passes straight to ``sentence-transformers`` (``None`` auto-selects
    CUDA when present, else CPU). Deferred import so the heavy ``embed`` group is
    only needed on a ``--rerank`` run (mirrors ``_load_query_embedder``). The
    cross-encoder's *scores* -- and so every Recall / MRR / nDCG / exclusion
    number -- are device-independent, so ``make eval-retrieval-rerank`` takes the
    default (the dev GPU when present: ~10x faster, and ``_load_query_embedder``
    is lazy now so the card is free).
    The *latency* the ``docs/RERANKING.md`` trade-off turns on is a CPU number by
    design -- a reranker in an interactive path, no GPU serving infra -- so
    ``scripts/tune_reranking.py`` measures that on a ``device="cpu"`` instance.
    M4 constructs its own reranker for its serving hardware.
    """
    from infrastructure.rag.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(device=device)


def _build_filter_for(
    filter_mode: str, document_meta: dict[str, dict[str, str]]
) -> FilterFor | None:
    """`default` -> a per-question SUSEP process + CNPJ filter; `none` -> None."""
    if filter_mode != "default":
        return None

    def filter_for(question: GoldenQuestion) -> RetrievalFilter | None:
        return RetrievalFilter.from_manifest_row(document_meta[question.document_id])

    return filter_for


def _output_stem(args: argparse.Namespace) -> str:
    """`eval/runs/` basename, tagged by the run's distinguishing options.

    ``random`` and the unfiltered ``lexical`` baseline keep their pre-M3-04
    names (``retrieval_eval_random`` / ``retrieval_eval_lexical``) so the
    ``docs/LEXICAL_RETRIEVAL.md`` reference and the eval smoke test stay valid;
    every other combination is tagged so the four hybrid comparison runs do not
    overwrite each other.
    """
    parts = [args.retriever]
    if args.retriever == "hybrid":
        parts.append(args.fusion)
    if args.rerank:
        parts.append("rerank")
    if args.co_retrieval:
        parts.append("co-retrieval")
    if args.filter_mode != "none":
        parts.append(f"filter-{args.filter_mode}")
    return "retrieval_eval_" + "_".join(parts)


def main() -> None:
    """Score the chosen retriever against the golden set."""
    args = _parse_args()

    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)
    filter_for = _build_filter_for(args.filter_mode, document_meta)

    with _open_retriever(args, corpus) as (retriever, extra_config):
        rows, unanswerable_count = evaluate_questions(
            questions, retriever, document_meta, filter_for=filter_for
        )

    config = RetrievalRunConfig(
        schema_version=SCHEMA_VERSION,
        retriever_name=args.retriever,
        k_values=list(K_VALUES),
        ndcg_k=NDCG_K,
        golden_set_dir=str(GOLDEN_SET_DIR),
        golden_set_question_count=len(questions),
        corpus_path=str(JSONL_PATH),
        corpus_clause_count=len(corpus),
        run_at_utc=datetime.now(UTC),
        filter_mode=args.filter_mode,
        **extra_config,
    )
    report = build_report(config, rows, unanswerable_count, clause_by_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(args)
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    overall = report["overall"]
    print(
        f"Overall: Recall@{max(K_VALUES)}={_fmt(overall[f'recall@{max(K_VALUES)}'])} "
        f"MRR={_fmt(overall['mrr'])} nDCG@{NDCG_K}={_fmt(overall[f'ndcg@{NDCG_K}'])}"
    )
    print(f"Exclusion-clause recall: {report['exclusion_clause_recall']['recall']}")
    print(f"Foreign-document rate: {report['foreign_document_rate']['rate']}")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
