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
its ``source_clause_ids``. No metadata filter (that is [M3-04]'s), so this
is the standalone lexical baseline the M3-03 DoD asks for; the committed
numbers and the verdict on which question types it wins live in
``docs/LEXICAL_RETRIEVAL.md``.

For every golden question except ``unanswerable`` ones (which carry no
``reference_clause_ids`` by schema construction, so Recall/MRR/nDCG are
undefined for them -- their count is still reported, not silently
dropped), retrieves the top 10 clause ids and computes Recall@{1,5,10},
MRR and nDCG@10 against ``reference_clause_ids``, each broken down by
``question_type``, ``product_line`` and extraction mode (joined onto each
question via its ``document_id`` against ``data/policies/manifest.csv``),
plus a separate exclusion-clause recall pooled across every reference
clause whose ``clause_type`` is ``exclusion``.

Writes ``eval/runs/retrieval_eval_<retriever>.json`` (machine-readable)
and ``eval/runs/retrieval_eval_<retriever>.md`` (human-readable), both
stamped with the [infrastructure.evaluation.retrieval_run_schema.
RetrievalRunConfig] that produced them. Run via ``make eval-retrieval``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from domain.clause_classification import ClauseType
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
from infrastructure.evaluation.retriever import Retriever
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.lexical_analyzer import build_analyzer
from infrastructure.rag.lexical_config import (
    BM25_B,
    BM25_K1,
    IDF_VARIANT,
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_INDEX_TEXT_FIELD,
    LEXICAL_STEMMING_EXCEPTIONS_PATH,
    config_fingerprint,
)
from infrastructure.rag.lexical_retriever import LexicalRetriever
from infrastructure.rag.lexical_stemming_exceptions import load_stemming_exceptions

GOLDEN_SET_DIR = Path("data/golden_set")
MANIFEST_PATH = Path("data/policies/manifest.csv")
CHUNKS_PATH = CHUNKS_JSONL_PATH
OUTPUT_DIR = Path("eval/runs")
K_VALUES: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10
RETRIEVE_K = max(*K_VALUES, NDCG_K)
RETRIEVER_NAMES = ("random", "lexical")


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
    return parser.parse_args()


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


@dataclass(frozen=True)
class ScoredQuestion:
    """One golden question's retrieval scores plus its breakdown metadata."""

    question_id: str
    question_type: str
    product_line: str
    extraction_mode: str
    reference_clause_ids: tuple[str, ...]
    retrieved: tuple[str, ...]
    recall: dict[int, float]
    mrr: float
    ndcg: float


def evaluate_questions(
    questions: Sequence[GoldenQuestion],
    retriever: Retriever,
    document_meta: dict[str, dict[str, str]],
    *,
    k_values: Sequence[int] = K_VALUES,
    ndcg_k: int = NDCG_K,
    retrieve_k: int = RETRIEVE_K,
) -> tuple[list[ScoredQuestion], int]:
    """Score every non-unanswerable question; return (rows, unanswerable_count).

    ``unanswerable`` questions are skipped entirely -- not retrieved
    against, not scored, since Recall/MRR/nDCG are undefined for an empty
    reference set -- and only counted, so callers can report that count
    instead of silently dropping it.
    """
    rows: list[ScoredQuestion] = []
    unanswerable_count = 0
    for question in questions:
        if question.question_type is QuestionType.UNANSWERABLE:
            unanswerable_count += 1
            continue
        meta = document_meta[question.document_id]
        retrieved = retriever.retrieve(question.question, k=retrieve_k)
        recall = {
            k: recall_at_k(retrieved, question.reference_clause_ids, k)
            for k in k_values
        }
        rows.append(
            ScoredQuestion(
                question_id=question.question_id,
                question_type=question.question_type.value,
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
        "Generated by `scripts/eval_retrieval.py` (`make eval-retrieval` / "
        "`make eval-retrieval-lexical`) against the golden set in "
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
    lines += [
        "## Exclusion-clause recall",
        "",
        "Of every reference clause across all scored questions whose "
        f"`clause_type == exclusion`, the fraction retrieved in the top-"
        f"{exclusion['k']}: **{exclusion_line}** "
        f"({exclusion['hits']}/{exclusion['total']}).",
        "",
        "## Summary",
        "",
        f"- Retriever: `{config['retriever_name']}`",
        f"- Questions scored: {int(report['overall']['n'])} "
        f"(of {config['golden_set_question_count']} total; "
        f"{report['by_question_type']['unanswerable']['n']} unanswerable excluded)",
        f"- Overall Recall@{max(k_values)}: "
        f"{_fmt(report['overall'][f'recall@{max(k_values)}'])}",
        f"- Overall MRR: {_fmt(report['overall']['mrr'])}",
        f"- Overall nDCG@{ndcg_k}: {_fmt(report['overall'][f'ndcg@{ndcg_k}'])}",
        f"- Exclusion-clause recall: {exclusion_line}",
        "",
    ]
    return "\n".join(lines)


def _random_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The `--retriever random` slice of RetrievalRunConfig: just the seed."""
    return {"seed": args.seed}


def _lexical_config_fields(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """The `--retriever lexical` slice of RetrievalRunConfig: the BM25 contract."""
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
        "lexical_config_fingerprint": config_fingerprint(
            exception_tokens=exception_tokens
        ),
    }


def main() -> None:
    """Score the chosen retriever against the golden set."""
    args = _parse_args()

    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)

    retriever_name = args.retriever
    retriever: Retriever
    if retriever_name == "lexical":
        chunks = load_chunk_corpus()
        retriever = build_lexical_retriever(chunks)
        extra_config = _lexical_config_fields(chunks)
    else:
        retriever = RandomRetriever(
            [record.clause_id for record in corpus], seed=args.seed
        )
        extra_config = _random_config_fields(args)

    rows, unanswerable_count = evaluate_questions(questions, retriever, document_meta)

    config = RetrievalRunConfig(
        schema_version=SCHEMA_VERSION,
        retriever_name=retriever_name,
        k_values=list(K_VALUES),
        ndcg_k=NDCG_K,
        golden_set_dir=str(GOLDEN_SET_DIR),
        golden_set_question_count=len(questions),
        corpus_path=str(JSONL_PATH),
        corpus_clause_count=len(corpus),
        run_at_utc=datetime.now(UTC),
        **extra_config,
    )
    report = build_report(config, rows, unanswerable_count, clause_by_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"retrieval_eval_{retriever_name}.json"
    md_path = OUTPUT_DIR / f"retrieval_eval_{retriever_name}.md"
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
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
