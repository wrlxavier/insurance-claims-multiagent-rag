"""Tests for the retrieval evaluation script [M2-06]."""

from pathlib import Path

import pytest
from scripts.eval_retrieval import (
    K_VALUES,
    NDCG_K,
    ScoredQuestion,
    _build_filter_for,
    _output_stem,
    _parse_args,
    aggregate,
    build_clause_text_map,
    build_lexical_retriever,
    compute_exclusion_clause_recall,
    compute_foreign_document_rate,
    evaluate_questions,
    load_golden_questions,
    render_markdown_report,
    resolve_source,
)

from infrastructure.evaluation.golden_set_schema import GoldenQuestion
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.retrieval_filter import RetrievalFilter


def make_question(**overrides: object) -> GoldenQuestion:
    """Build a valid direct_lookup GoldenQuestion, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "question_id": "direct_lookup-001",
        "document_id": "1",
        "question": "O que caracteriza perda total do veículo?",
        "reference_clause_ids": ["1:a"],
        "question_type": "direct_lookup",
        "difficulty": "easy",
        "expected_verdict": None,
        "notes": "",
        "authored_at": "2026-08-21",
    }
    fields.update(overrides)
    return GoldenQuestion.model_validate(fields)


def _clause(clause_id: str, clause_type: str) -> ParsedClauseRecord:
    """Build a minimal ParsedClauseRecord with the given id and clause_type."""
    return ParsedClauseRecord.model_validate(
        {
            "schema_version": "v1",
            "clause_id": clause_id,
            "document_id": "1",
            "parent_id": None,
            "path": clause_id.split(":", 1)[1],
            "title": "Title",
            "text": "Text.",
            "clause_type": clause_type,
            "type_source": "rule",
            "confidence": 1.0,
            "bundle_section": None,
            "page_start": 1,
            "page_end": 1,
            "source": "text",
            "susep_process": "123",
            "insurer": "Insurer",
            "cnpj": "123",
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2020",
        }
    )


class FakeRetriever:
    """A retriever test double that always returns the same fixed sequence."""

    def __init__(self, result: list[str]) -> None:
        self._result = result
        self.seen_filters: list[RetrievalFilter | None] = []

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        del question
        self.seen_filters.append(metadata_filter)
        return self._result[:k]


@pytest.mark.unit
def test_resolve_source_maps_ocr_required_to_ocr() -> None:
    assert resolve_source("ocr_required") == "ocr"


@pytest.mark.unit
def test_resolve_source_maps_text_and_anything_else_to_text() -> None:
    assert resolve_source("text") == "text"
    assert resolve_source("something_else") == "text"


@pytest.mark.unit
def test_load_golden_questions_reads_all_files_under_dir(tmp_path: Path) -> None:
    direct_lookup = make_question()
    unanswerable = make_question(
        question_id="unanswerable-001",
        question_type="unanswerable",
        reference_clause_ids=[],
        expected_verdict="insufficient_information",
    )
    (tmp_path / "direct_lookup.jsonl").write_text(
        direct_lookup.model_dump_json() + "\n", encoding="utf-8"
    )
    (tmp_path / "unanswerable.jsonl").write_text(
        unanswerable.model_dump_json() + "\n", encoding="utf-8"
    )

    questions = load_golden_questions(tmp_path)

    assert {q.question_id for q in questions} == {
        "direct_lookup-001",
        "unanswerable-001",
    }


@pytest.mark.unit
def test_evaluate_questions_skips_unanswerable_and_scores_others() -> None:
    scorable = make_question()
    unanswerable = make_question(
        question_id="unanswerable-001",
        question_type="unanswerable",
        reference_clause_ids=[],
        expected_verdict="insufficient_information",
    )
    document_meta = {"1": {"product_line": "CASCO", "extraction_mode": "text"}}
    retrieved_sequence = ["1:a"] + [f"other-{i}" for i in range(9)]
    retriever = FakeRetriever(retrieved_sequence)

    rows, unanswerable_count = evaluate_questions(
        [scorable, unanswerable], retriever, document_meta
    )

    assert unanswerable_count == 1
    assert len(rows) == 1
    row = rows[0]
    assert row.question_id == "direct_lookup-001"
    assert row.document_id == "1"
    assert row.product_line == "CASCO"
    assert row.extraction_mode == "text"
    assert row.recall == {1: 1.0, 5: 1.0, 10: 1.0}
    assert row.mrr == 1.0
    assert row.ndcg == 1.0
    assert retriever.seen_filters == [None]  # no filter_for -> unfiltered path


@pytest.mark.unit
def test_evaluate_questions_threads_a_per_question_filter() -> None:
    question = make_question()
    document_meta = {
        "1": {
            "product_line": "CASCO",
            "extraction_mode": "text",
            "susep_process": "P1",
            "cnpj": "C1",
        }
    }
    retriever = FakeRetriever(["1:a"])

    evaluate_questions(
        [question],
        retriever,
        document_meta,
        filter_for=_build_filter_for("default", document_meta),
    )

    assert retriever.seen_filters == [RetrievalFilter(susep_process="P1", cnpj="C1")]


@pytest.mark.unit
def test_compute_foreign_document_rate_pools_wrong_document_hits() -> None:
    row = ScoredQuestion(
        question_id="cross_document-001",
        question_type="cross_document",
        document_id="10",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("10:x",),
        retrieved=("10:x", "17:y", "10:z", "17:w"),
        recall={1: 1.0, 5: 1.0, 10: 1.0},
        mrr=1.0,
        ndcg=1.0,
    )

    result = compute_foreign_document_rate([row], k=10)

    assert result == {"k": 10, "hits": 2, "total": 4, "rate": 0.5}


@pytest.mark.unit
def test_output_stem_keeps_random_and_lexical_names_and_tags_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stem_for(*argv: str) -> str:
        monkeypatch.setattr("sys.argv", ["eval_retrieval.py", *argv])
        return _output_stem(_parse_args())

    assert stem_for("--retriever", "lexical") == "retrieval_eval_lexical"
    assert stem_for("--retriever", "random") == "retrieval_eval_random"
    assert stem_for("--retriever", "dense") == "retrieval_eval_dense"
    assert stem_for("--retriever", "lexical", "--filter", "default") == (
        "retrieval_eval_lexical_filter-default"
    )
    assert (
        stem_for("--retriever", "hybrid", "--fusion", "weighted", "--filter", "default")
        == "retrieval_eval_hybrid_weighted_filter-default"
    )
    assert (
        stem_for("--retriever", "hybrid", "--filter", "default", "--rerank")
        == "retrieval_eval_hybrid_rrf_rerank_filter-default"
    )


@pytest.mark.unit
def test_aggregate_computes_mean_across_rows() -> None:
    row_hit = ScoredQuestion(
        question_id="q1",
        question_type="direct_lookup",
        document_id="1",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("1:a",),
        retrieved=("1:a",),
        recall={1: 1.0, 5: 1.0, 10: 1.0},
        mrr=1.0,
        ndcg=1.0,
    )
    row_miss = ScoredQuestion(
        question_id="q2",
        question_type="direct_lookup",
        document_id="1",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("1:b",),
        retrieved=(),
        recall={1: 0.0, 5: 0.0, 10: 0.0},
        mrr=0.0,
        ndcg=0.0,
    )

    result = aggregate([row_hit, row_miss], K_VALUES, NDCG_K)

    assert result["n"] == 2.0
    assert result["recall@1"] == pytest.approx(0.5)
    assert result["recall@10"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
    assert result[f"ndcg@{NDCG_K}"] == pytest.approx(0.5)


@pytest.mark.unit
def test_aggregate_empty_group_returns_zero_n() -> None:
    result = aggregate([], K_VALUES, NDCG_K)

    assert result["n"] == 0.0
    assert all(value == 0.0 for value in result.values())


@pytest.mark.unit
def test_compute_exclusion_clause_recall_counts_only_exclusion_type() -> None:
    clause_by_id = {
        "1:a": _clause("1:a", "exclusion"),
        "1:b": _clause("1:b", "coverage"),
        "1:c": _clause("1:c", "exclusion"),
    }
    row_hit = ScoredQuestion(
        question_id="q1",
        question_type="coverage_with_exclusion",
        document_id="1",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("1:a", "1:b"),
        retrieved=("1:a", "1:x"),
        recall={1: 0.5, 5: 0.5, 10: 0.5},
        mrr=1.0,
        ndcg=1.0,
    )
    row_miss = ScoredQuestion(
        question_id="q2",
        question_type="coverage_with_exclusion",
        document_id="1",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("1:c",),
        retrieved=("1:y",),
        recall={1: 0.0, 5: 0.0, 10: 0.0},
        mrr=0.0,
        ndcg=0.0,
    )

    result = compute_exclusion_clause_recall([row_hit, row_miss], clause_by_id, k=10)

    # "1:a" and "1:c" are exclusion clauses; "1:b" is coverage and excluded.
    assert result["total"] == 2
    # Only "1:a" is retrieved.
    assert result["hits"] == 1
    assert result["recall"] == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_exclusion_clause_recall_returns_none_without_exclusion_refs() -> None:
    clause_by_id = {"1:b": _clause("1:b", "coverage")}
    row = ScoredQuestion(
        question_id="q1",
        question_type="direct_lookup",
        document_id="1",
        product_line="CASCO",
        extraction_mode="text",
        reference_clause_ids=("1:b",),
        retrieved=("1:b",),
        recall={1: 1.0, 5: 1.0, 10: 1.0},
        mrr=1.0,
        ndcg=1.0,
    )

    result = compute_exclusion_clause_recall([row], clause_by_id, k=10)

    assert result["total"] == 0
    assert result["recall"] is None


@pytest.mark.unit
def test_render_markdown_report_includes_config_and_all_sections() -> None:
    metrics_row = {
        "n": 1.0,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr": 0.0,
        "ndcg@10": 0.0,
    }
    report = {
        "config": {
            "schema_version": "v1",
            "retriever_name": "random",
            "k_values": [1, 5, 10],
            "ndcg_k": 10,
            "golden_set_dir": "data/golden_set",
            "golden_set_question_count": 140,
            "corpus_path": "build/parsed_clauses.jsonl",
            "corpus_clause_count": 4925,
            "seed": 42,
            "run_at_utc": "2026-08-24T00:00:00+00:00",
        },
        "overall": metrics_row,
        "by_question_type": {
            "direct_lookup": metrics_row,
            "unanswerable": {"n": 23, "excluded_from_scoring": True},
        },
        "by_product_line": {"CASCO": metrics_row},
        "by_extraction_mode": {"text": metrics_row},
        "exclusion_clause_recall": {"k": 10, "hits": 1, "total": 27, "recall": 0.037},
        "foreign_document_rate": {"k": 10, "hits": 0, "total": 90, "rate": 0.0},
    }

    markdown = render_markdown_report(report)

    assert "random" in markdown
    assert "42" in markdown
    for header in (
        "## Run configuration",
        "## Overall metrics",
        "## By question_type",
        "## By product line",
        "## By extraction mode",
        "## Exclusion-clause recall",
        "## Foreign-document rate",
        "## Summary",
    ):
        assert header in markdown
    assert "Chunk corpus" not in markdown  # lexical block absent for `random`
    assert "Dense model" not in markdown  # dense block absent for `random`


def _chunk(chunk_id: str, clause_id: str, text: str) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": "v1",
            "chunk_id": chunk_id,
            "document_id": "1",
            "clause_id": clause_id,
            "source_clause_ids": [clause_id],
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": text,
            "display_text": text,
            "char_count": len(text),
            "rule": "single",
            "clause_type": "coverage",
            "type_source": "rule",
            "confidence": 1.0,
            "bundle_section": None,
            "source": "text",
            "susep_process": "123",
            "insurer": "Insurer",
            "cnpj": "123",
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2020",
        }
    )


@pytest.mark.unit
def test_parse_args_defaults_and_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py"])
    args = _parse_args()
    assert (args.retriever, args.filter_mode, args.fusion) == ("random", "none", "rrf")

    for retriever in ("lexical", "dense", "hybrid"):
        monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--retriever", retriever])
        assert _parse_args().retriever == retriever

    monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--retriever", "nope"])
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.unit
def test_parse_args_rejects_default_filter_with_the_random_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["eval_retrieval.py", "--retriever", "random", "--filter", "default"],
    )
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.unit
def test_build_lexical_retriever_ranks_an_exact_term_clause_first() -> None:
    # Thin integration: real analyzer + the real committed exception CSV.
    chunks = [
        _chunk("1:franquia", "1:franquia", "A franquia reduzida para vidros."),
        _chunk("1:reboque", "1:reboque", "Servico de reboque e guincho na pane."),
    ]
    retriever = build_lexical_retriever(chunks)
    assert retriever.retrieve("Qual o valor da franquia?", k=2)[0] == "1:franquia"


@pytest.mark.unit
def test_render_markdown_report_includes_the_lexical_config_block() -> None:
    metrics_row = {
        "n": 1.0,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr": 0.0,
        "ndcg@10": 0.0,
    }
    report = {
        "config": {
            "schema_version": "v2",
            "retriever_name": "lexical",
            "k_values": [1, 5, 10],
            "ndcg_k": 10,
            "golden_set_dir": "data/golden_set",
            "golden_set_question_count": 140,
            "corpus_path": "build/parsed_clauses.jsonl",
            "corpus_clause_count": 4925,
            "seed": None,
            "run_at_utc": "2026-08-28T00:00:00+00:00",
            "chunk_corpus_path": "build/chunks.jsonl",
            "chunk_corpus_chunk_count": 4540,
            "lexical_analyzer_version": "v1",
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "lexical_idf_variant": "lucene_plus_one",
            "lexical_index_text_field": "text",
            "stemming_exception_count": 5,
            "lexical_config_fingerprint": "deadbeefdeadbeef",
        },
        "overall": metrics_row,
        "by_question_type": {
            "direct_lookup": metrics_row,
            "unanswerable": {"n": 23, "excluded_from_scoring": True},
        },
        "by_product_line": {"CASCO": metrics_row},
        "by_extraction_mode": {"text": metrics_row},
        "exclusion_clause_recall": {"k": 10, "hits": 1, "total": 27, "recall": 0.037},
        "foreign_document_rate": {"k": 10, "hits": 0, "total": 90, "rate": 0.0},
    }

    markdown = render_markdown_report(report)

    assert "Chunk corpus (BM25-indexed): `build/chunks.jsonl` (4540 chunks)" in markdown
    assert "BM25 k1=1.5, b=0.75" in markdown
    assert "deadbeefdeadbeef" in markdown


@pytest.mark.unit
def test_parse_args_rejects_rerank_with_the_random_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["eval_retrieval.py", "--retriever", "random", "--rerank"]
    )
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.unit
def test_render_markdown_report_includes_the_reranker_config_block() -> None:
    metrics_row = {
        "n": 1.0,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr": 0.0,
        "ndcg@10": 0.0,
    }
    report = {
        "config": {
            "schema_version": "v4",
            "retriever_name": "hybrid",
            "k_values": [1, 5, 10],
            "ndcg_k": 10,
            "golden_set_dir": "data/golden_set",
            "golden_set_question_count": 140,
            "corpus_path": "build/parsed_clauses.jsonl",
            "corpus_clause_count": 4925,
            "seed": None,
            "run_at_utc": "2026-08-28T00:00:00+00:00",
            "filter_mode": "default",
            "reranker_model_id": "Alibaba-NLP/gte-multilingual-reranker-base",
            "reranker_model_revision": "8215cf04",
            "rerank_candidate_depth": 10,
            "reranker_config_fingerprint": "777c0503f1073d52",
        },
        "overall": metrics_row,
        "by_question_type": {
            "direct_lookup": metrics_row,
            "unanswerable": {"n": 23, "excluded_from_scoring": True},
        },
        "by_product_line": {"CASCO": metrics_row},
        "by_extraction_mode": {"text": metrics_row},
        "exclusion_clause_recall": {"k": 10, "hits": 25, "total": 27, "recall": 0.925},
        "foreign_document_rate": {"k": 10, "hits": 0, "total": 90, "rate": 0.0},
    }

    markdown = render_markdown_report(report)

    assert "gte-multilingual-reranker-base` @ `8215cf04`" in markdown
    assert "candidate depth 10" in markdown
    assert "777c0503f1073d52" in markdown


@pytest.mark.unit
def test_build_clause_text_map_joins_a_split_clause_in_chunk_index_order() -> None:
    base = _chunk("1:c#0", "1:c", "primeira parte")
    part0 = base.model_copy(update={"chunk_index": 0})
    part1 = base.model_copy(
        update={"chunk_id": "1:c#1", "chunk_index": 1, "text": "segunda parte"}
    )
    merged = _chunk("1:d", "1:d", "corpo de d").model_copy(
        update={"source_clause_ids": ["1:d", "1:e"]}
    )

    text_map = build_clause_text_map([part1, part0, merged])

    assert text_map["1:c"] == "primeira parte\n\nsegunda parte"
    # A short clause merged into a neighbour's chunk gets that chunk's text.
    assert text_map["1:e"] == "corpo de d"
