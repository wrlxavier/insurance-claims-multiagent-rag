"""Tests for the retrieval evaluation script [M2-06]."""

from pathlib import Path

import pytest
from scripts.eval_retrieval import (
    K_VALUES,
    NDCG_K,
    ScoredQuestion,
    _parse_args,
    aggregate,
    build_lexical_retriever,
    compute_exclusion_clause_recall,
    evaluate_questions,
    load_golden_questions,
    render_markdown_report,
    resolve_source,
)

from infrastructure.evaluation.golden_set_schema import GoldenQuestion
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.rag.chunk_schema import ChunkRecord


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

    def retrieve(self, question: str, *, k: int) -> list[str]:
        del question
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
    assert row.product_line == "CASCO"
    assert row.extraction_mode == "text"
    assert row.recall == {1: 1.0, 5: 1.0, 10: 1.0}
    assert row.mrr == 1.0
    assert row.ndcg == 1.0


@pytest.mark.unit
def test_aggregate_computes_mean_across_rows() -> None:
    row_hit = ScoredQuestion(
        question_id="q1",
        question_type="direct_lookup",
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
        "## Summary",
    ):
        assert header in markdown
    assert "Chunk corpus" not in markdown  # lexical block absent for `random`


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
def test_parse_args_defaults_to_random_and_accepts_lexical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py"])
    assert _parse_args().retriever == "random"
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--retriever", "lexical"])
    assert _parse_args().retriever == "lexical"
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--retriever", "nope"])
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
    }

    markdown = render_markdown_report(report)

    assert "Chunk corpus (BM25-indexed): `build/chunks.jsonl` (4540 chunks)" in markdown
    assert "BM25 k1=1.5, b=0.75" in markdown
    assert "deadbeefdeadbeef" in markdown
