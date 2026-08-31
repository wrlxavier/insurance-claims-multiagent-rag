"""Unit tests for scripts/embed_chunks.py -- [M3-02].

The dollar-cost and report-shape logic, with no model and no database. The real
cold run is a manual step (like scripts/benchmark_ann_index.py's real-DB run).
"""

from dataclasses import dataclass

import pytest
from scripts.embed_chunks import (
    build_cost_note,
    build_report,
    dry_run_summary,
    render_markdown_report,
)

from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.embedding_config import EMBEDDING_MODEL_ID
from infrastructure.rag.embedding_pipeline import EmbeddingRun

_VERSIONS = {"pgvector_version": "pgvector 0.8.6", "postgres_version": "PostgreSQL 17"}


@dataclass
class _FakeEncoding:
    ids: list[int]


class _FakeTokenizer:
    """One token per whitespace group -- duck-types ``tokenizers.Tokenizer``."""

    def encode(self, sequence: str) -> _FakeEncoding:
        return _FakeEncoding(ids=list(range(len(sequence.split()))))


def _record(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "chunk_id": chunk_id,
            "document_id": "1",
            "clause_id": chunk_id,
            "source_clause_ids": [chunk_id],
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": text,
            "display_text": text,
            "char_count": len(text),
            "rule": "single",
            "clause_type": "coverage",
            "type_source": "rule",
            "confidence": None,
            "bundle_section": None,
            "source": "text",
            "susep_process": "15414.900666/2014-89",
            "insurer": "Bradesco Seguros",
            "cnpj": "12345678000199",
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2019",
        }
    )


@pytest.mark.unit
def test_dry_run_summary_is_offline_and_zero_dollars() -> None:
    records = [_record("1:0", "uma duas tres"), _record("1:1", "quatro")]

    summary = dry_run_summary(records, _FakeTokenizer())

    assert "2 chunks" in summary
    assert "4 passage tokens" in summary
    assert "$0.00" in summary
    assert EMBEDDING_MODEL_ID in summary


@pytest.mark.unit
def test_cost_note_is_dated_and_states_zero_dollars() -> None:
    note = build_cost_note(
        run_date="2026-08-28",
        chunks_embedded=4540,
        wall_clock_seconds=600.0,
        processor="AMD Ryzen 5 5600H",
    )

    assert "2026-08-28" in note
    assert "$0.00" in note
    assert EMBEDDING_MODEL_ID in note
    assert "no price constant" in note
    assert "~10.0 min" in note


@pytest.mark.unit
def test_cost_note_when_nothing_was_embedded() -> None:
    note = build_cost_note(
        run_date="2026-08-28",
        chunks_embedded=0,
        wall_clock_seconds=0.0,
        processor="x",
    )

    assert "$0.00" in note
    assert "clear" in note.lower()
    assert "min for a" not in note


@pytest.mark.unit
def test_build_report_dollar_cost_is_zero_and_tokens_reconcile() -> None:
    report = build_report(
        run=EmbeddingRun(embedded=4, already_present=0, batches=2),
        wall_clock_seconds=8.0,
        token_counts=[10, 20, 30, 40],
        chunk_count=4,
        cache_hits=0,
        cache_misses=4,
        batch_size=2,
        device="cpu",
        versions=_VERSIONS,
    )

    assert report.dollar_cost_usd == 0.0
    assert report.total_passage_tokens == 100
    assert report.max_passage_tokens == 40
    assert report.cache_hits == 0
    assert report.cache_misses == 4
    assert report.chunks_embedded == 4
    assert report.tokens_per_second == pytest.approx(12.5)


@pytest.mark.unit
def test_markdown_report_has_the_headline_sections() -> None:
    md = render_markdown_report(
        build_report(
            run=EmbeddingRun(embedded=4540, already_present=0, batches=71),
            wall_clock_seconds=600.0,
            token_counts=[260] * 4540,
            chunk_count=4540,
            cache_hits=0,
            cache_misses=4540,
            batch_size=64,
            device="cpu",
            versions=_VERSIONS,
        )
    )

    assert "# Corpus embedding cost" in md
    assert "**Cold run**" in md
    assert "$0.00" in md
    assert "## Run configuration" in md
    assert "0 / 4540" in md
