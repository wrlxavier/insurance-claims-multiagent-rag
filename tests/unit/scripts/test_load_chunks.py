"""Unit tests for scripts/load_chunks.py -- [M3-02]."""

from pathlib import Path

import pytest
from scripts.load_chunks import load_records

from infrastructure.rag.chunk_artifact import write_chunks_jsonl
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord


def _record(chunk_id: str) -> ChunkRecord:
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
            "text": f"texto {chunk_id}",
            "display_text": f"texto {chunk_id}",
            "char_count": 10,
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
def test_load_records_missing_file_points_at_build_chunks(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make build-chunks"):
        load_records(tmp_path / "chunks.jsonl")


@pytest.mark.unit
def test_load_records_reads_every_row(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl([_record("1:0"), _record("1:1")], path)

    records = load_records(path)

    assert [record.chunk_id for record in records] == ["1:0", "1:1"]
