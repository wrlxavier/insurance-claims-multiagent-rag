"""Round-trip tests for the chunk corpus artifact (Parquet/JSONL/build manifest)."""

import json
from pathlib import Path

import pytest

from infrastructure.rag.chunk_artifact import (
    ChunksBuildManifest,
    read_chunks_jsonl,
    read_chunks_parquet,
    utc_now,
    write_chunks_jsonl,
    write_chunks_manifest,
    write_chunks_parquet,
)
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord


def _record(**overrides: object) -> ChunkRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": "1:2",
        "document_id": "1",
        "clause_id": "1:2",
        "source_clause_ids": ["1:2"],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "1. CONDIÇÕES GERAIS",
        "text": "1. CONDIÇÕES GERAIS\n2. COBERTURAS\n\nTexto da cobertura.",
        "display_text": "2. COBERTURAS\n\nTexto da cobertura.",
        "char_count": 56,
        "rule": "single",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": 1.0,
        "bundle_section": None,
        "source": "text",
        "susep_process": "15414900666201489",
        "insurer": "Bradesco Seguros",
        "cnpj": "12345678000199",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2019",
    }
    base.update(overrides)
    return ChunkRecord.model_validate(base)


@pytest.mark.unit
def test_write_then_read_parquet_round_trips(tmp_path: Path) -> None:
    records = [
        _record(),
        _record(chunk_id="1:2#0", clause_id="1:2", source_clause_ids=["1:2", "1:2.1"]),
    ]
    path = tmp_path / "chunks.parquet"

    write_chunks_parquet(records, path)
    restored = read_chunks_parquet(path)

    assert restored == records


@pytest.mark.unit
def test_write_then_read_jsonl_round_trips(tmp_path: Path) -> None:
    records = [_record(), _record(chunk_id="1:3", clause_id="1:3")]
    path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(records, path)
    restored = read_chunks_jsonl(path)

    assert restored == records
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


@pytest.mark.unit
def test_write_chunks_manifest_round_trips(tmp_path: Path) -> None:
    manifest = ChunksBuildManifest(
        schema_version=SCHEMA_VERSION,
        chunking_version="v1",
        clause_segmentation_version="v2",
        built_at_utc=utc_now(),
        chunk_counts_by_document={"1": 2, "2": 5},
        total_chunk_count=7,
    )
    path = tmp_path / "chunks_manifest.json"

    write_chunks_manifest(manifest, path)
    restored = ChunksBuildManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert restored == manifest
