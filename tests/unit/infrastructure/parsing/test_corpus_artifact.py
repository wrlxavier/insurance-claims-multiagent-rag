"""Round-trip tests for the final corpus artifact (Parquet/JSONL/build manifest)."""

import json
from pathlib import Path

import pytest

from infrastructure.parsing.clause_schema import SCHEMA_VERSION, ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import (
    BuildManifest,
    read_parsed_clauses_jsonl,
    read_parsed_clauses_parquet,
    utc_now,
    write_build_manifest,
    write_parsed_clauses_jsonl,
    write_parsed_clauses_parquet,
)


def _record(**overrides: object) -> ParsedClauseRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "clause_id": "1:2",
        "document_id": "1",
        "parent_id": None,
        "path": "2",
        "title": "2. COBERTURAS",
        "text": "Texto da cobertura.",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": 1.0,
        "bundle_section": None,
        "page_start": 3,
        "page_end": 4,
        "source": "text",
        "susep_process": "15414900666201489",
        "insurer": "Bradesco Seguros",
        "cnpj": "12345678000199",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2019",
    }
    base.update(overrides)
    return ParsedClauseRecord.model_validate(base)


@pytest.mark.unit
def test_write_then_read_parquet_round_trips(tmp_path: Path) -> None:
    records = [_record(), _record(clause_id="1:3", path="3", parent_id="1:2")]
    path = tmp_path / "parsed_clauses.parquet"

    write_parsed_clauses_parquet(records, path)
    restored = read_parsed_clauses_parquet(path)

    assert restored == records


@pytest.mark.unit
def test_write_then_read_jsonl_round_trips(tmp_path: Path) -> None:
    records = [_record(), _record(clause_id="1:3", path="3", parent_id="1:2")]
    path = tmp_path / "parsed_clauses.jsonl"

    write_parsed_clauses_jsonl(records, path)
    restored = read_parsed_clauses_jsonl(path)

    assert restored == records
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


@pytest.mark.unit
def test_write_build_manifest_round_trips(tmp_path: Path) -> None:
    manifest = BuildManifest(
        schema_version=SCHEMA_VERSION,
        clause_segmentation_version="v2",
        boilerplate_removal_version="v1",
        llm_classification_enabled=False,
        built_at_utc=utc_now(),
        clause_counts_by_document={"1": 2, "2": 5},
        total_clause_count=7,
    )
    path = tmp_path / "manifest.json"

    write_build_manifest(manifest, path)
    restored = BuildManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert restored == manifest
