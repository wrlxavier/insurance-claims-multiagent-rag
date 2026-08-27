"""Write-path construction guarantees for the chunk repository -- [M3-02]."""

import pytest

from infrastructure.database.chunk_repository import _UPDATE_COLUMNS, _row_values
from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord

_CHUNK_TABLE = ChunkRow.metadata.tables["chunk"]


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
        "text": "1. CONDIÇÕES GERAIS\n2. COBERTURAS\n\nTexto.",
        "display_text": "2. COBERTURAS\n\nTexto.",
        "char_count": 42,
        "rule": "single",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": 1.0,
        "bundle_section": None,
        "source": "text",
        "susep_process": "15414.900666/2014-89",
        "insurer": "Bradesco Seguros",
        "cnpj": "12345678000199",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2019",
    }
    base.update(overrides)
    return ChunkRecord.model_validate(base)


@pytest.mark.unit
def test_row_values_renames_text_and_preserves_nulls() -> None:
    values = _row_values(_record(confidence=None))

    assert values["embedded_text"] == "1. CONDIÇÕES GERAIS\n2. COBERTURAS\n\nTexto."
    assert "text" not in values
    assert values["display_text"] == "2. COBERTURAS\n\nTexto."
    assert values["bundle_section"] is None
    assert values["confidence"] is None
    assert values["source_clause_ids"] == ["1:2"]


@pytest.mark.unit
def test_row_values_keys_match_the_table_columns_exactly() -> None:
    assert set(_row_values(_record())) == {
        column.name for column in _CHUNK_TABLE.columns
    }


@pytest.mark.unit
def test_upsert_refreshes_every_non_key_column() -> None:
    # A column left out of the ON CONFLICT SET list would silently keep its
    # stale value on a re-run.
    non_key_columns = {
        column.name for column in _CHUNK_TABLE.columns if column.name != "chunk_id"
    }
    assert set(_UPDATE_COLUMNS) == non_key_columns
