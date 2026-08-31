"""scripts/load_chunks.py against a real Postgres -- [M3-02].

The upsert write path itself is covered by tests/integration/test_chunk_embedding.py;
this pins the two things load_chunks adds -- the shared
``assert_chunk_table_ready`` guard against the migrated schema, and that a
straight re-load is a no-op on the row count.
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import (
    assert_chunk_table_ready,
    upsert_chunks,
)
from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord

pytestmark = pytest.mark.integration


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


def test_assert_chunk_table_ready_passes_on_the_migrated_schema(
    db_session: Session,
) -> None:
    assert_chunk_table_ready(db_session)  # must not raise


def test_reloading_the_same_corpus_keeps_the_row_count_stable(
    db_session: Session,
) -> None:
    records = [_record(f"1:{i}") for i in range(3)]

    upsert_chunks(db_session, records)
    db_session.flush()
    upsert_chunks(db_session, records)
    db_session.flush()

    count = db_session.execute(select(func.count()).select_from(ChunkRow)).scalar_one()
    assert count == 3
