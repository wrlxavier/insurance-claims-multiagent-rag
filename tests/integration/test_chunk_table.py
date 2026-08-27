"""The chunk table against a real Postgres -- [M3-02].

Covers the three schema-half DoD items that only SQL can prove: the write path
is idempotent (upsert by ``chunk_id``), ``bundle_section`` is genuinely
nullable so a strict filter excludes unknown-bundle rows, and the metadata
filter indexes exist.
"""

import pytest
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import _row_values, upsert_chunks
from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord

pytestmark = pytest.mark.integration

_EXPECTED_INDEXES = {
    "ix_chunk_clause_type",
    "ix_chunk_bundle_section",
    "ix_chunk_susep_process",
    "ix_chunk_cnpj",
    "ix_chunk_product_line",
    "ix_chunk_susep_process_cnpj",
}


def _record(**overrides: object) -> ChunkRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": "1:2",
        "document_id": "1",
        "clause_id": "1:2",
        "source_clause_ids": ["1:2", "1:2.1"],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "1. CONDIÇÕES GERAIS",
        "text": "1. CONDIÇÕES GERAIS\n2. COBERTURAS\n\nTexto da cobertura.",
        "display_text": "2. COBERTURAS\n\nTexto da cobertura.",
        "char_count": 53,
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
    base.update(overrides)
    return ChunkRecord.model_validate(base)


def _count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(ChunkRow)).scalar_one()


def test_chunk_rows_round_trip(db_session: Session) -> None:
    written = upsert_chunks(db_session, [_record()])
    db_session.commit()

    assert written == 1
    row = db_session.execute(
        select(ChunkRow).where(ChunkRow.chunk_id == "1:2")
    ).scalar_one()
    assert row.source_clause_ids == ["1:2", "1:2.1"]
    assert row.embedded_text == f"{row.parent_path}\n{row.display_text}"
    assert row.bundle_section is None
    assert row.confidence is None
    assert row.source == "text"


def test_upsert_is_idempotent(db_session: Session) -> None:
    records = [_record(chunk_id="1:2"), _record(chunk_id="1:3", clause_id="1:3")]

    upsert_chunks(db_session, records)
    db_session.commit()
    upsert_chunks(db_session, records)
    db_session.commit()

    assert _count(db_session) == 2

    # A changed chunk keeps its id and overwrites in place -- no duplicate, no
    # wipe required.
    upsert_chunks(
        db_session,
        [_record(chunk_id="1:2", display_text="Texto corrigido.", confidence=0.5)],
    )
    db_session.commit()

    assert _count(db_session) == 2
    updated = db_session.execute(
        select(ChunkRow).where(ChunkRow.chunk_id == "1:2")
    ).scalar_one()
    assert updated.display_text == "Texto corrigido."
    assert updated.confidence == 0.5


def test_strict_bundle_section_filter_excludes_null_rows(db_session: Session) -> None:
    upsert_chunks(
        db_session,
        [
            _record(chunk_id="10:a", bundle_section="Motocicletas"),
            _record(chunk_id="10:b", bundle_section="Motocicletas"),
            _record(chunk_id="10:c", bundle_section=None),
        ],
    )
    db_session.commit()

    strict = (
        db_session.execute(
            text("SELECT chunk_id FROM chunk WHERE bundle_section = :section"),
            {"section": "Motocicletas"},
        )
        .scalars()
        .all()
    )
    with_fallback = (
        db_session.execute(
            text(
                "SELECT chunk_id FROM chunk "
                "WHERE bundle_section = :section OR bundle_section IS NULL"
            ),
            {"section": "Motocicletas"},
        )
        .scalars()
        .all()
    )

    assert set(strict) == {"10:a", "10:b"}
    assert set(with_fallback) == {"10:a", "10:b", "10:c"}


def test_filter_indexes_exist(db_session: Session) -> None:
    index_names = set(
        db_session.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk'")
        ).scalars()
    )

    assert _EXPECTED_INDEXES <= index_names


def test_duplicate_plain_insert_violates_primary_key(db_session: Session) -> None:
    upsert_chunks(db_session, [_record(chunk_id="1:2")])
    db_session.commit()

    with pytest.raises(IntegrityError):
        db_session.execute(insert(ChunkRow), [_row_values(_record(chunk_id="1:2"))])
        db_session.flush()
