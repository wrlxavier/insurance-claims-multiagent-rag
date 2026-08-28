"""Schema-level guarantees for the ``chunk`` table model -- [M3-02]."""

import re

import pytest
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import CheckConstraint

from domain.chunk import ChunkRule
from domain.clause_classification import ClauseType, TypeSource
from infrastructure.database.models import ChunkRow
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

# A properly-typed Table handle (``ChunkRow.__table__`` is typed as a bare
# FromClause).
_CHUNK_TABLE = ChunkRow.metadata.tables["chunk"]

_EXPECTED_INDEXES = {
    "ix_chunk_clause_type",
    "ix_chunk_bundle_section",
    "ix_chunk_susep_process",
    "ix_chunk_cnpj",
    "ix_chunk_product_line",
    "ix_chunk_susep_process_cnpj",
}


def _check_constraint_values(name: str) -> set[str]:
    for constraint in _CHUNK_TABLE.constraints:
        if isinstance(constraint, CheckConstraint) and str(constraint.name) == name:
            return set(re.findall(r"'([^']*)'", str(constraint.sqltext)))
    raise AssertionError(f"no CHECK constraint named {name}")


@pytest.mark.unit
def test_table_name_and_primary_key() -> None:
    assert ChunkRow.__tablename__ == "chunk"
    assert [column.name for column in _CHUNK_TABLE.primary_key.columns] == ["chunk_id"]


@pytest.mark.unit
def test_bundle_section_is_genuinely_nullable_with_no_sentinel() -> None:
    column = _CHUNK_TABLE.c.bundle_section

    assert column.nullable is True
    # A server_default or a Python default would be a sentinel by another name;
    # M3-04's strict `WHERE bundle_section = :x` must be able to exclude these
    # rows in plain SQL.
    assert column.server_default is None
    assert column.default is None


@pytest.mark.unit
def test_confidence_is_nullable() -> None:
    assert _CHUNK_TABLE.c.confidence.nullable is True


@pytest.mark.unit
def test_provenance_and_attribution_columns_are_present_and_not_null() -> None:
    required = [
        "document_id",
        "clause_id",
        "susep_process",
        "insurer",
        "cnpj",
        "product_line",
        "indemnity_regime",
        "filing_year",
        "source",
        "type_source",
    ]
    for name in required:
        assert _CHUNK_TABLE.c[name].nullable is False


@pytest.mark.unit
def test_embedded_and_display_text_are_separate_columns() -> None:
    assert "embedded_text" in _CHUNK_TABLE.c
    assert "display_text" in _CHUNK_TABLE.c


@pytest.mark.unit
def test_embedding_is_a_nullable_halfvec_of_the_pinned_dimension() -> None:
    column = _CHUNK_TABLE.c.embedding

    # Width comes from the pinned model contract, not a literal here -- the
    # migration's literal `768` is caught by `alembic check` if it drifts.
    assert isinstance(column.type, HALFVEC)
    assert column.type.dim == EMBEDDING_DIMENSIONS
    # An un-embedded chunk is `NULL`; that is the pipeline's resumable cursor,
    # so no server_default / Python default may fill it in.
    assert column.nullable is True
    assert column.server_default is None
    assert column.default is None


@pytest.mark.unit
def test_embedding_column_is_not_indexed_yet() -> None:
    # The ANN index over `embedding` is a later M3-02 slice; until then exact
    # `<=>` ordering runs on the bare column.
    indexed_columns = {
        column.name for index in _CHUNK_TABLE.indexes for column in index.columns
    }
    assert "embedding" not in indexed_columns


@pytest.mark.unit
def test_check_constraints_match_the_domain_enums() -> None:
    assert _check_constraint_values("ck_chunk_rule_valid") == {
        member.value for member in ChunkRule
    }
    assert _check_constraint_values("ck_chunk_clause_type_valid") == {
        member.value for member in ClauseType
    }
    assert _check_constraint_values("ck_chunk_type_source_valid") == {
        member.value for member in TypeSource
    }
    assert _check_constraint_values("ck_chunk_source_valid") == {"text", "ocr"}


@pytest.mark.unit
def test_filter_indexes_are_declared() -> None:
    declared = {str(index.name) for index in _CHUNK_TABLE.indexes}
    assert declared == _EXPECTED_INDEXES


@pytest.mark.unit
def test_insurer_is_not_indexed() -> None:
    # M3-04 filters insurers by CNPJ, never by name.
    indexed_columns = {
        column.name for index in _CHUNK_TABLE.indexes for column in index.columns
    }
    assert "insurer" not in indexed_columns
