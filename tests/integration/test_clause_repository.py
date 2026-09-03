"""The SQLAlchemy ``ClauseRepository`` against a real Postgres -- [M5-03].

The port reads the existing ``chunk`` table. What only SQL proves: a clause is
reassembled from every chunk whose ``source_clause_ids`` array contains its id
(the ``= ANY`` match), a split clause rejoins its ``display_text`` in
``chunk_index`` order, a merged sibling id resolves to the same clause,
``get_many`` preserves request order and omits gaps, and ``list_for_policy``
filters by SUSEP process.
"""

import pytest
from sqlalchemy.orm import Session

from domain.clause_classification import ClauseType
from domain.susep_process import SusepProcess
from infrastructure.database.chunk_repository import upsert_chunks
from infrastructure.database.clause_repository import SqlAlchemyClauseRepository
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord

pytestmark = pytest.mark.integration

_SUSEP_A = "15414.900666/2014-89"
_SUSEP_B = "15414.610650/2024-59"


def _chunk(**overrides: object) -> ChunkRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": "docA:2.1",
        "document_id": "docA",
        "clause_id": "docA:2.1",
        "source_clause_ids": ["docA:2.1"],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "2. COBERTURAS",
        "text": "2. COBERTURAS\n\nTexto.",
        "display_text": "2.1 Colisao\n\nTexto da cobertura.",
        "char_count": 21,
        "rule": "single",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": None,
        "source": "text",
        "susep_process": _SUSEP_A,
        "insurer": "Bradesco Seguros",
        "cnpj": "12345678000199",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2019",
    }
    base.update(overrides)
    return ChunkRecord.model_validate(base)


def _solo(clause_id: str, **overrides: object) -> ChunkRecord:
    """A one-chunk clause whose id is its own anchor and only source id."""
    return _chunk(
        chunk_id=clause_id,
        clause_id=clause_id,
        source_clause_ids=[clause_id],
        **overrides,
    )


def _load(session: Session, *records: ChunkRecord) -> None:
    upsert_chunks(session, list(records))
    session.commit()


def test_get_returns_a_single_chunk_clause(db_session: Session) -> None:
    _load(db_session, _chunk())

    clause = SqlAlchemyClauseRepository(db_session).get("docA:2.1")

    assert clause is not None
    assert clause.clause_id == "docA:2.1"
    assert clause.document_id == "docA"
    assert clause.susep_process == SusepProcess(_SUSEP_A)
    assert clause.clause_type is ClauseType.COVERAGE
    assert clause.text == "2.1 Colisao\n\nTexto da cobertura."


def test_get_rejoins_a_split_clause_in_chunk_order(db_session: Session) -> None:
    _load(
        db_session,
        _chunk(
            chunk_id="docA:3#1",
            clause_id="docA:3",
            source_clause_ids=["docA:3"],
            chunk_index=1,
            chunk_count=2,
            display_text="segunda parte",
        ),
        _chunk(
            chunk_id="docA:3#0",
            clause_id="docA:3",
            source_clause_ids=["docA:3"],
            chunk_index=0,
            chunk_count=2,
            display_text="primeira parte",
        ),
    )

    clause = SqlAlchemyClauseRepository(db_session).get("docA:3")

    assert clause is not None
    assert clause.text == "primeira parte\n\nsegunda parte"


def test_get_resolves_a_merged_sibling_id(db_session: Session) -> None:
    _load(
        db_session,
        _chunk(
            chunk_id="docA:4",
            clause_id="docA:4",
            source_clause_ids=["docA:4", "docA:4.1"],
            display_text="clausula 4 com o item 4.1 dobrado",
        ),
    )

    repo = SqlAlchemyClauseRepository(db_session)
    anchor = repo.get("docA:4")
    sibling = repo.get("docA:4.1")
    assert anchor is not None and sibling is not None
    # Same underlying clause -- only the id the caller asked by differs.
    assert anchor.clause_id == "docA:4"
    assert sibling.clause_id == "docA:4.1"
    assert (anchor.text, anchor.document_id) == (sibling.text, sibling.document_id)


def test_get_unknown_clause_returns_none(db_session: Session) -> None:
    _load(db_session, _chunk())
    assert SqlAlchemyClauseRepository(db_session).get("docA:9.9") is None


def test_get_many_preserves_order_and_omits_gaps(db_session: Session) -> None:
    _load(db_session, _solo("docA:2.1"), _solo("docA:2.2"))

    clauses = SqlAlchemyClauseRepository(db_session).get_many(
        ["docA:2.2", "docA:missing", "docA:2.1"]
    )

    assert [c.clause_id for c in clauses] == ["docA:2.2", "docA:2.1"]


def test_list_for_policy_filters_by_susep_process(db_session: Session) -> None:
    _load(
        db_session,
        _solo("docA:2.1"),
        _solo("docA:2.2"),
        _solo("docB:1", document_id="docB", susep_process=_SUSEP_B),
    )

    clauses = SqlAlchemyClauseRepository(db_session).list_for_policy(
        SusepProcess(_SUSEP_A)
    )

    assert {c.clause_id for c in clauses} == {"docA:2.1", "docA:2.2"}
