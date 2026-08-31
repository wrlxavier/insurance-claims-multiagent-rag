"""Dense retrieval + the metadata pre-filter against a real Postgres -- [M3-04].

Covers what only SQL + pgvector can prove: the ``<=>`` cosine search orders by
the stored vectors, the [RetrievalFilter] translates to a ``WHERE`` that pushes
down (a same-CNPJ decoy document is excluded by SUSEP process -- the M2-03
cross-document case), insurers are matched by CNPJ not name, a stacked filter
legitimately returns fewer than ``k``, and the ``bundle_section`` lenient/strict
NULL rule behaves. Synthetic vectors, built in-test -- the CI integration job
does not fetch the corpus. The embedder is a deterministic fake; no live model
call anywhere in the suite.
"""

from collections.abc import Sequence
from math import cos, sin

import pytest
from sqlalchemy.orm import Session

from domain.clause_classification import ClauseType
from infrastructure.database.chunk_repository import (
    upsert_chunks,
    write_chunk_embeddings,
)
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.dense_retriever import DenseRetriever
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS
from infrastructure.rag.retrieval_filter import RetrievalFilter

pytestmark = pytest.mark.integration

_TARGET = {"susep_process": "10000.111111/2024-11", "cnpj": "11111111000111"}
# Same CNPJ, different SUSEP process -- the M2-03 near-duplicate sibling.
_DECOY = {"susep_process": "10000.222222/2024-22", "cnpj": "11111111000111"}
# A different insurer sharing a brand name would still differ by CNPJ.
_OTHER_INSURER = {"susep_process": "10000.333333/2024-33", "cnpj": "99999999000199"}


def _planar(theta: float) -> list[float]:
    """A unit vector at angle ``theta`` in the axis 0/1 plane; farther = larger."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = cos(theta)
    vector[1] = sin(theta)
    return vector


class FakeEmbedder:
    """Embeds any text to the fixed query direction (axis 0)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_planar(0.0) for _ in texts]


def _record(
    chunk_id: str,
    partition: dict[str, str],
    *,
    clause_type: str = "coverage",
    bundle_section: str | None = None,
) -> ChunkRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": partition["susep_process"],
        "clause_id": chunk_id,
        "source_clause_ids": [chunk_id],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "",
        "text": "Texto da cláusula.",
        "display_text": "Texto da cláusula.",
        "char_count": 18,
        "rule": "single",
        "clause_type": clause_type,
        "type_source": "rule",
        "confidence": None,
        "bundle_section": bundle_section,
        "source": "text",
        "insurer": "Seguradora Compartilhada",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2024",
        **partition,
    }
    return ChunkRecord.model_validate(base)


def _load(
    session: Session,
    records: list[ChunkRecord],
    vectors: dict[str, list[float]],
) -> None:
    upsert_chunks(session, records)
    write_chunk_embeddings(session, vectors)
    session.flush()


def _retriever(session: Session) -> DenseRetriever:
    return DenseRetriever(session, FakeEmbedder())


def test_process_filter_excludes_the_same_cnpj_decoy_document(
    db_session: Session,
) -> None:
    # 12 decoys at the query direction (closest); 12 targets farther out.
    records: list[ChunkRecord] = []
    vectors: dict[str, list[float]] = {}
    for i in range(12):
        records.append(_record(f"d:{i}", _DECOY))
        vectors[f"d:{i}"] = _planar(0.01 * (i + 1))
        records.append(_record(f"t:{i}", _TARGET))
        vectors[f"t:{i}"] = _planar(0.5 + 0.01 * (i + 1))
    _load(db_session, records, vectors)
    retriever = _retriever(db_session)

    # Unfiltered: the decoys are nearer, so they win.
    assert all(cid.startswith("d:") for cid in retriever.retrieve("q", k=10))

    # Filtered to the target's SUSEP process: only target clauses, k of them,
    # in distance order.
    filt = RetrievalFilter.from_manifest_row(_TARGET)
    hits = retriever.retrieve("q", k=10, metadata_filter=filt)
    assert hits == [f"t:{i}" for i in range(10)]


def test_cnpj_filter_matches_by_cnpj_never_by_insurer_name(
    db_session: Session,
) -> None:
    records = [_record("a:0", _TARGET), _record("b:0", _OTHER_INSURER)]
    vectors = {"a:0": _planar(0.1), "b:0": _planar(0.2)}
    _load(db_session, records, vectors)

    # Both rows share the `insurer` string; only the CNPJ distinguishes them.
    hits = _retriever(db_session).retrieve(
        "q", k=10, metadata_filter=RetrievalFilter(cnpj=_TARGET["cnpj"])
    )
    assert hits == ["a:0"]


def test_exact_search_fills_k_when_the_partition_has_enough_rows(
    db_session: Session,
) -> None:
    records = [_record(f"t:{i}", _TARGET) for i in range(15)]
    vectors = {f"t:{i}": _planar(0.01 * (i + 1)) for i in range(15)}
    _load(db_session, records, vectors)

    hits = _retriever(db_session).retrieve(
        "q", k=10, metadata_filter=RetrievalFilter.from_manifest_row(_TARGET)
    )
    assert len(hits) == 10


def test_a_stacked_clause_type_filter_can_return_fewer_than_k(
    db_session: Session,
) -> None:
    records = [_record(f"cov:{i}", _TARGET) for i in range(10)]
    records += [_record(f"exc:{i}", _TARGET, clause_type="exclusion") for i in range(2)]
    vectors = {r.chunk_id: _planar(0.01 * (i + 1)) for i, r in enumerate(records)}
    _load(db_session, records, vectors)

    filt = RetrievalFilter(
        susep_process=_TARGET["susep_process"], clause_type=ClauseType.EXCLUSION
    )
    hits = _retriever(db_session).retrieve("q", k=10, metadata_filter=filt)
    assert sorted(hits) == ["exc:0", "exc:1"]


def test_bundle_section_lenient_keeps_null_chunks_strict_drops_them(
    db_session: Session,
) -> None:
    records = [
        _record("moto:0", _TARGET, bundle_section="Moto"),
        _record("null:0", _TARGET, bundle_section=None),
        _record("carga:0", _TARGET, bundle_section="Carga"),
    ]
    vectors = {
        "moto:0": _planar(0.1),
        "null:0": _planar(0.2),
        "carga:0": _planar(0.3),
    }
    _load(db_session, records, vectors)
    retriever = _retriever(db_session)

    lenient = retriever.retrieve(
        "q", k=10, metadata_filter=RetrievalFilter(bundle_section="Moto")
    )
    assert sorted(lenient) == ["moto:0", "null:0"]

    strict = retriever.retrieve(
        "q",
        k=10,
        metadata_filter=RetrievalFilter(bundle_section="Moto", strict_bundle=True),
    )
    assert strict == ["moto:0"]


def test_chunks_without_an_embedding_are_never_returned(db_session: Session) -> None:
    records = [_record("has_vec", _TARGET), _record("no_vec", _TARGET)]
    upsert_chunks(db_session, records)
    write_chunk_embeddings(db_session, {"has_vec": _planar(0.1)})
    db_session.flush()

    hits = _retriever(db_session).retrieve("q", k=10)
    assert hits == ["has_vec"]
