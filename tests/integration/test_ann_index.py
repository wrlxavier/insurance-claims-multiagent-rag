"""The HNSW ANN index against a real Postgres -- [M3-02].

The committed proof for the [M3-02] DoD's filtered-search question: a metadata
pre-filter combined with an HNSW index scan can return fewer than ``k`` rows even
when ``>= k`` rows match the filter, and ``hnsw.iterative_scan = strict_order``
restores ``k``. Exact ``<=>`` search over the filtered partition never has this
problem. Synthetic vectors, built in-test -- the CI integration job does not
fetch the corpus.

The vectors are constructed so the shortfall is deterministic *and* the HNSW
graph stays navigable end to end: all 215 rows lie on a single arc from the
query direction (axis 0) toward axis 1, spaced by a small fixed angle. The 200
"decoy" rows occupy the near end of the arc (one ``(susep_process, cnpj)``
partition) and the 15 "target" rows the far end (another partition). The
``ef_search`` (40) nearest the query are therefore all decoys -- which the
target-partition filter discards -- but a strict iterative scan can walk the arc
all the way to the targets, because consecutive points are near-identical so the
graph is one connected chain rather than two islands. (Two orthogonal clusters
produce an HNSW graph the smaller cluster is simply unreachable from.)
"""

import json
from collections.abc import Sequence
from math import cos, sin

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import (
    upsert_chunks,
    write_chunk_embeddings,
)
from infrastructure.rag.ann_index import (
    INDEX_NAME,
    apply_ann_search_gucs,
    create_hnsw_index,
)
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.integration

K = 10
DECOY_COUNT = 200  # >= 5x the default hnsw.ef_search (40)
TARGET_COUNT = 15  # > K, so exact search over this partition always fills K
_ARC_STEP_RADIANS = 0.004  # angle between consecutive points on the arc

_DECOY_PARTITION = {"susep_process": "99999.111111/2011-11", "cnpj": "11111111000111"}
_TARGET_PARTITION = {"susep_process": "99999.222222/2022-22", "cnpj": "22222222000122"}
_TARGET_WHERE = "WHERE susep_process = :susep_process AND cnpj = :cnpj"

# The btree indexes over the filter columns: dropped inside the test transaction
# so the planner cannot satisfy a filtered `ORDER BY <=> LIMIT k` by reading the
# partition via btree and sorting it (which is exact and always fills k). That
# leaves the HNSW index as the access path -- the "the planner chose HNSW"
# branch the DoD is about. See docs/EMBEDDINGS.md.
_FILTER_INDEXES = (
    "ix_chunk_susep_process",
    "ix_chunk_cnpj",
    "ix_chunk_susep_process_cnpj",
)


def _arc_vector(rank: int) -> list[float]:
    """Unit vector at angle ``_ARC_STEP_RADIANS * (rank + 1)`` in the 0/1 plane.

    Monotonic in cosine distance from the query (axis 0): a higher rank is
    strictly farther. Consecutive ranks are near-identical, so the HNSW graph
    over the whole set is one navigable chain.
    """
    theta = _ARC_STEP_RADIANS * (rank + 1)
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = cos(theta)
    vector[1] = sin(theta)
    return vector


def _query_vector() -> list[float]:
    """Points straight along axis 0 -- the near end of the arc."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = 1.0
    return vector


def _decoy_vector(i: int) -> list[float]:
    """Arc point near the query -- ranks 0..DECOY_COUNT-1."""
    return _arc_vector(i)


def _target_vector(i: int) -> list[float]:
    """Arc point past every decoy -- ranks DECOY_COUNT..DECOY_COUNT+TARGET_COUNT-1."""
    return _arc_vector(DECOY_COUNT + i)


def _vector_literal(values: Sequence[float]) -> str:
    """Render a vector as pgvector's ``[v1,v2,...]`` text form."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def _record(chunk_id: str, partition: dict[str, str]) -> ChunkRecord:
    """A minimal valid ChunkRecord in the given ``(susep_process, cnpj)`` partition."""
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": "1",
        "clause_id": chunk_id,
        "source_clause_ids": [chunk_id],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "",
        "text": "Texto da cláusula.",
        "display_text": "Texto da cláusula.",
        "char_count": 18,
        "rule": "single",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": None,
        "source": "text",
        "insurer": "Seguradora",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2020",
        **partition,
    }
    return ChunkRecord.model_validate(base)


def _load(session: Session) -> None:
    """Upsert 200 decoy + 15 target rows and their embeddings; flush, no commit."""
    records: list[ChunkRecord] = []
    vectors: dict[str, list[float]] = {}
    for i in range(DECOY_COUNT):
        chunk_id = f"d:{i}"
        records.append(_record(chunk_id, _DECOY_PARTITION))
        vectors[chunk_id] = _decoy_vector(i)
    for i in range(TARGET_COUNT):
        chunk_id = f"t:{i}"
        records.append(_record(chunk_id, _TARGET_PARTITION))
        vectors[chunk_id] = _target_vector(i)
    upsert_chunks(session, records)
    write_chunk_embeddings(session, vectors)
    session.flush()


def _topk(session: Session, *, where: str = "") -> list[str]:
    """Run ``ORDER BY embedding <=> query LIMIT K`` with an optional filter."""
    params: dict[str, object] = {"q": _vector_literal(_query_vector()), "k": K}
    if where:
        params.update(_TARGET_PARTITION)
    sql = (
        f"SELECT chunk_id FROM chunk {where} "
        "ORDER BY embedding <=> CAST(:q AS halfvec) LIMIT :k"
    )
    return list(session.execute(text(sql), params).scalars())


def _plan_uses_index(plan: object, index_name: str) -> bool:
    """True if any node in an ``EXPLAIN (FORMAT JSON)`` plan is an index scan on it."""
    if isinstance(plan, str):
        plan = json.loads(plan)
    assert isinstance(plan, list)
    root = plan[0]
    assert isinstance(root, dict)
    stack: list[dict[str, object]] = [root["Plan"]]
    while stack:
        node = stack.pop()
        node_type = str(node.get("Node Type", ""))
        if node.get("Index Name") == index_name and "Index Scan" in node_type:
            return True
        children = node.get("Plans", [])
        if isinstance(children, list):
            stack.extend(node for node in children if isinstance(node, dict))
    return False


def test_hnsw_index_builds_and_serves_cosine_search(db_session: Session) -> None:
    _load(db_session)
    create_hnsw_index(db_session)

    index_names = set(
        db_session.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk'")
        ).scalars()
    )
    assert INDEX_NAME in index_names

    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    plan = db_session.execute(
        text(
            "EXPLAIN (FORMAT JSON) SELECT chunk_id FROM chunk "
            "ORDER BY embedding <=> CAST(:q AS halfvec) LIMIT :k"
        ),
        {"q": _vector_literal(_query_vector()), "k": K},
    ).scalar_one()
    assert _plan_uses_index(plan, INDEX_NAME)

    # The unfiltered nearest neighbours are all decoys.
    assert all(chunk_id.startswith("d:") for chunk_id in _topk(db_session))


def test_filtered_ann_can_return_fewer_than_k_and_iterative_scan_restores_it(
    db_session: Session,
) -> None:
    _load(db_session)
    create_hnsw_index(db_session)
    for index_name in _FILTER_INDEXES:
        db_session.execute(text(f"DROP INDEX {index_name}"))

    # Exact search over the 15-row target partition: seq-scans the partition and
    # sorts, so it always returns K.
    db_session.execute(text("SET LOCAL enable_indexscan = off"))
    db_session.execute(text("SET LOCAL enable_indexonlyscan = off"))
    exact = _topk(db_session, where=_TARGET_WHERE)
    assert len(exact) == K

    # ANN with no iterative scan: the ef_search nearest are all decoys, which the
    # target-partition filter discards -> fewer than K rows come back.
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    db_session.execute(text("SET LOCAL enable_indexonlyscan = on"))
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    apply_ann_search_gucs(db_session, iterative_scan="off")
    underfilled = _topk(db_session, where=_TARGET_WHERE)
    assert len(underfilled) < K

    # ANN with strict_order: the index keeps scanning until K post-filter matches
    # are found, in exact distance order.
    apply_ann_search_gucs(db_session, iterative_scan="strict_order")
    restored = _topk(db_session, where=_TARGET_WHERE)
    assert len(restored) == K
    assert set(restored) == set(exact)
