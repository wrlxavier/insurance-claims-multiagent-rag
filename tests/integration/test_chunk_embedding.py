"""The embedding column and pipeline against a real Postgres -- [M3-02].

Covers what only SQL + pgvector can prove: vectors land in ``chunk.embedding``
and round-trip, the ``<=>`` cosine operator orders by them, a metadata re-upsert
does not wipe them, a re-run embeds nothing, and -- the DoD's explicit case --
an interrupted run resumes with the corpus complete, nothing duplicated or
skipped.

The embedder is a deterministic fake; no live model call anywhere in the suite.
The ANN index over ``embedding`` is a later [M3-02] slice, so "the index" here
is exact ``<=>`` ordering on the bare column.
"""

import hashlib
import math
from collections.abc import Sequence

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import (
    fetch_chunks_missing_embedding,
    upsert_chunks,
)
from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS
from infrastructure.rag.embedding_pipeline import embed_missing_chunks

pytestmark = pytest.mark.integration


def _unit_vector(text: str) -> list[float]:
    """Deterministic, L2-normalised pseudo-embedding of the pinned width."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] + 1 for i in range(EMBEDDING_DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


class FakeEmbedder:
    """Records every ``embed`` call; deterministic vector per text."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [_unit_vector(text) for text in texts]


class EmbedderFailingAfter:
    """Delegates for the first ``ok_calls`` batches, then raises every time."""

    def __init__(self, *, ok_calls: int) -> None:
        self._remaining_ok = ok_calls
        self._inner = FakeEmbedder()
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._remaining_ok <= 0:
            raise RuntimeError("embedder down")
        self._remaining_ok -= 1
        self.seen.extend(texts)
        return self._inner.embed(texts)


def _record(**overrides: object) -> ChunkRecord:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": "1:1",
        "document_id": "1",
        "clause_id": "1:1",
        "source_clause_ids": ["1:1"],
        "chunk_index": 0,
        "chunk_count": 1,
        "parent_path": "",
        "text": "Texto da cláusula de cobertura.",
        "display_text": "Texto da cláusula de cobertura.",
        "char_count": 31,
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


def _load(session: Session, count: int) -> list[ChunkRecord]:
    records = [
        _record(chunk_id=f"1:{i}", clause_id=f"1:{i}", text=f"cláusula número {i}")
        for i in range(count)
    ]
    upsert_chunks(session, records)
    session.commit()
    return records


def _embedding_of(session: Session, chunk_id: str) -> list[float] | None:
    return session.execute(
        select(ChunkRow.embedding).where(ChunkRow.chunk_id == chunk_id)
    ).scalar_one()


def _count(session: Session) -> int:
    return session.execute(select(func.count()).select_from(ChunkRow)).scalar_one()


def test_write_path_populates_and_round_trips(db_session: Session) -> None:
    records = _load(db_session, 3)

    run = embed_missing_chunks(db_session, FakeEmbedder(), batch_size=2)

    assert run.embedded == 3
    assert run.already_present == 0
    assert run.batches == 2
    for record in records:
        stored = _embedding_of(db_session, record.chunk_id)
        assert stored is not None
        assert len(stored) == EMBEDDING_DIMENSIONS
        # halfvec is half-precision -- compare with tolerance.
        assert stored == pytest.approx(_unit_vector(record.text), rel=1e-2)


def test_cosine_operator_orders_by_the_stored_vectors(db_session: Session) -> None:
    _load(db_session, 3)
    embed_missing_chunks(db_session, FakeEmbedder(), batch_size=10)

    query = _unit_vector("cláusula número 1")
    ordered = (
        db_session.execute(
            select(ChunkRow.chunk_id).order_by(
                ChunkRow.embedding.cosine_distance(query)
            )
        )
        .scalars()
        .all()
    )

    # The exact match ranks first; an ordering only a working `<=>` produces.
    assert ordered[0] == "1:1"
    assert set(ordered) == {"1:0", "1:1", "1:2"}


def test_rerun_embeds_nothing_and_leaves_vectors_untouched(
    db_session: Session,
) -> None:
    _load(db_session, 3)
    embed_missing_chunks(db_session, FakeEmbedder(), batch_size=2)
    before = _embedding_of(db_session, "1:1")

    second = FakeEmbedder()
    run = embed_missing_chunks(db_session, second, batch_size=2)

    assert run.embedded == 0
    assert run.already_present == 3
    assert second.seen == []
    assert _embedding_of(db_session, "1:1") == pytest.approx(before)


def test_metadata_upsert_preserves_existing_embeddings(db_session: Session) -> None:
    records = _load(db_session, 2)
    embed_missing_chunks(db_session, FakeEmbedder(), batch_size=10)
    before = _embedding_of(db_session, "1:0")

    # A metadata refresh over the same corpus (M3-01 re-run) must not null the
    # vector column -- `upsert_chunks` leaves `embedding` out of its update set.
    upsert_chunks(db_session, records)
    db_session.commit()

    assert _embedding_of(db_session, "1:0") == pytest.approx(before)
    assert fetch_chunks_missing_embedding(db_session) == []


def test_interrupt_mid_corpus_then_resume(db_session: Session) -> None:
    _load(db_session, 5)

    # batch_size 2 over 5 rows = 3 batches; fail from the second on. The retry
    # exhausts (no real sleeps) and the exception propagates out.
    with pytest.raises(RuntimeError, match="embedder down"):
        embed_missing_chunks(
            db_session,
            EmbedderFailingAfter(ok_calls=1),
            batch_size=2,
            sleep=lambda _: None,
        )

    done = [cid for cid in ("1:0", "1:1") if _embedding_of(db_session, cid)]
    todo = [
        cid for cid in ("1:2", "1:3", "1:4") if _embedding_of(db_session, cid) is None
    ]
    assert done == ["1:0", "1:1"]  # first committed batch survived
    assert todo == ["1:2", "1:3", "1:4"]  # the rest untouched
    assert _count(db_session) == 5  # nothing duplicated

    resume = FakeEmbedder()
    run = embed_missing_chunks(db_session, resume, batch_size=2)

    assert run.embedded == 3
    assert run.already_present == 2
    # The resume pass only ever saw the three still-missing chunks.
    assert sorted(resume.seen) == [f"cláusula número {i}" for i in (2, 3, 4)]
    assert fetch_chunks_missing_embedding(db_session) == []
    assert _count(db_session) == 5
    for i in range(5):
        assert _embedding_of(db_session, f"1:{i}") is not None
